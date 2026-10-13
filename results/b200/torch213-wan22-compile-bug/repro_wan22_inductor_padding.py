"""Standalone repro: torch.compile (Inductor) miscompiles a slice of the Wan 2.2 VAE decoder
(diffusers AutoencoderKLWan, is_residual=True) when comprehensive stride padding is on.

Only torch is needed (no diffusers, no weights): this is the FX graph torch's minifier left
of the decoder's mid block -> up_blocks[0] -> up_blocks[1] on one steady 480x832 latent
(82 nodes), with readable names and random inputs.

What goes wrong (torch 2.13.0+cu130, B200): ``x = up0_out`` is written channels-last (it is
fused in place into the channels-last output of the upsample conv2d), but the kernel that
reads ``x_copy = x.clone()`` through DupUp3D (repeat_interleave -> view -> permute ->
contiguous -> view) indexes it with the padded channels-first strides Inductor chose for the
clone (C stride 12544, T stride 6272 instead of 12480 / 6240 / or channels-last 1). The
result is wrong, and the largest offset read is past the end of the buffer.

  python repro_wan22_inductor_padding.py
exits 1 if the default compile differs from eager while comprehensive_padding=False matches.
"""
import math
import sys

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F

C, H, W = 1024, 30, 52  # Wan 2.2 decoder width at its 1/16 latent grid (480x832 output)


def dup_up(x, out_channels, factor_t, factor_s):
    """diffusers DupUp3D.forward (first_chunk=False)."""
    repeats = out_channels * factor_t * factor_s * factor_s // x.size(1)
    x = x.repeat_interleave(repeats, dim=1)
    x = x.view(x.size(0), out_channels, factor_t, factor_s, factor_s, x.size(2), x.size(3), x.size(4))
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.view(x.size(0), out_channels, x.size(2) * factor_t, x.size(4) * factor_s, x.size(6) * factor_s)


def rms_silu(x, gamma):
    """WanRMS_norm (channel_first, no bias) then SiLU."""
    return F.silu(F.normalize(x.float(), dim=1).to(x.dtype) * x.size(1) ** 0.5 * gamma + 0.0)


def decoder_slice(w_mid, b_mid, w_rs0, b_rs0, g1, w1, b1, g2, w2, b2, w_tc, b_tc, w_rs1, b_rs1,
                  mid_residual, mid_conv_in, cache1, cache_tc, up0_frames, cache0, res1_out, res2_out, tc_cache_frame):
    # mid block, last conv + residual: the mid block output
    mid_out = F.conv3d(mid_conv_in, w_mid, b_mid) + mid_residual                 # [1, C, 1, 30, 52]
    # up_blocks[0]: (spatial upsample conv2d on 2 frames) + DupUp3D shortcut of its input
    x_copy0 = mid_out.clone()
    y = F.conv2d(up0_frames, w_rs0, b_rs0, padding=1)                            # [2, C, 60, 104]
    up0_out = y.view(1, 2, C, 2 * H, 2 * W).permute(0, 2, 1, 3, 4) + dup_up(x_copy0, C, 2, 2)
    # up_blocks[1]: x_copy = x.clone(), first resnet on x, ...
    x_copy1 = up0_out.clone()
    h = rms_silu(up0_out, g1)
    h = F.conv3d(F.pad(torch.cat([cache0, h], 2), [1, 1, 1, 1, 0, 0]), w1, b1)
    h = rms_silu(h, g2)
    h = F.conv3d(F.pad(torch.cat([cache1, h], 2), [1, 1, 1, 1, 0, 0]), w2, b2)
    x = res2_out + (res1_out + (h + up0_out))
    # ... temporal upsample (time_conv, interleave), spatial upsample conv2d ...
    t = F.conv3d(torch.cat([cache_tc, x], 2), w_tc, b_tc).reshape(1, 2, C, 2, 2 * H, 2 * W)
    x = torch.stack((tc_cache_frame, t[:, 1]), 3).reshape(1, C, 4, 2 * H, 2 * W)
    x = x.permute(0, 2, 1, 3, 4).reshape(4, C, 2 * H, 2 * W)
    x = F.interpolate(x.float(), scale_factor=(2.0, 2.0), mode="nearest-exact").type_as(x)
    y = F.conv2d(x, w_rs1, b_rs1, padding=1)                                     # [4, C, 120, 208]
    # ... + DupUp3D shortcut of x_copy: the read that goes wrong
    return y.view(1, 4, C, 4 * H, 4 * W).permute(0, 2, 1, 3, 4) + dup_up(x_copy1, C, 2, 2)


def inputs(device="cuda", dtype=torch.bfloat16):
    g = torch.Generator(device).manual_seed(0)

    def weight(*shape):
        return (torch.randn(shape, generator=g, device=device) / math.sqrt(math.prod(shape[1:]))).to(dtype)

    def act(*shape):
        return torch.randn(shape, generator=g, device=device).to(dtype)

    zeros, ones = (lambda: torch.zeros(C, device=device, dtype=dtype)), (lambda: torch.ones(C, 1, 1, 1, device=device, dtype=dtype))
    f = (1, C, 2, 2 * H, 2 * W)
    return [weight(C, C, 3, 3, 3), zeros(), weight(C, C, 3, 3), zeros(), ones(), weight(C, C, 3, 3, 3), zeros(),
            ones(), weight(C, C, 3, 3, 3), zeros(), weight(2 * C, C, 3, 1, 1), torch.zeros(2 * C, device=device, dtype=dtype),
            weight(C, C, 3, 3), zeros(),
            act(1, C, 1, H, W), act(1, C, 3, H + 2, W + 2), act(*f), act(*f), act(2, C, 2 * H, 2 * W), act(*f), act(*f), act(*f),
            act(1, C, 2, 2 * H, 2 * W)]


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    args = inputs()
    with torch.inference_mode():
        ref = decoder_slice(*args)
        res = {}
        for name, cfg in (("default", {}), ("comprehensive_padding=False", {"comprehensive_padding": False})):
            torch._dynamo.reset()
            with inductor_config.patch(cfg):
                out = torch.compile(decoder_slice, dynamic=False)(*args)
            torch.cuda.synchronize()
            res[name] = rel_l2(out, ref)
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}")
    for name, r in res.items():
        print(f"  compiled ({name}) vs eager: rel L2 {r:.4f}")
    bad = res["default"] > 0.1 and res["comprehensive_padding=False"] < 0.05
    print("MISCOMPILED" if bad else "ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
