"""Shared pieces for the Wan VAE streaming-decode benchmark.

Wan 2.1 VAE decoder, batch 1, one steady chunk = 3 latent frames
``[1, 16, 3, 60, 104]`` -> 12 RGB frames at 480x832, bf16, causal conv cache
warm across chunks, timed as decode + conversion to 8-bit frames, median /
p90 over 75 steady chunks.
"""

import os
from pathlib import Path

import torch
from torch import nn

WAN21_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN22_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

CHUNK_LATENTS = 3      # latent frames per steady chunk (z in R^{16x3x60x104})
FRAMES_PER_CHUNK = 12  # pixel frames per steady chunk (4x temporal)


def hf_snapshot(repo: str) -> Path:
    """Local snapshot dir of ``repo`` from the HF cache; never downloads."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=["vae/*"], local_files_only=True))


def load_vae(repo: str, dtype: torch.dtype, device: str, keep_encoder: bool = False):
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(str(hf_snapshot(repo) / "vae"), torch_dtype=dtype)
    if not keep_encoder:
        vae.encoder = None
        vae.quant_conv = None
    return vae.to(device).eval()


def latent_stats(vae, device, dtype=torch.float32):
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, z, 1, 1, 1)
    return mean, std


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float -> uint8, the formula both M* decoder nodes use."""
    return (video / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)


def bf16_nearest_upsample():
    """Run Wan's nearest-exact upsample in the activation dtype instead of fp32.

    Nearest-exact only copies values, so skipping the fp32 round trip is exact.
    """
    from diffusers.models.autoencoders import autoencoder_kl_wan as akw

    akw.WanUpsample.forward = nn.Upsample.forward


class StreamingDecoder:
    """Chunked causal decode of a Wan VAE with the conv feature cache kept warm
    across chunks. Mirrors ``AutoencoderKLWan._decode`` minus ``clear_cache``
    between calls.

    per_frame=True is stock diffusers behaviour (one latent frame per decoder
    call); per_frame=False decodes the whole chunk in one call.
    """

    def __init__(self, vae, *, per_frame: bool, channels_last: bool, compile_decoder: bool,
                 compile_mode: str | None = None):
        self.vae = vae
        self.per_frame = per_frame
        self.channels_last = channels_last
        self.patch_size = vae.config.patch_size
        if channels_last:
            for m in vae.modules():
                if isinstance(m, nn.Conv3d):
                    m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
                elif isinstance(m, nn.Conv2d):
                    m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
        self.decoder = vae.decoder
        self.step = (
            torch.compile(vae.decoder, fullgraph=False, dynamic=False, mode=compile_mode)
            if compile_decoder else vae.decoder
        )
        if compile_decoder and compile_mode == "reduce-overhead":
            self.step = self._cudagraph_tree_step(self.step)
        self.reset()

    @staticmethod
    def _cudagraph_tree_step(step):
        """torch's CUDA-graph trees overwrite a replay's outputs on the next replay,
        but the cache written by one chunk is read by the next: start a new step per
        call and move the cache out of the graph pool with a clone."""
        def run(x, feat_cache, feat_idx, **kw):
            torch.compiler.cudagraph_mark_step_begin()
            out = step(x, feat_cache=feat_cache, feat_idx=feat_idx, **kw)
            for i, t in enumerate(feat_cache):
                if torch.is_tensor(t):
                    feat_cache[i] = t.clone()
            return out.clone()
        return run

    def reset(self):
        self.feat = [None] * self.vae._cached_conv_counts["decoder"]

    def _prep(self, z):
        x = self.vae.post_quant_conv(z)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return x

    def _finish(self, out):
        if self.patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

            out = unpatchify(out, patch_size=self.patch_size)
        return out.clamp(-1.0, 1.0)

    def first(self, z1):
        """The stream's first latent frame (decodes to 1 pixel frame)."""
        x = self._prep(z1)
        out = self.step(x, feat_cache=self.feat, feat_idx=[0], first_chunk=True)
        return self._finish(out)

    def chunk(self, z, feat=None):
        feat = self.feat if feat is None else feat
        x = self._prep(z)
        if self.per_frame:
            outs = [
                self.step(x[:, :, i:i + 1], feat_cache=feat, feat_idx=[0])
                for i in range(x.shape[2])
            ]
            out = torch.cat(outs, 2)
        else:
            out = self.step(x, feat_cache=feat, feat_idx=[0])
        return self._finish(out)


class GraphedStreamingDecoder:
    """Replays ``StreamingDecoder.chunk`` from one manually captured CUDA graph once
    the stream is steady (every cache slot holds a fixed-shape tensor).

    The cache lives in static buffers: the captured step reads them, and copies the
    new cache back into them in place, so a replay is exactly one eager chunk.
    """

    def __init__(self, sd: StreamingDecoder, eager_chunks: int = 3):
        self.sd = sd
        self.vae = sd.vae
        self.eager_chunks = eager_chunks
        self.graph = None
        self.reset()

    def reset(self):
        self.sd.reset()
        self.n = 0
        self.need_load = True

    def first(self, z1):
        return self.sd.first(z1)

    def _body(self):
        feat = list(self.static_feat)
        out = self.sd.chunk(self.z, feat)
        for dst, src in zip(self.static_feat, feat, strict=True):
            if dst is not None:
                dst.copy_(src)
        return out

    def _capture(self, z):
        # Slots of convs that keep no cache (1x1 shortcuts) stay None for good.
        assert all(t is None or torch.is_tensor(t) for t in self.sd.feat), "cache not steady yet"
        self.z = z.clone()
        self.static_feat = [t.clone() if t is not None else None for t in self.sd.feat]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._body()

    def chunk(self, z):
        self.n += 1
        if self.n <= self.eager_chunks:
            return self.sd.chunk(z)
        if self.graph is None:
            self._capture(z)
        if self.need_load:
            for dst, src in zip(self.static_feat, self.sd.feat, strict=True):
                if dst is not None:
                    dst.copy_(src)
            self.need_load = False
        self.z.copy_(z)
        self.graph.replay()
        return self.out


def patch_causal_conv_native_pad():
    """Causal conv without a materialized padded copy: only the temporal context is
    concatenated (cached frames, or zero frames at stream start) and the H/W zero
    padding is done inside the conv kernel. Inputs of the causal convs (when their
    weights are channels-last) and of the 2D conv after each upsample are forced to
    channels-last, so no conv sees a channels-first activation. Same math as
    ``F.pad`` + conv with padding 0. Apply after ``bf16_nearest_upsample``."""
    import torch.nn.functional as F
    from diffusers.models.autoencoders import autoencoder_kl_wan as akw

    def forward(self, x, cache_x=None):
        pw, _, ph, _, pt, _ = self._padding
        if pt > 0:
            if cache_x is not None:
                x = torch.cat([cache_x.to(x.device), x], dim=2)
                pt -= cache_x.shape[2]
            if pt > 0:
                x = torch.cat([x.new_zeros(*x.shape[:2], pt, *x.shape[3:]), x], dim=2)
        if self.weight.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return F.conv3d(x, self.weight, self.bias, self.stride, (0, ph, pw), self.dilation, self.groups)

    akw.WanCausalConv3d.forward = forward

    upsample = akw.WanUpsample.forward  # feeds the 2D conv after each upsample

    def upsample_cl(self, x):
        out = upsample(self, x)
        return out.contiguous(memory_format=torch.channels_last) if out.dim() == 4 else out

    akw.WanUpsample.forward = upsample_cl


def mstar_path() -> str:
    """M* checkout whose model code the M* variants import: $MSTAR, else the repo
    this script lives in, else ~/oss/mstar."""
    if "MSTAR" in os.environ:
        return os.environ["MSTAR"]
    here = Path(__file__).resolve().parents[2]
    if (here / "mstar" / "model").is_dir():
        return str(here)
    return str(Path.home() / "oss" / "mstar")
