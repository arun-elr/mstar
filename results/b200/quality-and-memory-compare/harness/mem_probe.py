"""Peak GPU memory of one streaming decoder, split into resident weights, latents,
persistent per-stream state (conv cache and anything else kept between chunks)
and the transient activation peak of one steady chunk.

Same decoder objects and stream as the timing runs: latent 0, 10 warm-up chunks,
then 5 measured chunks with the peak counter reset first (bench_wan_decoder's
reset_peak / peak, i.e. torch.cuda.max_memory_allocated).

  vLLM-Omni (container, PYTHONPATH=/bench):  mem_probe.py --engine vllm_omni --variant vllm_omni_best --vae wan21 --latents L
  SGLang (~/bench, sgl-venv):                 mem_probe.py --engine sglang --variant sglang_best --vae wan21 --latents L
  M* (benchmark/wan_vae_decoder, .venv):      mem_probe.py --engine mstar --variant mstar_warm --vae wan21 --latents L
                                              mstar_warm_lpc1 = mstar_warm with windowed_decode_latents_per_call=1
"""

import argparse
import json
import sys

import torch

GiB = 2**30
SGLANG = {  # variant: (quality, decode_precision), as in the SGLang handoff's runs
    "sglang_default": ("lossless", None),
    "sglang_default_bf16": ("lossless", "bf16"),
    "sglang_best_wan21": ("extra-high", None),
    "sglang_best_wan22": ("extra-high", "bf16"),
}


def alloc():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def module_bytes(mod):
    """(total, encoder-side) bytes of CUDA parameters + buffers, by module name."""
    tot = enc = 0
    seen = set()
    for name, t in list(mod.named_parameters()) + list(mod.named_buffers()):
        if t is None or not t.is_cuda or t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        b = t.numel() * t.element_size()
        tot += b
        if {"encoder", "quant_conv"} & set(name.split(".")):
            enc += b
    return tot, enc


def held_tensors(mod, extra):
    """CUDA tensors reachable from Python between chunks, other than the module's
    parameters / buffers and ``extra`` (the latents): (count, logical bytes,
    bytes of their unique storages). A cache entry that is a view of a larger
    activation keeps that whole storage alive."""
    import gc

    skip = {t.untyped_storage().data_ptr() for t in list(mod.parameters()) + list(mod.buffers()) if t is not None}
    skip |= {t.untyped_storage().data_ptr() for t in extra if t is not None}
    n = logical = 0
    storages = {}
    for o in gc.get_objects():
        try:
            if not (torch.is_tensor(o) and o.is_cuda) or isinstance(o, torch.nn.Parameter):
                continue
            s = o.untyped_storage()
            ptr, nb = s.data_ptr(), s.nbytes()
        except Exception:  # compile-time fake tensors have no storage
            continue
        if ptr in skip or nb == 0:
            continue
        n += 1
        logical += o.numel() * o.element_size()
        storages[ptr] = nb
    return n, logical, sum(storages.values())


def build(args, raw):
    """(start(), chunk(c), vae module, info) for the engine."""
    if args.engine == "vllm_omni":
        sys.argv, argv = sys.argv[:1], sys.argv
        import bench_vllm_omni
        import bench_wan_decoder as B
        from common import WAN21_REPO, WAN22_REPO, to_uint8
        sys.argv = argv
        B.REPO = WAN21_REPO if args.vae == "wan21" else WAN22_REPO
        sd, dtype = B.build_streaming(args.variant, "cuda")
        mods = sd.vae
        w = alloc()
        z = raw.to("cuda", dtype)
        info = {"latents_per_decoder_call": 1, "decode_path": bench_vllm_omni.RUN_INFO.get("decode_path")}

        def start():
            sd.reset()
            return to_uint8(sd.first(z[:, :, :1]))

        def chunk(c):
            return to_uint8(sd.chunk(z[:, :, 1 + 3 * c: 4 + 3 * c]))
        return start, chunk, mods, w, z, info
    if args.engine == "sglang":
        sys.argv = sys.argv[:1]
        import bench_sglang_wan as B
        key = args.variant if args.variant in SGLANG else f"{args.variant}_{args.vae}"
        quality, prec = SGLANG[key]
        a = argparse.Namespace(vae=args.vae, path="stage", quality=quality, compile=False, compile_mode=None,
                               decode_precision=prec)
        st, ch, vae, sa, settings = B.build(a)
        w = alloc()
        z = B.normalized_latents(raw, vae, sa, "cuda")
        info = {"latents_per_decoder_call": 1, "decode_path": "DecodingStage.decode -> causal_decode",
                "decode_dtype": settings.get("decode_dtype")}

        def u8(v):
            return (v * 255).clamp(0, 255).to(torch.uint8)

        def start():
            return u8(st(z[:, :, :1]))

        def chunk(c):
            return u8(ch(z[:, :, 1 + 3 * c: 4 + 3 * c]))
        return start, chunk, vae, w, z, info
    import bench_wan_decoder as B
    from common import WAN21_REPO, WAN22_REPO
    B.REPO = WAN21_REPO if args.vae == "wan21" else WAN22_REPO
    over, compiled = B.node_overrides(args.variant.removesuffix("_lpc1"))
    if args.variant.endswith("_lpc1"):
        over = dict(over, windowed_decode_latents_per_call=1)
    sub, start, chunk = B.mstar_warm_node(raw, "cuda", compiled=compiled, n_steady=10 + 5, overrides=over)
    info = {"latents_per_decoder_call": over.get("windowed_decode_latents_per_call", 0), "compiled": compiled,
            "overrides": over}
    return start, chunk, sub, None, None, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm_omni", "sglang", "mstar"], required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--vae", choices=["wan21", "wan22"], required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    raw = torch.load(args.latents)["raw"]
    a0 = alloc()
    with torch.inference_mode():
        start, chunk, mod, a_w, z, info = build(args, raw)
        a_z = alloc()
        if a_w is None:  # M*: the normalized fp32 latents land on the GPU inside mstar_warm_node
            a_w = a_z - raw.numel() * 4
        wbytes, enc = module_bytes(mod)
        x = start()
        del x
        for c in range(10):
            x = chunk(c)
            del x
        a_ret = alloc()
        w_steady, _ = module_bytes(mod)
        lat = [z] if z is not None else [c.cell_contents for c in chunk.__closure__ or ()
                                          if torch.is_tensor(c.cell_contents)]
        n_held, held_logical, held_storage = held_tensors(mod, lat)
        torch.cuda.reset_peak_memory_stats()
        for c in range(10, 15):
            x = chunk(c)
            del x
        peak = torch.cuda.max_memory_allocated()
        a_end = alloc()
    res = {
        "engine": args.engine, "variant": args.variant, "vae": args.vae, **info,
        "weights_resident_gib": round(wbytes / GiB, 3), "encoder_weights_resident_gib": round(enc / GiB, 3),
        "allocated_after_load_gib": round((a_w - a0) / GiB, 3),
        "latents_on_gpu_gib": round((a_z - a_w) / GiB, 3),
        "allocated_after_load_and_latents_gib": round((a_z - a0) / GiB, 3),
        "persistent_state_gib": round((a_ret - a_z) / GiB, 3),
        "weights_resident_steady_gib": round(w_steady / GiB, 3),
        "state_between_chunks_gib": round((a_ret - a_z + wbytes - w_steady) / GiB, 3),
        "held_tensors": n_held, "held_logical_gib": round(held_logical / GiB, 3),
        "held_storage_gib": round(held_storage / GiB, 3),
        "allocated_between_chunks_gib": round((a_ret - a0) / GiB, 3),
        "peak_allocated_gib": round(peak / GiB, 2),
        "transient_peak_gib": round((peak - a_ret) / GiB, 3),
        "allocated_end_minus_between_chunks_gib": round((a_end - a_ret) / GiB, 3),
        "torch": torch.__version__,
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
