"""Measurement hook for the e2e Cosmos3 streaming runs (put this dir first on
PYTHONPATH of the server only). Nothing here touches the decoder code.

E2E_MEM_DIR: every process that initializes CUDA writes <pid>.json there every
0.5 s with torch.cuda.max_memory_allocated / max_memory_reserved. Touching
<E2E_MEM_DIR>/RESET (a new mtime) resets the peak counters; the peak seen
before each reset is kept in the record.

E2E_DUMP_DIR: while <E2E_DUMP_DIR>/ENABLE exists, Cosmos3Model.postprocess saves
every video chunk it encodes as raw uint8 [C, T, H, W] to
<E2E_DUMP_DIR>/<tag>/<seq>.npy (tag = ENABLE's contents), before the mp4 encode,
and Cosmos3VAEDecoderARSubmodule.forward saves each window's input latents (as the
DiT hands them over, with the window metadata) to <E2E_DUMP_DIR>/<tag>_latents/<seq>.pt.
"""
import json
import os
import sys
import threading
import time

_MEM = os.environ.get("E2E_MEM_DIR")
_DUMP = os.environ.get("E2E_DUMP_DIR")


def _mem_poll():
    out = os.path.join(_MEM, f"{os.getpid()}.json")
    reset = os.path.join(_MEM, "RESET")
    last_reset, resets, before_reset = None, 0, []
    while True:
        time.sleep(0.5)
        torch = sys.modules.get("torch")
        if torch is None:
            continue
        try:
            if not torch.cuda.is_initialized():
                continue
            mt = os.stat(reset).st_mtime if os.path.exists(reset) else None
            if mt is not None and mt != last_reset:
                before_reset.append({"max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                     "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
                torch.cuda.reset_peak_memory_stats()
                last_reset, resets = mt, resets + 1
            rec = {
                "pid": os.getpid(), "argv": " ".join(sys.argv)[:300], "t": time.time(), "resets": resets,
                "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "before_resets": before_reset,
            }
            tmp = out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rec, f)
            os.replace(tmp, out)
        except Exception as e:  # noqa: BLE001 — never take the server down
            print(f"[e2e-mem] {e!r}", file=sys.stderr)
            time.sleep(5)


_TARGET = "mstar.model.cosmos3.cosmos3_model"
_seq = [0]


def _patch(mod):
    import numpy as np

    cls = mod.Cosmos3Model
    orig = cls.postprocess

    def postprocess(self, output, modality, request_kwargs=None):
        flag = os.path.join(_DUMP, "ENABLE")
        if modality == "video" and os.path.exists(flag):
            tag = open(flag).read().strip() or "dump"
            x = output[0] if output.ndim == 5 else output
            d = os.path.join(_DUMP, tag)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"{_seq[0]:04d}.npy"), x.detach().cpu().numpy())
            _seq[0] += 1
        return orig(self, output, modality, request_kwargs)

    cls.postprocess = postprocess
    print(f"[e2e-dump] patched {_TARGET}.Cosmos3Model.postprocess in pid {os.getpid()}", file=sys.stderr)


_SUB = "mstar.model.cosmos3.submodules"
_lseq = [0]


def _patch_sub(mod):
    import torch

    cls = mod.Cosmos3VAEDecoderARSubmodule
    orig = cls.forward

    def forward(self, graph_walk, engine_inputs, latents, **kwargs):
        flag = os.path.join(_DUMP, "ENABLE")
        if latents.numel() and os.path.exists(flag):
            tag = (open(flag).read().strip() or "dump") + "_latents"
            rid = engine_inputs.request_ids[0]
            md = engine_inputs.per_request_info[rid].step_metadata
            d = os.path.join(_DUMP, tag)
            os.makedirs(d, exist_ok=True)
            torch.save({"latents": latents.detach().cpu(), "overlap_latent_units": int(md["overlap_latent_units"]),
                        "num_windows": int(md["num_windows"]), "num_frames": int(md["num_frames"]),
                        "resume_latent_units": int(md.get("resume_latent_units", 0) or 0)},
                       os.path.join(d, f"{_lseq[0]:04d}.pt"))
            _lseq[0] += 1
        return orig(self, graph_walk, engine_inputs, latents, **kwargs)

    cls.forward = forward
    print(f"[e2e-dump] patched {_SUB}.Cosmos3VAEDecoderARSubmodule.forward in pid {os.getpid()}", file=sys.stderr)


_PATCHES = {_TARGET: _patch, _SUB: _patch_sub}


class _Finder:
    def find_spec(self, name, path=None, target=None):
        if name not in _PATCHES:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _PATCHES[name](module)

        spec.loader.exec_module = exec_module
        return spec


if _MEM:
    os.makedirs(_MEM, exist_ok=True)
    threading.Thread(target=_mem_poll, daemon=True, name="e2e-mem").start()
if _DUMP:
    sys.meta_path.insert(0, _Finder())
