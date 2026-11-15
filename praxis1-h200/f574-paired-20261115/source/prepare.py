#!/usr/bin/env python3
"""Make a verified full-suite copy of the pinned simulator entrypoint."""
from __future__ import annotations

import pathlib
import sys


def replace_one(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one occurrence of {old!r}; got {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    ref, out = map(pathlib.Path, sys.argv[1:3])
    out.mkdir(parents=True, exist_ok=True)
    runner = (ref / "scripts/run_sim_eval.py").read_text()
    runner = replace_one(
        runner,
        "        units = units[shard::shards]\n",
        "        units = units[shard::shards]\n"
        "        expected = [(task, seed) for seed in range(suite.scene_seeds) for task in suite.tasks]\n"
        "        if units != expected:\n"
        "            raise RuntimeError(f'full-suite unit mismatch: {units!r} != {expected!r}')\n",
    )
    runner = replace_one(
        runner,
        '    print(f"policy {policy.name} ready in {time.monotonic() - t0:.0f} s", flush=True)\n',
        "    if not args.mstar_url:\n"
        "        from praxis.layers import Attention\n"
        "        attn = [m for m in policy.pipe.policy.modules() if isinstance(m, Attention)]\n"
        "        if not attn or any(m.backend != 'triton' or m.effective_backend('cuda') != 'triton' for m in attn):\n"
        "            raise RuntimeError('reference attention is not all effective Triton')\n"
        "        print(f'reference effective Triton modules: {len(attn)}', flush=True)\n"
        "    else:\n"
        "        if policy.name != 'mstar':\n"
        "            raise RuntimeError(f'unexpected M* policy: {policy.name}')\n"
        '    print(f"policy {policy.name} ready in {time.monotonic() - t0:.0f} s", flush=True)\n',
    )
    runner = replace_one(
        runner,
        '                    policy.prompt = prompt\n',
        '                    policy.prompt = prompt\n'
        '                    if policy.prompt != prompt:\n'
        "                        raise RuntimeError('task prompt was not set')\n",
    )
    runner = replace_one(
        runner,
        '                    "mean_call_s": round(r.mean_call_s, 3),\n',
        '                    "mean_call_s": round(r.mean_call_s, 3),\n'
        '                    "call_s": [float(value) for value in r.call_s],\n',
    )
    (out / "run_sim_eval_verified.py").write_text(runner)


if __name__ == "__main__":
    main()
