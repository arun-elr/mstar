#!/usr/bin/env python3
"""Make verified, scene-filtered copies of the pinned simulator entrypoints."""
from __future__ import annotations

import pathlib
import sys


def replace_one(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one occurrence of {old!r}; got {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    ref, trace, out = map(pathlib.Path, sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    runner = (ref / "scripts/run_sim_eval.py").read_text()
    runner = replace_one(runner, "import math\n", "import math\nimport os\n")
    runner = replace_one(
        runner,
        "        units = units[shard::shards]\n",
        "        scene_mode = os.environ['DIAG_SCENE_MODE']\n"
        "        if scene_mode == 'original':\n"
        "            units = units[0::8]\n"
        "            expected = [(task, i) for i, task in enumerate(suite.tasks) if task in tasks]\n"
        "        else:\n"
        "            scene_seed = int(scene_mode)\n"
        "            units = [(task, seed) for task, seed in units if seed == scene_seed]\n"
        "            expected = [(task, scene_seed) for task in suite.tasks if task in tasks]\n"
        "        if units != expected:\n"
        "            raise RuntimeError(f'unexpected scene units: {units!r} != {expected!r}')\n",
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
    (out / "run_sim_eval_verified.py").write_text(runner)

    tracer = trace.read_text()
    tracer = replace_one(
        tracer,
        '        policy = make_policy(cfg, "")\n',
        '        policy = make_policy(cfg, "")\n'
        '        if args.mode == "reference":\n'
        '            from praxis.layers import Attention\n'
        '            attn = [m for m in policy.pipe.policy.modules() if isinstance(m, Attention)]\n'
        "            if not attn or any(m.backend != 'triton' or m.effective_backend('cuda') != 'triton' for m in attn):\n"
        "                raise RuntimeError('reference trace attention is not all effective Triton')\n"
        "            print(f'reference trace effective Triton modules: {len(attn)}', flush=True)\n",
    )
    (out / "trace_lift_verified.py").write_text(tracer)


if __name__ == "__main__":
    main()
