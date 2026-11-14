#!/usr/bin/env python3
"""Diagnostic lift_cube episode trace for pinned Praxis 68f4d7a and M* 5902054c0f.

CPU-only in scripted mode. Reference and M* modes need a GPU model or running M* server.
Writes one JSONL row per policy call and simulation tick plus the simulator's
standard per-tick NPZ arrays. This script does not modify either checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from praxis.sim.config import load_config
from praxis.sim.env import SimEnv
from praxis.sim.loop import make_policy, run_episode


def _fingerprint(value: object) -> dict:
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "shape": list(arr.shape), "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    return {"type": type(value).__name__, "value": str(value)[:160]}


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", choices=("scripted", "reference", "mstar"), required=True)
    ap.add_argument("--mstar-client", type=Path, help="pinned copied sim_client.py")
    ap.add_argument("--mstar-url", help="M* Positronic v2 WebSocket URL")
    ap.add_argument("--checkpoint", type=Path, help="pinned Praxis checkpoint for reference mode")
    ap.add_argument("--scene-seed", type=int, default=0)
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    overrides = ["task.name=lift_cube", "task.prompt=null", "task.time_limit_s=null", "viewer.enabled=false", "policy.decode_video=false"]
    if args.mode == "scripted":
        overrides.append("policy.type=scripted")
    else:
        overrides.append("policy.type=model")
    if args.checkpoint:
        overrides.append(f"model.checkpoint={args.checkpoint}")
    cfg = load_config(args.config, overrides)
    cfg.session.seed = args.sampler_seed
    env = SimEnv(cfg.env_config())
    if args.mode == "mstar":
        if not args.mstar_url or not args.mstar_client:
            ap.error("mstar mode requires --mstar-url and --mstar-client")
        spec = importlib.util.spec_from_file_location("_pinned_mstar_sim_client", args.mstar_client)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        policy = module.MStarPolicy(args.mstar_url)
        policy.seed_stride = cfg.session.seed_stride
        policy.sampler_seed = args.sampler_seed
    else:
        if args.mode == "reference" and not args.checkpoint:
            ap.error("reference mode requires --checkpoint")
        policy = make_policy(cfg, "")
    if args.mode != "scripted":
        # Match run_sim_eval.py, which assigns the task prompt before each episode.
        policy.prompt = env.prompt
    current_call = -1
    current_row = 0
    output = args.out / f"{args.mode}-scene{args.scene_seed}-sample{args.sampler_seed}.jsonl"
    arrays = args.out / f"{args.mode}-scene{args.scene_seed}-sample{args.sampler_seed}.npz"
    with output.open("w") as log:
        def write(record: dict) -> None:
            log.write(json.dumps(_jsonable(record), allow_nan=False, sort_keys=True) + "\n")
            log.flush()

        original_act = policy.act
        def traced_act(history):
            nonlocal current_call, current_row
            obs = history.obs_dict()
            out = original_act(history)
            current_call += 1
            current_row = 0
            poses = [np.asarray(row["robot_command"]["pose"], np.float32) for row in out.commands]
            grips = [float(row["target_grip"]) for row in out.commands]
            write({
                "kind": "call", "call": current_call, "start_tick": env.tick_count,
                "observation": {k: _fingerprint(v) for k, v in sorted(obs.items())},
                "info": out.info, "pose12": poses, "grip": grips,
                "command_sha256": hashlib.sha256(np.stack(poses).tobytes() + np.asarray(grips, np.float32).tobytes()).hexdigest(),
            })
            return out
        policy.act = traced_act
        original_step = env.step
        def traced_step(pose12, grip):
            nonlocal current_row
            obs = original_step(pose12, grip)
            status = env.task.status
            cube_z = float(env.task.cube_pos(env.data)[2])
            lift_m = cube_z - float(env.task.rest_z)
            write({
                "kind": "tick", "tick": env.tick_count, "call": current_call,
                "row": current_row, "cube_z": cube_z, "lift_m": lift_m,
                "lift_threshold_m": float(env.task.cfg.lift_height),
                "above_threshold": bool(env.task.condition(env.data, float(grip))),
                "hold_ticks": status.hold_ticks, "required_hold_ticks": env.task.hold_ticks,
                "success": status.success, "success_tick": status.success_tick,
                "failed": status.failed, "cmd_grip": float(grip), "observed_grip": float(obs.grip),
            })
            current_row += 1
            return obs
        env.step = traced_step
        try:
            result = run_episode(
                env, policy, args.scene_seed, prompt=env.prompt,
                stop_on_success=True, after_success_s=1.0,
                arrays_path=arrays, video=None,
            )
            write({"kind": "summary", "result": result.to_json(), "arrays": str(arrays)})
        finally:
            policy.close()
            env.close()
    print(json.dumps({"trace": str(output), "arrays": str(arrays), "success": result.success, "ticks": result.ticks, "calls": result.calls}))


if __name__ == "__main__":
    main()
