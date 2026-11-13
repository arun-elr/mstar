"""Matched DROID warm-session WebSocket benchmark for reference Praxis and M*.

Run the two servers separately with the same DROID checkpoint and recipe. This
client sends an identical synthetic, continuous 46-frame observation trajectory
to each. It measures served latency, not robot success or numeric model parity.

Examples:
  PYTHONPATH=/path/to/praxis-reference python /tmp/mstar_praxis_warm_benchmark.py \
    --mode reference --url ws://127.0.0.1:8000/api/v1/session --out /tmp/ref-warm.json
  PYTHONPATH=/path/to/praxis-reference python /tmp/mstar_praxis_warm_benchmark.py \
    --mode mstar --url ws://127.0.0.1:8126/api/v1/session --out /tmp/mstar-warm.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from PIL import Image
from websockets.sync.client import connect

from praxis.single_arm import control_mode_payload
from praxis.serve import wire


KEYS = ("image.exterior", "image.wrist", "robot_state.ee_pose", "grip")
SPANS = ((0, 1), (0, 22), (21, 46))
EXPECTED_ROWS = (21, 24, 24)
RECIPE = {
    "max_context_chunks": 1,
    "num_steps": 2,
    "t_shift": 12.5,
    "cfg_scale": 1.0,
    "cfg_video_scale": 4.0,
    "skip_video_decoder": True,
    "sample_seed_base": 0,
}
EXPECTED_COMMAND_MODE = control_mode_payload("droid_impedance")


def trajectory(prompt: str) -> list[dict]:
    """One observed trajectory; call 2 overlaps call 1 at its first frame."""
    # Same deterministic construction as praxis.serve.policy.warmup_obs(0, 46),
    # inlined so the benchmark client need not import the serving/model stack.
    t = np.arange(46)
    exterior = np.zeros((46, 288, 512, 3), dtype=np.uint8)
    exterior[..., 0] = np.linspace(0, 255, 512, dtype=np.uint8)[None, None, :]
    exterior[..., 1] = np.linspace(0, 255, 288, dtype=np.uint8)[None, :, None]
    exterior[..., 2] = ((t * 37) % 255)[:, None, None]
    poses = np.zeros((46, 7), dtype=np.float32)
    poses[:, 0] = 0.4 + 0.0005 * t
    poses[:, 2] = 0.4
    poses[:, 3] = 1.0
    complete = {
        "image.exterior": exterior,
        "image.wrist": exterior.copy(),
        "robot_state.ee_pose": poses,
        "grip": np.zeros(46, dtype=np.float32),
    }
    # The DROID handshake asks the client to restrict each frame to 416x234
    # before stacking. Encode each resized frame once, then slice the stack.
    encoded: dict[str, list[bytes]] = {}
    for key in KEYS[:2]:
        frames = []
        for frame in complete[key]:
            image = Image.fromarray(frame).resize((416, 234), Image.Resampling.BILINEAR)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            frames.append(buffer.getvalue())
        encoded[key] = frames
    observations = []
    for start, stop in SPANS:
        obs = {
            key: {b"__jpeg__": True, b"frames": encoded[key][start:stop], b"ndim": 4}
            for key in KEYS[:2]
        }
        obs.update({key: complete[key][start:stop].copy() for key in KEYS[2:]})
        obs["task"] = prompt
        obs["descriptor"] = "praxis.sim"
        observations.append(obs)
    assert observations[0]["image.exterior"][b"frames"][0] == observations[1]["image.exterior"][b"frames"][0]
    assert np.array_equal(observations[0]["robot_state.ee_pose"][0], observations[1]["robot_state.ee_pose"][0])
    return observations


def with_reference_recipe(url: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({
        "sample_seed": "0", "schedule": "chunked", "context_mode": "observed",
        "max_context_chunks": "1", "num_steps": "2", "t_shift": "12.5",
        "cfg_scale": "1.0", "cfg_video_scale": "4.0", "cfg_batched": "false",
        "control_mode": "droid_impedance", "exec_horizon": "24", "skip_video_decoder": "on",
        "precompute": "off",
    })
    return urlunsplit(parts._replace(query=urlencode(query)))


def with_mstar_seed(url: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["sample_seed"] = "0"
    return urlunsplit(parts._replace(query=urlencode(query)))


def receive(socket, timeout: float) -> dict:
    frame = socket.recv(timeout=timeout)
    if not isinstance(frame, bytes):
        raise ValueError("expected binary msgpack frame")
    decoded = wire.deserialise(frame)
    if not isinstance(decoded, dict):
        raise ValueError("expected msgpack map")
    return decoded


def ready(socket, mode: str, timeout: float) -> tuple[dict, str | None]:
    for _ in range(8):
        frame = receive(socket, timeout)
        if frame.get("status") == "loading":
            continue
        if frame.get("status") == "error":
            raise RuntimeError(f"server refused session: {frame.get('error')}")
        if frame.get("status") != "ready" or not isinstance(frame.get("meta"), dict):
            raise ValueError(f"expected ready frame; got {frame}")
        meta = frame["meta"]
        break
    else:
        raise TimeoutError("server did not send ready frame")
    version = frame.get("protocol_version", 1)
    if version != (1 if mode == "reference" else 2):
        raise ValueError(f"wrong protocol version {version} for {mode}")
    if mode == "reference" and meta.get("context_mode") != "observed":
        raise ValueError("reference is not in observed-context mode")
    for key, value in RECIPE.items():
        if meta.get(key) != value:
            raise ValueError(f"metadata {key}={meta.get(key)!r}; expected {value!r}")
    if mode == "reference" and meta.get("control_mode") != "droid_impedance":
        raise ValueError("reference control mode is not droid_impedance")
    session_id = frame.get("session_id") if mode == "mstar" else None
    if mode == "mstar" and (not isinstance(session_id, str) or frame.get("max_in_flight") != 1):
        raise ValueError("M* did not provide a serial Positronic v2 session")
    return meta, session_id


def summary(rows: list[dict], warmup_sessions: int) -> dict:
    counted = [row for row in rows if row["session"] >= warmup_sessions]
    out = {}
    for call in range(3):
        group = [row for row in counted if row["call"] == call]
        times = np.asarray([row["round_trip_ms"] for row in group], dtype=np.float64)
        out[str(call)] = {
            "n": len(group), "rows": EXPECTED_ROWS[call],
            "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "max_ms": float(times.max()),
        }
    return out


def numeric_commands(commands: list[dict]) -> list[list[float]]:
    """Absolute pose12, target grip, and timestamp for each command row."""
    values = []
    for row in commands:
        command = row.get("robot_command")
        if not isinstance(command, dict) or command.get("type") != "cartesian_pos":
            raise ValueError("expected a single-arm Cartesian command")
        pose = np.asarray(command.get("pose"), dtype=np.float64).reshape(-1)
        if pose.shape != (12,):
            raise ValueError(f"command pose is {pose.shape}, expected (12,)")
        values_row = [*pose.tolist(), float(row["target_grip"]), float(row["timestamp"])]
        if not np.isfinite(values_row).all():
            raise ValueError("non-finite command row")
        values.append(values_row)
    return values


def check_command_mode(commands: list[dict]) -> None:
    """Both servers must send the same DROID command law on every row."""
    for row_index, row in enumerate(commands):
        command = row.get("robot_command")
        if not isinstance(command, dict) or command.get("mode") != EXPECTED_COMMAND_MODE:
            raise ValueError(f"command row {row_index} is not DROID impedance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("reference", "mstar"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup-sessions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--prompt", default="Pick up the red cube.")
    parser.add_argument("--expect-code-version", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    obs = trajectory(args.prompt)
    raw = [wire.serialise(value) for value in obs]
    for count, payload in zip((1, 22, 25), raw, strict=True):
        decoded = wire.deserialise(payload)
        assert all(decoded[key].shape[0] == count for key in KEYS)
    hashes = [hashlib.sha256(payload).hexdigest() for payload in raw]
    if args.self_test:
        query = dict(parse_qsl(urlsplit(with_reference_recipe("ws://127.0.0.1/api/v1/session?control_mode=none")).query))
        assert query["control_mode"] == "droid_impedance"
        command = {"robot_command": {"type": "cartesian_pos", "pose": list(range(12)),
                                     "mode": EXPECTED_COMMAND_MODE},
                   "target_grip": 0.5, "timestamp": 0.0}
        check_command_mode([command])
        bare = {**command, "robot_command": {"type": "cartesian_pos", "pose": list(range(12))}}
        assert numeric_commands([command]) == numeric_commands([bare])
        try:
            check_command_mode([bare])
        except ValueError:
            pass
        else:
            raise AssertionError("missing DROID mode was accepted")
        print(json.dumps({"frames": [1, 22, 25], "observation_sha256": hashes}))
        return
    if args.sessions < 2 or not 0 < args.warmup_sessions < args.sessions:
        parser.error("need at least two sessions and 0 < warmup_sessions < sessions")
    url = with_reference_recipe(args.url) if args.mode == "reference" else with_mstar_seed(args.url)
    rows = []
    first_meta = None
    for index in range(args.sessions):
        with connect(url, open_timeout=args.timeout, max_size=None, compression=None) as socket:
            meta, session_id = ready(socket, args.mode, args.timeout)
            if args.expect_code_version and meta.get("code_version") != args.expect_code_version:
                raise ValueError(f"code_version={meta.get('code_version')!r}, expected {args.expect_code_version!r}")
            stable = {key: value for key, value in meta.items() if key not in ("rollout_id", "rollout_id_source")}
            if first_meta is None:
                first_meta = stable
            elif stable != first_meta:
                raise ValueError("server metadata changed between sessions")
            requests = [
                wire.serialise({"session_id": session_id, "observation": item}) if session_id else payload
                for item, payload in zip(obs, raw, strict=True)
            ]
            for call, request in enumerate(requests):
                start = time.monotonic_ns()
                socket.send(request)
                answer = receive(socket, args.timeout)
                elapsed = (time.monotonic_ns() - start) / 1e6
                if answer.get("status") == "error":
                    raise RuntimeError(f"session {index} call {call}: {answer.get('error')}")
                result = answer.get("result")
                if not isinstance(result, list) or len(result) != EXPECTED_ROWS[call] or not all(isinstance(row, dict) for row in result):
                    raise ValueError(f"session {index} call {call}: expected {EXPECTED_ROWS[call]} command rows")
                check_command_mode(result)
                timing = answer.get("server_timing") if args.mode == "reference" else answer.get("timing")
                rows.append({
                    "session": index, "call": call, "frames": (1, 22, 25)[call],
                    "rows": len(result), "request_bytes": len(request),
                    "round_trip_ms": elapsed,
                    "server_reported_ms": timing.get("policy_ms" if args.mode == "reference" else "infer_ms") if isinstance(timing, dict) else None,
                    "command_sha256": hashlib.sha256(wire.serialise(result)).hexdigest(),
                    "command_keys": sorted(result[0]),
                    "numeric_commands": numeric_commands(result),
                    "control_mode": result[0]["robot_command"].get("mode"),
                })
            if session_id is not None:
                socket.send(wire.serialise({"session_id": session_id, "end_session": True}))
                if receive(socket, args.timeout) != {"session_id": session_id, "end_session": True}:
                    raise ValueError("M* did not acknowledge session cleanup")
    source = Path(__file__).read_bytes()
    gpu_probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ) if shutil.which("nvidia-smi") else None
    client_root = Path(wire.__file__).resolve().parents[2]
    result = {
        "mode": args.mode, "measured_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(), "script_sha256": hashlib.sha256(source).hexdigest(),
        "reference_client_commit": subprocess.run(
            ["git", "-C", str(client_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "client_dirty": bool(subprocess.run(
            ["git", "-C", str(client_root), "status", "--porcelain"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()),
        "gpu": gpu_probe.stdout.strip() if gpu_probe is not None and gpu_probe.returncode == 0 else None,
        "server_meta": first_meta, "observation_sha256": hashes,
        "trajectory": "shared continuous 46-frame DROID synthetic stream, JPEG quality 90, 416x234",
        "sessions": args.sessions, "warmup_sessions": args.warmup_sessions,
        "calls": rows, "summary": summary(rows, args.warmup_sessions),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
