"""Time the same cold Praxis observation through reference or M* serving.

This is a sequential, end-to-end WebSocket measurement. It excludes server
startup, handshake, and client encoding. It does not measure model parity or
closed-loop success. Positronic modes send the same raw observation payload;
the openpi mode converts that observation into its FR3 hand wire format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.serve import wire

ROWS = 21
REQUIRED_META = (
    "checkpoint",
    "code_version",
    "max_context_chunks",
    "num_steps",
    "t_shift",
    "cfg_scale",
    "cfg_video_scale",
    "skip_video_decoder",
)
VOLATILE_META = ("rollout_id", "rollout_id_source")


def stable_meta(meta: dict) -> dict:
    """Remove per-connection reference IDs before checking server identity."""
    return {key: value for key, value in meta.items() if key not in VOLATILE_META}


def cold_observation(prompt: str) -> dict:
    """One fixed 512x288 observation in the Praxis sim pose frame."""
    height, width = 288, 512
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    exterior = np.empty((1, height, width, 3), dtype=np.uint8)
    exterior[0, :, :, 0] = x
    exterior[0, :, :, 1] = y
    exterior[0, :, :, 2] = 64
    wrist = exterior.copy()
    wrist[..., 2] = 192
    return {
        "image.exterior": exterior,
        "image.wrist": wrist,
        "robot_state.ee_pose": np.array([[0.5, 0, 0.4, 1, 0, 0, 0]], dtype=np.float32),
        "grip": np.array([[0]], dtype=np.float32),
        "descriptor": "praxis.sim",
        "task": prompt,
    }


def reference_url(url: str, *, skip_video_decoder: bool = False) -> str:
    """Pin the reference to cold mode and the selected decoder work."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(
        context_mode="stateless",
        max_context_chunks="0",
        sample_seed="0",
        schedule="chunked",
        skip_video_decoder="on" if skip_video_decoder else "off",
    )
    return urlunsplit(parts._replace(query=urlencode(query)))


def mstar_url(url: str) -> str:
    """Pin the Positronic base seed across benchmark sessions."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["sample_seed"] = "0"
    return urlunsplit(parts._replace(query=urlencode(query)))


def _receive(socket, timeout: float) -> dict:
    data = socket.recv(timeout=timeout)
    if not isinstance(data, bytes):
        raise ValueError("server sent a nonbinary frame")
    frame = wire.deserialise(data)
    if not isinstance(frame, dict):
        raise ValueError("server sent a non-map frame")
    return frame


def _ready(
    socket, timeout: float, mode: str, *, reference_skip_video_decoder: bool = False,
) -> tuple[dict, str | None]:
    for _ in range(8):
        frame = _receive(socket, timeout)
        if mode == "mstar-openpi":
            meta = frame
            break
        if frame.get("status") == "loading":
            continue
        if frame.get("status") == "error":
            raise RuntimeError(f"server refused session: {frame.get('error')}")
        meta = frame.get("meta")
        if frame.get("status") != "ready" or not isinstance(meta, dict):
            raise ValueError("server did not send a ready frame with metadata")
        break
    else:
        raise ValueError("server did not finish the handshake")
    missing = [key for key in REQUIRED_META if key not in meta]
    if missing:
        raise ValueError(f"ready metadata lacks {missing}; add deployment provenance")
    if meta["max_context_chunks"] != 0:
        raise ValueError("this benchmark requires max_context_chunks=0")
    if mode == "mstar-openpi":
        return meta, None
    version = frame.get("protocol_version", 1)
    if mode == "reference" and (
        version != 1
        or meta.get("context_mode") != "stateless"
        or meta.get("skip_video_decoder") is not reference_skip_video_decoder
    ):
        raise ValueError(
            "reference must serve protocol v1 stateless mode with "
            f"skip_video_decoder={str(reference_skip_video_decoder).lower()}"
        )
    session_id = frame.get("session_id")
    if mode == "mstar" and (
        version != 2
        or not isinstance(session_id, str)
        or not session_id
        or frame.get("max_in_flight") != 1
    ):
        raise ValueError("M* must serve a Positronic v2 session")
    return meta, session_id if mode == "mstar" else None


def openpi_payload(observation: dict, prompt: str, seed: int) -> bytes:
    """Convert the same semantic observation to the root openpi schema."""
    from praxis.serve.openpi_sim_client import openpi_observation

    latest = SimpleNamespace(
        exterior=observation["image.exterior"][-1],
        wrist=observation["image.wrist"][-1],
        ee_pose=observation["robot_state.ee_pose"][-1],
        grip=float(observation["grip"][-1, 0]),
    )
    request, _ = openpi_observation(SimpleNamespace(latest=latest), prompt, seed)
    return wire.serialise(request)


def measure(
    url: str,
    mode: str,
    *,
    sessions: int,
    calls: int,
    warmup_sessions: int,
    timeout: float,
    prompt: str,
    reference_skip_video_decoder: bool = False,
) -> dict:
    """Return raw calls and provenance. Any failed call aborts the run."""
    if mode not in ("reference", "mstar", "mstar-openpi"):
        raise ValueError("mode must be reference, mstar or mstar-openpi")
    if mode != "reference" and reference_skip_video_decoder:
        raise ValueError("reference_skip_video_decoder applies only to reference mode")
    if sessions < 2 or calls < 1 or not 0 < warmup_sessions < sessions or timeout <= 0:
        raise ValueError(
            "need sessions >= 2, calls >= 1, 0 < warmup < sessions, timeout > 0"
        )
    observation = cold_observation(prompt)
    obs_bytes = wire.serialise(observation)
    address = (
        reference_url(url, skip_video_decoder=reference_skip_video_decoder) if mode == "reference" else
        mstar_url(url) if mode == "mstar" else url
    )
    records: list[dict] = []
    first_meta = None
    for session in range(sessions):
        with connect(
            address, open_timeout=timeout, max_size=None, compression=None
        ) as socket:
            meta, session_id = _ready(
                socket, timeout, mode,
                reference_skip_video_decoder=reference_skip_video_decoder,
            )
            if first_meta is None:
                first_meta = meta
            elif stable_meta(meta) != stable_meta(first_meta):
                raise ValueError("server metadata changed during the benchmark")
            payload = (
                wire.serialise({"session_id": session_id, "observation": observation})
                if session_id is not None
                else obs_bytes
            )
            for call in range(calls):
                request = (
                    openpi_payload(observation, prompt, call)
                    if mode == "mstar-openpi"
                    else payload
                )
                start = time.monotonic_ns()
                socket.send(request)
                answer = _receive(socket, timeout)
                elapsed_ms = (time.monotonic_ns() - start) / 1e6
                if answer.get("status") == "error":
                    raise RuntimeError(
                        f"session {session} call {call}: {answer.get('error')}"
                    )
                rows = answer.get("actions" if mode == "mstar-openpi" else "result")
                if mode == "mstar-openpi":
                    valid = isinstance(rows, np.ndarray) and rows.shape == (ROWS, 10)
                else:
                    valid = (
                        isinstance(rows, list)
                        and len(rows) == ROWS
                        and all(isinstance(row, dict) for row in rows)
                    )
                if not valid:
                    raise ValueError(
                        f"session {session} call {call}: expected {ROWS} action rows"
                    )
                timing = (
                    answer.get("timing")
                    if mode == "mstar"
                    else answer.get("server_timing")
                )
                records.append(
                    {
                        "session": session,
                        "call": call,
                        "warmup": session < warmup_sessions,
                        "round_trip_ms": elapsed_ms,
                        "server_infer_ms": (
                            timing.get(
                                "policy_ms" if mode == "reference" else "infer_ms"
                            )
                            if isinstance(timing, dict)
                            else None
                        ),
                        "rows": len(rows),
                        "request_bytes": len(request),
                    }
                )
            if session_id is not None:
                socket.send(
                    wire.serialise({"session_id": session_id, "end_session": True})
                )
                _receive(socket, timeout)
    assert first_meta is not None
    client_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    client_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    gpu = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if shutil.which("nvidia-smi")
        else None
    )
    return {
        "mode": mode,
        "server_meta": first_meta,
        "client_commit": client_commit,
        "client_dirty": client_dirty,
        "python": platform.python_version(),
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "gpu": gpu.stdout.strip() if gpu is not None and gpu.returncode == 0 else None,
        "seed_controlled": True,
        "observation_sha256": hashlib.sha256(obs_bytes).hexdigest(),
        "observation_format": "raw uint8 1x288x512x3 per view, fixed one-frame calls; openpi uses FR3 hand keys",
        "sessions": sessions,
        "calls_per_session": calls,
        "warmup_sessions": warmup_sessions,
        "calls": records,
    }


def summarize(result: dict) -> dict:
    """Summarize counted calls only; retain cold request sizes for audit."""
    counted = [row for row in result["calls"] if not row["warmup"]]
    times = np.asarray([row["round_trip_ms"] for row in counted], dtype=np.float64)
    return {
        "n": len(counted),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_ms": float(times.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode", required=True, choices=("reference", "mstar", "mstar-openpi")
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--calls", type=int, default=4)
    parser.add_argument("--warmup-sessions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--prompt", default="Pick up the red cube.")
    parser.add_argument(
        "--reference-skip-video-decoder", choices=("off", "on"), default="off",
        help="reference decoder setting to request and verify (default: off)",
    )
    args = parser.parse_args()
    result = measure(
        args.url,
        args.mode,
        sessions=args.sessions,
        calls=args.calls,
        warmup_sessions=args.warmup_sessions,
        timeout=args.timeout,
        prompt=args.prompt,
        reference_skip_video_decoder=args.reference_skip_video_decoder == "on",
    )
    result["summary"] = summarize(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
