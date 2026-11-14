"""Use an M* Positronic session as a policy in the DROID simulation.

This is a benchmark client. It does not load a Praxis checkpoint locally.
The session base seed follows the simulation's sampler seed and scene seed.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.sync.client import connect

from praxis.serve import wire

FIRST_ROWS = 21
LATER_ROWS = 24


def stable_server_meta(meta: dict) -> dict:
    """Exclude the per-episode seed from the server build identity."""
    return {key: value for key, value in meta.items() if key != "sample_seed_base"}


@dataclass
class RemoteOutput:
    commands: list[dict]
    predicted: None = None
    info: dict[str, float | int | str] = field(default_factory=dict)


class MStarPolicy:
    """One WebSocket session per simulated episode, one call per action chunk."""

    name = "mstar"
    decode_video = False

    def __init__(self, url: str, prompt: str = "", timeout: float = 300.0) -> None:
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("M* policy URL must use ws:// or wss://")
        if timeout <= 0:
            raise ValueError("M* policy timeout must be positive")
        self.url = url
        self.prompt = prompt
        self.timeout = timeout
        self._socket = None
        self._session_id: str | None = None
        self.meta: dict = {}
        self.calls = 0
        self.restarts = 0
        self.sampler_seed = 0
        self.seed_stride = 1000

    def _recv(self) -> dict:
        assert self._socket is not None
        frame = self._socket.recv(timeout=self.timeout)
        if not isinstance(frame, bytes):
            raise ValueError("M* sent a nonbinary session frame")
        value = wire.deserialise(frame)
        if not isinstance(value, dict):
            raise ValueError("M* sent a non-map session frame")
        return value

    def reset(self, env: object, seed: int) -> None:
        """Open a fresh episode with the reference's base sampling seed."""
        self.close()
        self.calls = 0
        self.restarts = 0
        self.meta = {}
        seed_base = int(self.sampler_seed) + int(self.seed_stride) * int(seed)
        if not -(2**63) <= seed_base < 2**63:
            raise ValueError("sample_seed must fit in a signed 64-bit integer")
        parts = urlsplit(self.url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["sample_seed"] = str(seed_base)
        address = urlunsplit(parts._replace(query=urlencode(query)))
        try:
            self._socket = connect(
                address,
                open_timeout=self.timeout,
                max_size=None,
                compression=None,
            )
            for _ in range(4):
                answer = self._recv()
                if answer.get("status") == "loading":
                    continue
                if answer.get("status") == "error":
                    raise RuntimeError(f"M* refused the session: {answer.get('error')}")
                session_id = answer.get("session_id")
                if (
                    answer.get("status") != "ready"
                    or answer.get("protocol_version") != 2
                    or not isinstance(session_id, str)
                    or not session_id
                    or answer.get("max_in_flight") != 1
                    or not isinstance(answer.get("meta"), dict)
                    or answer["meta"].get("sample_seed_base") != seed_base
                    or answer["meta"].get("seed_source") != "session_param"
                ):
                    raise ValueError("M* did not send a Positronic v2 ready frame")
                self._session_id = session_id
                self.meta = answer["meta"]
                return
            raise ValueError("M* did not finish the session handshake")
        except Exception:
            self.close()
            raise

    def act(self, history: object) -> RemoteOutput:
        if self._socket is None or self._session_id is None:
            raise RuntimeError("call reset before act")
        observation = dict(history.obs_dict())
        observation["task"] = self.prompt
        frame = wire.serialise(
            {
                "session_id": self._session_id,
                "observation": observation,
            }
        )
        started = time.monotonic()
        try:
            self._socket.send(frame)
            answer = self._recv()
            if answer.get("status") == "error":
                raise RuntimeError(f"M* policy call failed: {answer.get('error')}")
            rows = answer.get("result")
            lengths = (FIRST_ROWS,) if self.calls == 0 else (FIRST_ROWS, LATER_ROWS)
            if (
                not isinstance(rows, list)
                or len(rows) not in lengths
                or not all(isinstance(row, dict) for row in rows)
            ):
                raise ValueError(
                    f"M* returned invalid commands: expected {lengths} rows"
                )
        except Exception:
            self.close()
            raise
        if self.calls and len(rows) == FIRST_ROWS:
            self.restarts += 1
        self.calls += 1
        return RemoteOutput(
            rows,
            info={
                "step_s": time.monotonic() - started,
                "depth": 0 if len(rows) == FIRST_ROWS else -1,
                "restarts": self.restarts,
            },
        )

    def close(self) -> None:
        socket, session_id = self._socket, self._session_id
        self._socket = None
        self._session_id = None
        if socket is None:
            return
        if session_id is not None:
            with contextlib.suppress(Exception):
                socket.send(
                    wire.serialise({"session_id": session_id, "end_session": True})
                )
        with contextlib.suppress(Exception):
            socket.close()
