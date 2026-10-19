"""Server turn detection for ``/v1/realtime`` (OpenAI's ``server_vad``).

``TurnSegmenter`` cuts a 16 kHz stream into turns from a per-frame speech
probability, and returns what to do in stream order: start a turn, add audio
to it, stop it, commit it. A commit covers exactly the frames up to the end
of the turn's silence window; audio after that point, even in the same
append, is segmented for the next turn. sglang-omni#1322 lost that suffix.

The probability comes from Silero VAD (MIT), a 2 MB TorchScript model run on
CPU, about 0.3 ms per 32 ms frame. Its weights are read from disk only:
``MSTAR_SILERO_VAD_PATH``, else the copy bundled with the ``silero-vad``
package (``pip install 'mstar[vad]'``).
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = 512              # samples per decision: Silero's window at 16 kHz, 32 ms
# A frame inside a turn counts as silence below threshold - HYSTERESIS, as in
# Silero's own segmentation, so a turn does not flicker on a borderline frame.
HYSTERESIS = 0.15


def samples_to_ms(n: int) -> int:
    return n * 1000 // SAMPLE_RATE


class SpeechProbability(Protocol):
    """Per-session speech probability of one FRAME of 16 kHz audio; stateful."""

    def __call__(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class ServerVAD:
    """OpenAI's ``server_vad`` settings. ``create_response`` and
    ``interrupt_response`` are accepted and echoed; a transcription session
    has no response to create or interrupt."""

    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True
    interrupt_response: bool = True

    def view(self) -> dict:
        return {"type": "server_vad", **asdict(self)}


class SileroVAD:
    """Silero VAD for one stream (the model carries the stream's state)."""

    def __init__(self, model):
        self._model = model

    def __call__(self, frame: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            return float(self._model(torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))[None],
                                     SAMPLE_RATE))

    def reset(self) -> None:
        self._model.reset_states()


def silero_weights() -> Path | None:
    """The Silero TorchScript file this server uses, or None."""
    path = os.environ.get("MSTAR_SILERO_VAD_PATH")
    if path:
        return Path(path)
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        return None
    bundled = Path(next(iter(spec.submodule_search_locations))) / "data" / "silero_vad.jit"
    return bundled if bundled.is_file() else None


@functools.lru_cache(maxsize=1)
def silero_factory() -> Callable[[], SpeechProbability] | None:
    """Make one SileroVAD per session from a model loaded once, or return
    None (with one warning) when no weights are available."""
    path = silero_weights()
    if path is None:
        logger.warning("server_vad is unavailable: no Silero VAD weights (set MSTAR_SILERO_VAD_PATH "
                       "or install mstar[vad])")
        return None
    try:
        import torch

        model = torch.jit.load(str(path), map_location="cpu").eval()
    except Exception:  # noqa: BLE001 — a bad file disables server_vad, not the server
        logger.exception("server_vad is unavailable: cannot load Silero VAD weights from %s", path)
        return None
    return lambda: SileroVAD(copy.deepcopy(model))


class TurnSegmenter:
    """Splits a 16 kHz stream into turns (OpenAI ``server_vad`` semantics).

    ``feed`` takes audio of any length and returns operations in stream
    order: ``("speech_started", audio_start_ms)``, ``("audio", samples)`` for
    the open turn, ``("speech_stopped", audio_end_ms)``, ``("commit",)``.
    Times are milliseconds from the start of the session's audio. A turn
    starts on a frame at or above ``threshold`` and takes up to
    ``prefix_padding_ms`` of the audio before it; it ends once
    ``silence_duration_ms`` of whole silent frames have passed (they belong
    to the turn), or when it reaches ``max_turn_samples``. Audio between
    turns, beyond the prefix, is in no turn. A partial frame waits for the
    next ``feed``.
    """

    def __init__(self, vad: SpeechProbability, cfg: ServerVAD, max_turn_samples: int, pos: int = 0):
        self.vad, self.cfg = vad, cfg
        self.max_turn = max_turn_samples
        self._prefix_cap = cfg.prefix_padding_ms * SAMPLE_RATE // 1000
        self._silence_cap = cfg.silence_duration_ms * SAMPLE_RATE // 1000
        self._neg = max(cfg.threshold - HYSTERESIS, 0.01)
        self._rest = np.zeros(0, dtype=np.float32)   # the partial frame
        self._pos = pos                              # stream index of _rest[0]
        self._prefix = np.zeros(0, dtype=np.float32)
        self.speaking = False
        self._silence = 0                            # trailing silent samples in the turn
        self._turn = 0                               # samples in the turn

    def feed(self, audio: np.ndarray) -> list[tuple]:
        ops: list[tuple] = []
        buf = np.concatenate([self._rest, np.asarray(audio, dtype=np.float32)]) if self._rest.size else \
            np.asarray(audio, dtype=np.float32)
        whole = buf.size - buf.size % FRAME
        for i in range(0, whole, FRAME):
            self._frame(buf[i:i + FRAME], self._pos + i, ops)
        self._rest = buf[whole:].copy()
        self._pos += whole
        return ops

    def _frame(self, frame: np.ndarray, start: int, ops: list) -> None:
        p = self.vad(frame)
        if not self.speaking:
            if p >= self.cfg.threshold:
                self.speaking = True
                self._turn, self._silence = self._prefix.size + FRAME, 0
                ops.append(("speech_started", samples_to_ms(start - self._prefix.size)))
                ops.append(("audio", np.concatenate([self._prefix, frame])))
                self._prefix = np.zeros(0, dtype=np.float32)
            elif self._prefix_cap:
                self._prefix = np.concatenate([self._prefix, frame])[-self._prefix_cap:]
            return
        ops.append(("audio", frame))
        self._turn += FRAME
        self._silence = self._silence + FRAME if p < self._neg else 0
        if self._silence >= self._silence_cap or self._turn >= self.max_turn:
            self._stop(start + FRAME, ops)

    def _stop(self, end: int, ops: list) -> None:
        ops.append(("speech_stopped", samples_to_ms(end)))
        ops.append(("commit",))
        self.speaking, self._silence, self._turn = False, 0, 0

    def end_turn(self) -> list[tuple]:
        """A client commit: an open turn takes the partial frame and stops
        there. Between turns, nothing changes."""
        ops: list[tuple] = []
        if self.speaking:
            if self._rest.size:
                ops.append(("audio", self._rest))
            self._pos += self._rest.size
            self._rest = np.zeros(0, dtype=np.float32)
            self._stop(self._pos, ops)
            ops.pop()   # the client's commit is the commit
        return ops

    def reset(self) -> None:
        """A clear: drop buffered audio, keep the timeline."""
        self._pos += self._rest.size
        self._rest = np.zeros(0, dtype=np.float32)
        self._prefix = np.zeros(0, dtype=np.float32)
        self.speaking, self._silence, self._turn = False, 0, 0
        self.vad.reset()
