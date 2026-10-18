"""A request's media is refused by what its files add up to, and by the
model's per-modality file count, before the data worker decodes the files."""
import importlib
import queue
import types
import wave

import pytest
import torch

from mstar.model.base import TensorAndMetadata
from mstar.model.media_limits import RequestMediaBudget, check_media_limits

RATE = 16_000


def _wav(path, seconds, rate=RATE):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x00\x00" * int(seconds * rate))
    return str(path)


@pytest.fixture
def headers(monkeypatch):
    """What each test file's header declares, by path: CI's install has neither
    soundfile nor torchcodec to read real ones."""
    from mstar.model import media_limits

    declared = {}
    monkeypatch.setattr(media_limits, "probe_audio_seconds", declared.get, raising=False)
    return declared


def _clips(tmp_path, seconds, headers, n):
    paths = []
    for i in range(n):
        path = tmp_path / f"clip{i}.wav"
        path.write_bytes(b"")
        headers[str(path)] = seconds
        paths.append(str(path))
    return paths


class _ReachedPromptError(Exception):
    """Raised by the fake ``process_prompt``: every file loaded."""


def _preprocess_thread(model):
    from mstar.api_server.data_worker import PreprocessWorkerThread

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.out_queue = queue.Queue()
    wt.in_flight_requests = set()
    wt.request_output_state = {}
    wt.device = "cpu"
    wt.model = model
    return wt


def _audio_request(paths):
    from mstar.api_server.request_types import PreprocessInput

    return PreprocessInput(
        request_id="r1", text="say this", file_paths={"audio": paths},
        input_modalities=["audio", "text"], output_modalities=["audio"], model_kwargs={},
    )


def _recording_loader(loaded, seconds):
    """A ``load_audio`` that records each file and returns ``seconds`` of silence."""
    def load_audio(filepath, device):
        loaded.append(filepath)
        audio = torch.zeros(int(seconds * RATE))
        return TensorAndMetadata(data=audio, metadata={"sample_rate": RATE, "num_channels": 1})

    return load_audio


def _fake_model(loaded, seconds, max_input_files=None):
    def process_prompt(*_args, **_kwargs):
        raise _ReachedPromptError

    return types.SimpleNamespace(
        load_audio=_recording_loader(loaded, seconds),
        max_input_files=lambda: dict(max_input_files or {}),
        process_prompt=process_prompt,
    )


def _status(wt, error):
    wt._fail_request("r1", error, "preprocessing")
    return wt.out_queue.get_nowait().metadata["status"]


ONE_CLIP_MODELS = [
    ("mstar.model.qwen3_tts.qwen3_tts_model", "Qwen3TTSModel"),
    ("mstar.model.chatterbox.chatterbox_model", "ChatterboxModel"),
    ("mstar.model.whisper.whisper_model", "WhisperModel"),
    ("mstar.model.higgs_audio.higgs_audio_model", "HiggsAudioModel"),
]


def _one_clip_model(module, cls, loaded):
    model = object.__new__(getattr(importlib.import_module(module), cls))
    model.load_audio = _recording_loader(loaded, 1.0)

    def process_prompt(*_args, **_kwargs):
        raise _ReachedPromptError

    model.process_prompt = process_prompt
    return model


@pytest.mark.parametrize("module, cls", ONE_CLIP_MODELS)
def test_a_second_clip_is_refused_before_either_is_decoded(tmp_path, headers, module, cls):
    loaded = []
    wt = _preprocess_thread(_one_clip_model(module, cls, loaded))
    with pytest.raises(ValueError, match="attaches 2 audio files; this model takes at most 1") as info:
        wt._process_input(_audio_request(_clips(tmp_path, 1.0, headers, 2)))
    assert loaded == []
    assert _status(wt, info.value) == 400


@pytest.mark.parametrize("module, cls", ONE_CLIP_MODELS)
def test_one_ordinary_clip_still_loads(tmp_path, headers, module, cls):
    loaded = []
    paths = _clips(tmp_path, 1.0, headers, 1)
    with pytest.raises(_ReachedPromptError):
        _preprocess_thread(_one_clip_model(module, cls, loaded))._process_input(_audio_request(paths))
    assert loaded == paths


def test_audio_clips_share_the_audio_limit(tmp_path, headers, monkeypatch):
    """Each 2 s clip fits a 3 s limit; the second is refused from its header
    before it is decoded."""
    monkeypatch.setenv("MSTAR_MAX_AUDIO_SECONDS", "3")
    paths = _clips(tmp_path, 2.0, headers, 3)
    loaded = []
    wt = _preprocess_thread(_fake_model(loaded, 2.0))
    with pytest.raises(ValueError, match="audio clips decode to more than 3 seconds together") as info:
        wt._process_input(_audio_request(paths))
    assert loaded == paths[:1]
    assert _status(wt, info.value) == 400


@pytest.mark.parametrize("header", [None, 1.0], ids=["unreadable", "understated"])
def test_audio_the_header_does_not_vouch_for_is_counted_once_decoded(tmp_path, headers, monkeypatch, header):
    monkeypatch.setenv("MSTAR_MAX_AUDIO_SECONDS", "3")
    paths = _clips(tmp_path, 1.0, headers, 2)
    headers[paths[0]] = header
    loaded = []
    wt = _preprocess_thread(_fake_model(loaded, 4.0))
    with pytest.raises(ValueError, match="MSTAR_MAX_AUDIO_SECONDS"):
        wt._process_input(_audio_request(paths))
    assert loaded == paths[:1]


def test_clips_within_the_limit_all_load(tmp_path, headers, monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_AUDIO_SECONDS", "3")
    paths = _clips(tmp_path, 1.5, headers, 2)
    loaded = []
    with pytest.raises(_ReachedPromptError):
        _preprocess_thread(_fake_model(loaded, 1.5))._process_input(_audio_request(paths))
    assert loaded == paths


def test_a_clip_is_charged_its_header_or_its_decode_not_both(tmp_path, headers):
    from mstar.model.media_limits import charge_decoded_audio, check_audio_duration

    budget = RequestMediaBudget(num_files=1)
    with budget.file():
        check_audio_duration(_clips(tmp_path, 2.0, headers, 1)[0])
        charge_decoded_audio(torch.zeros(2 * RATE), RATE)
    assert budget.used["audio"] == pytest.approx(2.0)


def test_probe_reads_the_header(tmp_path):
    pytest.importorskip("soundfile")
    from mstar.model.media_limits import probe_audio_seconds

    assert probe_audio_seconds(_wav(tmp_path / "clip.wav", 2.5, rate=24_000)) == pytest.approx(2.5)
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not audio")
    assert probe_audio_seconds(str(junk)) is None


def test_a_bad_audio_limit_stops_startup(monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_AUDIO_SECONDS", "0")
    with pytest.raises(ValueError, match="MSTAR_MAX_AUDIO_SECONDS"):
        check_media_limits()
