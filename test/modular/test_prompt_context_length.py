"""A request whose prompt and output cannot fit a KV cache's positions is
refused with a 400 before the conductor sees it, for every model that counts
its positions (``Model.context_use``).

The data worker compares each count with the cache's ``max_seq_len`` as the
deployment's ``resources:`` block leaves it. Nothing did before: a prompt of
any length was forwarded to the conductor.

The served cases run the API server and its data worker loop for real, with
each model's real ``process_prompt`` over a stand-in tokenizer (one id per
word), on ``/generate`` and through the ``mstar-server`` bridge. The counting
cases pin each model's count to the code that lays its prefill out.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
from test_fish_s2_pro_model import tiny_ar as _fish_ar
from test_fish_s2_pro_model import tiny_model as _fish
from test_orpheus_prompt_length import _Server
from test_qwen3_5_config import _qwen3_8_config
from test_qwen3_5_video import LAYOUTS as VIDEO_LAYOUTS
from test_qwen3_5_video import a_video, an_image, sampled, write_synthetic_checkpoint
from test_qwen3_tts_model import _make_model as _qwen3_tts
from test_qwen3_tts_model import _tiny_model_config

from mstar.api_server import entrypoint
from mstar.api_server.data_worker import _paged_kv_configs
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.base import ContextUse, Model, TensorAndMetadata
from mstar.model.fish_s2_pro.config import AR_KV
from mstar.model.multimodal import TEXT, parts_from_modalities, prefill_plan
from mstar.model.qwen3_5.config import ROPE, Qwen3_5Config, Qwen3_5VisionConfig
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_5.submodules import LLMSubmodule
from mstar.model.qwen3_omni.config import THINKER_POS, Qwen3OmniModelConfig
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule, talker_text_mask
from mstar.model.qwen3_tts.components.talker import Qwen3TTSCodePredictor, Qwen3TTSTalkerModel
from mstar.model.qwen3_tts.qwen3_tts_model import talker_prefill_len
from mstar.model.qwen3_tts.submodules import TalkerSubmodule
from mstar.utils.ipc_format import ConductorMessageType

OMNI = Qwen3OmniModelConfig()


class _ChatTokenizer:
    """A Qwen chat template and tokenizer: specials and role names are single
    ids (the role ids are Qwen3-Omni's, which its Talker mask reads), any other
    word is one id."""

    SPECIALS = {
        "<|im_start|>": OMNI.im_start_token_id, "<|im_end|>": OMNI.im_end_token_id,
        "<|vision_start|>": 151652, "<|vision_end|>": 151653,
        "<|image_pad|>": 151655, "<|video_pad|>": 151656,
        "<|audio_start|>": 151669, "<|audio_end|>": 151670, "<|audio_pad|>": 151675,
    }
    ROLES = {"system": OMNI.system_token_id, "user": OMNI.user_token_id, "assistant": OMNI.assistant_token_id}
    ATTACHMENTS = {
        "image": "<|vision_start|><|image_pad|><|vision_end|>",
        "video": "<|vision_start|><|video_pad|><|vision_end|>",
        "audio": "<|audio_start|><|audio_pad|><|audio_end|>",
    }
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.SPECIALS.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        out = []
        for m in messages:
            content = m["content"]
            if not isinstance(content, str):
                content = "".join(
                    item["text"] if item["type"] == TEXT else self.ATTACHMENTS[item["type"]]
                    for item in content
                )
            out.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>\n")
        return "".join(out) + "<|im_start|>assistant\n"

    def __call__(self, text, return_tensors=None, **kwargs):
        pattern = "|".join(re.escape(t) for t in self.SPECIALS)
        ids = []
        for piece in re.split(f"({pattern})", text):
            if piece in self.SPECIALS:
                ids.append(self.SPECIALS[piece])
            else:
                ids.extend(self.ROLES.get(word, 1000) for word in piece.split())
        return _Encoding(input_ids=torch.tensor([ids]) if return_tensors == "pt" else ids)


class _Encoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class _OmniProcessor(_ChatTokenizer):
    """Qwen3-Omni's processor: fixed grids, and a clip of 250 feature frames."""

    def image_processor(self, images, return_tensors):
        return {"pixel_values": torch.zeros(96, 8), "image_grid_thw": torch.tensor([[1, 8, 12]])}

    class feature_extractor:
        sampling_rate = 16000

        def __new__(cls, audio, **kwargs):
            return {"input_features": torch.zeros(1, 128, 250), "attention_mask": torch.ones(1, 250)}

    class video_processor:
        temporal_patch_size = 2

        def __new__(cls, videos, size):
            return {"video_grid_thw": torch.tensor([[3, 6, 4]]), "pixel_values_videos": torch.zeros(72, 8)}


class _ImageProcessor:
    """Qwen3.5's image processor: one 8 x 12 patch grid per image."""

    size = {"shortest_edge": 65536}
    patch_size = 16

    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.zeros(96, 8), "image_grid_thw": torch.tensor([[1, 8, 12]])}


def _qwen3_5(tmp_path: Path) -> Qwen3_5DenseModel:
    (tmp_path / "config.json").write_text(json.dumps(_qwen3_8_config()))
    model = object.__new__(Qwen3_5DenseModel)
    model.config = Qwen3_5Config.from_hf(tmp_path)
    model.vision_config = Qwen3_5VisionConfig(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=1, in_channels=3,
        patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=64, out_hidden_size=16,
    )
    model.tokenizer = _ChatTokenizer()
    model._image_processor = _ImageProcessor()
    return model


class _Bagel(BagelModel):
    """BAGEL's prompt and resource code over a stand-in tokenizer."""

    def __init__(self, think_mode=False):
        from test_multimodal_ordering import _StubTokenizer

        self.config = SimpleNamespace(
            think_mode=think_mode, num_hidden_layers=2, num_key_value_heads=1,
            hidden_size=16, num_attention_heads=2, max_position_embeddings=32768,
            rope_theta=10000.0, vocab_size=128, num_timesteps=50,
            vit_config=SimpleNamespace(num_attention_heads=1, hidden_size=16, patch_size=14),
            vit_max_num_patch_per_side=70,
        )
        self._has_cfg_parallel = False
        self._image_gen_remote_handoff = False
        self.tokenizer = _StubTokenizer()
        self.boi_token_id = _StubTokenizer.SPECIALS["<|vision_start|>"]
        self.eoi_token_id = _StubTokenizer.SPECIALS["<|vision_end|>"]


def _omni() -> Qwen3OmniModel:
    model = object.__new__(Qwen3OmniModel)
    model.config = Qwen3OmniModelConfig()
    model.tokenizer = model._processor = _OmniProcessor()
    return model


# --------------------------------------------------------------------------
# Served: one position over the limit is a 400 on both frontends, never
# forwarded; a prompt exactly at the limit is admitted.
# --------------------------------------------------------------------------

# (name, model factory, serving YAML, output modalities, one position of
# text). Qwen3.5 and BAGEL take their limit from the YAML's `resources:`
# block, the others from the checkpoint's context.
CASES = {
    "qwen3_5": (_qwen3_5, {"resources": {"kv_cache": {"max_seq_len": 8192}}}, ["text"], "a "),
    "bagel": (lambda _: _Bagel(), {"resources": {"kv": {"max_seq_len": 6000}}}, ["text"], "a"),
    "qwen3_omni": (lambda _: _omni(), None, ["text"], "a "),
    "qwen3_tts": (lambda _: _qwen3_tts(), None, ["audio"], "a "),
    # one id per character; the tiny checkpoint's 512-position context
    "fish_s2_pro": (lambda _: _fish(), None, ["audio"], "a"),
}


def _use(model: Model, text: str, out: list[str]) -> dict[str, ContextUse]:
    """The model's own count for a text prompt."""
    result = model.process_prompt(text, ["text"], out)
    tensors, kwargs = (result, {}) if isinstance(result, dict) else result
    return model.context_use(dict(tensors), ["text"], out, dict(kwargs))


def _prompt_for(name: str, model: Model, over: int) -> tuple[str, int, int]:
    """A prompt ``over`` positions past the limit, its length and the limit.

    One ``unit`` of text is one position in every model here, so the prompt
    for any length is the one-unit prompt plus that many units."""
    _, model_config, out, unit = CASES[name]
    (key, one), *_ = _use(model, unit, out).items()
    context = _paged_kv_configs(model, model_config)[key].max_seq_len
    limit = context - one.generated
    text = unit * (limit + over - one.prompt + 1)
    assert _use(model, text, out)[key].prompt == limit + over
    return text, limit + over, limit


def _server(name, tmp_path, monkeypatch):
    factory, model_config, out, _ = CASES[name]
    s = _Server(model=factory(tmp_path), model_config=model_config, model_name=name)
    s.api.upload_dir = tmp_path
    monkeypatch.setattr(entrypoint, "api_server", s.api)
    return s, out


@pytest.fixture(params=sorted(CASES))
def served(request, tmp_path, monkeypatch):
    s, out = _server(request.param, tmp_path, monkeypatch)
    yield request.param, s, out
    s.close()


def _post(form):
    async def run():
        transport = httpx.ASGITransport(app=entrypoint.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/generate", data=form)

    return asyncio.run(run())


def _refusal(length: int, limit: int) -> str:
    return f"The prompt takes {length} positions; the model takes at most {limit} "


def test_generate_one_past_the_context_is_a_400(served):
    name, server, out = served
    text, length, limit = _prompt_for(name, server.api.model, over=1)

    r = _post({"text": text, "output_modalities": ",".join(out), "streaming": "false"})

    assert r.status_code == 400, r.text
    assert _refusal(length, limit) in r.json()["detail"]
    assert server.socket.sent_of(ConductorMessageType.NEW_REQUEST) == []


def test_rust_frontend_submit_one_past_the_context_is_a_400(served):
    """``mstar-server`` reaches the same data worker through the bridge."""
    from mstar.api_server.rust_frontend import RustFrontendBridge

    name, server, out = served
    text, length, limit = _prompt_for(name, server.api.model, over=1)
    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = server.api
    bridge._keys = {}
    sent = []
    bridge._send = sent.append

    async def run():
        bridge._submit({
            "t": "submit", "rid": "X", "sub": 1, "text": text,
            "input_modalities": ["text"], "output_modalities": out,
        })
        deadline = time.monotonic() + 10
        while not any(m["t"] in ("done", "err") for m in sent):
            assert time.monotonic() < deadline, sent
            await asyncio.sleep(0.005)

    asyncio.run(run())

    (chunk,) = [m for m in sent if m["t"] == "chunk"]
    assert chunk["modality"] == "error" and chunk["metadata"] == {"status": 400}
    assert _refusal(length, limit) in chunk["data"].decode()
    assert server.socket.sent_of(ConductorMessageType.NEW_REQUEST) == []


def test_a_prompt_at_the_limit_is_admitted(served):
    name, server, out = served
    text, _, _ = _prompt_for(name, server.api.model, over=0)

    server.api.submit_request(text=text, input_modalities=["text"], output_modalities=out, streaming=False)

    deadline = time.monotonic() + 10
    while not server.socket.sent_of(ConductorMessageType.NEW_REQUEST):
        assert time.monotonic() < deadline, "the prompt at the limit never reached the conductor"
        time.sleep(0.005)
    (key,) = server.api.pending_requests
    assert server.api.pending_requests[key].error is None


def test_a_larger_output_cap_lowers_the_limit(tmp_path, monkeypatch):
    """The output counted is the request's own cap, as the conductor resolves it."""
    server, _ = _server("qwen3_5", tmp_path, monkeypatch)
    try:
        r = _post({
            "text": "a " * 100, "output_modalities": "text", "streaming": "false",
            "model_kwargs": json.dumps({"max_output_tokens": 8190}),
        })
        assert r.status_code == 400, r.text
        assert "the model takes at most 2 (its 8192-position context less the 8190 its" in r.json()["detail"]
        assert server.socket.sent_of(ConductorMessageType.NEW_REQUEST) == []
    finally:
        server.close()


def _submit(server, text: str, out: list[str], model_kwargs: dict, input_modalities=("text",), **request):
    """Submit a request; return its refusal, or None once it reaches the conductor."""
    server.api.submit_request(
        text=text, input_modalities=list(input_modalities), output_modalities=out,
        model_kwargs=model_kwargs, streaming=False, **request,
    )
    deadline = time.monotonic() + 10
    while True:
        (pending,) = server.api.pending_requests.values()
        if pending.error is not None:
            assert server.socket.sent_of(ConductorMessageType.NEW_REQUEST) == []
            return str(pending.error)
        if server.socket.sent_of(ConductorMessageType.NEW_REQUEST):
            return None
        assert time.monotonic() < deadline, "the request was neither refused nor forwarded"
        time.sleep(0.005)


# A cap past the decode Loop's max_iters: decode stops at the Loop, so the
# output counted is the Loop's bound, and a short prompt is served as before.
# Every cache here holds 32,768 positions.
LOOP_CASES = {
    "qwen3_omni": (_omni, ["text"], {"max_output_tokens": 65536}),
    "qwen3_omni_talker": (_omni, ["text", "audio"], {"talker_max_output_tokens": 65536}),
    "bagel": (_Bagel, ["text"], {"max_output_tokens": 32768}),
    "qwen3_tts": (_qwen3_tts, ["audio"], {"max_output_tokens": 32768}),
}


@pytest.mark.parametrize("name", sorted(LOOP_CASES))
def test_a_cap_past_the_decode_loop_is_served(name, tmp_path, monkeypatch):
    factory, out, model_kwargs = LOOP_CASES[name]
    server = _Server(model=factory(), model_name=name)
    server.api.upload_dir = tmp_path
    monkeypatch.setattr(entrypoint, "api_server", server.api)
    try:
        assert _submit(server, "describe a cat", out, model_kwargs) is None
    finally:
        server.close()


@pytest.mark.parametrize(("name", "walk"), [
    ("qwen3_omni", "thinker_decode"),
    ("qwen3_omni_talker", "talker_decode"),
    ("bagel", "decode"),
    ("qwen3_tts", "talker_decode"),
])
def test_the_output_counted_is_the_decode_loops_bound(name, walk):
    factory, out, model_kwargs = LOOP_CASES[name]
    model = factory()
    result = model.process_prompt("describe a cat", ["text"], out)
    tensors, kwargs = (result, {}) if isinstance(result, dict) else result
    uses = model.context_use(dict(tensors), ["text"], out, {**kwargs, **model_kwargs})
    key = "talker_kv" if name == "qwen3_omni_talker" else next(iter(uses))
    assert uses[key].generated == model.get_graph_walk_graphs()[walk].max_iters < 32768


def test_bagel_think_mode_image_output_counts_the_decode_loops_thinking():
    model = _Bagel(think_mode=True)
    tensors = model.process_prompt("a cat", ["text"], ["image"])
    use = model.context_use(tensors, ["text"], ["image"], {"max_output_tokens": 32768})
    # the thinking stops at the decode Loop's 2,048, the latents sit one past it
    assert use["kv"].generated == 2048 + 1


@pytest.mark.parametrize(("cap", "out"), [
    ("abc", ["text", "audio"]), (-1, ["text", "audio"]), (0, ["text", "audio"]),
    (True, ["text", "audio"]), (1.5, ["text", "audio"]),
    # the conductor reads the Talker cap for text output too
    ("abc", ["text"]),
])
def test_a_talker_cap_that_is_not_a_positive_integer_is_a_400(cap, out, tmp_path, monkeypatch):
    server, _ = _server("qwen3_omni", tmp_path, monkeypatch)
    try:
        error = _submit(server, "describe a cat", out, {"talker_max_output_tokens": cap})
        assert error is not None
        assert f"talker_max_output_tokens must be an integer >= 1; got {cap!r}" in error
    finally:
        server.close()


def test_a_cap_past_the_context_names_the_kwarg(tmp_path, monkeypatch):
    """A cap that alone fills the context leaves no room for any prompt; the
    refusal says which kwarg to lower."""
    server, _ = _server("qwen3_5", tmp_path, monkeypatch)
    try:
        error = _submit(server, "a " * 10, ["text"], {"max_output_tokens": 9000})
        assert error is not None
        assert "max_output_tokens lets the output take 9000 positions" in error
        assert "8192-position context" in error
    finally:
        server.close()


def test_qwen3_omni_audio_output_counts_the_talkers_cache(tmp_path, monkeypatch):
    """With audio output the Talker's cache is checked too, against its own
    output cap, and the refusal names that cache. A one-token Thinker cap
    leaves the Thinker room, so only the Talker is past its limit."""
    server, _ = _server("qwen3_omni", tmp_path, monkeypatch)
    model, out = server.api.model, ["text", "audio"]
    kwargs = {"max_output_tokens": 1, "talker_max_output_tokens": 1000}
    one = _use(model, "a ", out)["talker_kv"]
    limit = _paged_kv_configs(model, None)["talker_kv"].max_seq_len - 1000
    try:
        error = _submit(server, "a " * (limit - one.prompt + 2), out, kwargs)
        assert error is not None
        assert (
            f"The prompt takes {limit + 1} positions in its 'talker_kv' cache; "
            f"the model takes at most {limit} "
        ) in error
    finally:
        server.close()


# --------------------------------------------------------------------------
# Counting: each model's count matches the positions its prefill takes.
# --------------------------------------------------------------------------

def test_qwen3_5_counts_an_image_by_its_mrope_span(tmp_path):
    """Text and image in one prompt: the count is where `LLMSubmodule`'s vision
    prefill leaves the position cursor, not the image's token count."""
    model = _qwen3_5(tmp_path)
    mods = ["image", "text"]
    tensors, kwargs = model.process_prompt(
        "look at this", mods, ["text"], tensors={"image_inputs": [torch.zeros(3, 32, 48)]},
    )
    use = model.context_use(tensors, mods, ["text"], kwargs)
    order = tuple(0 if s.modality == TEXT else 1 for s in prefill_plan(parts_from_modalities(mods)))

    assert use == {"kv_cache": ContextUse(_qwen3_5_vision_prefill(model, order, tensors), 2048)}
    # max(4, 6) positions and two sentinels, against 24 tokens and two sentinels
    text = sum(t.numel() for t in tensors["text_inputs"])
    assert use["kv_cache"].prompt == text + 6 + 2


def _qwen3_5_vision_prefill(model: Qwen3_5DenseModel, order: tuple, inputs: dict) -> int:
    """How far `LLMSubmodule`'s vision prefill moves the position cursor."""
    hidden, merge = 4, model.vision_config.spatial_merge_size ** 2
    stub = SimpleNamespace(
        vision_config=model.vision_config,
        get_device=lambda: torch.device("cpu"),
        _sentinel_embeds=lambda: (torch.zeros(1, hidden), torch.zeros(1, hidden)),
        node_resources={ROPE: SimpleNamespace(position=lambda rid, label: 0)},
        model=SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: torch.zeros(len(ids), hidden))),
    )
    tokens = sum(int(grid.prod()) // merge for grid in inputs.get("image_grid_thw", []))
    prefilled = LLMSubmodule._vision_inputs(
        stub,
        SimpleNamespace(request_id="r", step_metadata={"prefill_order": order}),
        {**inputs, "vision_embeds": [torch.zeros(tokens, hidden)]},
    )
    return prefilled.tensor_inputs["mrope_advance"]


@pytest.fixture(scope="module")
def qwen3_5_video(tmp_path_factory) -> Qwen3_5DenseModel:
    """Qwen3.5 over the video tests' synthetic checkpoint: a real tokenizer,
    chat template and video processor."""
    return Qwen3_5DenseModel(str(write_synthetic_checkpoint(tmp_path_factory.mktemp("qwen3_5_video"))))


def _video_request(model: Qwen3_5DenseModel, layout: str):
    """One of the video tests' layouts: its modalities, texts, sampled videos
    and images."""
    modalities, names = VIDEO_LAYOUTS[layout]
    texts = [f"Question {i}: what happens?" for i in range(modalities.count(TEXT))]
    videos = [sampled(*a_video(name), model.video_processor) for name in names]
    images = [an_image() for _ in range(modalities.count("image"))]
    return modalities, texts, videos, images


@pytest.mark.parametrize("layout", sorted(VIDEO_LAYOUTS))
def test_qwen3_5_counts_each_video_group_where_the_vision_prefill_leaves_it(qwen3_5_video, layout):
    """A video's temporal groups are placed one by one like one-frame images,
    each with two sentinels; its timestamps are text. The count is where the
    vision prefill of the request's schedule leaves the cursor
    (`test_qwen3_5_video` checks that row against transformers)."""
    model = qwen3_5_video
    modalities, texts, videos, images = _video_request(model, layout)
    tensors, kwargs = model.process_prompt(
        "\n".join(texts), modalities, ["text"],
        tensors={"image_inputs": images, "video_inputs": [frames for frames, _ in videos]},
        prompt_parts=parts_from_modalities(modalities, texts),
        input_metadata={"video_inputs": [metadata for _, metadata in videos]},
    )
    use = model.context_use(tensors, modalities, ["text"], kwargs)
    (step,) = model._prefill_schedule(modalities, tensors, kwargs["video_groups"])

    assert use["kv_cache"].prompt == _qwen3_5_vision_prefill(model, step.order, {
        "text_inputs": step.input_tensors["text_inputs"],
        "image_grid_thw": step.input_tensors["image_grid_thw"],
    })


@pytest.mark.parametrize("over", [0, 1], ids=["at-the-limit", "one-over"])
def test_a_video_that_pushes_the_prompt_past_the_context_is_a_400(qwen3_5_video, over, tmp_path, monkeypatch):
    """A one-line question with a video, served: the video's positions count,
    so a context that fits the text alone refuses the request one position
    past the limit and admits it at the limit."""
    model = qwen3_5_video
    modalities, (text,), ((frames, metadata),), _ = _video_request(model, "video_then_question")
    tensors, kwargs = model.process_prompt(
        text, modalities, ["text"], tensors={"video_inputs": [frames]},
        input_metadata={"video_inputs": [metadata]},
    )
    use = model.context_use(tensors, modalities, ["text"], kwargs)["kv_cache"]
    # so a count of the text alone would admit the one-over case
    assert use.prompt - 1 > sum(t.numel() for t in tensors["text_inputs"])
    monkeypatch.setattr(model, "load_video", lambda path, device: TensorAndMetadata(frames, metadata))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\0")

    context = use.prompt + use.generated - over
    server = _Server(
        model=model, model_config={"resources": {"kv_cache": {"max_seq_len": context}}},
        model_name="qwen3_5",
    )
    server.api.upload_dir = tmp_path
    monkeypatch.setattr(entrypoint, "api_server", server.api)
    try:
        error = _submit(
            server, text, ["text"], {}, input_modalities=modalities, file_paths={"video": [str(video)]},
        )
    finally:
        server.close()
    if over:
        assert error is not None and _refusal(use.prompt, use.prompt - 1) in error
    else:
        assert error is None


def test_fish_s2_pro_counts_a_clones_reference_frames():
    """A clone's reference frames take one position each in the slow AR's
    prefill, spliced in after the system prefix."""
    model, ar = _fish(), _fish_ar()
    clip = torch.zeros(20 * model.config.codec.hop_length + 1)
    tensors, kwargs = model.process_prompt(
        "hi", ["text", "audio"], ["audio"], {"audio_inputs": [clip]}, ref_text="r",
    )
    prefix_len, ref_frames = kwargs["prompt_layout"]
    codes = torch.zeros(ref_frames, model.config.fast_ar.num_codebooks, dtype=torch.long)
    # the stand-in tokenizer's ids are past the tiny AR's vocabulary; only the length matters
    ids = torch.ones_like(tensors["prompt_ids"][0])
    with torch.no_grad():
        prefill = ar._build_prefill(ids, prefix_len, ref_frames, codes)

    use = model.context_use(tensors, ["text", "audio"], ["audio"], kwargs)

    assert ref_frames == 21
    assert use == {AR_KV: ContextUse(prefill.shape[0], 64)}


@pytest.mark.parametrize(("out", "walks"), [(["text"], 1), (["image"], 2)])
def test_bagel_counts_each_image_walk_one_position(out, walks):
    """An understood image takes its ViT walk; one being edited its VAE walk
    too. Each walk advances the position by one, whatever its slots."""
    from mstar.model.bagel.submodules import IMAGE_ADVANCE

    model = _Bagel()
    mods = ["image", "image", "text"]
    images = [torch.rand(3, 28, 28), torch.rand(3, 28, 42)]
    result = model.process_prompt("compare them", mods, out, tensors={"image_inputs": images})
    tensors = result if isinstance(result, dict) else result.new_input_tensors
    tensors = {"image_inputs": images, **tensors}
    schedule = model._build_prefill_schedule(
        mods, {k: list(range(len(v))) for k, v in tensors.items()}, is_understanding=out == ["text"],
    )
    image_walks = [walk for walk, _ in schedule if walk in ("prefill_vit", "prefill_vae")]
    assert len(image_walks) == 2 * walks

    use = model.context_use(tensors, mods, out, {})

    text = sum(t.numel() for t in tensors["text_inputs"])
    generated = ContextUse(0, 2048) if out == ["text"] else ContextUse(0, 1, kwarg=None)
    assert use == {"kv": generated._replace(prompt=text + len(image_walks) * IMAGE_ADVANCE)}


def test_bagel_think_mode_image_output_counts_its_thinking():
    model = _Bagel(think_mode=True)
    tensors = model.process_prompt("a cat", ["text"], ["image"])
    use = model.context_use(tensors, ["text"], ["image"], {"max_output_tokens": 300})
    # the think span is prefilled first, then the prompt; up to 300 thought
    # tokens are decoded and the latents sit one position past them
    assert use["kv"] == ContextUse(sum(t.numel() for t in tensors["text_inputs"]), 301)


def test_qwen3_omni_counts_match_the_thinker_prefill():
    """Thinker positions and the Talker's kept states, prefilling a text,
    audio, image and video prompt walk by walk through
    `ThinkerSubmodule.prepare_inputs`."""
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        _get_feat_extract_output_lengths,
    )

    model = _omni()
    mods = ["audio", "image", "video", "text"]
    tensors = model.process_prompt(
        "what do you hear and see", mods, ["text", "audio"],
        tensors={
            "audio_inputs": [torch.zeros(16000)],
            "image_inputs": [torch.rand(3, 32, 48)],
            "video_inputs": [torch.zeros(4, 3, 24, 16)],
        },
        input_metadata={"video_inputs": [{"video_sample_fps": 4.0}]},
    )
    use = model.context_use(tensors, mods, ["text", "audio"], {})

    hidden, cursor, kept = 4, [0], 0
    stub = SimpleNamespace(
        config=model.config,
        get_device=lambda: torch.device("cpu"),
        node_resources={THINKER_POS: SimpleNamespace(position=lambda rid, label: cursor[0])},
        model=SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: torch.zeros(len(ids), hidden))),
        _get_talker_text_mask=lambda ids: talker_text_mask(ids, model.config),
        _wrap_audio_input=lambda e: torch.cat([torch.zeros(1, hidden), e, torch.zeros(1, hidden)]),
        _wrap_vision_input=lambda e: torch.cat([torch.zeros(1, hidden), e, torch.zeros(1, hidden)]),
    )
    for step in prefill_plan(parts_from_modalities(mods)):
        info = SimpleNamespace(rid_handle="r")
        if step.modality == TEXT:
            walk, inputs = "prefill_text", {"text_inputs": [tensors["text_inputs"][step.index]]}
        elif step.modality == "audio":
            # the audio encoder's output length (HF's)
            n = int(_get_feat_extract_output_lengths(tensors["audio_seqlens"][step.index]))
            walk, inputs = "prefill_audio", {"audio_embeds": [torch.zeros(n, hidden)]}
        else:
            key = "image_grid_thw" if step.modality == "image" else "video_grid_thw"
            grid = tensors[key][step.index]
            n = int(grid[0] * (grid[1] // 2) * (grid[2] // 2))
            inputs = {"vision_embeds": [torch.zeros(n, hidden)], "image_grid_thw": [grid], "deepstack": []}
            if step.modality == "video":
                inputs["video_second_per_grid"] = tensors["video_second_per_grid"]
            walk = "prefill_vision"
        prefilled = ThinkerSubmodule.prepare_inputs(stub, walk, info, inputs)
        cursor[0] += int(prefilled.tensor_inputs.get("mrope_pos_advance", prefilled.input_seq_len))
        mm, text_kept = prefilled.tensor_inputs["masks_for_talker"].bool()
        kept += int(((text_kept & ~mm) | mm).sum())

    assert use["thinker_kv"] == ContextUse(cursor[0], 2048)
    assert use["talker_kv"] == ContextUse(kept + 6, 2048, kwarg="talker_max_output_tokens")


@pytest.mark.parametrize(
    ("layout", "speaker_id", "language_id", "xvec"),
    [
        ([0, 4, 1], 40, -1, False),            # built-in speaker, streamed text
        ([0, 4, 0], 40, 41, False),            # with a language tag, whole text
        ([6, 4, 1], -1, -1, False),            # an instruction, no speaker
        ([0, 4, 1, 0, 0], -1, -1, True),       # x-vector clone
        ([0, 4, 1, 7, 5], -1, -1, True),       # in-context clone, streamed
        ([0, 4, 0, 7, 5], -1, -1, True),       # in-context clone, whole text
        ([0, 1, 1, 7, 5], -1, -1, True),       # text shorter than the reference frames
    ],
)
def test_qwen3_tts_count_matches_the_talker_prefill(layout, speaker_id, language_id, xvec):
    config = _tiny_model_config()
    submodule = TalkerSubmodule(Qwen3TTSTalkerModel(config), Qwen3TTSCodePredictor(config), config)
    submodule.CHATML_ASSISTANT_PREFIX_TOKEN_IDS = (1, 2, 3)
    submodule.CHATML_ASSISTANT_SUFFIX_TOKEN_IDS = (8, 9, 10, 11, 12)
    instruct = torch.arange(20, 20 + layout[0])
    text = torch.arange(100, 100 + layout[1])
    assistant = torch.cat([torch.tensor([1, 2, 3]), text, torch.tensor([8, 9, 10, 11, 12])])
    # two transcript tokens in the reference turn
    clone = len(layout) == 5 and layout[4]
    reference = torch.tensor([1, 2, 3, 50, 51, 8, 9]) if clone else torch.tensor([], dtype=torch.long)

    prefill = submodule._build_prefill(
        request_id="r", text_ids=torch.cat([instruct, assistant, reference]), prompt_layout=layout,
        speaker_id=speaker_id, language_id=language_id,
        speaker_embed=torch.randn(config.talker.hidden_size) if xvec else None,
        ref_codes=torch.randint(0, 32, (5, config.talker.num_code_groups)) if xvec else None,
    )

    assert talker_prefill_len(layout, speaker_id, language_id, xvec, prefix_len=3) == prefill.shape[0]


def test_a_model_that_declares_nothing_is_not_checked(tmp_path, monkeypatch):
    """No count, no check: the request goes to the conductor as before."""
    server, _ = _server("qwen3_5", tmp_path, monkeypatch)
    monkeypatch.setattr(Qwen3_5DenseModel, "context_use", Model.context_use)
    try:
        server.api.submit_request(
            text="a " * 9000, input_modalities=["text"], output_modalities=["text"], streaming=False,
        )
        deadline = time.monotonic() + 10
        while not server.socket.sent_of(ConductorMessageType.NEW_REQUEST):
            assert time.monotonic() < deadline
            time.sleep(0.005)
    finally:
        server.close()
