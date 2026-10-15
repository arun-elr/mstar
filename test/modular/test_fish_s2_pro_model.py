"""Fish S2-Pro model contract, weight maps, AR submodule and adapter (CPU, tiny random weights)."""

from __future__ import annotations

import base64
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from mstar.api_server.openai import adapters
from mstar.api_server.openai.protocol import SpeechRequest
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.graph.base import TensorPointerInfo
from mstar.model.fish_s2_pro.components.dac import apply_rotary_emb
from mstar.model.fish_s2_pro.components.fast_ar import FishFastAR
from mstar.model.fish_s2_pro.components.slow_ar import FishSlowAR, remap_slow_ar_weights
from mstar.model.fish_s2_pro.config import (
    AR_SAMPLER,
    FishDACConfig,
    FishFastARConfig,
    FishGenerationConfig,
    FishS2ProConfig,
    FishSlowARConfig,
)
from mstar.model.fish_s2_pro.fish_s2_pro_model import FishS2ProModel, load_fast_ar_weights
from mstar.model.fish_s2_pro.submodules import ARSubmodule
from mstar.model.registry import HF_MODELS, get_model_class

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "fish_s2_pro.yaml"
SEMANTIC_START, SEMANTIC_CODES, EOS_ID = 100, 32, 99


def tiny_config(**generation) -> FishS2ProConfig:
    return FishS2ProConfig(
        semantic_start_token_id=SEMANTIC_START, semantic_end_token_id=SEMANTIC_START + SEMANTIC_CODES - 1,
        eos_token_id=EOS_ID,
        slow_ar=FishSlowARConfig(dim=32, n_layer=2, n_head=4, n_local_heads=2, head_dim=8,
                                 intermediate_size=48, vocab_size=140, max_seq_len=512),
        fast_ar=FishFastARConfig(dim=32, text_dim=32, n_layer=2, n_head=4, n_local_heads=2, head_dim=8,
                                 intermediate_size=48, vocab_size=SEMANTIC_CODES, num_codebooks=4),
        codec=FishDACConfig(semantic_codebook_size=SEMANTIC_CODES, codebook_size=8, n_residual_codebooks=3),
        generation=FishGenerationConfig(**{"max_new_tokens": 64, **generation}),
    )


class _TokenizerStub:
    """Special tokens are single ids; other text is one id per character, so a
    tag encoded apart from its text gives other ids than the two together."""

    SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2, "<|voice|>": 3, "<|phoneme_start|>": 4, "<|phoneme_end|>": 5}

    all_special_ids = [1, 2]
    added_tokens_decoder = {3: None, 4: None, 5: None}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in re.split(r"(<\|(?:im_start|im_end|voice|phoneme_start|phoneme_end)\|>)", text):
            ids += [self.SPECIAL[piece]] if piece in self.SPECIAL else [1000 + ord(c) for c in piece]
        return ids

    def convert_tokens_to_ids(self, tokens):
        return [self.SPECIAL[t] for t in tokens]

    def convert_ids_to_tokens(self, token_id):
        special = {v: k for k, v in self.SPECIAL.items()}
        return special[token_id] if token_id in special else chr(token_id - 1000)


def tiny_model(**generation) -> FishS2ProModel:
    model = object.__new__(FishS2ProModel)
    model.config = tiny_config(**generation)
    model.tokenizer = _TokenizerStub()
    model._special_ids = {1, 2, 3}
    model._submodule_cache = {}
    return model


def tiny_ar(seed: int = 0) -> ARSubmodule:
    torch.manual_seed(seed)
    config = tiny_config()
    slow_ar, fast_ar = FishSlowAR(config), FishFastAR(config.fast_ar)
    with torch.no_grad():
        for param in [*slow_ar.parameters(), *fast_ar.parameters()]:
            param.normal_(0, 0.5)
    slow_ar.build_semantic_head()
    return ARSubmodule(slow_ar.double().eval(), fast_ar.double().eval(), config)


# ---------------------------------------------------------------------------
# Registry, deployment config, adapter
# ---------------------------------------------------------------------------


def test_registry_and_deployment_config():
    assert get_model_class("fish_s2_pro") is FishS2ProModel
    assert HF_MODELS["fish_s2_pro"]["model_path_hf"] == "fishaudio/s2-pro"
    deployment = yaml.safe_load(CONFIG_PATH.read_text())
    assert deployment["model"] == "fish_s2_pro"
    walks = {walk for group in deployment["node_groups"] for walk in group["graph_walks"]}
    assert walks == set(tiny_model().get_graph_walk_graphs())


def test_adapter_clones_from_a_data_url_and_drops_a_lone_ref_text(tmp_path):
    adapter = adapters.get_adapter("fish_s2_pro")
    assert isinstance(adapter, adapters.FishS2ProAdapter)
    wav = "data:audio/wav;base64," + base64.b64encode(b"RIFF0000WAVE").decode()
    clone = adapter.speech_to_request(
        SpeechRequest(input="hi", ref_audio=wav, ref_text="ref words", temperature=0.5, top_k=10), tmp_path)
    assert clone.input_modalities == ["text", "audio"] and clone.output_modalities == ["audio"]
    assert Path(clone.file_paths["audio"][0]).read_bytes() == b"RIFF0000WAVE"
    assert clone.model_kwargs["ref_text"] == "ref words"
    assert clone.model_kwargs["temperature"] == 0.5 and clone.model_kwargs["top_k"] == 10

    plain = adapter.speech_to_request(SpeechRequest(input="hi", ref_text="orphan"), tmp_path)
    assert plain.input_modalities == ["text"] and "ref_text" not in plain.model_kwargs


def test_adapter_refuses_server_paths(tmp_path):
    with pytest.raises(ValueError, match="paths on the server are refused"):
        adapters.FishS2ProAdapter().speech_to_request(SpeechRequest(input="hi", ref_audio="/etc/passwd"), tmp_path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_reads_fish_field_names(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "fish_qwen3_omni", "semantic_start_token_id": 151678, "semantic_end_token_id": 155773,
        "eos_token_id": 151645,
        "text_config": {"dim": 2560, "n_layer": 36, "n_head": 32, "n_local_heads": 8, "rope_base": 1e6},
        "audio_decoder_config": {"n_layer": 4, "max_seq_len": 11, "vocab_size": 4096, "num_codebooks": 10},
    }))
    config = FishS2ProConfig.from_pretrained(tmp_path, codec_overrides={"chunk_frames": 16})
    assert (config.slow_ar.n_layer, config.slow_ar.n_local_heads, config.fast_ar.n_layer) == (36, 8, 4)
    assert config.eos_code == 4096 and config.semantic_head_size == 4097
    assert config.codec.hop_length == 2048 and config.codec.chunk_frames == 16
    assert config.codec.left_context_frames == 8 * 127 + 10 and config.codec.exact_streaming
    assert config.vq_scale == pytest.approx(1 / math.sqrt(11))


def test_config_rejects_inconsistent_checkpoints():
    with pytest.raises(ValueError, match="model_type"):
        FishS2ProConfig(model_type="fish_qwen3_5_moe")
    with pytest.raises(ValueError, match="Fast AR vocabulary"):
        FishS2ProConfig(semantic_end_token_id=151678 + 1000)
    with pytest.raises(ValueError, match="RoPE table"):
        FishS2ProConfig(generation=FishGenerationConfig(max_new_tokens=4097))


# ---------------------------------------------------------------------------
# Weight maps
# ---------------------------------------------------------------------------


def fish_slow_ar_checkpoint(config: FishS2ProConfig) -> dict[str, torch.Tensor]:
    s = config.slow_ar
    q, kv = s.n_head * s.head_dim, s.n_local_heads * s.head_dim
    state = {"text_model.model.embeddings.weight": torch.randn(s.vocab_size, s.dim),
             "text_model.model.norm.weight": torch.randn(s.dim)}
    for i in range(s.n_layer):
        p = f"text_model.model.layers.{i}."
        state.update({
            p + "attention.wqkv.weight": torch.randn(q + 2 * kv, s.dim),
            p + "attention.wo.weight": torch.randn(s.dim, q),
            p + "attention.q_norm.weight": torch.randn(s.head_dim),
            p + "attention.k_norm.weight": torch.randn(s.head_dim),
            p + "attention_norm.weight": torch.randn(s.dim),
            p + "ffn_norm.weight": torch.randn(s.dim),
            p + "feed_forward.w1.weight": torch.randn(s.intermediate_size, s.dim),
            p + "feed_forward.w3.weight": torch.randn(s.intermediate_size, s.dim),
            p + "feed_forward.w2.weight": torch.randn(s.dim, s.intermediate_size),
        })
    return state


def test_slow_ar_weight_map_splits_wqkv_and_renames():
    config = tiny_config()
    state = fish_slow_ar_checkpoint(config)
    mapped = dict(remap_slow_ar_weights(
        [*state.items(), ("audio_decoder.output.weight", torch.zeros(1))], config))
    wqkv = state["text_model.model.layers.1.attention.wqkv.weight"]
    assert torch.equal(torch.cat([mapped[f"layers.1.self_attn.{n}_proj.weight"] for n in "qkv"]), wqkv)
    assert mapped["layers.1.self_attn.q_proj.weight"].shape[0] == 4 * 8
    layer0 = "text_model.model.layers.0."
    assert torch.equal(mapped["layers.0.mlp.gate_proj.weight"], state[layer0 + "feed_forward.w1.weight"])
    assert torch.equal(mapped["layers.0.mlp.up_proj.weight"], state[layer0 + "feed_forward.w3.weight"])
    assert torch.equal(mapped["layers.0.post_attention_layernorm.weight"],
                       state["text_model.model.layers.0.ffn_norm.weight"])
    assert "embed_tokens.weight" in mapped and "norm.weight" in mapped
    assert not any(name.startswith("audio_decoder") for name in mapped)


def test_slow_ar_loads_completely_and_scores_only_semantic_tokens_and_eos():
    config = tiny_config()
    state = fish_slow_ar_checkpoint(config)
    slow_ar = FishSlowAR(config)
    slow_ar.load_weights(state.items())
    lm = dict(slow_ar.lm.named_parameters())
    wqkv = state["text_model.model.layers.0.attention.wqkv.weight"]
    assert torch.equal(lm["layers.0.self_attn.qkv_proj.weight"], wqkv)
    embed = state["text_model.model.embeddings.weight"]
    assert torch.equal(slow_ar.semantic_head[:SEMANTIC_CODES], embed[SEMANTIC_START:SEMANTIC_START + SEMANTIC_CODES])
    assert torch.equal(slow_ar.semantic_head[-1], embed[EOS_ID])
    hidden = torch.randn(3, config.slow_ar.dim)
    full = hidden @ embed.t()
    rows = [*range(SEMANTIC_START, SEMANTIC_START + SEMANTIC_CODES), EOS_ID]
    torch.testing.assert_close(slow_ar.semantic_logits(hidden), full[:, rows])

    del state["text_model.model.layers.1.feed_forward.w2.weight"]
    with pytest.raises(RuntimeError, match="unloaded"):
        FishSlowAR(config).load_weights(state.items())


def test_fast_ar_loader_rejects_gaps_extras_and_shape_mismatches():
    cfg = tiny_config().fast_ar
    state = {name: torch.randn_like(p) for name, p in FishFastAR(cfg).named_parameters()}
    fast_ar = FishFastAR(cfg)
    load_fast_ar_weights(fast_ar, ((f"audio_decoder.{k}", v) for k, v in state.items()))
    assert torch.equal(fast_ar.output.weight, state["output.weight"])
    with pytest.raises(RuntimeError, match="unloaded"):
        load_fast_ar_weights(FishFastAR(cfg), [(f"audio_decoder.{k}", v) for k, v in state.items()
                                               if k != "norm.weight"])
    with pytest.raises(RuntimeError, match="no place"):
        load_fast_ar_weights(FishFastAR(cfg), [("audio_decoder.extra.weight", torch.zeros(1))])
    with pytest.raises(RuntimeError, match="checkpoint"):
        load_fast_ar_weights(FishFastAR(cfg), [("audio_decoder.norm.weight", torch.zeros(3))])


# ---------------------------------------------------------------------------
# Fast AR and the AR submodule
# ---------------------------------------------------------------------------


def fast_ar_without_cache(fast_ar: FishFastAR, slow_hidden: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
    """Recompute the whole codebook sequence with causal attention at every position."""
    cfg = fast_ar.config
    codes = [semantic]
    seq = [fast_ar.project_in(slow_hidden), fast_ar.embeddings(semantic)]
    for pos in range(1, cfg.num_codebooks):
        x = torch.stack(seq, dim=1)
        freqs = fast_ar.freqs_cis[:x.shape[1]].unsqueeze(0)
        for layer in fast_ar.layers:
            att = layer.attention
            h = layer.attention_norm(x)
            q, k, v = att.wqkv(h).split([cfg.n_head * cfg.head_dim, *[cfg.n_local_heads * cfg.head_dim] * 2], -1)
            q = apply_rotary_emb(q.view(*x.shape[:2], cfg.n_head, -1), freqs)
            k = apply_rotary_emb(k.view(*x.shape[:2], cfg.n_local_heads, -1), freqs)
            v = v.view(*x.shape[:2], cfg.n_local_heads, -1)
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               is_causal=True, enable_gqa=True)
            x = x + att.wo(y.transpose(1, 2).reshape(*x.shape[:2], -1))
            x = x + layer.feed_forward(layer.ffn_norm(x))
        code = fast_ar.output(fast_ar.norm(x[:, pos])).argmax(-1)
        codes.append(code)
        seq.append(fast_ar.embeddings(code))
    return torch.stack(codes, dim=1)


def test_fast_ar_cached_loop_equals_a_full_causal_recompute():
    ar = tiny_ar()
    hidden = torch.randn(5, 32, dtype=torch.float64)
    semantic = torch.tensor([0, 3, 31, 7, 7])
    kv = torch.zeros(ar.fast_ar.kv_cache_shape(5), dtype=torch.float64)
    with torch.no_grad():
        codes = ar.fast_ar(hidden, semantic, kv)
        expected = fast_ar_without_cache(ar.fast_ar, hidden, semantic)
    assert torch.equal(codes, expected)
    assert torch.equal(codes[:, 0], semantic)


def test_frame_embeddings_follow_the_reference_formula():
    ar = tiny_ar()
    codes = torch.tensor([[3, 1, 2, 0], [31, 7, 7, 7]])
    offsets = torch.arange(4) * SEMANTIC_CODES
    vq = ar.fast_ar.codebook_embeddings.weight[codes + offsets].sum(1)
    token = ar.slow_ar.semantic_head[codes[:, 0]]
    torch.testing.assert_close(ar.reference_embeds(codes), (token + vq) / math.sqrt(5))
    # <|im_end|> (head index 32) embeds its own row, with codebook 0 zeroed as the Fast AR saw it
    head_index = torch.tensor([32, 31])
    torch.testing.assert_close(
        ar.next_input_embeds(head_index, codes),
        (ar.slow_ar.semantic_head[head_index] + vq) * (1 / math.sqrt(5)),
    )


def test_clone_prefill_splices_reference_frames_after_the_prefix():
    ar = tiny_ar()
    ids = torch.tensor([5, 6, 7, 8, 9])
    ref_codes = torch.tensor([[1, 0, 0, 0], [2, 1, 1, 1], [9, 9, 9, 9]])
    with torch.no_grad():
        embeds = ar._build_prefill(ids, torch.tensor([2, 2]), ref_codes)
        plain = ar._build_prefill(ids, torch.tensor([0, 0]), None)
    assert embeds.shape == (7, 32)
    torch.testing.assert_close(embeds[:2], plain[:2])
    torch.testing.assert_close(embeds[2:4], ar.reference_embeds(ref_codes[:2]))
    torch.testing.assert_close(embeds[4:], plain[2:])
    with pytest.raises(ValueError, match="needs 4 reference frames"):
        ar._build_prefill(ids, torch.tensor([2, 4]), ref_codes)


def test_check_stop_on_im_end_on_the_frame_limit_and_not_with_ignore_eos():
    ar = tiny_ar()

    def info(ignore_eos=False):
        return SimpleNamespace(step_metadata={"ar_max_tokens": 3}, max_tokens=99,
                               resource_configs={AR_SAMPLER: SamplingReqConfig(ignore_eos=ignore_eos)})

    def step(rid, head, ignore_eos=False):
        outputs = {"new_token": [torch.tensor([head])]}
        ar.postprocess(rid, info(ignore_eos), outputs)
        return ar.check_stop(rid, info(ignore_eos), outputs)

    assert step("a", 4) == set()
    assert step("a", ar.eos_code) == {"ar_decode_loop"}
    assert step("b", ar.eos_code, ignore_eos=True) == set()
    assert step("b", 1) == set()
    assert step("b", 2) == {"ar_decode_loop"}

    # speculation: step N+1's postprocess runs before step N's stop check
    frames = [{"new_token": [torch.tensor([head])]} for head in (4, 5, 6)]
    ar.postprocess("c", info(), frames[0])
    assert ar.check_stop("c", info(), frames[0]) == set()
    ar.postprocess("c", info(), frames[1])
    ar.postprocess("c", info(), frames[2])
    assert ar.check_stop("c", info(), frames[1]) == set()
    assert ar.check_stop("c", info(), frames[2]) == {"ar_decode_loop"}


def test_eos_is_not_sampled_on_the_prefill_frame_or_with_ignore_eos():
    ar = tiny_ar()

    def info(ignore_eos=False):
        return SimpleNamespace(rid_handle=0, request_id="r",
                               resource_configs={AR_SAMPLER: SamplingReqConfig(ignore_eos=ignore_eos)})

    dim = ar.config.slow_ar.dim
    prefill = ar.prepare_inputs("ar_prefill", info(), {
        "prompt_ids": [torch.tensor([4, 5, 6])], "prompt_layout": [torch.tensor([3, 0])]})
    decode = ar.prepare_inputs("ar_decode", info(), {"ar_input_embeds": [torch.zeros(1, dim)]})
    ignoring = ar.prepare_inputs("ar_decode", info(ignore_eos=True), {"ar_input_embeds": [torch.zeros(1, dim)]})
    assert [item.tensor_inputs["suppress_eos"].item() for item in (prefill, decode, ignoring)] == [True, False, True]

    # EOS scores highest in every row; only the unsuppressed row may sample it
    ar.slow_ar.forward = lambda embeds, label: embeds
    rows = torch.zeros(3, ar.eos_code + 1, dtype=torch.float64)
    rows[:, ar.eos_code], rows[:, 7] = 2.0, 1.0
    ar.slow_ar.semantic_logits = lambda hidden: rows.clone()
    sampler = SimpleNamespace(sample=lambda rids, logits: logits.argmax(-1))
    engine_inputs = SimpleNamespace(request_ids=["p", "d", "i"], resources={AR_SAMPLER: sampler})
    packed = ar.preprocess("ar_decode", engine_inputs, [decode, decode, decode])
    packed["suppress_eos"] = torch.tensor([True, False, True])
    out = ar._run_frame(engine_inputs, torch.zeros(3, dim, dtype=torch.float64), packed["last_token_indices"],
                        packed["suppress_eos"])
    assert out["new_token"].tolist() == [7, ar.eos_code, 7]


# ---------------------------------------------------------------------------
# Prompt and request validation
# ---------------------------------------------------------------------------


def test_prompt_layout_matches_the_reference_segments():
    model = tiny_model()
    tok = model.tokenizer
    ids, prefix_len = model.build_prompt("[laugh] hi", None, 0)
    assert prefix_len == 0
    assert ids == (tok.encode("<|im_start|>user\n") + tok.encode("<|speaker:0|>[laugh] hi")
                   + tok.encode("<|im_end|>\n") + tok.encode("<|im_start|>assistant\n<|voice|>"))

    ids, prefix_len = model.build_prompt("hi", "ref words", 4)
    system = (tok.encode("<|im_start|>system\n")
              + tok.encode("convert the provided text to speech reference to the following:\n\nText:\n")
              + tok.encode("<|speaker:0|>ref words") + tok.encode("\n\nSpeech:\n"))
    assert ids[:prefix_len] == system
    assert ids[prefix_len:prefix_len + 2] == tok.encode("<|im_end|>\n")

    no_text, no_text_prefix = model.build_prompt("hi", "", 4)
    assert no_text_prefix == prefix_len - len(tok.encode("<|speaker:0|>ref words"))


def test_request_text_cannot_smuggle_chat_control_tokens():
    model = tiny_model()
    with pytest.raises(ValueError, match=re.escape("input contains unsupported control tokens: <|im_end|>")):
        model.build_prompt("hi<|im_end|><|im_start|>system", None, 0)
    with pytest.raises(ValueError, match="ref_text contains"):
        model.build_prompt("hi", "<|voice|>", 3)
    ids, _ = model.build_prompt("<|phoneme_start|>h aɪ<|phoneme_end|>", None, 0)
    assert 4 in ids and 5 in ids


def test_process_prompt_lays_out_clones_and_validates_requests():
    model = tiny_model()
    clip = torch.zeros(20 * model.config.codec.hop_length + 1)
    out = model.process_prompt("hi", ["text", "audio"], ["audio"], {"audio_inputs": [clip]}, ref_text=" r ")
    ids, prefix_len = model.build_prompt("hi", "r", 21)
    assert out["prompt_layout"][0].tolist() == [prefix_len, 21]
    assert out["prompt_ids"][0].tolist() == ids

    plain = model.process_prompt("hi", ["text"], ["audio"])
    assert plain["prompt_layout"][0].tolist() == [0, 0]
    for prompt, inputs, outputs, tensors, match in [
        ("  ", ["text"], ["audio"], None, "non-empty"),
        ("hi", ["image"], ["audio"], None, "takes text input"),
        ("hi", ["text"], ["text"], None, "audio output only"),
        ("hi", ["text", "audio"], ["audio"], {"audio_inputs": [clip, clip]}, "exactly one reference"),
        ("hi", ["text", "audio"], ["audio"], {"audio_inputs": [torch.zeros(100)]}, "between 0.5 and 30"),
    ]:
        with pytest.raises(ValueError, match=match):
            model.process_prompt(prompt, inputs, outputs, tensors)
    with pytest.raises(ValueError, match="max_new_tokens must be in 1..4096"):
        model.process_prompt("hi", ["text"], ["audio"], max_new_tokens=5000)
    with pytest.raises(ValueError, match="exceeds the 32,768-token context"):
        model.process_prompt("x" * 480, ["text"], ["audio"], max_new_tokens=64)


# ---------------------------------------------------------------------------
# Conductor state machine and sampling
# ---------------------------------------------------------------------------


def pointers(*names):
    return {
        name: [TensorPointerInfo(dims=[1], dtype=torch.long, nbytes=8, address=0, stride=[1], uuid=i,
                                 source_session_id="host:0", source_entity="api_server")]
        for i, name in enumerate(names)
    }


def test_partition_state_machine_routes_clones_through_the_ref_encoder():
    model = tiny_model()
    signals = pointers("prompt_ids", "prompt_layout", "audio_inputs")
    clone = model.get_initial_forward_pass_args("AR", ["text", "audio"], ["audio"], signals, {"max_new_tokens": 7})
    assert clone.full_metadata.graph_walk == "ar_prefill_clone" and clone.full_metadata.is_prefill
    assert {(edge.name, edge.next_node) for edge in clone.inputs} == {
        ("prompt_ids", "AR"), ("prompt_layout", "AR"), ("audio_inputs", "RefEncoder")}
    assert clone.step_metadata == {"is_prefill": True, "ar_max_tokens": 7}
    plain = model.get_initial_forward_pass_args("AR", ["text"], ["audio"], signals)
    assert plain.full_metadata.graph_walk == "ar_prefill"
    assert all(edge.next_node == "AR" for edge in plain.inputs)

    meta = clone.full_metadata
    decode = model.get_partition_forward_pass_args("AR", meta, pointers("ar_input_embeds"))
    assert meta.graph_walk == "ar_decode" and not meta.is_prefill and not decode.request_done
    assert decode.inputs[0].name == "ar_input_embeds" and decode.step_metadata["is_prefill"] is False
    assert model.get_partition_forward_pass_args("AR", meta, {}).request_done

    # one frame allowed: the prefill sampled it, so the request ends without a decode walk
    single = model.get_initial_forward_pass_args("AR", ["text"], ["audio"], signals, {"max_new_tokens": 1})
    embeds = pointers("ar_input_embeds")
    done = model.get_partition_forward_pass_args("AR", single.full_metadata, embeds)
    assert done.request_done and done.unpersist_tensors == embeds["ar_input_embeds"]
    assert model.get_graph_walk_graphs()["ar_decode"].max_iters == model.config.codec.max_frames

    codec = model.get_initial_forward_pass_args("Codec", ["text"], ["audio"], {})
    assert codec.full_metadata.graph_walk == "codec_chunk" and not codec.request_done
    with pytest.raises(ValueError, match="Unknown Fish S2-Pro partition"):
        model.get_partition_forward_pass_args("Talker", CurrentForwardConductorMetadata(
            input_modalities=["text"], output_modalities=["audio"], graph_walk="x", is_prefill=False), {})


def test_sampling_defaults_and_greedy():
    model = tiny_model()
    default = model.get_request_resource_configs({})[AR_SAMPLER]
    assert (default.temperature, default.top_p, default.top_k, default.repetition_penalty) == (0.8, 0.8, 30, 1.0)
    greedy = model.get_request_resource_configs({}, {"do_sample": False, "temperature": 0.7, "top_k": 5})[AR_SAMPLER]
    assert greedy.temperature == 0.0 and greedy.top_k == 5


def test_postprocess_emits_pcm16_bytes():
    model = tiny_model()
    out = model.postprocess(torch.tensor([0.0, 1.0, -1.0, 2.0]), "audio")
    assert list(torch.frombuffer(bytearray(out), dtype=torch.int16)) == [0, 32767, -32767, 32767]
    assert model.postprocess(torch.tensor([7, -3], dtype=torch.int16), "audio") == b"\x07\x00\xfd\xff"
    with pytest.raises(ValueError, match="modality"):
        model.postprocess(torch.zeros(1), "text")


# ---------------------------------------------------------------------------
# RoPE table rounding
# ---------------------------------------------------------------------------


def test_rope_table_dtype_rounds_the_cos_sin_table():
    config = PositionConfig(kv_cache="kv", interleave=True, rope_theta=1e6, rotary_dim=8,
                            rope_table_dtype=torch.bfloat16)
    manager = RopeManager(config, torch.device("cpu"), max_seq_len=64, head_dim=8)
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    angles = torch.outer(torch.arange(64, dtype=torch.float32), inv_freq)
    expected = torch.cat([angles.cos(), angles.sin()], dim=-1).to(torch.bfloat16).float()
    assert manager._rope_table.dtype == torch.float32
    assert torch.equal(manager._rope_table, expected)
    exact = RopeManager(PositionConfig(kv_cache="kv", rotary_dim=8), torch.device("cpu"), max_seq_len=64, head_dim=8)
    assert exact._rope_table is None
