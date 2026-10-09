"""Configuration for LTX-2.5, read from the diffusers snapshot's component configs.

Architecture comes from ``transformer/config.json``, ``connectors/config.json``,
``text_encoder/config.json`` and the VAE / vocoder configs, so nothing structural is
hardcoded here. What diffusers keeps in pipeline code rather than in a config (the
1024-token prompt length, the distilled sigma schedules, the request defaults of the
model card's quick start) is below, each named for the constant it mirrors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

LTX25_REPO = "Lightricks/LTX-2.5-Diffusers"

# The components this model serves. The repo also ships ``transformer_full/`` (the SFT
# DiT, 38 GB), a prompt enhancer and a diffusion decoder, none of which the distilled
# recipe loads.
SNAPSHOT_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "connectors/config.json",
    "connectors/*-of-*.safetensors",
    "connectors/*.index.json",
    "transformer/config.json",
    "transformer/*-of-00004.safetensors",
    "transformer/*.index.json",
    "vae/*",
    "audio_vae/*",
    "vocoder/*",
    "latent_upsampler/*",
]

# Graph names shared by the model file and the submodules.
TEXT_ENCODER_NODE = "text_encoder"
DIT_NODE = "dit"
UPSAMPLER_NODE = "latent_upsampler"
VAE_DECODER_NODE = "vae_decoder"
AUDIO_DECODER_NODE = "audio_decoder"
# One loop per denoising stage; the two-stage recipe runs the dit node in both.
DENOISE_LOOP = "denoise_loop"
REFINE_LOOP = "refine_loop"
# Resource key of the dit's ragged self-attention.
DIT_ATTN = "dit_attn"

# diffusers.pipelines.ltx2.utils
DISTILLED_SIGMA_VALUES = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875)
STAGE_2_DISTILLED_SIGMA_VALUES = (0.909375, 0.725, 0.421875)
# LTX2Pipeline.encode_prompt(max_sequence_length=1024)
TEXT_MAX_SEQ_LEN = 1024


@dataclass(frozen=True)
class LTX2TransformerConfig:
    """``transformer/config.json`` of ``LTX2VideoTransformer3DModel``.

    Only the LTX-2.5 configuration is implemented; ``validate`` rejects flags this port
    does not carry (an LTX-2.0 caption projection, interleaved RoPE, ...).
    """

    in_channels: int
    out_channels: int
    num_attention_heads: int
    attention_head_dim: int
    cross_attention_dim: int
    audio_in_channels: int
    audio_out_channels: int
    audio_num_attention_heads: int
    audio_attention_head_dim: int
    audio_cross_attention_dim: int
    num_layers: int
    norm_eps: float
    ff_bias: bool
    audio_ff_bias: bool
    attention_bias: bool
    attention_out_bias: bool
    gated_attn: bool
    audio_gated_attn: bool
    cross_attn_mod: bool
    audio_cross_attn_mod: bool
    use_prompt_adaln_single: bool
    rope_theta: float
    rope_type: str
    rope_double_precision: bool
    causal_offset: int
    pos_embed_max_pos: int
    audio_pos_embed_max_pos: int
    base_height: int
    base_width: int
    vae_scale_factors: tuple[int, int, int]
    audio_sampling_rate: int
    audio_hop_length: int
    audio_scale_factor: int
    timestep_scale_multiplier: int
    cross_attn_timestep_scale_multiplier: int
    patch_size: int
    patch_size_t: int
    audio_patch_size: int
    audio_patch_size_t: int
    use_prompt_embeddings: bool
    qk_norm: str
    activation_fn: str
    norm_elementwise_affine: bool

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def audio_inner_dim(self) -> int:
        return self.audio_num_attention_heads * self.audio_attention_head_dim

    @classmethod
    def from_dict(cls, cfg: dict) -> "LTX2TransformerConfig":
        names = cls.__dataclass_fields__
        values = {k: cfg[k] for k in names if k in cfg}
        values["vae_scale_factors"] = tuple(cfg["vae_scale_factors"])
        values["out_channels"] = cfg.get("out_channels") or cfg["in_channels"]
        values["audio_out_channels"] = cfg.get("audio_out_channels") or cfg["audio_in_channels"]
        missing = sorted(set(names) - set(values))
        if missing:
            raise ValueError(f"LTX-2.5 transformer config is missing {missing}")
        out = cls(**values)
        out.validate()
        return out

    def validate(self) -> None:
        expected = {
            "rope_type": "split", "qk_norm": "rms_norm_across_heads", "activation_fn": "gelu-approximate",
            "use_prompt_embeddings": False, "norm_elementwise_affine": False, "patch_size": 1,
            "patch_size_t": 1, "audio_patch_size": 1, "audio_patch_size_t": 1,
            "cross_attn_mod": True, "audio_cross_attn_mod": True,
        }
        wrong = {k: getattr(self, k) for k, v in expected.items() if getattr(self, k) != v}
        if wrong:
            raise NotImplementedError(
                f"LTX-2.5 port implements {expected}; this checkpoint has {wrong}"
            )


@dataclass(frozen=True)
class GemmaTextConfig:
    """``text_encoder/config.json``'s ``text_config`` (Gemma-4 unified), text tower only."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    global_head_dim: int
    num_global_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    layer_types: tuple[str, ...]
    sliding_window: int
    attention_k_eq_v: bool
    sliding_rope_theta: float
    global_rope_theta: float
    global_partial_rotary_factor: float
    pad_token_id: int

    @classmethod
    def from_dict(cls, cfg: dict) -> "GemmaTextConfig":
        tc = cfg.get("text_config", cfg)
        rope = tc["rope_parameters"]
        if rope["sliding_attention"]["rope_type"] != "default" or rope["full_attention"]["rope_type"] != "proportional":
            raise NotImplementedError(f"Gemma-4 port implements default/proportional RoPE; got {rope}")
        for flag, ok in (("num_kv_shared_layers", 0), ("hidden_size_per_layer_input", 0),
                         ("enable_moe_block", False), ("use_double_wide_mlp", False), ("attention_bias", False)):
            if tc.get(flag, ok) != ok:
                raise NotImplementedError(f"Gemma-4 port does not implement {flag}={tc[flag]!r}")
        if tc["hidden_activation"] != "gelu_pytorch_tanh":
            raise NotImplementedError(f"Gemma-4 port implements gelu_pytorch_tanh; got {tc['hidden_activation']}")
        return cls(
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            global_head_dim=tc["global_head_dim"],
            num_global_key_value_heads=tc["num_global_key_value_heads"],
            intermediate_size=tc["intermediate_size"],
            vocab_size=tc["vocab_size"],
            rms_norm_eps=tc["rms_norm_eps"],
            layer_types=tuple(tc["layer_types"]),
            sliding_window=tc["sliding_window"],
            attention_k_eq_v=bool(tc["attention_k_eq_v"]),
            sliding_rope_theta=rope["sliding_attention"]["rope_theta"],
            global_rope_theta=rope["full_attention"]["rope_theta"],
            global_partial_rotary_factor=rope["full_attention"].get("partial_rotary_factor", 1.0),
            pad_token_id=tc.get("pad_token_id", 0),
        )


@dataclass(frozen=True)
class ConnectorConfig:
    """``connectors/config.json`` of ``LTX2TextConnectors`` (LTX-2.3+ per-modality projections)."""

    caption_channels: int
    text_proj_in_factor: int
    video_heads: int
    video_head_dim: int
    video_layers: int
    video_registers: int
    video_gated_attn: bool
    audio_heads: int
    audio_head_dim: int
    audio_layers: int
    audio_registers: int
    audio_gated_attn: bool
    rope_base_seq_len: int
    rope_theta: float
    video_hidden_dim: int
    audio_hidden_dim: int
    proj_bias: bool

    @classmethod
    def from_dict(cls, cfg: dict) -> "ConnectorConfig":
        if not cfg.get("per_modality_projections", False) or cfg.get("rope_type") != "split":
            raise NotImplementedError("LTX-2.5 connector port implements per-modality projections with split RoPE")
        if cfg.get("causal_temporal_positioning", False):
            raise NotImplementedError("causal_temporal_positioning is not implemented")
        return cls(
            caption_channels=cfg["caption_channels"],
            text_proj_in_factor=cfg["text_proj_in_factor"],
            video_heads=cfg["video_connector_num_attention_heads"],
            video_head_dim=cfg["video_connector_attention_head_dim"],
            video_layers=cfg["video_connector_num_layers"],
            video_registers=cfg["video_connector_num_learnable_registers"],
            video_gated_attn=bool(cfg["video_gated_attn"]),
            audio_heads=cfg["audio_connector_num_attention_heads"],
            audio_head_dim=cfg["audio_connector_attention_head_dim"],
            audio_layers=cfg["audio_connector_num_layers"],
            audio_registers=cfg["audio_connector_num_learnable_registers"],
            audio_gated_attn=bool(cfg["audio_gated_attn"]),
            rope_base_seq_len=cfg["connector_rope_base_seq_len"],
            rope_theta=cfg["rope_theta"],
            video_hidden_dim=cfg["video_hidden_dim"],
            audio_hidden_dim=cfg["audio_hidden_dim"],
            proj_bias=bool(cfg["proj_bias"]),
        )


@dataclass(frozen=True)
class LatentGeometry:
    """Pixel <-> latent arithmetic of the video and audio VAEs."""

    spatial_compression: int = 32
    temporal_compression: int = 8
    latent_channels: int = 128
    audio_sample_rate: int = 16000
    audio_hop_length: int = 160
    audio_temporal_compression: int = 4
    audio_mel_bins: int = 64
    audio_mel_compression: int = 4
    audio_latent_channels: int = 8
    output_sample_rate: int = 48000
    output_audio_channels: int = 2

    @property
    def audio_latents_per_second(self) -> float:
        return self.audio_sample_rate / self.audio_hop_length / float(self.audio_temporal_compression)

    @property
    def audio_latent_mel_bins(self) -> int:
        return self.audio_mel_bins // self.audio_mel_compression

    @property
    def audio_token_channels(self) -> int:
        return self.audio_latent_channels * self.audio_latent_mel_bins

    def latent_frames(self, num_frames: int) -> int:
        return (num_frames - 1) // self.temporal_compression + 1

    def audio_frames(self, num_frames: int, fps: float) -> int:
        # LTX2Pipeline.__call__: round(duration_s * audio_latents_per_second)
        return round(num_frames / fps * self.audio_latents_per_second)


@dataclass(frozen=True)
class LTX25Config:
    transformer: LTX2TransformerConfig
    text: GemmaTextConfig
    connectors: ConnectorConfig
    geometry: LatentGeometry = field(default_factory=LatentGeometry)

    # Request defaults: the model card's quick start (single stage) and two-stage recipe.
    default_height: int = 544
    default_width: int = 960
    default_num_frames: int = 121
    default_fps: float = 24.0
    text_max_seq_len: int = TEXT_MAX_SEQ_LEN

    @classmethod
    def from_snapshot(cls, snapshot: Path) -> "LTX25Config":
        def read(rel: str) -> dict:
            with open(snapshot / rel) as f:
                return json.load(f)

        vae, audio_vae, vocoder = read("vae/config.json"), read("audio_vae/config.json"), read("vocoder/config.json")
        geometry = LatentGeometry(
            spatial_compression=int(vae["spatial_compression_ratio"]),
            temporal_compression=int(vae["temporal_compression_ratio"]),
            latent_channels=int(vae["latent_channels"]),
            audio_sample_rate=int(audio_vae["sample_rate"]),
            audio_hop_length=int(audio_vae["mel_hop_length"]),
            # AutoencoderKLLTX2Audio downsamples once per ch_mult level past the first
            audio_temporal_compression=2 ** (len(audio_vae["ch_mult"]) - 1),
            audio_mel_bins=int(audio_vae["mel_bins"]),
            audio_mel_compression=2 ** (len(audio_vae["ch_mult"]) - 1),
            audio_latent_channels=int(audio_vae["latent_channels"]),
            output_sample_rate=int(vocoder["output_sampling_rate"]),
            output_audio_channels=int(vocoder.get("bwe_out_channels", vocoder["out_channels"])),
        )
        return cls(
            transformer=LTX2TransformerConfig.from_dict(read("transformer/config.json")),
            text=GemmaTextConfig.from_dict(read("text_encoder/config.json")),
            connectors=ConnectorConfig.from_dict(read("connectors/config.json")),
            geometry=geometry,
        )
