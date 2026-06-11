import torch
import transformers
from transformers import Gemma3Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.gemma3 import Gemma3ForConditionalGeneration

from ltx_core.loader import KeyValueOperationResult
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_core.text_encoders.gemma.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
from ltx_core.text_encoders.gemma.feature_extractor import (
    FeatureExtractorV1,
    FeatureExtractorV2,
)

_TRANSFORMERS_V5: bool = int(transformers.__version__.split(".", 1)[0]) >= 5


class GemmaTextEncoderConfigurator(ModelConfigurator[GemmaTextEncoder]):
    @classmethod
    def from_config(cls, config: dict) -> GemmaTextEncoder:  # noqa: ARG003
        gemma_config = Gemma3Config.from_dict(GEMMA3_CONFIG_FOR_LTX.to_dict())
        with torch.device("meta"):
            model = Gemma3ForConditionalGeneration(gemma_config)

        return GemmaTextEncoder(model=model)


class EmbeddingsProcessorConfigurator(ModelConfigurator[EmbeddingsProcessor]):
    @classmethod
    def from_config(cls, config: dict) -> EmbeddingsProcessor:
        transformer_config = config.get("transformer", {})

        # Create video embeddings connector (always needed)
        video_connector = Embeddings1DConnectorConfigurator.from_config(config)

        # Create audio embeddings connector
        audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(config)

        # Create feature extractor
        feature_extractor = _create_feature_extractor(transformer_config)

        return EmbeddingsProcessor(
            video_connector=video_connector,
            audio_connector=audio_connector,
            feature_extractor=feature_extractor,
        )


_V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


def _create_feature_extractor(transformer_config: dict) -> torch.nn.Module:
    """Select and create the appropriate feature extractor based on config.
    Detection logic:
    - V1: V2 config keys absent → projection lives in transformer
    - V2: V2 config keys present with exact expected values → per-token RMS norm with dual aggregate embeds
    - Anything else: NotImplementedError (config drift)
    """
    gemma_text_config = GEMMA3_CONFIG_FOR_LTX.text_config
    embedding_dim = gemma_text_config.hidden_size
    num_layers = gemma_text_config.num_hidden_layers + 1  # +1 for the embedding layer
    flat_dim = embedding_dim * num_layers

    overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
    if not overlapping_keys:
        aggregate_embed = torch.nn.Linear(flat_dim, embedding_dim, bias=False)
        return FeatureExtractorV1(aggregate_embed=aggregate_embed, is_av=True)

    missing_keys = _V2_EXPECTED_CONFIG.keys() - overlapping_keys
    if missing_keys:
        raise NotImplementedError("Partial V2 config — missing keys: " + ", ".join(sorted(missing_keys)))

    unexpected_value_keys = {k for k in overlapping_keys if transformer_config[k] != _V2_EXPECTED_CONFIG[k]}
    if unexpected_value_keys:
        raise NotImplementedError(
            "Unknown config: "
            + ", ".join(
                f"{k}={transformer_config[k]!r} (expected {_V2_EXPECTED_CONFIG[k]!r})" for k in unexpected_value_keys
            )
        )

    video_inner_dim = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
    audio_inner_dim = transformer_config["audio_num_attention_heads"] * transformer_config["audio_attention_head_dim"]
    return FeatureExtractorV2(
        video_aggregate_embed=torch.nn.Linear(flat_dim, video_inner_dim, bias=True),
        embedding_dim=embedding_dim,
        audio_aggregate_embed=torch.nn.Linear(flat_dim, audio_inner_dim, bias=True),
    )


# --- Split SDOps: Gemma LLM keys vs Embeddings Processor keys ---


def _build_gemma_llm_key_ops(*, transformers_v5: bool) -> SDOps:
    """Build the checkpoint-key remapping for the Gemma multimodal encoder.
    The vision-tower mapping differs between transformers <5 and >=5 because
    upstream PR https://github.com/huggingface/transformers/pull/39847 flattened
    ``Gemma3ForConditionalGeneration.model.vision_tower.vision_model`` into
    ``model.vision_tower``. Checkpoints continue to ship the legacy
    ``vision_tower.vision_model.*`` prefix, so we strip the inner ``vision_model.``
    when targeting v5 and pass it through unchanged for v4.
    """
    base = (
        SDOps("GEMMA_LLM_KEY_OPS")
        # 1. Map language model layers (note the double .model prefix)
        .with_matching(prefix="language_model.model.")
        .with_replacement("language_model.model.", "model.model.language_model.")
        # 2. Map the Vision Tower (version-dependent — see docstring)
        .with_matching(prefix="vision_tower.")
    )
    if transformers_v5:
        base = base.with_replacement("vision_tower.vision_model.", "model.model.vision_tower.")
    else:
        base = base.with_replacement("vision_tower.", "model.model.vision_tower.")
    return (
        base
        # 3. Map the Multi-Modal Projector
        .with_matching(prefix="multi_modal_projector.")
        .with_replacement("multi_modal_projector.", "model.model.multi_modal_projector.")
        # 4. Duplicate embed_tokens to lm_head (needed for prompt enhancement via generate())
        .with_kv_operation(
            operation=lambda key, value: [
                KeyValueOperationResult(key, value),
                KeyValueOperationResult("model.lm_head.weight", value),
            ],
            key_prefix="model.model.language_model.embed_tokens.weight",
        )
    )


GEMMA_LLM_KEY_OPS = _build_gemma_llm_key_ops(transformers_v5=_TRANSFORMERS_V5)

EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("EMBEDDINGS_PROCESSOR_KEY_OPS")
    # 1. Map the feature extractor (V1: aggregate_embed inside feature_extractor)
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    # V2 dual aggregate embeds
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    .with_matching(prefix="text_embedding_projection.audio_aggregate_embed.")
    .with_replacement("text_embedding_projection.audio_aggregate_embed.", "feature_extractor.audio_aggregate_embed.")
    # 2. Map the connectors
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "video_connector.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_connector.")
)

VIDEO_ONLY_EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("VIDEO_ONLY_EMBEDDINGS_PROCESSOR_KEY_OPS")
    # 1. Map the feature extractor (V1: aggregate_embed inside feature_extractor)
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    # V2 video aggregate embed
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    # 2. Map the connectors
    .with_matching(prefix="model.diffusion_model.embeddings_connector.")
    .with_replacement("model.diffusion_model.embeddings_connector.", "embeddings_processor.video_connector.")
)


def _resolve_local_base_freq(config: object) -> float:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "sliding_attention" in rope_parameters:
        sliding = rope_parameters["sliding_attention"]
        if isinstance(sliding, dict) and "rope_theta" in sliding:
            return float(sliding["rope_theta"])
    if hasattr(config, "rope_local_base_freq"):
        return float(config.rope_local_base_freq)
    raise AttributeError(
        "Gemma text_config exposes neither rope_local_base_freq nor rope_parameters['sliding_attention']['rope_theta']"
    )


def _resolve_full_rope_type(config: object) -> str:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "full_attention" in rope_parameters:
        full = rope_parameters["full_attention"]
        if isinstance(full, dict) and "rope_type" in full:
            return str(full["rope_type"])
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        if isinstance(rope_scaling, dict):
            if "rope_type" in rope_scaling:
                return str(rope_scaling["rope_type"])
        elif hasattr(rope_scaling, "rope_type"):
            return str(rope_scaling.rope_type)
    raise AttributeError(
        "Gemma text_config exposes neither rope_scaling.rope_type nor rope_parameters['full_attention']['rope_type']"
    )


def _populate_rotary_v4(l_model: torch.nn.Module, config: object) -> None:
    """transformers <5 layout: separate ``rotary_emb_local`` + ``rotary_emb``."""
    dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    base = _resolve_local_base_freq(config)
    local_inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
    full_inv, _ = ROPE_INIT_FUNCTIONS[_resolve_full_rope_type(config)](config)
    l_model.rotary_emb_local.register_buffer("inv_freq", local_inv)
    l_model.rotary_emb.register_buffer("inv_freq", full_inv)


def _populate_rotary_v5(l_model: torch.nn.Module, config: object) -> None:
    """transformers >=5 layout: single ``rotary_emb`` with per-layer-type buffers.
    Mirrors ``Gemma3PreTrainedModel._init_weights`` for ``Gemma3RotaryEmbedding``
    so meta-built models reach the same numerical state as a from_pretrained load.
    """
    rope_emb = l_model.rotary_emb
    for layer_type in dict.fromkeys(config.layer_types):
        rope_params = config.rope_parameters[layer_type]
        if rope_params is None:
            continue
        rope_type = rope_params["rope_type"]
        if rope_type == "default":
            inv_freq, attn_scaling = rope_emb.compute_default_rope_parameters(config, layer_type=layer_type)
        else:
            inv_freq, attn_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, layer_type=layer_type)
        rope_emb.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
        rope_emb.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
        setattr(rope_emb, f"{layer_type}_attention_scaling", attn_scaling)


def create_and_populate(module: GemmaTextEncoder) -> GemmaTextEncoder:
    model = module.model
    v_tower = model.model.vision_tower
    v_model = v_tower.vision_model if hasattr(v_tower, "vision_model") else v_tower
    l_model = model.model.language_model

    config = model.config.text_config

    if hasattr(l_model, "rotary_emb_local"):
        _populate_rotary_v4(l_model, config)
    else:
        _populate_rotary_v5(l_model, config)

    positions_length = len(v_model.embeddings.position_ids[0])
    position_ids = torch.arange(positions_length, dtype=torch.long, device="cpu").unsqueeze(0)
    v_model.embeddings.register_buffer("position_ids", position_ids)
    embed_scale = torch.tensor(config.hidden_size**0.5, device="cpu")
    l_model.embed_tokens.register_buffer("embed_scale", embed_scale)

    return module


GEMMA_MODEL_OPS = ModuleOps(
    name="GemmaModel",
    matcher=lambda module: hasattr(module, "model") and isinstance(module.model, Gemma3ForConditionalGeneration),
    mutator=create_and_populate,
)
