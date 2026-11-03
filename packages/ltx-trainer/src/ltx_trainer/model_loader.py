# ruff: noqa: PLC0415

"""
Model loader for LTX-2 trainer using the ltx-core package.
This module provides a unified interface for loading LTX-2 model components
for training, using SimpleBuilder from ltx-core.
Example usage:
    vae_encoder = load_video_vae_encoder("/path/to/checkpoint.safetensors", device="cuda")
    vae_decoder = load_video_vae_decoder("/path/to/checkpoint.safetensors", device="cuda")
    text_encoder = load_text_encoder("/path/to/gemma", device="cuda")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch

# Type alias for device specification
Device = str | torch.device

# Type checking imports (not loaded at runtime)
if TYPE_CHECKING:
    from ltx_core.model.audio_vae import AudioDecoder, AudioEncoder, Vocoder
    from ltx_core.model.transformer import LTXModel
    from ltx_core.model.video_vae import VideoDecoder, VideoEncoder
    from ltx_core.text_encoders.gemma import LTXGemmaTextEncoder
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
    from ltx_core.types import SpatioTemporalScaleFactors


# =============================================================================
# Individual Component Loaders
# =============================================================================


def load_transformer(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "LTXModel":
    """Load the LTX transformer model.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded LTXModel transformer
    """
    from ltx_core.loader.simple_builder import SimpleBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
        transformer_checkpoint_selection_ops,
    )

    builder = SimpleBuilder(LTXModelConfigurator, (Path(checkpoint_path),), (LTXV_MODEL_COMFY_RENAMING_MAP,))
    # As DiffusionStage does: connector weights belong to the embeddings processor when the checkpoint says so.
    selection = transformer_checkpoint_selection_ops(builder.model_config())
    if selection is not None:
        builder = builder.with_sd_ops((*builder.sd_ops, selection))
    return builder.build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoEncoder":
    """Load the video VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoEncoder
    """
    from ltx_core.loader.simple_builder import SimpleBuilder
    from ltx_core.model.video_vae import VAE_ENCODER_COMFY_KEYS_FILTER, VideoEncoderConfigurator

    return SimpleBuilder(
        VideoEncoderConfigurator,
        (Path(checkpoint_path),),
        (VAE_ENCODER_COMFY_KEYS_FILTER,),
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoDecoder":
    """Load the video VAE decoder (for inference/validation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded VideoDecoder
    """
    from ltx_core.loader.simple_builder import SimpleBuilder
    from ltx_core.model.video_vae import (
        VideoDecoderConfigurator,
        is_diffusion_video_vae,
        video_decoder_sd_ops_for_checkpoint,
    )
    from ltx_core.model.video_vae.transformer import build_cutlass_fna_diffusion_decoder_op

    checkpoint_path = str(checkpoint_path)
    diffusion_vae = is_diffusion_video_vae(checkpoint_path)
    module_ops = (build_cutlass_fna_diffusion_decoder_op(),) if diffusion_vae else ()

    return SimpleBuilder(
        VideoDecoderConfigurator,
        (Path(checkpoint_path),),
        video_decoder_sd_ops_for_checkpoint(checkpoint_path, diffusion_vae=diffusion_vae),
        module_ops=module_ops,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioEncoder":
    """Load the audio VAE encoder (for preprocessing).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights (default bfloat16, but float32 recommended for quality)
    Returns:
        Loaded AudioEncoder
    """
    from ltx_core.loader.simple_builder import SimpleBuilder
    from ltx_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator

    return SimpleBuilder(
        AudioEncoderConfigurator,
        (Path(checkpoint_path),),
        (AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,),
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioDecoder":
    """Load the audio VAE decoder.
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded AudioDecoder
    """
    from ltx_core.loader.simple_builder import SimpleBuilder
    from ltx_core.model.audio_vae import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, AudioDecoderConfigurator

    return SimpleBuilder(
        AudioDecoderConfigurator,
        (Path(checkpoint_path),),
        (AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,),
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_vocoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "Vocoder":
    """Load the vocoder (for audio waveform generation).
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded Vocoder
    """
    from ltx_core.loader.simple_builder import SimpleBuilder
    from ltx_core.model.audio_vae import VOCODER_COMFY_KEYS_FILTER, VocoderConfigurator

    return SimpleBuilder(
        VocoderConfigurator,
        (Path(checkpoint_path),),
        (VOCODER_COMFY_KEYS_FILTER,),
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_text_encoder(
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
) -> "LTXGemmaTextEncoder":
    """Load the Gemma text encoder.
    Args:
        gemma_model_path: Gemma model directory, or packed text-encoder safetensors
        device: Device to load model on
        dtype: Data type for model weights
        load_in_8bit: Whether to quantize the Gemma backbone to 8-bit with bitsandbytes.
    Returns:
        Loaded LTXGemmaTextEncoder
    """
    from ltx_core.loader.simple_builder import SimpleBuilder
    from ltx_core.text_encoders.gemma import (
        GemmaTextEncoderConfigurator,
        get_gemma_ops,
    )
    from ltx_core.text_encoders.gemma.gemma_assets import resolve_gemma_weight_paths

    torch_device = _to_torch_device(device)

    gemma_weight_paths = resolve_gemma_weight_paths(str(gemma_model_path))
    gemma_sd_ops, gemma_module_ops = get_gemma_ops(str(gemma_model_path))
    build_device = torch.device("cpu") if load_in_8bit else torch_device

    text_encoder = SimpleBuilder(
        GemmaTextEncoderConfigurator.with_gemma_model_path(str(gemma_model_path)),
        tuple(Path(path) for path in gemma_weight_paths),
        (gemma_sd_ops,),
        module_ops=gemma_module_ops,
    ).build(device=build_device, dtype=dtype)

    if load_in_8bit:
        from ltx_trainer.gemma_8bit import quantize_gemma_to_8bit

        return quantize_gemma_to_8bit(text_encoder, device=torch_device)

    return text_encoder


def load_embeddings_processor(
    checkpoint_path: str | Path | Sequence[str | Path],
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "EmbeddingsProcessor":
    """Load the embeddings processor (feature extractor + video/audio connectors).
    Args:
        checkpoint_path: Path to the LTX-2 safetensors checkpoint file
        gemma_model_path: Path to the Gemma model directory. Used to read the
            text encoder's hidden_size / num_hidden_layers when sizing the
            feature-extractor projections.
        device: Device to load model on
        dtype: Data type for model weights
    Returns:
        Loaded EmbeddingsProcessor with feature extractor and connectors
    """
    from ltx_core.loader.simple_builder import SimpleBuilder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        EmbeddingsProcessorConfigurator,
    )

    torch_device = _to_torch_device(device)

    paths = (
        tuple(Path(path) for path in checkpoint_path)
        if isinstance(checkpoint_path, (list, tuple))
        else (Path(checkpoint_path),)
    )
    return SimpleBuilder(
        EmbeddingsProcessorConfigurator.with_gemma_model_path(str(gemma_model_path)),
        paths,
        (EMBEDDINGS_PROCESSOR_KEY_OPS,),
    ).build(device=torch_device, dtype=dtype)


# =============================================================================
# Helper Functions
# =============================================================================


def _to_torch_device(device: Device) -> torch.device:
    """Convert device specification to torch.device."""
    return torch.device(device) if isinstance(device, str) else device


def read_video_scale_factors(model_path: str | Path) -> "SpatioTemporalScaleFactors":
    """Read the video VAE compression factors from a checkpoint's embedded config.
    The VAE is not always loaded where these factors are needed (during training the
    latents are precomputed), so the trainer reads flat or nested encoder metadata
    and derives its compression using the core scale-factor helpers. Checkpoints
    without a VAE config (e.g. audio-only) retain the default 32x32x8 layout.
    A failure to *read* the metadata is deliberately not swallowed: it means the
    checkpoint path is wrong or the file is corrupt, and silently falling back to the
    default layout would train a non-default VAE (e.g. 16x16x4) on the wrong RoPE
    positions without ever surfacing the error. Fail loudly instead.
    """
    from ltx_core.loader.checkpoint import CheckpointReader
    from ltx_core.types import SpatioTemporalScaleFactors

    with CheckpointReader.open((Path(model_path),)) as reader:
        metadata = reader.read_checkpoint_metadata()
    model_config = metadata.get("config", {})
    vae_config = model_config.get("vae", {})
    if "encoder" in vae_config:
        encoder_config = vae_config["encoder"]
        blocks = encoder_config.get("blocks", encoder_config.get("encoder_blocks", []))
        return SpatioTemporalScaleFactors.from_blocks(blocks, encoder_config.get("patch_size", 4))
    return SpatioTemporalScaleFactors.from_model_config(model_config)


def is_split_transformer(checkpoint_path: str | Path) -> bool:
    """Return whether ``checkpoint_path`` is the transformer file of a split checkpoint pack."""
    from ltx_core.loader.checkpoint import CheckpointReader

    try:
        with CheckpointReader.open((Path(checkpoint_path),)) as reader:
            metadata = reader.read_checkpoint_metadata()
    except (OSError, ValueError):
        return False
    config = metadata.get("config", {})
    sections = set(config) if isinstance(config, dict) else set()
    return "transformer" in sections and not sections & {"vae", "audio_vae", "vocoder"}


def resolve_video_vae_path(model_path: str | Path, video_vae_path: str | Path | None = None) -> str:
    """Return the file holding the video VAE."""
    return _resolve_component_path(model_path, video_vae_path, "video VAE", "video_vae_path")


def resolve_audio_vae_path(model_path: str | Path, audio_vae_path: str | Path | None = None) -> str:
    """Return the file holding the audio VAE and vocoder."""
    return _resolve_component_path(model_path, audio_vae_path, "audio VAE", "audio_vae_path")


def embedding_weight_paths(
    transformer_path: str | Path,
    text_encoder_path: str | Path | None,
) -> str | tuple[str, str]:
    """Return the files containing embeddings-processor weights for either checkpoint layout."""
    from ltx_core.text_encoders.gemma.gemma_assets import is_safetensors_file

    transformer_path = str(transformer_path)
    if text_encoder_path is not None and is_safetensors_file(text_encoder_path):
        return transformer_path, str(text_encoder_path)
    if is_split_transformer(transformer_path):
        raise ValueError(
            f"{transformer_path} is the transformer of a split checkpoint pack, whose "
            "text_embedding_projection weights live in the packed text-encoder file. "
            "Set text_encoder_path to that .safetensors instead of a Gemma directory."
        )
    return transformer_path


def _resolve_component_path(
    model_path: str | Path,
    component_path: str | Path | None,
    component: str,
    flag: str,
) -> str:
    if component_path is not None:
        return str(component_path)
    if is_split_transformer(model_path):
        raise ValueError(
            f"{model_path} is the transformer of a split checkpoint pack and carries no "
            f"{component}. Pass the standalone {component} safetensors via {flag}."
        )
    return str(model_path)
