#!/usr/bin/env python3
"""
LTX-2 Video Generation Script

A comprehensive CLI tool for generating videos using the LTX-2 model with:
- Two-stage pipeline (low-res generation + hi-res refinement)
- Joint audio-video generation
- Intelligent CPU/GPU offloading
- Block swapping for memory efficiency
- Image conditioning (I2V mode)
- LoRA support
- FP8 quantization

Based on the LTX-2 ti2vid_two_stages pipeline with advanced memory management
features inspired by Kandinsky5.
"""

import argparse
import gc
import logging
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import torch
from tqdm import tqdm

# Import LTX-2 components
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import AudioProcessor, decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape

from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.block_swap import (
    enable_block_swap,
    offload_all_blocks,
    enable_text_encoder_block_swap,
    offload_all_text_encoder_blocks,
)
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_CFG_GUIDANCE_SCALE,
    DEFAULT_FRAME_RATE,
    DEFAULT_LORA_STRENGTH,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    guider_denoising_func,
    image_conditionings_by_replacing_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video, load_video_conditioning
from ltx_pipelines.utils.types import PipelineComponents


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_GEMMA_ROOT = "./gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_CHECKPOINT_PATH = "./weights/ltx-2-19b-dev.safetensors"
DEFAULT_SPATIAL_UPSAMPLER_PATH = "./weights/ltx-2-spatial-upscaler-x2-1.0.safetensors"
DEFAULT_DISTILLED_LORA_PATH = "./weights/ltx-2-19b-distilled-lora-384.safetensors"


# =============================================================================
# Argument Parser
# =============================================================================

def resolve_path(path: str) -> str:
    """Resolve a path to an absolute path."""
    return str(Path(path).expanduser().resolve().as_posix())


class ImageAction(argparse.Action):
    """Parse image conditioning arguments: PATH FRAME_IDX STRENGTH"""
    def __call__(self, parser, namespace, values, option_string=None):
        path, frame_idx, strength_str = values
        resolved_path = resolve_path(path)
        frame_idx = int(frame_idx)
        strength = float(strength_str)
        current = getattr(namespace, self.dest) or []
        current.append((resolved_path, frame_idx, strength))
        setattr(namespace, self.dest, current)


class LoraAction(argparse.Action):
    """Parse LoRA arguments: PATH [STRENGTH]"""
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > 2:
            msg = f"{option_string} accepts at most 2 arguments (PATH and optional STRENGTH), got {len(values)} values"
            raise argparse.ArgumentError(self, msg)

        path = values[0]
        strength_str = values[1] if len(values) > 1 else str(DEFAULT_LORA_STRENGTH)

        resolved_path = resolve_path(path)
        strength = float(strength_str)

        current = getattr(namespace, self.dest) or []
        current.append(LoraPathStrengthAndSDOps(resolved_path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
        setattr(namespace, self.dest, current)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LTX-2 Video Generation with advanced memory management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic text-to-video generation
  python ltx_generate_video.py \\
    --checkpoint-path model.safetensors \\
    --spatial-upsampler-path upsampler.safetensors \\
    --distilled-lora distilled.safetensors \\
    --prompt "A cat playing piano" \\
    --output-path output.mp4

  # Image-to-video with conditioning
  python ltx_generate_video.py \\
    --checkpoint-path model.safetensors \\
    --spatial-upsampler-path upsampler.safetensors \\
    --distilled-lora distilled.safetensors \\
    --prompt "The cat starts playing" \\
    --image input.jpg 0 0.9 \\
    --output-path output.mp4

  # Memory-optimized generation with offloading
  python ltx_generate_video.py \\
    --checkpoint-path model.safetensors \\
    --spatial-upsampler-path upsampler.safetensors \\
    --distilled-lora distilled.safetensors \\
    --prompt "A beautiful sunset" \\
    --offload \\
    --enable-fp8 \\
    --output-path output.mp4
        """
    )

    # ==========================================================================
    # Required Model Paths
    # ==========================================================================
    model_group = parser.add_argument_group("Model Paths")
    model_group.add_argument(
        "--checkpoint-path",
        type=resolve_path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to LTX-2 model checkpoint (default: {DEFAULT_CHECKPOINT_PATH}).",
    )
    model_group.add_argument(
        "--gemma-root",
        type=resolve_path,
        default=DEFAULT_GEMMA_ROOT,
        help=f"Path to Gemma text encoder directory (default: {DEFAULT_GEMMA_ROOT}).",
    )
    model_group.add_argument(
        "--spatial-upsampler-path",
        type=resolve_path,
        default=DEFAULT_SPATIAL_UPSAMPLER_PATH,
        help=f"Path to spatial upsampler model (default: {DEFAULT_SPATIAL_UPSAMPLER_PATH}).",
    )
    model_group.add_argument(
        "--distilled-lora",
        dest="distilled_lora",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=None,
        help=f"Distilled LoRA for stage 2 (default: {DEFAULT_DISTILLED_LORA_PATH} 1.0).",
    )

    # ==========================================================================
    # Generation Parameters
    # ==========================================================================
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired video content.",
    )
    gen_group.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for unwanted content (default: comprehensive quality prompt).",
    )
    gen_group.add_argument(
        "--output-path",
        type=resolve_path,
        required=True,
        help="Output video file path (.mp4).",
    )
    gen_group.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    gen_group.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Video width in pixels, must be divisible by 64 (default: {DEFAULT_WIDTH}).",
    )
    gen_group.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Video height in pixels, must be divisible by 64 (default: {DEFAULT_HEIGHT}).",
    )
    gen_group.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Number of frames (8*K + 1, e.g., 121 = 5s at 24fps) (default: {DEFAULT_NUM_FRAMES}).",
    )
    gen_group.add_argument(
        "--frame-rate",
        type=float,
        default=DEFAULT_FRAME_RATE,
        help=f"Video frame rate in fps (default: {DEFAULT_FRAME_RATE}).",
    )
    gen_group.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help=f"Number of denoising steps for stage 1 (default: {DEFAULT_NUM_INFERENCE_STEPS}).",
    )
    gen_group.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=DEFAULT_CFG_GUIDANCE_SCALE,
        help=f"CFG guidance scale (default: {DEFAULT_CFG_GUIDANCE_SCALE}).",
    )

    # ==========================================================================
    # Image Conditioning (I2V)
    # ==========================================================================
    i2v_group = parser.add_argument_group("Image Conditioning (I2V)")
    i2v_group.add_argument(
        "--image",
        dest="images",
        action=ImageAction,
        nargs=3,
        metavar=("PATH", "FRAME_IDX", "STRENGTH"),
        default=[],
        help="Image conditioning: path, frame index, strength. Can be repeated.",
    )

    # ==========================================================================
    # LoRA Support
    # ==========================================================================
    lora_group = parser.add_argument_group("LoRA Support")
    lora_group.add_argument(
        "--lora",
        dest="loras",
        action=LoraAction,
        nargs="+",
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="User LoRA: path and optional strength (default: 1.0). Can be repeated.",
    )

    # ==========================================================================
    # Memory Optimization
    # ==========================================================================
    mem_group = parser.add_argument_group("Memory Optimization")
    mem_group.add_argument(
        "--offload",
        action="store_true",
        help="Enable CPU/GPU offloading to reduce VRAM usage.",
    )
    mem_group.add_argument(
        "--enable-fp8",
        action="store_true",
        help="Enable FP8 mode for transformer (reduces memory, calculations in bfloat16).",
    )
    # DiT (main transformer) block swapping
    mem_group.add_argument(
        "--enable-dit-block-swap",
        action="store_true",
        help="Enable block swapping for main DiT transformer (stage 1).",
    )
    mem_group.add_argument(
        "--dit-blocks-in-memory",
        type=int,
        default=22,
        help="Number of DiT transformer blocks to keep in GPU (default: 22).",
    )
    # Text encoder block swapping
    mem_group.add_argument(
        "--enable-text-encoder-block-swap",
        action="store_true",
        help="Enable block swapping for text encoder (Gemma).",
    )
    mem_group.add_argument(
        "--text-encoder-blocks-in-memory",
        type=int,
        default=6,
        help="Number of text encoder layers to keep in GPU (default: 6). "
             "The Gemma-3-12B text encoder has 48 layers.",
    )
    # Refiner block swapping
    mem_group.add_argument(
        "--enable-refiner-block-swap",
        action="store_true",
        help="Enable block swapping for refiner transformer (stage 2).",
    )
    mem_group.add_argument(
        "--refiner-blocks-in-memory",
        type=int,
        default=22,
        help="Number of refiner transformer blocks to keep in GPU (default: 22).",
    )

    # ==========================================================================
    # Audio Control
    # ==========================================================================
    audio_group = parser.add_argument_group("Audio Control")
    audio_group.add_argument(
        "--disable-audio",
        action="store_true",
        help="Disable audio generation (video only output).",
    )

    # ==========================================================================
    # Pipeline Selection
    # ==========================================================================
    pipeline_group = parser.add_argument_group("Pipeline Selection")
    pipeline_group.add_argument(
        "--one-stage",
        action="store_true",
        help="Use one-stage pipeline (faster, generates at full resolution directly). "
             "Default is two-stage (half res + upsample + refine).",
    )
    pipeline_group.add_argument(
        "--refine-only",
        action="store_true",
        help="Use refine-only pipeline (stage 2 only on input video). "
             "Requires --input-video.",
    )

    # ==========================================================================
    # Video Input (V2V / Refine)
    # ==========================================================================
    v2v_group = parser.add_argument_group("Video Input (V2V / Refine)")
    v2v_group.add_argument(
        "--input-video",
        type=resolve_path,
        default=None,
        help="Input video path for video-to-video refinement.",
    )
    v2v_group.add_argument(
        "--refine-strength",
        type=float,
        default=0.3,
        help="Amount of noise to add before refinement (0=none, 1=full denoise). Default: 0.3",
    )
    v2v_group.add_argument(
        "--refine-steps",
        type=int,
        default=10,
        help="Number of refinement steps for refine-only mode. Default: 10",
    )

    # ==========================================================================
    # Advanced Options
    # ==========================================================================
    adv_group = parser.add_argument_group("Advanced Options")
    adv_group.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Use Gemma to enhance the prompt before generation.",
    )
    adv_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


# =============================================================================
# Memory Management Utilities
# =============================================================================

def offload_model(model: torch.nn.Module, target: str = "cpu") -> torch.nn.Module:
    """Offload a model to CPU or back to GPU."""
    if model is not None:
        model = model.to(target, non_blocking=True)
    return model


def synchronize_and_cleanup():
    """Synchronize CUDA and cleanup memory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_loras_chunked_gpu(
    model: torch.nn.Module,
    lora_state_dicts: list,
    lora_strengths: list[float],
    gpu_device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Apply LoRAs to a model in a chunked manner using GPU for computation.

    This function processes each layer's weight individually:
    1. Move weight and LoRA matrices to GPU
    2. Compute LoRA delta on GPU: delta = lora_B @ lora_A * strength
    3. Add delta to weight on GPU
    4. Move result back to CPU
    5. DELETE the LoRA weights from the state dict to free RAM

    This keeps peak GPU memory low while using GPU for fast computation,
    and progressively frees RAM as LoRA weights are consumed.

    Key naming conventions:
    - Model parameter: velocity_model.adaln_single.xxx.weight
    - LoRA keys (after sd_ops): adaln_single.xxx.lora_A.weight
    - So we strip "velocity_model." prefix when looking up LoRA keys
    """
    from tqdm import tqdm

    # Build a map of LoRA weights for quick lookup
    # We'll delete entries as we use them to free RAM
    lora_maps = []
    for lsd in lora_state_dicts:
        lora_map = {}
        for key in list(lsd.sd.keys()):
            if ".lora_A.weight" in key or ".lora_B.weight" in key:
                lora_map[key] = lsd.sd[key]
        lora_maps.append(lora_map)

    # Get all named parameters that might have LoRAs
    params_to_process = []
    for name, param in model.named_parameters():
        if param is not None and name.endswith(".weight"):
            # Model param: velocity_model.adaln_single.xxx.weight
            # LoRA key: adaln_single.xxx.lora_A.weight
            # Strip velocity_model. prefix if present
            lora_prefix = name[:-len(".weight")]
            if lora_prefix.startswith("velocity_model."):
                lora_prefix = lora_prefix[len("velocity_model."):]

            key_a = f"{lora_prefix}.lora_A.weight"
            key_b = f"{lora_prefix}.lora_B.weight"
            # Check if any LoRA has weights for this layer
            has_lora = any(key_a in lora_map and key_b in lora_map for lora_map in lora_maps)
            if has_lora:
                params_to_process.append((name, param, lora_prefix, key_a, key_b))

    print(f">>> Applying LoRAs to {len(params_to_process)} layers using GPU...")

    for name, param, lora_prefix, key_a, key_b in tqdm(params_to_process, desc="Applying LoRAs"):
        # Collect all LoRA deltas for this weight
        deltas = []
        for lora_map, lsd, strength in zip(lora_maps, lora_state_dicts, lora_strengths):
            if key_a in lora_map and key_b in lora_map:
                lora_a = lora_map[key_a].to(device=gpu_device, dtype=dtype)
                lora_b = lora_map[key_b].to(device=gpu_device, dtype=dtype)
                delta = torch.matmul(lora_b * strength, lora_a)
                deltas.append(delta)
                # Free GPU memory immediately
                del lora_a, lora_b
                # FREE RAM: Delete from BOTH the map and the original state dict
                del lora_map[key_a]
                del lora_map[key_b]
                # Delete from original state dict to actually free the tensor memory
                if key_a in lsd.sd:
                    del lsd.sd[key_a]
                if key_b in lsd.sd:
                    del lsd.sd[key_b]

        if deltas:
            # Move weight to GPU, apply deltas, move back to CPU
            weight_gpu = param.data.to(device=gpu_device, dtype=dtype)
            for delta in deltas:
                weight_gpu.add_(delta)
                del delta
            param.data = weight_gpu.to(device="cpu", dtype=dtype)
            del weight_gpu

            # Periodically clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Clean up any remaining LoRA state dict references
    for lsd in lora_state_dicts:
        lsd.sd.clear()
    gc.collect()

    print(">>> LoRA application complete")


# =============================================================================
# Chunked Video Encoding
# =============================================================================

def encode_video_chunked(
    video_tensor: torch.Tensor,
    video_encoder,
    chunk_frames: int = 65,  # 8*8 + 1
    overlap_frames: int = 8,
) -> torch.Tensor:
    """
    Encode video in temporal chunks to reduce memory usage.
    Uses trapezoidal blending for smooth transitions.

    Args:
        video_tensor: Shape (1, C, F, H, W) normalized video
        video_encoder: VideoEncoder model
        chunk_frames: Frames per chunk (must be 8*k+1)
        overlap_frames: Overlap between chunks (must be multiple of 8)

    Returns:
        Encoded latent tensor (1, latent_channels, F', H', W')
    """
    _, c, total_frames, h, w = video_tensor.shape

    # If video fits in one chunk, encode directly
    if total_frames <= chunk_frames:
        return video_encoder(video_tensor)

    # Validate
    assert (chunk_frames - 1) % 8 == 0, "chunk_frames must be 8*k+1"
    assert overlap_frames % 8 == 0, "overlap must be multiple of 8"

    # Calculate latent overlap (temporal compression is 8x)
    latent_overlap = overlap_frames // 8

    # Collect chunks
    chunks_info = []  # (start_frame, end_frame, pad_frames)
    start = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        actual_frames = end - start

        # Pad last chunk if needed to satisfy 8*k+1
        if (actual_frames - 1) % 8 != 0:
            # Find next valid size
            target = 8 * ((actual_frames - 1) // 8 + 1) + 1
            pad_frames = target - actual_frames
            chunks_info.append((start, end, pad_frames))
        else:
            chunks_info.append((start, end, 0))

        # Move to next chunk
        if end >= total_frames:
            break
        start = end - overlap_frames

    print(f">>> Encoding {len(chunks_info)} chunk(s) of {chunk_frames} frames each...")

    # Encode each chunk
    latent_chunks = []
    for i, (start, end, pad) in enumerate(chunks_info):
        print(f">>> Encoding chunk {i+1}/{len(chunks_info)} (frames {start}-{end})...")
        chunk = video_tensor[:, :, start:end, :, :]

        # Pad if necessary
        if pad > 0:
            last_frame = chunk[:, :, -1:, :, :]
            padding = last_frame.expand(-1, -1, pad, -1, -1)
            chunk = torch.cat([chunk, padding], dim=2)

        # Encode
        with torch.no_grad():
            latent = video_encoder(chunk)

        # Remove padded latent tokens if we padded
        if pad > 0:
            valid_tokens = (end - start - 1) // 8 + 1
            latent = latent[:, :, :valid_tokens, :, :]

        latent_chunks.append(latent)

        # Free memory
        del chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combine chunks with blending
    if len(latent_chunks) == 1:
        return latent_chunks[0]

    # Calculate total latent frames
    total_latent_frames = (total_frames - 1) // 8 + 1
    result_shape = list(latent_chunks[0].shape)
    result_shape[2] = total_latent_frames
    result = torch.zeros(result_shape, dtype=latent_chunks[0].dtype, device=latent_chunks[0].device)
    weight_sum = torch.zeros(total_latent_frames, dtype=torch.float32, device=result.device)

    latent_pos = 0
    for i, latent in enumerate(latent_chunks):
        chunk_len = latent.shape[2]

        # Create weight mask
        weight = torch.ones(chunk_len, device=latent.device)

        # Fade in for non-first chunks
        if i > 0 and latent_overlap > 0:
            fade_in = torch.linspace(0, 1, latent_overlap + 1, device=latent.device)[1:]
            weight[:latent_overlap] = fade_in

        # Fade out for non-last chunks
        if i < len(latent_chunks) - 1 and latent_overlap > 0:
            fade_out = torch.linspace(1, 0, latent_overlap + 1, device=latent.device)[1:]
            weight[-latent_overlap:] = fade_out

        # Add weighted latent to result
        end_pos = min(latent_pos + chunk_len, total_latent_frames)
        actual_len = end_pos - latent_pos

        weight_expanded = weight[:actual_len].view(1, 1, actual_len, 1, 1)
        result[:, :, latent_pos:end_pos, :, :] += latent[:, :, :actual_len, :, :] * weight_expanded
        weight_sum[latent_pos:end_pos] += weight[:actual_len]

        # Move position (accounting for overlap)
        if i < len(latent_chunks) - 1:
            latent_pos = latent_pos + chunk_len - latent_overlap

    # Normalize by weight sum
    weight_sum = weight_sum.clamp(min=1e-8).view(1, 1, -1, 1, 1)
    result = result / weight_sum

    return result


# =============================================================================
# Pipeline with Offloading
# =============================================================================

class LTXVideoGeneratorWithOffloading:
    """
    LTX-2 video generator with intelligent model offloading.

    This class wraps the two-stage pipeline and manages model lifecycle
    to minimize GPU memory usage through strategic CPU offloading.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: str = None,
        fp8transformer: bool = False,
        offload: bool = False,
        # Separate block swap controls
        enable_dit_block_swap: bool = False,
        dit_blocks_in_memory: int = 22,
        enable_text_encoder_block_swap: bool = False,
        text_encoder_blocks_in_memory: int = 6,
        enable_refiner_block_swap: bool = False,
        refiner_blocks_in_memory: int = 22,
        one_stage: bool = False,
        refine_only: bool = False,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.offload = offload
        # Separate block swap settings
        self.enable_dit_block_swap = enable_dit_block_swap
        self.dit_blocks_in_memory = dit_blocks_in_memory
        self.enable_text_encoder_block_swap = enable_text_encoder_block_swap
        self.text_encoder_blocks_in_memory = text_encoder_blocks_in_memory
        self.enable_refiner_block_swap = enable_refiner_block_swap
        self.refiner_blocks_in_memory = refiner_blocks_in_memory
        self.one_stage = one_stage
        self.refine_only = refine_only

        # Create model ledger for stage 1
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            fp8transformer=fp8transformer,
        )

        # Store params for stage 2 (create fresh ledger later to avoid shared state issues)
        self._stage_2_checkpoint_path = checkpoint_path
        self._stage_2_gemma_root = gemma_root
        self._stage_2_spatial_upsampler_path = spatial_upsampler_path
        self._stage_2_loras = loras
        self._stage_2_distilled_lora = distilled_lora
        self._stage_2_fp8transformer = fp8transformer

        # Create model ledger for stage 2 (with distilled LoRA)
        # Note: Creating via with_loras for now, will create fresh in generate if needed
        self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
            loras=distilled_lora,
        )

        # Pipeline components (patchifiers, scale factors)
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        disable_audio: bool = False,
        input_video: str | None = None,
        refine_strength: float = 0.3,
        refine_steps: int = 10,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor | None]:
        """
        Generate video with optional audio.

        Returns:
            Tuple of (video_iterator, audio_tensor or None)
        """
        # Validate resolution
        assert_resolution(height=height, width=width, is_two_stage=not self.one_stage)

        # Initialize diffusion components
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = self.dtype

        start_time = time.time()

        # =====================================================================
        # Phase 1: Text Encoding
        # =====================================================================
        print(">>> Loading text encoder...")
        text_encoder_block_swap = None
        if self.enable_text_encoder_block_swap:
            # Load text encoder to CPU first for block swapping
            original_device = self.stage_1_model_ledger.device
            self.stage_1_model_ledger.device = torch.device("cpu")
            text_encoder = self.stage_1_model_ledger.text_encoder()
            self.stage_1_model_ledger.device = original_device

            # Enable block swap for text encoder
            print(f">>> Enabling text encoder block swap ({self.text_encoder_blocks_in_memory} layers in GPU)...")
            text_encoder_block_swap = enable_text_encoder_block_swap(
                text_encoder,
                blocks_in_memory=self.text_encoder_blocks_in_memory,
                device=self.device,
            )
        else:
            text_encoder = self.stage_1_model_ledger.text_encoder()

        if enhance_prompt:
            print(">>> Enhancing prompt with Gemma...")
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
            print(f">>> Enhanced prompt: {prompt}")

        print(">>> Encoding prompts...")
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        # Offload text encoder
        print(">>> Releasing text encoder from GPU...")
        if text_encoder_block_swap:
            offload_all_text_encoder_blocks(text_encoder)
            text_encoder_block_swap = None
        else:
            text_encoder.to("cpu")
        del text_encoder
        synchronize_and_cleanup()

        print(f">>> Text encoding completed in {time.time() - start_time:.1f}s")

        # Initialize audio_latent for use in stage 2
        # Will be set by refine-only mode if encoding audio from input video
        audio_latent = None

        # =====================================================================
        # Refine-only mode: Skip stage 1 and use input video directly
        # =====================================================================
        if self.refine_only and input_video:
            print(">>> Refine-only mode: Loading and encoding input video...")
            refine_start = time.time()

            video_encoder = self.stage_1_model_ledger.video_encoder()

            # Load and encode input video (using chunked encoding to manage memory)
            video_tensor = load_video_conditioning(
                video_path=input_video,
                height=height,
                width=width,
                frame_cap=num_frames,
                dtype=dtype,
                device=self.device,
            )
            upscaled_video_latent = encode_video_chunked(
                video_tensor=video_tensor,
                video_encoder=video_encoder,
                chunk_frames=65,  # 8*8 + 1 frames per chunk
                overlap_frames=8,  # 1 latent token overlap
            )
            # Ensure video latent is in the correct dtype for the pipeline
            upscaled_video_latent = upscaled_video_latent.to(dtype=dtype)

            # Extract and encode audio from input video (like stage 1 would)
            audio_latent = None
            if not disable_audio:
                print(">>> Encoding audio from input video...")

                # Extract audio waveform from input video
                waveform = decode_audio_from_file(input_video, self.device)

                if waveform is not None:
                    import av
                    audio_encoder = self.stage_1_model_ledger.audio_encoder()

                    # Create audio processor with encoder's parameters
                    audio_processor = AudioProcessor(
                        sample_rate=audio_encoder.sample_rate,
                        mel_bins=audio_encoder.mel_bins,
                        mel_hop_length=audio_encoder.mel_hop_length,
                        n_fft=audio_encoder.n_fft,
                    ).to(self.device)

                    # Reshape waveform to [batch, channels, total_samples]
                    # decode_audio_from_file returns [num_frames, channels, samples_per_frame]
                    if waveform.dim() == 3:
                        # Flatten frames into samples: [num_frames, channels, samples] -> [1, channels, total_samples]
                        num_frames_audio, channels, samples_per_frame = waveform.shape
                        waveform = waveform.permute(1, 0, 2).reshape(channels, -1).unsqueeze(0)
                    elif waveform.dim() == 2:
                        # [channels, samples] -> [1, channels, samples]
                        waveform = waveform.unsqueeze(0)

                    # Get sample rate from the video file
                    container = av.open(input_video)
                    audio_stream = next(s for s in container.streams if s.type == "audio")
                    sample_rate = audio_stream.sample_rate
                    container.close()

                    # Convert waveform to mel spectrogram (use float32 for audio quality)
                    mel_spectrogram = audio_processor.waveform_to_mel(
                        waveform.to(dtype=torch.float32),
                        waveform_sample_rate=sample_rate
                    )

                    # Encode mel spectrogram to latents
                    audio_latent = audio_encoder(mel_spectrogram.to(dtype=torch.float32))
                    # Convert to bfloat16 for consistency with pipeline
                    audio_latent = audio_latent.to(dtype=dtype)

                    # Clean up audio encoder
                    del audio_encoder, audio_processor
                    cleanup_memory()
                    print(">>> Audio encoded successfully")
                else:
                    print(">>> Input video has no audio track, will generate fresh audio")

            print(f">>> Input video encoded in {time.time() - refine_start:.1f}s")

            # Clean up video encoder before loading stage 2
            del video_encoder
            cleanup_memory()

            # Skip Phase 2, 3 - go directly to Phase 4 (Stage 2)
            block_swap_manager = None
            video_encoder = None  # Not needed for refine-only

        # =====================================================================
        # Phase 2: Stage 1 - Low Resolution Generation (skip for refine-only)
        # =====================================================================
        skip_stage_1 = self.refine_only and input_video
        if not skip_stage_1:
            print(">>> Stage 1: Loading video encoder and transformer...")
            stage1_start = time.time()

            video_encoder = self.stage_1_model_ledger.video_encoder()

            # For block swapping, load transformer to CPU first, then selectively move blocks
            block_swap_manager = None
            if self.enable_dit_block_swap:
                print(f">>> Loading DiT transformer to CPU for block swapping...")
                # Temporarily override device to load to CPU
                original_device = self.stage_1_model_ledger.device
                self.stage_1_model_ledger.device = torch.device("cpu")
                transformer = self.stage_1_model_ledger.transformer()
                self.stage_1_model_ledger.device = original_device

                # Move non-block components to GPU, keep blocks on CPU
                print(f">>> Enabling DiT block swapping ({self.dit_blocks_in_memory} blocks in GPU)...")
                # Move the wrapper and non-block parts to GPU
                transformer.velocity_model.patchify_proj.to(self.device)
                transformer.velocity_model.adaln_single.to(self.device)
                transformer.velocity_model.caption_projection.to(self.device)
                transformer.velocity_model.norm_out.to(self.device)
                transformer.velocity_model.proj_out.to(self.device)
                transformer.velocity_model.scale_shift_table = torch.nn.Parameter(
                    transformer.velocity_model.scale_shift_table.to(self.device)
                )
                # Audio components
                if hasattr(transformer.velocity_model, "audio_patchify_proj"):
                    transformer.velocity_model.audio_patchify_proj.to(self.device)
                if hasattr(transformer.velocity_model, "audio_adaln_single"):
                    transformer.velocity_model.audio_adaln_single.to(self.device)
                if hasattr(transformer.velocity_model, "audio_caption_projection"):
                    transformer.velocity_model.audio_caption_projection.to(self.device)
                if hasattr(transformer.velocity_model, "audio_norm_out"):
                    transformer.velocity_model.audio_norm_out.to(self.device)
                if hasattr(transformer.velocity_model, "audio_proj_out"):
                    transformer.velocity_model.audio_proj_out.to(self.device)
                if hasattr(transformer.velocity_model, "audio_scale_shift_table"):
                    transformer.velocity_model.audio_scale_shift_table = torch.nn.Parameter(
                        transformer.velocity_model.audio_scale_shift_table.to(self.device)
                    )
                # Cross-attention adaln components
                if hasattr(transformer.velocity_model, "av_ca_video_scale_shift_adaln_single"):
                    transformer.velocity_model.av_ca_video_scale_shift_adaln_single.to(self.device)
                if hasattr(transformer.velocity_model, "av_ca_audio_scale_shift_adaln_single"):
                    transformer.velocity_model.av_ca_audio_scale_shift_adaln_single.to(self.device)
                if hasattr(transformer.velocity_model, "av_ca_a2v_gate_adaln_single"):
                    transformer.velocity_model.av_ca_a2v_gate_adaln_single.to(self.device)
                if hasattr(transformer.velocity_model, "av_ca_v2a_gate_adaln_single"):
                    transformer.velocity_model.av_ca_v2a_gate_adaln_single.to(self.device)

                block_swap_manager = enable_block_swap(
                    transformer,
                    blocks_in_memory=self.dit_blocks_in_memory,
                    device=self.device,
                )
            else:
                transformer = self.stage_1_model_ledger.transformer()

            # Create diffusion schedule
            sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
                dtype=torch.float32, device=self.device
            )

            # Define denoising function for stage 1 (with CFG)
            def first_stage_denoising_loop(
                sigmas: torch.Tensor,
                video_state: LatentState,
                audio_state: LatentState,
                stepper: DiffusionStepProtocol,
            ) -> tuple[LatentState, LatentState]:
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=guider_denoising_func(
                        cfg_guider,
                        v_context_p,
                        v_context_n,
                        a_context_p,
                        a_context_n,
                        transformer=transformer,
                    ),
                )

            # Stage 1 output shape (half resolution for two-stage, full for one-stage)
            if self.one_stage:
                stage_1_output_shape = VideoPixelShape(
                    batch=1,
                    frames=num_frames,
                    width=width,
                    height=height,
                    fps=frame_rate,
                )
            else:
                stage_1_output_shape = VideoPixelShape(
                    batch=1,
                    frames=num_frames,
                    width=width // 2,
                    height=height // 2,
                    fps=frame_rate,
                )

            # Image conditioning for stage 1
            stage_1_conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )

            stage_label = "One-stage" if self.one_stage else "Stage 1"
            print(f">>> {stage_label}: Generating at {stage_1_output_shape.width}x{stage_1_output_shape.height}...")
            video_state, audio_state = denoise_audio_video(
                output_shape=stage_1_output_shape,
                conditionings=stage_1_conditionings,
                noiser=noiser,
                sigmas=sigmas,
                stepper=stepper,
                denoising_loop_fn=first_stage_denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=self.device,
            )

            print(f">>> {stage_label} completed in {time.time() - stage1_start:.1f}s", flush=True)

            # Cleanup stage 1 transformer
            if block_swap_manager:
                offload_all_blocks(transformer)
                # Clear offloader references from transformer to break reference cycle
                if hasattr(transformer, 'velocity_model'):
                    if hasattr(transformer.velocity_model, '_block_swap_offloader'):
                        transformer.velocity_model._block_swap_offloader = None
                    if hasattr(transformer.velocity_model, '_blocks_ref'):
                        transformer.velocity_model._blocks_ref = None
                if hasattr(transformer, '_block_swap_offloader'):
                    transformer._block_swap_offloader = None
                if hasattr(transformer, '_blocks_ref'):
                    transformer._blocks_ref = None
                block_swap_manager = None
            if self.offload:
                print(">>> Offloading stage 1 transformer to CPU...")
            # Set to None instead of del to avoid GC issues
            transformer = None
            cleanup_memory()

            # For one-stage, skip upsampling and stage 2 refinement
            if self.one_stage:
                # Cleanup video encoder
                video_encoder = None
                cleanup_memory()

                # Skip directly to VAE decoding
                print(">>> Decoding video...")
                decode_start = time.time()

                decoded_video = vae_decode_video(
                    video_state.latent,
                    self.stage_1_model_ledger.video_decoder(),
                    tiling_config,
                )

                if not disable_audio:
                    print(">>> Decoding audio...")
                    decoded_audio = vae_decode_audio(
                        audio_state.latent,
                        self.stage_1_model_ledger.audio_decoder(),
                        self.stage_1_model_ledger.vocoder(),
                    )
                else:
                    decoded_audio = None

                print(f">>> Decoding completed in {time.time() - decode_start:.1f}s")
                print(f">>> Total generation time: {time.time() - start_time:.1f}s")

                return decoded_video, decoded_audio

            # =====================================================================
            # Phase 3: Spatial Upsampling (two-stage only)
            # =====================================================================
            print(">>> Upsampling latents (2x)...", flush=True)
            upsample_start = time.time()

            spatial_upsampler = self.stage_2_model_ledger.spatial_upsampler()
            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1],
                video_encoder=video_encoder,
                upsampler=spatial_upsampler,
            )

            torch.cuda.synchronize()
            cleanup_memory()
            print(f">>> Upsampling completed in {time.time() - upsample_start:.1f}s", flush=True)
        # End of skip_stage_1 block

        # =====================================================================
        # Phase 4: Stage 2 - High Resolution Refinement (two-stage only)
        # =====================================================================
        print(">>> Stage 2: Loading transformer with distilled LoRA...", flush=True)
        stage2_start = time.time()

        # Force complete cleanup before loading stage 2 transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # For block swapping with LoRAs, we need to:
        # 1. Load transformer WITHOUT LoRAs to CPU (fast)
        # 2. Load LoRA state dicts
        # 3. Apply LoRAs using chunked GPU computation (fast, low memory)
        # This avoids the slow CPU-only LoRA application that appears to hang.
        block_swap_manager = None
        if self.enable_refiner_block_swap:
            # Create ledger WITHOUT LoRAs - loading will be fast
            stage_2_ledger_no_lora = ModelLedger(
                dtype=self.dtype,
                device=torch.device("cpu"),
                checkpoint_path=self._stage_2_checkpoint_path,
                gemma_root_path=self._stage_2_gemma_root,
                spatial_upsampler_path=self._stage_2_spatial_upsampler_path,
                loras=(),  # No LoRAs - load base model only
                fp8transformer=self._stage_2_fp8transformer,
            )

            # Load transformer without LoRAs (fast - just loading weights)
            print(">>> Loading stage 2 transformer to CPU...", flush=True)
            transformer = stage_2_ledger_no_lora.transformer()

            # Now apply LoRAs using chunked GPU computation
            all_loras = (*self._stage_2_loras, *self._stage_2_distilled_lora) if self._stage_2_distilled_lora else self._stage_2_loras
            if all_loras:
                print(f">>> Loading {len(all_loras)} LoRA(s)...", flush=True)
                from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
                lora_loader = SafetensorsStateDictLoader()
                lora_state_dicts = []
                lora_strengths = []
                for lora in all_loras:
                    lora_sd = lora_loader.load(lora.path, sd_ops=lora.sd_ops, device=torch.device("cpu"))
                    lora_state_dicts.append(lora_sd)
                    lora_strengths.append(lora.strength)

                # Apply LoRAs using chunked GPU computation
                apply_loras_chunked_gpu(
                    model=transformer,
                    lora_state_dicts=lora_state_dicts,
                    lora_strengths=lora_strengths,
                    gpu_device=self.device,
                    dtype=self.dtype,
                )

                # Clean up LoRA state dicts
                del lora_state_dicts
                synchronize_and_cleanup()

            # For VAE operations later
            stage_2_ledger = stage_2_ledger_no_lora

            # Move non-block components to GPU
            print(f">>> Enabling refiner block swapping ({self.refiner_blocks_in_memory} blocks in GPU)...")
            transformer.velocity_model.patchify_proj.to(self.device)
            transformer.velocity_model.adaln_single.to(self.device)
            transformer.velocity_model.caption_projection.to(self.device)
            transformer.velocity_model.norm_out.to(self.device)
            transformer.velocity_model.proj_out.to(self.device)
            transformer.velocity_model.scale_shift_table = torch.nn.Parameter(
                transformer.velocity_model.scale_shift_table.to(self.device)
            )
            if hasattr(transformer.velocity_model, "audio_patchify_proj"):
                transformer.velocity_model.audio_patchify_proj.to(self.device)
            if hasattr(transformer.velocity_model, "audio_adaln_single"):
                transformer.velocity_model.audio_adaln_single.to(self.device)
            if hasattr(transformer.velocity_model, "audio_caption_projection"):
                transformer.velocity_model.audio_caption_projection.to(self.device)
            if hasattr(transformer.velocity_model, "audio_norm_out"):
                transformer.velocity_model.audio_norm_out.to(self.device)
            if hasattr(transformer.velocity_model, "audio_proj_out"):
                transformer.velocity_model.audio_proj_out.to(self.device)
            if hasattr(transformer.velocity_model, "audio_scale_shift_table"):
                transformer.velocity_model.audio_scale_shift_table = torch.nn.Parameter(
                    transformer.velocity_model.audio_scale_shift_table.to(self.device)
                )
            # Cross-attention adaln components
            if hasattr(transformer.velocity_model, "av_ca_video_scale_shift_adaln_single"):
                transformer.velocity_model.av_ca_video_scale_shift_adaln_single.to(self.device)
            if hasattr(transformer.velocity_model, "av_ca_audio_scale_shift_adaln_single"):
                transformer.velocity_model.av_ca_audio_scale_shift_adaln_single.to(self.device)
            if hasattr(transformer.velocity_model, "av_ca_a2v_gate_adaln_single"):
                transformer.velocity_model.av_ca_a2v_gate_adaln_single.to(self.device)
            if hasattr(transformer.velocity_model, "av_ca_v2a_gate_adaln_single"):
                transformer.velocity_model.av_ca_v2a_gate_adaln_single.to(self.device)

            block_swap_manager = enable_block_swap(
                transformer,
                blocks_in_memory=self.refiner_blocks_in_memory,
                device=self.device,
            )
        else:
            # Non-block-swap case: load with LoRAs directly to GPU (fast)
            stage_2_ledger = ModelLedger(
                dtype=self.dtype,
                device=self.device,
                checkpoint_path=self._stage_2_checkpoint_path,
                gemma_root_path=self._stage_2_gemma_root,
                spatial_upsampler_path=self._stage_2_spatial_upsampler_path,
                loras=(*self._stage_2_loras, *self._stage_2_distilled_lora) if self._stage_2_distilled_lora else self._stage_2_loras,
                fp8transformer=self._stage_2_fp8transformer,
            )
            transformer = stage_2_ledger.transformer()

        # For refine-only mode, use the configurable refine_steps
        # For normal two-stage, use the fixed distilled sigma values
        if self.refine_only and input_video:
            # Generate base sigma schedule (1.0 to 0.0)
            base_sigmas = LTX2Scheduler().execute(steps=refine_steps).to(
                dtype=torch.float32, device=self.device
            )
            # Scale sigmas by refine_strength so they start from refine_strength instead of 1.0
            # This preserves (1 - refine_strength) of the input content
            distilled_sigmas = base_sigmas * refine_strength
            print(f">>> Using {refine_steps} refinement steps with strength {refine_strength}")
        else:
            distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        # Define denoising function for stage 2 (no CFG, just positive)
        def second_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,
                ),
            )

        # Stage 2 output shape (full resolution)
        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )

        # Image conditioning for stage 2 (full resolution)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        print(f">>> Stage 2: Refining at {stage_2_output_shape.width}x{stage_2_output_shape.height}...")
        # For refine-only mode, use audio_latent from input video encoding
        # For normal two-stage, use audio_state.latent from stage 1
        stage_2_initial_audio = audio_latent if (self.refine_only and input_video) else audio_state.latent
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=stage_2_initial_audio,
        )

        print(f">>> Stage 2 completed in {time.time() - stage2_start:.1f}s")

        # Cleanup stage 2 models
        if block_swap_manager:
            offload_all_blocks(transformer)
            # Clear offloader references to break reference cycle
            if hasattr(transformer, 'velocity_model'):
                if hasattr(transformer.velocity_model, '_block_swap_offloader'):
                    transformer.velocity_model._block_swap_offloader = None
                if hasattr(transformer.velocity_model, '_blocks_ref'):
                    transformer.velocity_model._blocks_ref = None
            if hasattr(transformer, '_block_swap_offloader'):
                transformer._block_swap_offloader = None
            if hasattr(transformer, '_blocks_ref'):
                transformer._blocks_ref = None
            block_swap_manager = None
        torch.cuda.synchronize()
        transformer = None
        video_encoder = None
        cleanup_memory()

        # =====================================================================
        # Phase 5: VAE Decoding
        # =====================================================================
        print(">>> Decoding video...")
        decode_start = time.time()

        decoded_video = vae_decode_video(
            video_state.latent,
            self.stage_2_model_ledger.video_decoder(),
            tiling_config,
        )

        if not disable_audio:
            print(">>> Decoding audio...")
            decoded_audio = vae_decode_audio(
                audio_state.latent,
                self.stage_2_model_ledger.audio_decoder(),
                self.stage_2_model_ledger.vocoder(),
            )
        else:
            decoded_audio = None

        print(f">>> Decoding completed in {time.time() - decode_start:.1f}s")
        print(f">>> Total generation time: {time.time() - start_time:.1f}s")

        return decoded_video, decoded_audio


# =============================================================================
# Main Entry Point
# =============================================================================

def add_metadata_to_video(video_path: str, parameters: dict) -> None:
    """Add generation parameters to video metadata using ffmpeg."""
    import json
    import subprocess

    params_json = json.dumps(parameters, indent=2)
    temp_path = video_path.replace(".mp4", "_temp.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-metadata", f"comment={params_json}",
        "-codec", "copy",
        temp_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to add metadata: {e.stderr.decode() if e.stderr else e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Warning: Failed to add metadata: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


@torch.inference_mode()
def main():
    """Main entry point for LTX-2 video generation."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # Handle default distilled LoRA
    if args.distilled_lora is None:
        args.distilled_lora = [LoraPathStrengthAndSDOps(
            resolve_path(DEFAULT_DISTILLED_LORA_PATH),
            DEFAULT_LORA_STRENGTH,
            LTXV_LORA_COMFY_RENAMING_MAP
        )]

    pipeline_type = "refine-only" if args.refine_only else ("one-stage" if args.one_stage else "two-stage")
    print("=" * 60)
    print("LTX-2 Video Generation")
    print("=" * 60)
    print(f"Pipeline: {pipeline_type}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output: {args.output_path}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} ({args.num_frames / args.frame_rate:.1f}s at {args.frame_rate}fps)")
    print(f"Seed: {args.seed}")
    print(f"Offload: {args.offload}")
    print(f"FP8: {args.enable_fp8}")
    if args.images:
        print(f"Image conditioning: {len(args.images)} image(s)")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print("=" * 60)

    # Create generator
    generator = LTXVideoGeneratorWithOffloading(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.loras,
        fp8transformer=args.enable_fp8,
        offload=args.offload,
        enable_dit_block_swap=args.enable_dit_block_swap,
        dit_blocks_in_memory=args.dit_blocks_in_memory,
        enable_text_encoder_block_swap=args.enable_text_encoder_block_swap,
        text_encoder_blocks_in_memory=args.text_encoder_blocks_in_memory,
        enable_refiner_block_swap=args.enable_refiner_block_swap,
        refiner_blocks_in_memory=args.refiner_blocks_in_memory,
        one_stage=args.one_stage,
        refine_only=args.refine_only,
    )

    # Set up tiling config for VAE
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    # Generate video
    video, audio = generator.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_guidance_scale=args.cfg_guidance_scale,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        disable_audio=args.disable_audio,
        input_video=args.input_video,
        refine_strength=args.refine_strength,
        refine_steps=args.refine_steps,
    )

    # Encode and save video
    print(f">>> Encoding video to {args.output_path}...")
    encode_start = time.time()

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE if audio is not None else None,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )

    print(f">>> Video saved in {time.time() - encode_start:.1f}s")

    # Build and save metadata
    metadata = {
        "model_type": "LTX-2",
        "pipeline": "refine-only" if args.refine_only else ("one-stage" if args.one_stage else "two-stage"),
        "checkpoint_path": args.checkpoint_path,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "cfg_guidance_scale": args.cfg_guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "offload": args.offload,
        "enable_fp8": args.enable_fp8,
        # Separate block swap settings
        "enable_dit_block_swap": args.enable_dit_block_swap,
        "dit_blocks_in_memory": args.dit_blocks_in_memory if args.enable_dit_block_swap else None,
        "enable_text_encoder_block_swap": args.enable_text_encoder_block_swap,
        "text_encoder_blocks_in_memory": args.text_encoder_blocks_in_memory if args.enable_text_encoder_block_swap else None,
        "enable_refiner_block_swap": args.enable_refiner_block_swap,
        "refiner_blocks_in_memory": args.refiner_blocks_in_memory if args.enable_refiner_block_swap else None,
        "images": [(img[0], img[1], img[2]) for img in args.images] if args.images else None,
        "loras": [(lora.path, lora.strength) for lora in args.loras] if args.loras else None,
        "distilled_lora": [(lora.path, lora.strength) for lora in args.distilled_lora] if args.distilled_lora else None,
        "input_video": args.input_video,
        "refine_strength": args.refine_strength if args.input_video else None,
        "refine_steps": args.refine_steps if args.input_video else None,
        "disable_audio": args.disable_audio,
        "enhance_prompt": args.enhance_prompt,
    }

    print(">>> Adding metadata to video...")
    add_metadata_to_video(args.output_path, metadata)

    print(f">>> Output: {args.output_path}")
    print(">>> Done!")


if __name__ == "__main__":
    main()
