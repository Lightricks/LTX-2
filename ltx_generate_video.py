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
from ltx_pipelines.utils.custom_offloading_utils import clean_memory_on_device
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_CFG_GUIDANCE_SCALE,
    DEFAULT_FRAME_RATE,
    DEFAULT_LORA_STRENGTH,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DISTILLED_SIGMA_VALUES,
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
    image_conditionings_by_adding_guiding_latent,
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
# LTX-V RGB Factors for Latent Preview (128 channels)
# =============================================================================
# These factors enable fast latent-to-RGB conversion without VAE decoding.
# Source: Wan2GP/shared/RGB_factors.py (computed for LTXv model)
LATENT_RGB_FACTORS_LTXV = [
    [1.1202e-02, -6.3815e-04, -1.0021e-02],
    [8.6031e-02, 6.5813e-02, 9.5409e-04],
    [-1.2576e-02, -7.5734e-03, -4.0528e-03],
    [9.4063e-03, -2.1688e-03, 2.6093e-03],
    [3.7636e-03, 1.2765e-02, 9.1548e-03],
    [2.1024e-02, -5.2973e-03, 3.4373e-03],
    [-8.8896e-03, -1.9703e-02, -1.8761e-02],
    [-1.3160e-02, -1.0523e-02, 1.9709e-03],
    [-1.5152e-03, -6.9891e-03, -7.5810e-03],
    [-1.7247e-03, 4.6560e-04, -3.3839e-03],
    [1.3617e-02, 4.7077e-03, -2.0045e-03],
    [1.0256e-02, 7.7318e-03, 1.3948e-02],
    [-1.6108e-02, -6.2151e-03, 1.1561e-03],
    [7.3407e-03, 1.5628e-02, 4.4865e-04],
    [9.5357e-04, -2.9518e-03, -1.4760e-02],
    [1.9143e-02, 1.0868e-02, 1.2264e-02],
    [4.4575e-03, 3.6682e-05, -6.8508e-03],
    [-4.5681e-04, 3.2570e-03, 7.7929e-03],
    [3.3902e-02, 3.3405e-02, 3.7454e-02],
    [-2.3001e-02, -2.4877e-03, -3.1033e-03],
    [5.0265e-02, 3.8841e-02, 3.3539e-02],
    [-4.1018e-03, -1.1095e-03, 1.5859e-03],
    [-1.2689e-01, -1.3107e-01, -2.1005e-01],
    [2.6276e-02, 1.4189e-02, -3.5963e-03],
    [-4.8679e-03, 8.8486e-03, 7.8029e-03],
    [-1.6610e-03, -4.8597e-03, -5.2060e-03],
    [-2.1010e-03, 2.3610e-03, 9.3796e-03],
    [-2.2482e-02, -2.1305e-02, -1.5087e-02],
    [-1.5753e-02, -1.0646e-02, -6.5083e-03],
    [-4.6975e-03, 5.0288e-03, -6.7390e-03],
    [1.1951e-02, 2.0712e-02, 1.6191e-02],
    [-6.3704e-03, -8.4827e-03, -9.5483e-03],
    [7.2610e-03, -9.9326e-03, -2.2978e-02],
    [-9.1904e-04, 6.2882e-03, 9.5720e-03],
    [-3.7178e-02, -3.7123e-02, -5.6713e-02],
    [-1.3373e-01, -1.0720e-01, -5.3801e-02],
    [-5.3702e-03, 8.1256e-03, 8.8397e-03],
    [-1.5247e-01, -2.1437e-01, -2.1843e-01],
    [3.1441e-02, 7.0335e-03, -9.7541e-03],
    [2.1528e-03, -8.9817e-03, -2.1023e-02],
    [3.8461e-03, -5.8957e-03, -1.5014e-02],
    [-4.3470e-03, -1.2940e-02, -1.5972e-02],
    [-5.4781e-03, -1.0842e-02, -3.0204e-03],
    [-6.5347e-03, 3.0806e-03, -1.0163e-02],
    [-5.0414e-03, -7.1503e-03, -8.9686e-04],
    [-8.5851e-03, -2.4351e-03, 1.0674e-03],
    [-9.0016e-03, -9.6493e-03, 1.5692e-03],
    [5.0914e-03, 1.2099e-02, 1.9968e-02],
    [1.3758e-02, 1.1669e-02, 8.1958e-03],
    [-1.0518e-02, -1.1575e-02, -4.1307e-03],
    [-2.8410e-02, -3.1266e-02, -2.2149e-02],
    [2.9336e-03, 3.6511e-02, 1.8717e-02],
    [-1.6703e-02, -1.6696e-02, -4.4529e-03],
    [4.8818e-02, 4.0063e-02, 8.7410e-03],
    [-1.5066e-02, -5.7328e-04, 2.9785e-03],
    [-1.7613e-02, -8.1034e-03, 1.3086e-02],
    [-9.2633e-03, 1.0803e-02, -6.3489e-03],
    [3.0851e-03, 4.7750e-04, 1.2347e-02],
    [-2.2785e-02, -2.3043e-02, -2.6005e-02],
    [-2.4787e-02, -1.5389e-02, -2.2104e-02],
    [-2.3572e-02, 1.0544e-03, 1.2361e-02],
    [-7.8915e-03, -1.2271e-03, -6.0968e-03],
    [-1.1478e-02, -1.2543e-03, 6.2679e-03],
    [-5.4229e-02, 2.6644e-02, 6.3394e-03],
    [4.4216e-03, -7.3338e-03, -1.0464e-02],
    [-4.5013e-03, 1.6082e-03, 1.4420e-02],
    [1.3673e-02, 8.8877e-03, 4.1253e-03],
    [-1.0145e-02, 9.0072e-03, 1.5695e-02],
    [-5.6234e-03, 1.1847e-03, 8.1261e-03],
    [-3.7171e-03, -5.3538e-03, 1.2590e-03],
    [2.9476e-02, 2.1424e-02, 3.0424e-02],
    [-3.4925e-02, -2.4340e-02, -2.5316e-02],
    [-3.4127e-02, -2.2406e-02, -1.0589e-02],
    [-1.7342e-02, -1.3249e-02, -1.0719e-02],
    [-2.1478e-03, -8.6051e-03, -2.9878e-03],
    [1.2089e-03, -4.2391e-03, -6.8569e-03],
    [9.0411e-04, -6.6886e-03, -6.7547e-05],
    [1.6048e-02, -1.0057e-02, -2.8929e-02],
    [1.2290e-03, 1.0163e-02, 1.8861e-02],
    [1.7264e-02, 2.7257e-04, 1.3785e-02],
    [-1.3482e-02, -3.6427e-03, 6.7481e-04],
    [4.6782e-03, -5.2423e-03, 2.4467e-03],
    [-5.9113e-03, -6.2244e-03, -1.8162e-03],
    [1.5496e-02, 1.4582e-02, 1.9514e-03],
    [7.4958e-03, 1.5886e-03, -8.2305e-03],
    [1.9086e-02, 1.6360e-03, -3.9674e-03],
    [-5.7021e-03, -2.7307e-03, -4.1066e-03],
    [1.7450e-03, 1.4602e-02, 2.5794e-02],
    [-8.2788e-04, 2.2902e-03, 4.5161e-03],
    [1.1632e-02, 8.9193e-03, -7.2813e-03],
    [7.5721e-03, 2.6784e-03, 1.1393e-02],
    [5.1939e-03, 3.6903e-03, 1.4049e-02],
    [-1.8383e-02, -2.2529e-02, -2.4477e-02],
    [5.8842e-04, -5.7874e-03, -1.4770e-02],
    [-1.6125e-02, -8.6101e-03, -1.4533e-02],
    [2.0540e-02, 2.0729e-02, 6.4338e-03],
    [3.3587e-03, -1.1226e-02, -1.6444e-02],
    [-1.4742e-03, -1.0489e-02, 1.7097e-03],
    [2.8130e-02, 2.3546e-02, 3.2791e-02],
    [-1.8532e-02, -1.2842e-02, -8.7756e-03],
    [-8.0533e-03, -1.0771e-02, -1.7536e-02],
    [-3.9009e-03, 1.6150e-02, 3.3359e-02],
    [-7.4554e-03, -1.4154e-02, -6.1910e-03],
    [3.4734e-03, -1.1370e-02, -1.0581e-02],
    [1.1476e-02, 3.9281e-03, 2.8231e-03],
    [7.1639e-03, -1.4741e-03, -3.8066e-03],
    [2.2250e-03, -8.7552e-03, -9.5719e-03],
    [2.4146e-02, 2.1696e-02, 2.8056e-02],
    [-5.4365e-03, -2.4291e-02, -1.7802e-02],
    [7.4263e-03, 1.0510e-02, 1.2705e-02],
    [6.2669e-03, 6.2658e-03, 1.9211e-02],
    [1.6378e-02, 9.4933e-03, 6.6971e-03],
    [1.7173e-02, 2.3601e-02, 2.3296e-02],
    [-1.4568e-02, -9.8279e-03, -1.1556e-02],
    [1.4431e-02, 1.4430e-02, 6.6362e-03],
    [-6.8230e-03, 1.8863e-02, 1.4555e-02],
    [6.1156e-03, 3.4700e-03, -2.6662e-03],
    [-2.6983e-03, -5.9402e-03, -9.2276e-03],
    [1.0235e-02, 7.4173e-03, -7.6243e-03],
    [-1.3255e-02, 1.9322e-02, -9.2153e-04],
    [2.4222e-03, -4.8039e-03, -1.5759e-02],
    [2.6244e-02, 2.5951e-02, 2.0249e-02],
    [1.5711e-02, 1.8498e-02, 2.7407e-03],
    [-2.1714e-03, 4.7214e-03, -2.2443e-02],
    [-7.4747e-03, 7.4166e-03, 1.4430e-02],
    [-8.3906e-03, -7.9776e-03, 9.7927e-03],
    [3.8321e-02, 9.6622e-03, -1.9268e-02],
    [-1.4605e-02, -6.7032e-03, 3.9675e-03],
]
LATENT_RGB_FACTORS_BIAS_LTXV = [-0.0571, -0.1657, -0.2512]


# =============================================================================
# Helper Functions
# =============================================================================

def build_anchor_image_tuples(
    anchor_image: str | None,
    anchor_interval: int | None,
    anchor_strength: float,
    num_frames: int,
    images: list[tuple[str, int, float]],
) -> list[tuple[str, int, float]]:
    """
    Build anchor conditioning tuples for guiding latent injection.

    Args:
        anchor_image: Explicit anchor image path, or None to use first --image
        anchor_interval: Frame interval for anchor injection
        anchor_strength: Conditioning strength for anchors
        num_frames: Total number of frames in the video
        images: Existing image conditionings (to extract first image if needed)

    Returns:
        List of (image_path, frame_idx, strength) tuples for anchor conditioning
    """
    if anchor_interval is None:
        return []

    # Determine anchor path
    if anchor_image is not None:
        anchor_path = anchor_image
    elif images:
        anchor_path = images[0][0]  # Use first i2v image
    else:
        raise ValueError("--anchor-interval requires --anchor-image or at least one --image")

    # Generate frames: [interval, 2*interval, ...] (skip 0, that's handled by i2v)
    anchor_frames = list(range(anchor_interval, num_frames, anchor_interval))
    return [(anchor_path, frame_idx, anchor_strength) for frame_idx in anchor_frames]


class VideoConditionByMotionLatent:
    """
    SVI Pro motion conditioning using keyframe-style token appending.

    Adapts H1111's motion latent conditioning to LTX's keyframe system.
    Instead of concatenating motion latents directly to the latent tensor,
    this creates VideoConditionByKeyframeIndex items that append tokens
    with appropriate temporal position offsets.

    Motion latents guide temporal consistency by appearing at the END
    of the frame sequence, providing "future motion context" to the denoiser.
    """

    def __init__(
        self,
        motion_latent: torch.Tensor,  # Shape: [1, C, num_motion_latent, H, W]
        strength: float = 0.7,
        frame_offset: int = 0,
    ):
        """
        Args:
            motion_latent: Latent tensor from previous clip [1, C, N, H, W]
            strength: Conditioning strength (0=ignore, 1=fully condition)
            frame_offset: Additional offset for frame positioning
        """
        self.motion_latent = motion_latent
        self.strength = strength
        self.frame_offset = frame_offset

    def to_keyframe_conditionings(
        self,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list:
        """
        Convert motion latent to list of VideoConditionByKeyframeIndex items.

        Each latent frame becomes a separate keyframe conditioning at the
        appropriate temporal position (at END of sequence for motion context).

        Args:
            num_frames: Total pixel frames in the video
            dtype: Target dtype for conditioning tensors
            device: Target device

        Returns:
            List of VideoConditionByKeyframeIndex items
        """
        from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex

        conditionings = []
        motion_latent = self.motion_latent.to(device=device, dtype=dtype)
        num_motion_frames = motion_latent.shape[2]

        # Position motion frames at the END of the temporal sequence
        # This provides "where we're going" context for motion continuity
        for i in range(num_motion_frames):
            # Extract single latent frame: [1, C, 1, H, W]
            frame_latent = motion_latent[:, :, i:i+1, :, :]

            # Frame index at end: num_frames - num_motion_frames + i
            # Convert to pixel frame index for VideoConditionByKeyframeIndex
            frame_idx = num_frames - num_motion_frames + i + self.frame_offset

            conditionings.append(
                VideoConditionByKeyframeIndex(
                    keyframes=frame_latent,
                    frame_idx=frame_idx,
                    strength=self.strength,
                )
            )

        return conditionings


def _encode_frames_to_latent(
    frames: torch.Tensor,
    video_encoder,
    device: torch.device,
    dtype: torch.dtype,
    target_latent_frames: int,
) -> torch.Tensor:
    """
    Encode pixel frames to latent representation for motion conditioning.

    Args:
        frames: Video frames [F, H, W, C] in 0-255 uint8 or [0,1] float
        video_encoder: LTX video encoder instance
        device: Target device
        dtype: Target dtype
        target_latent_frames: Number of latent frames to return

    Returns:
        Latent tensor [1, C, target_latent_frames, H, W]
    """
    # Handle different input formats
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    elif frames.max() > 1.0:
        frames = frames / 255.0

    # Convert from [F, H, W, C] to [1, C, F, H, W]
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
    frames = frames.to(device=device, dtype=dtype)

    # Normalize to [-1, 1] for VAE
    frames = frames * 2.0 - 1.0

    # Encode
    with torch.no_grad():
        latent = video_encoder(frames)  # [1, latent_C, lat_F, lat_H, lat_W]

    # Return last N latent frames
    return latent[:, :, -target_latent_frames:, :, :]


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
    model_group.add_argument(
        "--distilled-checkpoint",
        dest="distilled_checkpoint",
        action="store_true",
        default=False,
        help="Use distilled model settings for stage 1. "
             "Use this when the main checkpoint is a distilled model (not requiring CFG).",
    )
    model_group.add_argument(
        "--stage2-checkpoint",
        type=resolve_path,
        default=None,
        help="Path to a separate checkpoint for stage 2 refinement (full model, not LoRA). "
             "If not specified, uses the main checkpoint with --distilled-lora applied.",
    )
    model_group.add_argument(
        "--stage2-steps",
        type=int,
        default=3,
        help="Number of denoising steps for stage 2 refinement (default: 3). "
             "Uses LTX2Scheduler to generate sigma schedule.",
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
    # Anchor Image Conditioning
    # ==========================================================================
    anchor_group = parser.add_argument_group("Anchor Image Conditioning")
    anchor_group.add_argument(
        "--anchor-image",
        type=resolve_path,
        default=None,
        help="Anchor image path for periodic guidance. If not provided but --anchor-interval is set, uses first --image.",
    )
    anchor_group.add_argument(
        "--anchor-interval",
        type=int,
        default=None,
        help="Frame interval for anchor injection (e.g., 60). Anchors injected at [interval, 2*interval, ...].",
    )
    anchor_group.add_argument(
        "--anchor-strength",
        type=float,
        default=0.8,
        help="Conditioning strength for anchor images (default: 0.8).",
    )
    anchor_group.add_argument(
        "--anchor-decay",
        type=str,
        choices=["none", "linear", "cosine", "sigmoid"],
        default="none",
        help="Decay schedule for anchor constraints. 'none' = constant strength, "
             "'cosine' = smooth decay (recommended), 'linear' = steady decay, "
             "'sigmoid' = late-stage release. Decay allows anchors to guide structure "
             "early but permit motion later. (default: none)",
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
    # SVI Pro (Multi-Clip) Mode
    # ==========================================================================
    svi_group = parser.add_argument_group("SVI Pro (Multi-Clip) Mode")
    svi_group.add_argument(
        "--svi-mode",
        action="store_true",
        help="Enable SVI Pro mode for multi-clip generation with motion continuity.",
    )
    svi_group.add_argument(
        "--num-clips",
        type=int,
        default=2,
        help="Number of clips to generate in SVI mode (default: 2).",
    )
    svi_group.add_argument(
        "--num-motion-latent",
        type=int,
        default=2,
        help="Number of latent frames from previous clip to use for motion conditioning (default: 2).",
    )
    svi_group.add_argument(
        "--num-motion-frame",
        type=int,
        default=1,
        help="Frame offset from end of clip for next input image (1=last, 4=4th from last). Default: 1.",
    )
    svi_group.add_argument(
        "--seed-multiplier",
        type=int,
        default=42,
        help="Per-clip seed variation: seed = base_seed + clip_idx * multiplier (default: 42).",
    )
    svi_group.add_argument(
        "--overlap-frames",
        type=int,
        default=1,
        help="Number of overlapping frames between clips for smooth transitions (default: 1).",
    )
    svi_group.add_argument(
        "--extend-video",
        type=resolve_path,
        default=None,
        help="Input video path to extend using SVI Pro. Enables video extension mode.",
    )
    svi_group.add_argument(
        "--prepend-original",
        action="store_true",
        default=True,
        help="Prepend original video frames when extending (default: True).",
    )
    svi_group.add_argument(
        "--prompt-list",
        type=str,
        nargs="*",
        default=None,
        help="List of prompts for multi-clip generation. One prompt per clip. If fewer prompts than clips, last prompt is repeated.",
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

    # ==========================================================================
    # Preview Generation
    # ==========================================================================
    preview_group = parser.add_argument_group("Preview Generation")
    preview_group.add_argument(
        "--preview-dir",
        type=str,
        default=None,
        help="Directory to save preview images during denoising (enables preview mode).",
    )
    preview_group.add_argument(
        "--preview-interval",
        type=int,
        default=1,
        help="Save preview every N denoising steps (default: 1).",
    )
    preview_group.add_argument(
        "--preview-max-height",
        type=int,
        default=200,
        help="Maximum height of preview images in pixels (default: 200).",
    )

    # ==========================================================================
    # Video Continuation (Enhanced)
    # ==========================================================================
    cont_group = parser.add_argument_group("Video Continuation (Enhanced)")
    cont_group.add_argument(
        "--freeze-frames",
        type=int,
        default=0,
        help="Number of frames from input video to freeze during denoising (default: 0 = disabled).",
    )
    cont_group.add_argument(
        "--freeze-transition",
        type=int,
        default=4,
        help="Number of frames for gradual transition from frozen to generated (default: 4).",
    )

    # ==========================================================================
    # Sliding Window (Long Video)
    # ==========================================================================
    sliding_group = parser.add_argument_group("Sliding Window (Long Video)")
    sliding_group.add_argument(
        "--sliding-window-size",
        type=int,
        default=129,
        help="Frames per window, must be 8n+1 for latent alignment (default: 129).",
    )
    sliding_group.add_argument(
        "--sliding-window-overlap",
        type=int,
        default=9,
        help="Overlapping frames between windows, should be 8n+1 (default: 9).",
    )
    sliding_group.add_argument(
        "--sliding-window-overlap-noise",
        type=float,
        default=0.0,
        help="Noise level (0-100) for overlap blending to reduce seams (default: 0.0).",
    )
    sliding_group.add_argument(
        "--sliding-window-color-correction",
        type=float,
        default=0.0,
        help="LAB color correction strength (0-1) for consistent colors across windows (default: 0.0).",
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


def reconfigure_block_swap(
    transformer,
    new_blocks_in_memory: int,
    device: torch.device,
) -> tuple:
    """
    Reconfigure block swapping with fewer blocks in GPU after an OOM.

    Args:
        transformer: The transformer model with block swap enabled
        new_blocks_in_memory: New (reduced) number of blocks to keep in GPU
        device: Target GPU device

    Returns:
        Tuple of (new_offloader, new_blocks_in_memory) or (None, 0) if at minimum
    """
    from ltx_core.model.transformer.model import X0Model

    # Get the underlying LTXModel
    if isinstance(transformer, X0Model):
        ltx_model = transformer.velocity_model
    else:
        ltx_model = transformer

    # Get current offloader
    offloader = getattr(ltx_model, "_block_swap_offloader", None)
    if offloader is None:
        print("[BlockSwap] No offloader found, cannot reconfigure")
        return None, 0

    num_blocks = len(ltx_model.transformer_blocks)

    # Ensure we have at least 1 block
    if new_blocks_in_memory < 1:
        print(f"[BlockSwap] Cannot reduce below 1 block in GPU")
        return None, 0

    print(f"[BlockSwap] Reconfiguring: {new_blocks_in_memory}/{num_blocks} blocks in GPU...")

    # Wait for any pending async operations
    for idx in range(num_blocks):
        if idx in offloader.futures:
            offloader._wait_blocks_move(idx)

    # Shutdown the ThreadPoolExecutor
    offloader.thread_pool.shutdown(wait=True)
    offloader.futures.clear()

    # Synchronize CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Move all blocks to CPU
    from ltx_pipelines.utils.custom_offloading_utils import weighs_to_device
    for block in ltx_model.transformer_blocks:
        weighs_to_device(block, "cpu")

    # Clear GPU memory
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    # Restore original method temporarily
    if hasattr(ltx_model, "_original_process_transformer_blocks"):
        ltx_model._process_transformer_blocks = ltx_model._original_process_transformer_blocks

    # Clean up old offloader attributes
    if hasattr(ltx_model, "_block_swap_offloader"):
        del ltx_model._block_swap_offloader
    if hasattr(ltx_model, "_blocks_to_swap"):
        del ltx_model._blocks_to_swap
    if hasattr(ltx_model, "_blocks_ref"):
        del ltx_model._blocks_ref

    if isinstance(transformer, X0Model):
        if hasattr(transformer, "_block_swap_offloader"):
            del transformer._block_swap_offloader
        if hasattr(transformer, "_blocks_to_swap"):
            del transformer._blocks_to_swap
        if hasattr(transformer, "_blocks_ref"):
            del transformer._blocks_ref

    # Re-enable block swap with new configuration
    new_offloader = enable_block_swap(
        transformer,
        blocks_in_memory=new_blocks_in_memory,
        device=device,
    )

    return new_offloader, new_blocks_in_memory


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

    for name, param, lora_prefix, key_a, key_b in tqdm(params_to_process, desc="Applying LoRAs", miniters=100):
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
        distilled_checkpoint: bool = False,
        stage2_checkpoint: str | None = None,
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
        self.distilled_checkpoint = distilled_checkpoint
        self.stage2_checkpoint = stage2_checkpoint

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
        # Use stage2_checkpoint if provided, otherwise use the main checkpoint
        self._stage_2_checkpoint_path = stage2_checkpoint if stage2_checkpoint else checkpoint_path
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
        stage2_steps: int = 3,
        anchor_image: str | None = None,
        anchor_interval: int | None = None,
        anchor_strength: float = 0.8,
        anchor_decay: str | None = None,
        # SVI Pro parameters
        _motion_latent: torch.Tensor | None = None,
        _num_motion_latent: int = 0,
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
        # Encode both prompts if CFG might be used (cfg_guidance_scale > 1)
        # Otherwise only encode positive prompt to save memory/time
        if cfg_guidance_scale > 1.0:
            # Encode both positive and negative prompts for CFG
            context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
            v_context_p, a_context_p = context_p
            v_context_n, a_context_n = context_n
        else:
            # Only encode positive prompt (no CFG needed)
            context_p = encode_text(text_encoder, prompts=[prompt])[0]
            v_context_p, a_context_p = context_p
            v_context_n, a_context_n = None, None

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
            # Both distilled and standard checkpoints use configurable LTX2Scheduler
            sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
                dtype=torch.float32, device=self.device
            )
            print(f">>> Using {num_inference_steps} inference steps")

            # Define denoising function for stage 1
            # Use CFG guidance if scale > 1 and we have negative context, otherwise simple denoising
            use_cfg = cfg_guidance_scale > 1.0 and v_context_n is not None
            # Convert anchor_decay "none" to None for the denoising loop
            effective_anchor_decay = anchor_decay if anchor_decay and anchor_decay != "none" else None
            if use_cfg:
                # CFG guidance with positive/negative prompts
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
                        anchor_decay=effective_anchor_decay,
                    )
            else:
                # No CFG guidance, single forward pass
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
                        denoise_fn=simple_denoising_func(
                            video_context=v_context_p,
                            audio_context=a_context_p,
                            transformer=transformer,
                        ),
                        anchor_decay=effective_anchor_decay,
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

            # Image conditioning for stage 1 (i2v - replaces latent at frame 0)
            stage_1_conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )

            # Anchor image conditioning for stage 1 (guiding latent - appends tokens)
            if anchor_interval is not None:
                anchor_tuples = build_anchor_image_tuples(
                    anchor_image=anchor_image,
                    anchor_interval=anchor_interval,
                    anchor_strength=anchor_strength,
                    num_frames=num_frames,
                    images=images,
                )
                if anchor_tuples:
                    anchor_conditionings = image_conditionings_by_adding_guiding_latent(
                        images=anchor_tuples,
                        height=stage_1_output_shape.height,
                        width=stage_1_output_shape.width,
                        video_encoder=video_encoder,
                        dtype=dtype,
                        device=self.device,
                    )
                    stage_1_conditionings = stage_1_conditionings + anchor_conditionings
                    print(f">>> Added {len(anchor_conditionings)} anchor points at frames {[t[1] for t in anchor_tuples]}")

            # SVI Pro: Add motion latent conditionings
            if _motion_latent is not None and _num_motion_latent > 0:
                motion_cond = VideoConditionByMotionLatent(
                    motion_latent=_motion_latent,
                    strength=0.7,
                    frame_offset=0,
                )
                motion_conditionings = motion_cond.to_keyframe_conditionings(
                    num_frames=num_frames,
                    dtype=dtype,
                    device=self.device,
                )
                stage_1_conditionings = stage_1_conditionings + motion_conditionings
                print(f">>> SVI Pro: Added {len(motion_conditionings)} motion keyframe conditionings")

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

        # For refine-only mode, use the configurable refine_steps with strength scaling
        # For normal two-stage, use stage2_steps (default 3)
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
            # Use pre-tuned sigma values for stage 2 refinement with distilled LoRA
            # These values are specifically calibrated for the distilled model
            if stage2_steps == 3:
                # Use exact tuned values for default 3 steps
                distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
            else:
                # Interpolate tuned values to support different step counts
                # Original tuned values: [0.909375, 0.725, 0.421875, 0.0]
                tuned = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES)
                # Create interpolation positions for original values (0 to 1)
                orig_positions = torch.linspace(0, 1, len(tuned))
                # Create new positions for desired step count
                new_positions = torch.linspace(0, 1, stage2_steps + 1)
                # Interpolate to get new sigma values
                distilled_sigmas = torch.zeros(stage2_steps + 1)
                for i, pos in enumerate(new_positions):
                    # Find surrounding original values and interpolate
                    idx = torch.searchsorted(orig_positions, pos, right=True)
                    if idx == 0:
                        distilled_sigmas[i] = tuned[0]
                    elif idx >= len(tuned):
                        distilled_sigmas[i] = tuned[-1]
                    else:
                        # Linear interpolation between tuned[idx-1] and tuned[idx]
                        t = (pos - orig_positions[idx-1]) / (orig_positions[idx] - orig_positions[idx-1])
                        distilled_sigmas[i] = tuned[idx-1] * (1 - t) + tuned[idx] * t
                distilled_sigmas = distilled_sigmas.to(self.device)
                print(f">>> Stage 2 using {stage2_steps} steps (interpolated from tuned schedule)")

        # Define denoising function for stage 2 (no CFG, just positive)
        # Convert anchor_decay "none" to None for the denoising loop (if not already done in stage 1)
        effective_anchor_decay = anchor_decay if anchor_decay and anchor_decay != "none" else None
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
                anchor_decay=effective_anchor_decay,
            )

        # Stage 2 output shape (full resolution)
        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )

        # Image conditioning for stage 2 (i2v - replaces latent at frame 0)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        # Anchor image conditioning for stage 2 (guiding latent - appends tokens)
        if anchor_interval is not None:
            anchor_tuples = build_anchor_image_tuples(
                anchor_image=anchor_image,
                anchor_interval=anchor_interval,
                anchor_strength=anchor_strength,
                num_frames=num_frames,
                images=images,
            )
            if anchor_tuples:
                anchor_conditionings = image_conditionings_by_adding_guiding_latent(
                    images=anchor_tuples,
                    height=stage_2_output_shape.height,
                    width=stage_2_output_shape.width,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=self.device,
                )
                stage_2_conditionings = stage_2_conditionings + anchor_conditionings

        # SVI Pro: Add motion latent conditionings for stage 2
        if _motion_latent is not None and _num_motion_latent > 0:
            motion_cond = VideoConditionByMotionLatent(
                motion_latent=_motion_latent,
                strength=0.7,
                frame_offset=0,
            )
            motion_conditionings = motion_cond.to_keyframe_conditionings(
                num_frames=num_frames,
                dtype=dtype,
                device=self.device,
            )
            stage_2_conditionings = stage_2_conditionings + motion_conditionings

        print(f">>> Stage 2: Refining at {stage_2_output_shape.width}x{stage_2_output_shape.height}...")
        # For refine-only mode, use audio_latent from input video encoding
        # For normal two-stage, use audio_state.latent from stage 1
        stage_2_initial_audio = audio_latent if (self.refine_only and input_video) else audio_state.latent

        # OOM retry loop - reduces blocks in GPU until denoising succeeds
        current_blocks = self.refiner_blocks_in_memory
        min_blocks = 1  # Minimum blocks to try before giving up
        max_retries = current_blocks - min_blocks + 1  # Maximum retry attempts

        for retry_attempt in range(max_retries):
            try:
                # Reset generator for reproducibility on retries
                if retry_attempt > 0:
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                    noiser = GaussianNoiser(generator=generator)

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
                # Success - break out of retry loop
                break

            except torch.OutOfMemoryError as e:
                # Check if we can retry with fewer blocks
                if not block_swap_manager:
                    print(f">>> OOM Error: Block swapping not enabled, cannot reduce memory usage")
                    raise

                current_blocks -= 1
                if current_blocks < min_blocks:
                    print(f">>> OOM Error: Already at minimum blocks ({min_blocks}), cannot reduce further")
                    raise

                print(f">>> OOM Error caught! Retrying with {current_blocks} blocks in GPU...")

                # Clear CUDA error state and memory
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()

                # Reconfigure block swap with fewer blocks
                block_swap_manager, current_blocks = reconfigure_block_swap(
                    transformer,
                    new_blocks_in_memory=current_blocks,
                    device=torch.device(self.device),
                )

                if block_swap_manager is None:
                    print(f">>> Failed to reconfigure block swap, cannot retry")
                    raise

                # Clear memory again after reconfiguration
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()

                print(f">>> Retry attempt {retry_attempt + 1}/{max_retries}...")

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
# SVI Pro Multi-Clip Functions
# =============================================================================

def generate_svi_multi_clip(
    generator: "LTXVideoGeneratorWithOffloading",
    args,
    initial_image_path: str,
    num_clips: int,
    num_motion_latent: int = 2,
    num_motion_frame: int = 1,
    seed_multiplier: int = 42,
    overlap_frames: int = 1,
    prompts: list[str] | None = None,
    initial_motion_latent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Generate multi-clip streaming video using SVI Pro approach.

    This implements the SVI Pro algorithm adapted for LTX:
    1. Generate first clip from initial image (with anchor at frame 0)
    2. For each subsequent clip:
       - Extract motion frame from previous clip as new input image
       - Extract last N latent frames as motion_latent
       - Use original image as anchor for style consistency
       - Add motion_latent as keyframe conditionings at END of sequence
       - Vary seed per clip for different motion dynamics

    Args:
        generator: LTXVideoGeneratorWithOffloading instance
        args: Command line arguments containing generation parameters
        initial_image_path: Path to the starting image (becomes anchor)
        num_clips: Number of clips to generate
        num_motion_latent: Latent frames from previous clip for motion context
        num_motion_frame: Pixel frame offset from end for next input image
        seed_multiplier: Per-clip seed variation
        overlap_frames: Overlapping frames between clips
        prompts: Optional list of per-clip prompts
        initial_motion_latent: Optional motion latent for video extension

    Returns:
        Tuple of (concatenated_video_tensor [F,H,W,C], combined_audio_tensor or None)
    """
    import tempfile
    import shutil
    import cv2
    from PIL import Image

    device = generator.device
    dtype = generator.dtype
    base_seed = args.seed

    # Storage for clips
    all_video_chunks = []
    all_audio_chunks = []

    # SVI anchor: the original image for style consistency
    anchor_image_path = initial_image_path
    current_input_image = initial_image_path

    # Motion latent from previous clip (for clip 2+)
    prev_motion_latent = initial_motion_latent

    # Temp directory for intermediate frames
    temp_dir = tempfile.mkdtemp()

    try:
        for clip_idx in range(num_clips):
            print("=" * 60)
            print(f"SVI Pro: Generating clip {clip_idx + 1}/{num_clips}")
            print("=" * 60)

            # Calculate seed for this clip
            clip_seed = base_seed + clip_idx * seed_multiplier
            print(f">>> Clip {clip_idx + 1} seed: {clip_seed}")

            # Select prompt for this clip
            if prompts and clip_idx < len(prompts):
                clip_prompt = prompts[clip_idx]
            elif prompts:
                clip_prompt = prompts[-1]  # Use last prompt
            else:
                clip_prompt = args.prompt
            print(f">>> Clip {clip_idx + 1} prompt: {clip_prompt[:80]}..." if len(clip_prompt) > 80 else f">>> Clip {clip_idx + 1} prompt: {clip_prompt}")

            # Build image conditionings for this clip
            # Frame 0: current input image (motion frame from prev clip, or initial)
            clip_images = [(current_input_image, 0, 0.95)]

            # Generate this clip
            video_iterator, audio = generator.generate(
                prompt=clip_prompt,
                negative_prompt=args.negative_prompt,
                seed=clip_seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=args.frame_rate,
                num_inference_steps=args.num_inference_steps,
                cfg_guidance_scale=args.cfg_guidance_scale,
                images=clip_images,
                tiling_config=TilingConfig.default(),
                enhance_prompt=False,
                disable_audio=args.disable_audio,
                stage2_steps=args.stage2_steps,
                anchor_image=anchor_image_path if clip_idx > 0 else None,
                anchor_interval=args.anchor_interval if clip_idx > 0 else None,
                anchor_strength=args.anchor_strength,
                anchor_decay=args.anchor_decay,
                # SVI-specific parameters
                _motion_latent=prev_motion_latent if clip_idx > 0 else None,
                _num_motion_latent=num_motion_latent if clip_idx > 0 else 0,
            )

            # Collect video frames from iterator
            video_frames = []
            for chunk in video_iterator:
                video_frames.append(chunk)
            video_tensor = torch.cat(video_frames, dim=0)  # [F, H, W, C]
            print(f">>> Clip {clip_idx + 1} generated: {video_tensor.shape}")

            # Free video_frames list immediately - chunks are now in video_tensor
            del video_frames

            # Extract motion latent for next clip
            if clip_idx < num_clips - 1 and num_motion_latent > 0:
                # We need frames for latent encoding
                # LTX temporal compression is 8x, so num_motion_latent * 8 pixel frames ≈ num_motion_latent latent frames
                frames_needed = num_motion_latent * 8 + 1  # Extra frame for VAE boundary
                frames_for_latent = video_tensor[-frames_needed:, :, :, :].clone()  # Clone to avoid view

                # Load video encoder for latent extraction
                video_encoder = generator.stage_1_model_ledger.video_encoder()
                new_motion_latent = _encode_frames_to_latent(
                    frames_for_latent,
                    video_encoder,
                    device,
                    dtype,
                    num_motion_latent,
                )
                print(f">>> Extracted motion latent: {new_motion_latent.shape}")

                # Cleanup encoder and intermediate tensors
                del frames_for_latent
                video_encoder.to("cpu")
                del video_encoder
                synchronize_and_cleanup()

                # Update motion latent (old one will be garbage collected)
                prev_motion_latent = new_motion_latent

                # Extract motion frame for next clip input
                frame_idx = -min(num_motion_frame, video_tensor.shape[0])
                motion_frame = video_tensor[frame_idx].cpu().numpy().astype("uint8")  # [H, W, C]

                temp_image_path = os.path.join(temp_dir, f"clip_{clip_idx}_motion.png")
                Image.fromarray(motion_frame).save(temp_image_path)
                current_input_image = temp_image_path
                print(f">>> Saved motion frame to: {temp_image_path}")

            # Store clip on CPU (skip overlap frames for non-first clips)
            # Use .cpu() to prevent GPU memory accumulation across clips
            # Use .clone() for slices to avoid keeping original tensor alive via view
            if clip_idx == 0:
                all_video_chunks.append(video_tensor.cpu())
            else:
                all_video_chunks.append(video_tensor[overlap_frames:].clone().cpu())

            if audio is not None:
                # Calculate corresponding audio samples to skip
                # audio is at AUDIO_SAMPLE_RATE, video at args.frame_rate
                # Store audio on CPU as well
                if clip_idx == 0:
                    all_audio_chunks.append(audio.cpu() if audio.device.type != "cpu" else audio)
                else:
                    samples_per_frame = int(24000 / args.frame_rate)  # AUDIO_SAMPLE_RATE
                    samples_to_skip = overlap_frames * samples_per_frame
                    audio_chunk = audio[samples_to_skip:]
                    all_audio_chunks.append(audio_chunk.cpu() if audio_chunk.device.type != "cpu" else audio_chunk)

            # Cleanup GPU memory before next iteration
            del video_tensor
            if audio is not None:
                del audio
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

        # Concatenate all clips
        print("=" * 60)
        print("SVI Pro: Concatenating clips...")
        print("=" * 60)
        final_video = torch.cat(all_video_chunks, dim=0)
        final_audio = torch.cat(all_audio_chunks, dim=0) if all_audio_chunks else None

        print(f">>> Final video shape: {final_video.shape}")
        if final_audio is not None:
            print(f">>> Final audio shape: {final_audio.shape}")

        return final_video, final_audio

    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def generate_svi_video_extension(
    generator: "LTXVideoGeneratorWithOffloading",
    args,
    input_video_path: str,
    num_clips: int,
    num_motion_latent: int = 2,
    num_motion_frame: int = 1,
    seed_multiplier: int = 42,
    overlap_frames: int = 1,
    anchor_image_path: str | None = None,
    prepend_original: bool = True,
    prompts: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Extend an existing video using SVI Pro approach.

    1. Load input video and extract transition frame (for next clip input)
    2. Extract and encode last N frames as initial motion latent
    3. Optionally extract first frame as anchor (or use provided)
    4. Call generate_svi_multi_clip with initial_motion_latent
    5. Concatenate original video with extension

    Args:
        generator: LTXVideoGeneratorWithOffloading instance
        args: Command line arguments
        input_video_path: Path to video to extend
        num_clips: Number of extension clips
        num_motion_latent: Motion latent frames
        num_motion_frame: Frame offset for input image
        seed_multiplier: Per-clip seed variation
        overlap_frames: Overlap for blending
        anchor_image_path: Optional explicit anchor image
        prepend_original: Whether to include original video
        prompts: Optional per-clip prompts

    Returns:
        Tuple of (extended_video, audio)
    """
    import tempfile
    import shutil
    import cv2
    from PIL import Image

    device = generator.device
    dtype = generator.dtype

    temp_dir = tempfile.mkdtemp()

    try:
        print("=" * 60)
        print("SVI Pro Video Extension Mode")
        print("=" * 60)

        # Load input video info
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f">>> Input video: {total_frames} frames at {fps:.1f} fps, {video_width}x{video_height}")

        # Extract transition frame (last frame for continuation)
        transition_frame_idx = total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, transition_frame_idx)
        ret, transition_frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {transition_frame_idx}")

        # Convert BGR to RGB
        transition_frame = cv2.cvtColor(transition_frame, cv2.COLOR_BGR2RGB)
        start_image_path = os.path.join(temp_dir, "transition_frame.png")
        Image.fromarray(transition_frame).save(start_image_path)
        print(f">>> Extracted transition frame to: {start_image_path}")

        # Extract anchor frame (first frame or provided)
        if anchor_image_path and os.path.exists(anchor_image_path):
            anchor_path = anchor_image_path
            print(f">>> Using provided anchor image: {anchor_path}")
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame = cap.read()
            if ret:
                first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                anchor_path = os.path.join(temp_dir, "anchor_frame.png")
                Image.fromarray(first_frame).save(anchor_path)
                print(f">>> Extracted anchor frame to: {anchor_path}")
            else:
                anchor_path = start_image_path

        # Load original video frames for prepending
        original_video = None
        if prepend_original:
            print(">>> Loading original video frames...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            original_frames = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to match target resolution
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LANCZOS4)
                original_frames.append(torch.from_numpy(frame))
            original_video = torch.stack(original_frames, dim=0)  # [F, H, W, C]
            print(f">>> Loaded original video: {original_video.shape}")

        cap.release()

        # Encode last N frames for initial motion latent
        initial_motion_latent = None
        if num_motion_latent > 0 and original_video is not None:
            print(">>> Encoding initial motion latent from video end...")
            frames_needed = num_motion_latent * 8 + 1
            frames_for_latent = original_video[-frames_needed:, :, :, :].float()

            # Need to resize to stage 1 dimensions if using two-stage
            if not generator.one_stage:
                # Resize frames for stage 1
                stage_1_h = args.height // 2
                stage_1_w = args.width // 2
                frames_for_latent_resized = []
                for i in range(frames_for_latent.shape[0]):
                    frame = frames_for_latent[i].numpy().astype("uint8")
                    frame_resized = cv2.resize(frame, (stage_1_w, stage_1_h), interpolation=cv2.INTER_LANCZOS4)
                    frames_for_latent_resized.append(torch.from_numpy(frame_resized).float())
                frames_for_latent = torch.stack(frames_for_latent_resized, dim=0)

            video_encoder = generator.stage_1_model_ledger.video_encoder()
            initial_motion_latent = _encode_frames_to_latent(
                frames_for_latent,
                video_encoder,
                device,
                dtype,
                num_motion_latent,
            )
            print(f">>> Extracted initial motion latent: {initial_motion_latent.shape}")

            video_encoder.to("cpu")
            del video_encoder
            synchronize_and_cleanup()

        # Update anchor path in args for generate
        original_anchor = getattr(args, 'anchor_image', None)
        args.anchor_image = anchor_path

        # Generate extension clips
        extension_video, extension_audio = generate_svi_multi_clip(
            generator=generator,
            args=args,
            initial_image_path=start_image_path,
            num_clips=num_clips,
            num_motion_latent=num_motion_latent,
            num_motion_frame=num_motion_frame,
            seed_multiplier=seed_multiplier,
            overlap_frames=overlap_frames,
            prompts=prompts,
            initial_motion_latent=initial_motion_latent,
        )

        # Restore args
        args.anchor_image = original_anchor

        # Concatenate with original if prepending
        if prepend_original and original_video is not None:
            print(">>> Concatenating original video with extension...")
            # Blend overlap region for smooth transition
            if overlap_frames > 0 and extension_video.shape[0] > overlap_frames:
                # Create linear crossfade weights
                weights = torch.linspace(1.0, 0.0, overlap_frames, device=original_video.device)
                weights = weights.view(-1, 1, 1, 1)

                # Get overlap regions
                overlap_orig = original_video[-overlap_frames:].float()
                overlap_ext = extension_video[:overlap_frames].float()

                # Blend: original fades out, extension fades in
                blended = (overlap_orig * weights + overlap_ext * (1 - weights)).to(original_video.dtype)

                # Concatenate: orig[:-overlap] + blended + ext[overlap:]
                final_video = torch.cat([
                    original_video[:-overlap_frames],
                    blended,
                    extension_video[overlap_frames:],
                ], dim=0)
            else:
                final_video = torch.cat([original_video, extension_video], dim=0)

            print(f">>> Final extended video: {final_video.shape}")
        else:
            final_video = extension_video

        return final_video, extension_audio

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# =============================================================================
# Preview Generation
# =============================================================================

def generate_preview(
    latents: torch.Tensor,
    max_height: int = 200,
    num_keyframes: int = 4,
) -> "Image.Image | None":
    """
    Generate RGB preview from latent tensor without VAE decoding.

    Uses RGB factor convolution (same technique as Wan2GP) for fast,
    memory-efficient preview during denoising.

    Args:
        latents: Latent tensor [B, C, T, H, W] or [C, T, H, W]
        max_height: Maximum height of output preview
        num_keyframes: Number of temporal keyframes to show

    Returns:
        PIL Image showing preview grid, or None if failed
    """
    from PIL import Image
    import numpy as np

    try:
        # Ensure batch dimension
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)

        B, C, T, H, W = latents.shape

        # Select keyframes (temporal subsampling)
        num_keyframes = min(T, num_keyframes)
        if num_keyframes <= 0:
            return None

        skip = max(1, T / num_keyframes)
        frame_indices = [min(int(i * skip), T - 1) for i in range(num_keyframes)]

        # Select frames
        selected = torch.stack([latents[:, :, i, :, :] for i in frame_indices], dim=2)

        # Create weight tensor from RGB factors [3, 128] -> [3, 128, 1, 1, 1]
        weight = torch.tensor(
            LATENT_RGB_FACTORS_LTXV,
            device=latents.device,
            dtype=latents.dtype,
        ).T.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        bias = torch.tensor(
            LATENT_RGB_FACTORS_BIAS_LTXV,
            device=latents.device,
            dtype=latents.dtype,
        )

        # Apply 3D convolution for latent-to-RGB: [B, 128, T, H, W] -> [B, 3, T, H, W]
        images = torch.nn.functional.conv3d(
            selected, weight, bias=bias, stride=1, padding=0
        )

        # Normalize to 0-255
        images = images.add_(1.0).mul_(127.5)
        images = images.detach().cpu()

        if images.dtype == torch.bfloat16:
            images = images.to(torch.float16)

        images = images.numpy().clip(0, 255).astype(np.uint8)

        # Rearrange to grid: [B, C, T, H, W] -> [B*H, T*W, C]
        # For batch=1: [1, 3, T, H, W] -> [H, T*W, 3]
        B, C, T_sel, H, W = images.shape
        images = images[0]  # Remove batch dim: [3, T, H, W]
        images = images.transpose(1, 2, 3, 0)  # [T, H, W, 3]
        images = images.transpose(0, 2, 1, 3)  # [T, W, H, 3]
        images = images.reshape(T_sel * W, H, 3).transpose(1, 0, 2)  # [H, T*W, 3]

        # Resize to max_height
        h, w, _ = images.shape
        scale = max_height / h
        pil_image = Image.fromarray(images)
        pil_image = pil_image.resize(
            (int(w * scale), int(h * scale)),
            resample=Image.Resampling.BILINEAR,
        )

        return pil_image

    except Exception as e:
        print(f"Warning: Preview generation failed: {e}")
        return None


def create_preview_callback(
    preview_dir: str,
    preview_interval: int = 1,
    max_height: int = 200,
    stage_name: str = "stage1",
):
    """
    Create a callback function for preview generation during denoising.

    Args:
        preview_dir: Directory to save preview images
        preview_interval: Save every N steps
        max_height: Maximum preview height
        stage_name: Prefix for filenames (e.g., "stage1", "stage2")

    Returns:
        Callback function compatible with euler_denoising_loop
    """
    os.makedirs(preview_dir, exist_ok=True)

    def callback(step_idx: int, video_state, sigmas: torch.Tensor):
        if step_idx % preview_interval != 0:
            return

        preview = generate_preview(
            latents=video_state.latent,
            max_height=max_height,
            num_keyframes=4,
        )

        if preview is not None:
            sigma = sigmas[step_idx].item() if step_idx < len(sigmas) else 0.0
            preview_path = os.path.join(
                preview_dir,
                f"{stage_name}_step_{step_idx:03d}_sigma_{sigma:.4f}.jpg",
            )
            preview.save(preview_path, quality=85)
            print(f">>> Preview saved: {preview_path}")

    return callback


# =============================================================================
# Video Continuation (Frame Freezing) Helpers
# =============================================================================

def prepare_frame_freezing(
    video_tensor: torch.Tensor,
    video_encoder,
    freeze_frames: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare frozen latent and freeze mask for video continuation.

    Args:
        video_tensor: Input video frames [F, H, W, C] in 0-255 uint8
        video_encoder: LTX video encoder
        freeze_frames: Number of pixel frames to freeze
        num_frames: Total pixel frames in output video
        device: Target device
        dtype: Target dtype

    Returns:
        Tuple of (frozen_latent [1, C, lat_T, lat_H, lat_W], freeze_mask [1, 1, lat_T, lat_H, lat_W])
    """
    # Extract frames to freeze
    frames_to_freeze = video_tensor[:freeze_frames].float()
    if frames_to_freeze.max() > 1.0:
        frames_to_freeze = frames_to_freeze / 255.0

    # Convert from [F, H, W, C] to [1, C, F, H, W]
    frames_to_freeze = frames_to_freeze.permute(3, 0, 1, 2).unsqueeze(0)
    frames_to_freeze = frames_to_freeze.to(device=device, dtype=dtype)

    # Normalize to [-1, 1] for VAE
    frames_to_freeze = frames_to_freeze * 2.0 - 1.0

    # Encode to latent
    with torch.no_grad():
        frozen_latent = video_encoder(frames_to_freeze)  # [1, C, lat_F, lat_H, lat_W]

    # Calculate latent dimensions
    lat_F = frozen_latent.shape[2]
    lat_H = frozen_latent.shape[3]
    lat_W = frozen_latent.shape[4]

    # Total latent frames for the full video
    total_lat_frames = (num_frames - 1) // 8 + 1

    # Create freeze mask: 1 for frozen frames, 0 for generated frames
    freeze_mask = torch.zeros(1, 1, total_lat_frames, lat_H, lat_W, device=device, dtype=dtype)
    freeze_mask[:, :, :lat_F, :, :] = 1.0

    # Pad frozen_latent to full temporal length (zeros for non-frozen parts)
    full_frozen_latent = torch.zeros(
        1, frozen_latent.shape[1], total_lat_frames, lat_H, lat_W,
        device=device, dtype=dtype
    )
    full_frozen_latent[:, :, :lat_F, :, :] = frozen_latent

    return full_frozen_latent, freeze_mask


# =============================================================================
# Sliding Window (Long Video) Generation
# =============================================================================

def prepare_overlap_injection(
    prev_latent: torch.Tensor,
    overlap_noise: float,
) -> torch.Tensor:
    """
    Prepare overlapped latents with noise for sliding window injection.

    Wan2GP approach: blend previous latent with noise based on overlap_noise.

    Args:
        prev_latent: Latent from end of previous window [1, C, T, H, W]
        overlap_noise: Noise level (0-100 scale)

    Returns:
        Prepared latent tensor for injection
    """
    # Convert overlap_noise from 0-100 scale to 0-1 sigma scale
    noise_sigma = overlap_noise / 100.0

    if noise_sigma > 0:
        noise = torch.randn_like(prev_latent) * noise_sigma
        return prev_latent * (1 - noise_sigma) + noise

    return prev_latent


def apply_lab_color_correction(
    current_window: torch.Tensor,
    reference_window: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Apply LAB color space transfer for consistent colors across windows.

    Args:
        current_window: Current window pixels [F, H, W, C] in 0-255 range
        reference_window: Reference pixels from previous window [F_ref, H, W, C]
        strength: Blending strength (0=none, 1=full)

    Returns:
        Color-corrected current window
    """
    if strength == 0:
        return current_window

    try:
        from skimage import color
        import numpy as np

        # Convert to float [0, 1] for color space conversion
        current_np = current_window.float().numpy() / 255.0
        reference_np = reference_window.float().numpy() / 255.0

        # Compute LAB statistics for reference
        reference_lab = color.rgb2lab(reference_np)
        ref_mean = np.mean(reference_lab, axis=(0, 1, 2))
        ref_std = np.std(reference_lab, axis=(0, 1, 2)) + 1e-8

        # Apply correction to entire current window
        current_lab = color.rgb2lab(current_np)
        cur_mean = np.mean(current_lab, axis=(0, 1, 2))
        cur_std = np.std(current_lab, axis=(0, 1, 2)) + 1e-8

        # Transfer statistics: (x - cur_mean) * (ref_std / cur_std) + ref_mean
        corrected_lab = (current_lab - cur_mean) * (ref_std / cur_std) + ref_mean

        # Convert back to RGB
        corrected_rgb = color.lab2rgb(corrected_lab)
        corrected_rgb = np.clip(corrected_rgb * 255, 0, 255).astype(np.uint8)

        # Blend with original based on strength
        original = current_window.numpy()
        result = (corrected_rgb * strength + original * (1 - strength)).astype(np.uint8)
        return torch.from_numpy(result)

    except ImportError:
        print("Warning: skimage not available, skipping color correction")
        return current_window
    except Exception as e:
        print(f"Warning: Color correction failed: {e}")
        return current_window


def sliding_window_generate(
    generator: "LTXVideoGeneratorWithOffloading",
    args,
    total_frames: int,
    window_size: int = 129,
    overlap: int = 9,
    overlap_noise: float = 0.0,
    color_correction_strength: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Generate long videos using sliding window approach.

    Divides the video into overlapping windows, generates each window
    sequentially, and concatenates with proper blending.

    Args:
        generator: LTXVideoGeneratorWithOffloading instance
        args: Command line arguments
        total_frames: Total frames to generate
        window_size: Frames per window (should be 8n+1 for latent alignment)
        overlap: Overlapping frames between windows (should be 8n+1)
        overlap_noise: Noise level (0-100) for overlapped frames
        color_correction_strength: LAB color correction strength (0-1)

    Returns:
        Tuple of (video_tensor [F, H, W, C], audio_tensor or None)
    """
    from PIL import Image

    # Validate alignment to LTX's 8-frame temporal compression
    if (window_size - 1) % 8 != 0:
        print(f"Warning: window_size {window_size} not aligned to 8n+1, adjusting...")
        window_size = ((window_size - 1) // 8) * 8 + 1

    if overlap > 0 and (overlap - 1) % 8 != 0:
        print(f"Warning: overlap {overlap} not aligned to 8n+1, adjusting...")
        overlap = ((overlap - 1) // 8) * 8 + 1

    # Calculate number of windows needed
    effective_window = window_size - overlap
    if effective_window <= 0:
        raise ValueError(f"Overlap ({overlap}) must be less than window_size ({window_size})")

    num_windows = max(1, (total_frames - overlap + effective_window - 1) // effective_window)

    print(f">>> Sliding Window: {num_windows} windows of {window_size} frames")
    print(f">>> Overlap: {overlap} frames, Effective: {effective_window} new frames/window")

    all_video_chunks = []
    all_audio_chunks = []
    prev_window_latent = None
    prev_window_pixels = None

    device = generator.device
    dtype = generator.dtype

    for window_idx in range(num_windows):
        print("=" * 60)
        print(f">>> Sliding Window {window_idx + 1}/{num_windows}")
        print("=" * 60)

        # Calculate frame range for this window
        start_frame = window_idx * effective_window
        end_frame = min(start_frame + window_size, total_frames)
        window_frames = end_frame - start_frame

        # Ensure window_frames is valid (8n+1)
        if window_frames < 9:
            window_frames = 9
        if (window_frames - 1) % 8 != 0:
            window_frames = ((window_frames - 1) // 8 + 1) * 8 + 1

        # Calculate seed for this window
        window_seed = args.seed + window_idx * 42

        # Build conditionings for this window
        window_images = list(args.images) if args.images else []

        # Prepare overlapped latents injection (if not first window)
        if window_idx > 0 and prev_window_latent is not None:
            # Inject previous window's ending latent as start conditioning
            overlap_latent = prepare_overlap_injection(prev_window_latent, overlap_noise)
            # Add as conditioning at frame 0 with high strength
            # This is handled internally by passing to generate()

        print(f">>> Window {window_idx + 1}: frames {start_frame}-{end_frame}, {window_frames} frames, seed {window_seed}")

        # Generate this window
        video_iterator, audio = generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=window_seed,
            height=args.height,
            width=args.width,
            num_frames=window_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            cfg_guidance_scale=args.cfg_guidance_scale,
            images=window_images,
            tiling_config=TilingConfig.default(),
            enhance_prompt=False,
            disable_audio=args.disable_audio,
            stage2_steps=args.stage2_steps,
            anchor_image=args.anchor_image,
            anchor_interval=args.anchor_interval,
            anchor_strength=args.anchor_strength,
            anchor_decay=args.anchor_decay,
        )

        # Collect video frames from iterator
        video_frames = []
        for chunk in video_iterator:
            video_frames.append(chunk)
        video_tensor = torch.cat(video_frames, dim=0)  # [F, H, W, C]
        print(f">>> Window {window_idx + 1} generated: {video_tensor.shape}")

        # Extract ending latent for next window (if not last window)
        if window_idx < num_windows - 1:
            # Extract last overlap frames for latent encoding
            overlap_pixel_frames = overlap if overlap > 0 else 8
            frames_for_latent = video_tensor[-overlap_pixel_frames:, :, :, :]

            # Encode to latent
            video_encoder = generator.stage_1_model_ledger.video_encoder()

            # Resize if two-stage pipeline
            if not generator.one_stage:
                import cv2
                stage_1_h = args.height // 2
                stage_1_w = args.width // 2
                frames_resized = []
                for i in range(frames_for_latent.shape[0]):
                    frame = frames_for_latent[i].numpy().astype("uint8")
                    frame_resized = cv2.resize(frame, (stage_1_w, stage_1_h), interpolation=cv2.INTER_LANCZOS4)
                    frames_resized.append(torch.from_numpy(frame_resized).float())
                frames_for_latent = torch.stack(frames_resized, dim=0)

            prev_window_latent = _encode_frames_to_latent(
                frames_for_latent,
                video_encoder,
                device,
                dtype,
                target_latent_frames=(overlap_pixel_frames - 1) // 8 + 1,
            )
            print(f">>> Extracted overlap latent: {prev_window_latent.shape}")

            video_encoder.to("cpu")
            del video_encoder
            synchronize_and_cleanup()

        # Apply color correction if enabled and not first window
        if color_correction_strength > 0 and prev_window_pixels is not None:
            video_tensor = apply_lab_color_correction(
                current_window=video_tensor,
                reference_window=prev_window_pixels,
                strength=color_correction_strength,
            )

        # Store reference for next window's color correction
        prev_window_pixels = video_tensor[-overlap:] if overlap > 0 else video_tensor[-8:]

        # Store chunk (strip overlap from non-first windows)
        if window_idx == 0:
            all_video_chunks.append(video_tensor)
        else:
            all_video_chunks.append(video_tensor[overlap:])

        if audio is not None:
            samples_per_frame = int(AUDIO_SAMPLE_RATE / args.frame_rate)
            if window_idx == 0:
                all_audio_chunks.append(audio)
            else:
                samples_to_skip = overlap * samples_per_frame
                all_audio_chunks.append(audio[samples_to_skip:])

    # Concatenate all windows
    print("=" * 60)
    print(">>> Sliding Window: Concatenating windows...")
    print("=" * 60)

    final_video = torch.cat(all_video_chunks, dim=0)
    # Trim to exact requested length
    if final_video.shape[0] > total_frames:
        final_video = final_video[:total_frames]

    final_audio = torch.cat(all_audio_chunks, dim=0) if all_audio_chunks else None

    print(f">>> Final video shape: {final_video.shape}")
    if final_audio is not None:
        print(f">>> Final audio shape: {final_audio.shape}")

    return final_video, final_audio


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
    # Don't auto-add distilled LoRA if:
    # - User explicitly set --distilled-lora (even to empty)
    # - Using a distilled checkpoint (checkpoint itself is distilled)
    # - Using a separate stage 2 checkpoint (assumed to be a complete model)
    if args.distilled_lora is None:
        if args.distilled_checkpoint or args.stage2_checkpoint:
            args.distilled_lora = []  # No distilled LoRA needed
        else:
            args.distilled_lora = [LoraPathStrengthAndSDOps(
                resolve_path(DEFAULT_DISTILLED_LORA_PATH),
                DEFAULT_LORA_STRENGTH,
                LTXV_LORA_COMFY_RENAMING_MAP
            )]

    pipeline_type = "refine-only" if args.refine_only else ("one-stage" if args.one_stage else "two-stage")
    checkpoint_type = "distilled" if args.distilled_checkpoint else "standard"
    print("=" * 60)
    print("LTX-2 Video Generation")
    print("=" * 60)
    print(f"Pipeline: {pipeline_type} ({checkpoint_type} checkpoint)")
    print(f"Checkpoint: {args.checkpoint_path}")
    if args.stage2_checkpoint:
        print(f"Stage 2 Checkpoint: {args.stage2_checkpoint}")
    if not args.one_stage:
        print(f"Stage 2 Steps: {args.stage2_steps}")
    print(f"Output: {args.output_path}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} ({args.num_frames / args.frame_rate:.1f}s at {args.frame_rate}fps)")
    print(f"Steps: {args.num_inference_steps}, CFG: {args.cfg_guidance_scale}")
    print(f"Seed: {args.seed}")
    print(f"Offload: {args.offload}")
    print(f"FP8: {args.enable_fp8}")
    if args.images:
        print(f"Image conditioning: {len(args.images)} image(s)")
    if args.anchor_interval:
        anchor_src = args.anchor_image if args.anchor_image else "first --image"
        num_anchors = len(range(args.anchor_interval, args.num_frames, args.anchor_interval))
        decay_info = f", decay: {args.anchor_decay}" if args.anchor_decay != "none" else ""
        print(f"Anchor conditioning: {num_anchors} anchor(s) every {args.anchor_interval} frames (source: {anchor_src}, strength: {args.anchor_strength}{decay_info})")
    # SVI Pro mode info
    if args.svi_mode or args.extend_video:
        svi_mode_type = "Video Extension" if args.extend_video else "Multi-Clip Generation"
        print(f"SVI Pro Mode: {svi_mode_type}")
        print(f"  Clips: {args.num_clips}")
        print(f"  Motion Latent Frames: {args.num_motion_latent}")
        print(f"  Seed Multiplier: {args.seed_multiplier}")
        if args.extend_video:
            print(f"  Input Video: {args.extend_video}")
            print(f"  Prepend Original: {args.prepend_original}")
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
        distilled_checkpoint=args.distilled_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
    )

    # Set up tiling config for VAE
    tiling_config = TilingConfig.default()

    # Determine if sliding window mode should be used
    use_sliding_window = (
        args.num_frames > args.sliding_window_size
        and not args.svi_mode
        and not args.extend_video
    )

    # Branch between sliding window, SVI Pro mode, and regular mode
    if use_sliding_window:
        # Sliding window mode for long videos
        print("=" * 60)
        print(f">>> Using Sliding Window mode: {args.num_frames} frames > {args.sliding_window_size} window size")
        print("=" * 60)

        video_tensor, audio = sliding_window_generate(
            generator=generator,
            args=args,
            total_frames=args.num_frames,
            window_size=args.sliding_window_size,
            overlap=args.sliding_window_overlap,
            overlap_noise=args.sliding_window_overlap_noise,
            color_correction_strength=args.sliding_window_color_correction,
        )

        # Convert to iterator for encode_video
        def tensor_to_iterator(tensor):
            yield tensor

        video = tensor_to_iterator(video_tensor)
        total_frames = video_tensor.shape[0]
        video_chunks_number = get_video_chunks_number(total_frames, tiling_config)

    elif args.svi_mode or args.extend_video:
        # SVI Pro mode: multi-clip or video extension
        # Parse prompt list if provided
        prompts = None
        if getattr(args, 'prompt_list', None) is not None and len(args.prompt_list) > 0:
            prompts = args.prompt_list
            print(f">>> Using {len(prompts)} prompts for multi-clip generation")
            for i, p in enumerate(prompts):
                print(f">>>   Clip {i+1}: {p[:50]}..." if len(p) > 50 else f">>>   Clip {i+1}: {p}")
        else:
            print(f">>> Using single prompt for all clips: {args.prompt}")

        if args.extend_video:
            # Video extension mode
            if not args.images:
                # No explicit image provided, will use frames from video
                args.images = []
            video_tensor, audio = generate_svi_video_extension(
                generator=generator,
                args=args,
                input_video_path=args.extend_video,
                num_clips=args.num_clips,
                num_motion_latent=args.num_motion_latent,
                num_motion_frame=args.num_motion_frame,
                seed_multiplier=args.seed_multiplier,
                overlap_frames=args.overlap_frames,
                anchor_image_path=args.anchor_image,
                prepend_original=args.prepend_original,
                prompts=prompts,
            )
        else:
            # Multi-clip generation mode
            if not args.images:
                raise ValueError("SVI mode requires at least one --image for the initial frame")
            initial_image = args.images[0][0]
            video_tensor, audio = generate_svi_multi_clip(
                generator=generator,
                args=args,
                initial_image_path=initial_image,
                num_clips=args.num_clips,
                num_motion_latent=args.num_motion_latent,
                num_motion_frame=args.num_motion_frame,
                seed_multiplier=args.seed_multiplier,
                overlap_frames=args.overlap_frames,
                prompts=prompts,
            )

        # SVI returns a decoded tensor [F, H, W, C], convert to iterator format for encode_video
        def tensor_to_iterator(tensor):
            """Convert tensor to chunk iterator for encode_video compatibility."""
            yield tensor

        video = tensor_to_iterator(video_tensor)
        # Calculate total frames for proper video chunk handling
        total_frames = video_tensor.shape[0]
        video_chunks_number = get_video_chunks_number(total_frames, tiling_config)
    else:
        # Regular single-clip generation
        video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
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
            stage2_steps=args.stage2_steps,
            anchor_image=args.anchor_image,
            anchor_interval=args.anchor_interval,
            anchor_strength=args.anchor_strength,
            anchor_decay=args.anchor_decay,
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
        "distilled_checkpoint": args.distilled_checkpoint,
        "checkpoint_path": args.checkpoint_path,
        "stage2_checkpoint": args.stage2_checkpoint,
        "stage2_steps": args.stage2_steps if not args.one_stage else None,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt if args.cfg_guidance_scale > 1.0 else None,
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
        "anchor_image": args.anchor_image,
        "anchor_interval": args.anchor_interval,
        "anchor_strength": args.anchor_strength if args.anchor_interval else None,
        "anchor_decay": args.anchor_decay if args.anchor_interval and args.anchor_decay != "none" else None,
        "disable_audio": args.disable_audio,
        "enhance_prompt": args.enhance_prompt,
        # SVI Pro metadata
        "svi_mode": args.svi_mode,
        "svi_num_clips": args.num_clips if (args.svi_mode or args.extend_video) else None,
        "svi_num_motion_latent": args.num_motion_latent if (args.svi_mode or args.extend_video) else None,
        "svi_num_motion_frame": args.num_motion_frame if (args.svi_mode or args.extend_video) else None,
        "svi_seed_multiplier": args.seed_multiplier if (args.svi_mode or args.extend_video) else None,
        "svi_overlap_frames": args.overlap_frames if (args.svi_mode or args.extend_video) else None,
        "svi_extend_video": args.extend_video,
        "svi_prepend_original": args.prepend_original if args.extend_video else None,
        # Sliding window metadata
        "sliding_window_mode": use_sliding_window,
        "sliding_window_size": args.sliding_window_size if use_sliding_window else None,
        "sliding_window_overlap": args.sliding_window_overlap if use_sliding_window else None,
        "sliding_window_overlap_noise": args.sliding_window_overlap_noise if use_sliding_window else None,
        "sliding_window_color_correction": args.sliding_window_color_correction if use_sliding_window else None,
        # Preview metadata
        "preview_dir": args.preview_dir,
        "preview_interval": args.preview_interval if args.preview_dir else None,
        # Video continuation metadata
        "freeze_frames": args.freeze_frames if args.freeze_frames > 0 else None,
        "freeze_transition": args.freeze_transition if args.freeze_frames > 0 else None,
    }

    print(">>> Adding metadata to video...")
    add_metadata_to_video(args.output_path, metadata)

    print(f">>> Output: {args.output_path}")
    print(">>> Done!")


if __name__ == "__main__":
    main()
