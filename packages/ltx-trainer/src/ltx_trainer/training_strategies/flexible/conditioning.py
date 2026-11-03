"""Stateless conditioning and flow-matching math for the flexible strategy.
These helpers are pure functions over tensors. The strategy orchestrates them; keeping
them free of strategy/instance state makes the masking and loss math independently testable.
"""

from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from torch import Tensor

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, get_pixel_coords
from ltx_core.model.transformer.modality import Modality, TokenTimesteps
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.flexible.config import (
    FirstFrameConditionConfig,
    IntrinsicConditionBase,
    MaskConditionConfig,
    ModalityConfig,
    PrefixConditionConfig,
    ReferenceConditionConfig,
    SpatialCropConditionConfig,
    SuffixConditionConfig,
)

# Default frames per second for video missing in the FPS metadata
DEFAULT_FPS = 24

# VAE scale factors for LTX-2
VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()

# Stateless patchifiers (patch_size=1) shared by the position/patchify helpers below.
_VIDEO_PATCHIFIER = VideoLatentPatchifier(patch_size=1)
_AUDIO_PATCHIFIER = AudioPatchifier(patch_size=1)

#: The two modalities the flexible strategy conditions over.
ModalityKey = Literal["video", "audio"]


# --- Data carriers ---


@dataclass
class LatentData:
    """Loaded and patchified latents with metadata."""

    latents: Tensor  # [B, seq_len, C]
    num_frames: int
    height: int
    width: int
    fps: float | Tensor  # scalar or one value per batch element


@dataclass
class ModalitySample:
    """Per-modality flow-matching tensors as they flow through the conditioning pipeline.
    Built once (noised for generated modalities, kept clean otherwise), then rewritten by each
    conditioning step, and finally read out into a transformer ``Modality`` plus loss targets/mask.
    """

    latents: Tensor  # working latents: noisy for generated modalities, clean for conditioning ones
    positions: Tensor  # patch positions
    timesteps: Tensor  # per-token timesteps (0 where conditioned / clean)
    sigmas: Tensor  # per-sample noise level (0 for conditioning modalities)
    targets: Tensor | None  # velocity targets; None when the modality is not generated
    loss_mask: Tensor | None  # True where the loss applies; None when the modality is not generated


@dataclass(frozen=True)
class ReferenceScaleFactors:
    """How much smaller a reference (IC-LoRA) clip is than its target, in latent space (1 = same size)."""

    spatial: int = 1
    temporal: int = 1


@dataclass
class ModalityProcessingResult:
    """Result of processing a single modality into a transformer-ready ``Modality``."""

    modality: Modality
    targets: Tensor | None
    loss_mask: Tensor | None


# --- Latents & positions ---


def patchify_latent_data(latent_data: dict[str, Any], modality_key: ModalityKey) -> LatentData:
    """Patchify a batch latent entry into ``[B, seq_len, C]`` and extract its shape metadata."""
    latents = latent_data["latents"]
    if modality_key == "video":
        num_frames, height, width = latents.shape[2:]
        for name, size in (("num_frames", num_frames), ("height", height), ("width", width)):
            if not torch.all(torch.as_tensor(latent_data[name]) == size):
                raise ValueError(f"Video latent metadata '{name}' does not match tensor dimension {size}")
        fps = latent_data.get("fps")
        fps = torch.as_tensor(DEFAULT_FPS if fps is None else fps, device=latents.device, dtype=torch.float64)
        if fps.numel() not in (1, latents.shape[0]) or not torch.all(torch.isfinite(fps) & (fps > 0)):
            raise ValueError("Video FPS must contain one positive, finite value per batch element")
        latents = _VIDEO_PATCHIFIER.patchify(latents)
    else:
        num_frames = latent_data.get("num_frames", [latents.shape[2]])[0]
        if isinstance(num_frames, Tensor):
            num_frames = num_frames.item()
        height = 1
        width = 1
        fps = 1.0
        latents = _AUDIO_PATCHIFIER.patchify(latents)
    return LatentData(latents=latents, num_frames=num_frames, height=height, width=width, fps=fps)


def positions_for(
    data: LatentData,
    modality_key: ModalityKey,
    video_scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> Tensor:
    """Generate patch positions for a modality (video uses a spatial grid; audio is 1-D).
    Uses ltx-core's native patchifier so train-time positions match inference exactly; video
    positions are float32 (RoPE temporal precision) with the time axis divided by fps.
    ``video_scale_factors`` are the VAE compression factors (read from the checkpoint) used to
    convert latent coords to pixel coords -- defaults to the standard 32x32x8 layout.
    """
    batch_size, seq_len, _ = data.latents.shape
    device = data.latents.device
    if modality_key == "video":
        latent_coords = _VIDEO_PATCHIFIER.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=data.num_frames, height=data.height, width=data.width, batch=batch_size, channels=128
            ),
            device=device,
        )
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords, scale_factors=video_scale_factors, causal_fix=True
        ).float()
        fps = torch.as_tensor(data.fps, device=device, dtype=torch.float64).reshape(-1, 1, 1)
        # Match inference's scalar division exactly: PyTorch's CUDA scalar path
        # computes the reciprocal in double before casting to the position dtype.
        # CPU scalar division instead casts the divisor before dividing.
        temporal = pixel_coords[:, 0, ...]
        pixel_coords[:, 0, ...] = (
            temporal * fps.reciprocal().float() if device.type == "cuda" else temporal / fps.float()
        )
        return pixel_coords
    return _AUDIO_PATCHIFIER.get_patch_grid_bounds(
        output_shape=AudioLatentShape(frames=seq_len, mel_bins=16, batch=batch_size, channels=8),
        device=device,
    )


# --- Masks ---


def compute_temporal_mask(
    batch_size: int,
    seq_len: int,
    height: int,
    width: int,
    num_frames: int,
    from_end: bool,
    device: torch.device,
) -> Tensor:
    """Compute float mask for temporal region (prefix or suffix). Returns [B, seq_len] in {0, 1}."""
    tokens_per_frame = height * width
    num_tokens = num_frames * tokens_per_frame
    mask = torch.zeros(batch_size, seq_len, device=device)
    if from_end:
        mask[:, -num_tokens:] = 1.0
    else:
        mask[:, :num_tokens] = 1.0
    return mask


def compute_spatial_crop_mask(
    batch_size: int,
    seq_len: int,
    height: int,
    width: int,
    region: tuple[int, int, int, int],
    device: torch.device,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> Tensor:
    """Compute float mask for spatial crop region (y1, x1, y2, x2) in pixel coords.
    Returns [B, seq_len] in {0, 1}. ``scale_factors`` are the VAE compression factors (read from
    the checkpoint) used to map pixel coords to latent coords -- defaults to the 32x32x8 layout.
    """
    y1, x1, y2, x2 = region
    num_frames = seq_len // (height * width)

    # Convert pixel to latent coordinates and clamp (per-axis VAE scale factor).
    def to_latent(v: int, scale: int, max_v: int) -> int:
        return max(0, min(v // scale, max_v))

    ly1 = to_latent(y1, scale_factors.height, height)
    ly2 = to_latent(y2, scale_factors.height, height)
    lx1 = to_latent(x1, scale_factors.width, width)
    lx2 = to_latent(x2, scale_factors.width, width)

    # Create spatial mask and tile across frames
    spatial_mask = torch.zeros(height, width, device=device)
    spatial_mask[ly1:ly2, lx1:lx2] = 1.0
    full_mask = spatial_mask.flatten().repeat(num_frames)

    return full_mask.unsqueeze(0).expand(batch_size, -1)


# --- Scale factors ---


def infer_scale_factor(cond_height: int, cond_width: int, target_height: int, target_width: int) -> int:
    """Infer spatial scale factor between condition and target resolutions."""
    if min(cond_height, cond_width, target_height, target_width) <= 0:
        raise ValueError("Condition and target dimensions must be positive")
    if target_height % cond_height or target_width % cond_width:
        raise ValueError("Target dimensions must be exact multiples of condition dimensions")
    if target_height == cond_height and target_width == cond_width:
        return 1
    scale_h = target_height // cond_height
    scale_w = target_width // cond_width
    if scale_h != scale_w:
        raise ValueError(
            f"Non-uniform scale factors between condition and target: height={scale_h}, width={scale_w}. "
            "Condition and target resolutions must scale uniformly."
        )
    return scale_h


def infer_temporal_scale_factor(cond_num_frames: int, target_num_frames: int) -> int:
    """Infer temporal scale factor between condition and target latent frame counts.
    The first latent frame encodes a single pixel frame (the VAE's causal structure),
    so the temporal groups count is (num_frames - 1).  The scale factor is the ratio
    of target groups to condition groups.
    """
    if target_num_frames == cond_num_frames:
        return 1
    target_groups = target_num_frames - 1
    cond_groups = cond_num_frames - 1
    if cond_groups <= 0 or target_groups <= 0:
        return 1
    if target_groups % cond_groups != 0:
        raise ValueError(
            f"Target temporal groups ({target_groups}) is not evenly divisible by "
            f"condition temporal groups ({cond_groups})."
        )
    return target_groups // cond_groups


def rescale_reference_positions(
    cond_positions: Tensor,
    target_positions: Tensor,
    spatial_scale_factor: int,
    temporal_scale_factor: int,
) -> Tensor:
    """Translate/rescale reduced-resolution reference patch positions into the target's frame.
    Returns a new tensor; a no-op returning the input unchanged when both factors are 1.
    """
    if spatial_scale_factor == 1 and temporal_scale_factor == 1:
        return cond_positions
    cond_positions = cond_positions.clone()
    if temporal_scale_factor != 1:
        # Ref positions are already at the ref's effective fps (source_fps / S, stored by
        # process_videos.py). Shift by (S - 1) / target_fps so the ref's last patch aligns
        # with the target's last; clamp the causal patch at 0.
        t_target = target_positions[:, 0, 0:1, 1:2]  # = 1 / target_fps
        cond_positions[:, 0, ...] = torch.clamp(
            cond_positions[:, 0, ...] - (temporal_scale_factor - 1) * t_target, min=0
        )
    if spatial_scale_factor != 1:
        cond_positions[:, 1, ...] *= spatial_scale_factor
        cond_positions[:, 2, ...] *= spatial_scale_factor
    return cond_positions


# --- Conditioning pipeline ---


def build_modality_sample(
    data: LatentData,
    modality_key: ModalityKey,
    timestep_sampler: TimestepSampler,
    is_generated: bool,
    video_scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    *,
    sigmas: Tensor | None = None,
) -> ModalitySample:
    """Create the initial per-modality sample: noise it (generated) or keep it clean, plus positions."""
    clean = data.latents
    batch_size, seq_len, _ = clean.shape
    device, dtype = clean.device, clean.dtype
    positions = positions_for(data, modality_key, video_scale_factors)

    if is_generated:
        if sigmas is None:
            sigmas = timestep_sampler.sample_for(clean)
        noise = torch.randn_like(clean)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        latents = (1 - sigmas_expanded) * clean + sigmas_expanded * noise
        targets = noise - clean  # velocity prediction
        timesteps = sigmas.view(-1, 1).expand(batch_size, seq_len).clone()
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    else:
        # Conditioning modality: keep clean (sigma=0), no loss.
        latents = clean
        sigmas = torch.zeros(batch_size, device=device, dtype=dtype)
        targets = None
        timesteps = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        loss_mask = None

    return ModalitySample(
        latents=latents,
        positions=positions,
        timesteps=timesteps,
        sigmas=sigmas,
        targets=targets,
        loss_mask=loss_mask,
    )


def apply_intrinsic_condition(
    sample: ModalitySample,
    data: LatentData,
    config: IntrinsicConditionBase,
    batch: dict[str, Any],
    video_scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> ModalitySample:
    """Apply intrinsic conditioning using a binary mask (returns an updated sample).
    For each token, the mask value determines conditioning strength:
    - mask=1: conditioned (clean latent, timestep=0, excluded from loss)
    - mask=0: generated (noisy latent, original timestep, contributes to loss)
    The conditioning decision is drawn independently per batch element so the training signal across
    samples in a batch is i.i.d. -- a single batch-wide draw would correlate gradient updates. Only
    valid on a generated sample (``loss_mask`` is set); the clean latents come from ``data``.
    """
    batch_size, seq_len, _ = sample.latents.shape
    device = sample.latents.device

    # An image has no continuation to predict. Preserve the legacy strategies' guard
    # rather than turning its entire video loss off when first-frame conditioning fires.
    if isinstance(config, FirstFrameConditionConfig) and data.num_frames == 1:
        return sample

    # Per-sample Bernoulli draw -- each element is independently conditioned.
    apply_per_sample = torch.rand(batch_size, device=device) < config.probability
    if not apply_per_sample.any():
        return sample

    h, w = data.height, data.width
    if isinstance(config, FirstFrameConditionConfig):
        mask = compute_temporal_mask(batch_size, seq_len, h, w, 1, False, device)
    elif isinstance(config, PrefixConditionConfig):
        mask = compute_temporal_mask(batch_size, seq_len, h, w, config.temporal_boundary, False, device)
    elif isinstance(config, SuffixConditionConfig):
        mask = compute_temporal_mask(batch_size, seq_len, h, w, config.temporal_boundary, True, device)
    elif isinstance(config, SpatialCropConditionConfig):
        mask = compute_spatial_crop_mask(batch_size, seq_len, h, w, config.spatial_region, device, video_scale_factors)
    elif isinstance(config, MaskConditionConfig):
        # Binarize to match inference, which thresholds masks at load time
        mask = (batch[config.mask_dir]["mask"].reshape(batch_size, seq_len) > 0.5).float()
    else:
        raise ValueError(f"Unknown intrinsic condition type: {type(config).__name__}")

    # Zero the mask for samples the per-sample draw did not select.
    mask = mask * apply_per_sample.view(-1, 1).to(mask.dtype)

    # Apply binary mask: clean conditioned tokens, noisy generated tokens.
    m = mask.unsqueeze(-1)
    return replace(
        sample,
        latents=m * data.latents + (1 - m) * sample.latents,
        timesteps=(1 - mask) * sample.timesteps,
        loss_mask=sample.loss_mask & (mask == 0),
    )


def apply_reference_condition(
    sample: ModalitySample,
    batch: dict[str, Any],
    config: ReferenceConditionConfig,
    modality_key: ModalityKey,
    scale_factors: ReferenceScaleFactors | None,
    video_scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> ModalitySample:
    """Prepend reference latents to the target sequence for reference conditioning (IC-LoRA style).
    The apply/skip decision is batch-wide (reference conditioning changes the sequence length, so it
    cannot be applied to only part of a batch) but is drawn from the torch RNG so runs are
    reproducible under ``torch.manual_seed`` -- mirroring the intrinsic per-sample draw rather than
    Python's unseeded ``random``. Reference (clean, timestep=0) tokens are prepended and excluded
    from the loss; their positions are rescaled into the target's frame for reduced-resolution refs.
    """
    if torch.rand((), device=sample.latents.device).item() >= config.probability:
        return sample

    cond = patchify_latent_data(batch[config.latents_dir], modality_key)
    cond_latents = cond.latents
    batch_size, cond_seq_len, _ = cond_latents.shape
    device = cond_latents.device
    dtype = cond_latents.dtype

    cond_positions = positions_for(cond, modality_key, video_scale_factors)
    if modality_key == "video":
        factors = scale_factors or ReferenceScaleFactors()
        cond_positions = rescale_reference_positions(
            cond_positions, sample.positions, factors.spatial, factors.temporal
        )

    cond_timesteps = torch.zeros(batch_size, cond_seq_len, device=device, dtype=dtype)
    cond_loss_mask = torch.zeros(batch_size, cond_seq_len, dtype=torch.bool, device=device)
    loss_mask = torch.cat([cond_loss_mask, sample.loss_mask], dim=1) if sample.loss_mask is not None else None
    # Targets stay unchanged (only the target portion contributes to the loss).
    return replace(
        sample,
        latents=torch.cat([cond_latents, sample.latents], dim=1),
        positions=torch.cat([cond_positions, sample.positions], dim=2),
        timesteps=torch.cat([cond_timesteps, sample.timesteps], dim=1),
        loss_mask=loss_mask,
    )


def process_modality(
    modality_config: ModalityConfig | None,
    batch: dict[str, Any],
    modality_key: ModalityKey,
    timestep_sampler: TimestepSampler,
    scale_factors: ReferenceScaleFactors | None,
    video_scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    *,
    sigmas: Tensor | None = None,
) -> ModalityProcessingResult | None:
    """Build a transformer-ready ``Modality`` for one modality: load latents, noise, condition, position.
    Returns ``None`` when the modality is absent. Generated modalities are noised and contribute to
    the loss; conditioning modalities stay clean (sigma=0) and are excluded. ``video_scale_factors``
    are the VAE compression factors (read from the checkpoint) driving latent<->pixel coordinate math.
    """
    if modality_config is None:
        return None

    data = patchify_latent_data(batch[f"{modality_key}_latents"], modality_key)
    sample = build_modality_sample(
        data, modality_key, timestep_sampler, modality_config.is_generated, video_scale_factors, sigmas=sigmas
    )

    # Intrinsic conditions must all apply before any reference condition (which changes seq length).
    for cond in modality_config.conditions:
        if isinstance(cond, IntrinsicConditionBase) and modality_config.is_generated:
            sample = apply_intrinsic_condition(sample, data, cond, batch, video_scale_factors)
    for cond in modality_config.conditions:
        if isinstance(cond, ReferenceConditionConfig):
            sample = apply_reference_condition(sample, batch, cond, modality_key, scale_factors, video_scale_factors)

    keyframes_mask = None
    if modality_key == "video":
        # Match VideoLatentTools: the causal encoder's first latent frame represents
        # one pixel frame. Reference tokens are prepended and remain unmarked.
        keyframes_mask = torch.zeros(*sample.latents.shape[:2], 1, device=sample.latents.device)
        target_start = sample.latents.shape[1] - data.latents.shape[1]
        keyframes_mask[:, target_start : target_start + data.height * data.width] = 1.0

    conditions = batch["conditions"]
    modality = Modality(
        enabled=True,
        latent=sample.latents,
        sigma=sample.sigmas,
        timesteps=TokenTimesteps(sample.timesteps),
        positions=sample.positions,
        context=conditions[f"{modality_key}_prompt_embeds"],
        context_mask=conditions["prompt_attention_mask"],
        keyframes_mask=keyframes_mask,
    )
    return ModalityProcessingResult(modality=modality, targets=sample.targets, loss_mask=sample.loss_mask)


# --- Loss ---


def compute_modality_loss(pred: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
    """Compute per-element MSE loss for a single modality. Returns [B,]."""
    # Slice prediction to match targets length (removes any prepended condition tokens)
    target_len = targets.shape[1]
    pred = pred[:, -target_len:, :]
    mask = loss_mask[:, -target_len:]

    # Compute masked MSE loss, reduce per-element [B,] over (seq, channels)
    mask_expanded = mask.unsqueeze(-1).float()
    squared_error = (pred - targets).pow(2)
    masked_loss = squared_error * mask_expanded
    return masked_loss.mean(dim=[-2, -1]) / mask_expanded.mean(dim=[-2, -1]).clamp(min=1e-8)
