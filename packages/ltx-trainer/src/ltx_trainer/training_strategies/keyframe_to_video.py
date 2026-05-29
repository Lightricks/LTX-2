"""Keyframe-to-video training strategy.

This strategy mirrors keyframe interpolation inference:
- Target video tokens are noised and trained normally.
- Clean first/last keyframe tokens are appended as conditioning tokens.
- Loss is computed only on target video tokens, not appended keyframes.
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import VideoLatentShape
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class KeyframeToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for keyframe-to-video interpolation training."""

    name: Literal["keyframe_to_video"] = "keyframe_to_video"

    first_frame_conditioning_p: float = Field(
        default=1.0,
        description="Batch-level probability of appending a clean first-frame keyframe condition",
        ge=0.0,
        le=1.0,
    )

    last_frame_conditioning_p: float = Field(
        default=1.0,
        description="Batch-level probability of appending a clean last-frame keyframe condition",
        ge=0.0,
        le=1.0,
    )

    random_keyframe_conditioning_p: float = Field(
        default=0.0,
        description="Batch-level probability of appending clean intermediate keyframe conditions",
        ge=0.0,
        le=1.0,
    )

    max_random_keyframes: int = Field(
        default=0,
        description="Maximum number of intermediate latent frames to append when random keyframe conditioning is used",
        ge=0,
    )


class KeyframeToVideoStrategy(TrainingStrategy):
    """Train generation conditioned on appended first and last keyframe tokens."""

    config: KeyframeToVideoConfig

    def __init__(self, config: KeyframeToVideoConfig):
        super().__init__(config)

    def get_data_sources(self) -> dict[str, str]:
        """Keyframe training reuses standard video latents and text conditions."""
        return {
            "latents": "latents",
            "conditions": "conditions",
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare noised target tokens plus appended clean endpoint keyframe tokens."""
        latents = batch["latents"]
        target_latents = latents["latents"]

        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        target_tokens = self._video_patchifier.patchify(target_latents)

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size, target_seq_len, _channels = target_tokens.shape
        device = target_tokens.device
        dtype = target_tokens.dtype

        sigmas = timestep_sampler.sample_for(target_tokens)
        noise = torch.randn_like(target_tokens)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_target = (1 - sigmas_expanded) * target_tokens + sigmas_expanded * noise
        targets = noise - target_tokens

        target_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        latent_parts = [noisy_target]
        positions_parts = [target_positions]
        conditioning_masks = [torch.zeros(batch_size, target_seq_len, dtype=torch.bool, device=device)]
        loss_masks = [torch.ones(batch_size, target_seq_len, dtype=torch.bool, device=device)]

        if self._sample_batch_condition(self.config.first_frame_conditioning_p, device):
            self._append_keyframe(
                target_latents=target_latents[:, :, :1],
                frame_idx=0,
                fps=fps,
                latent_parts=latent_parts,
                positions_parts=positions_parts,
                conditioning_masks=conditioning_masks,
                loss_masks=loss_masks,
            )

        if self._sample_batch_condition(self.config.last_frame_conditioning_p, device):
            last_frame_idx = (num_frames - 1) * VIDEO_SCALE_FACTORS.time
            self._append_keyframe(
                target_latents=target_latents[:, :, -1:],
                frame_idx=last_frame_idx,
                fps=fps,
                latent_parts=latent_parts,
                positions_parts=positions_parts,
                conditioning_masks=conditioning_masks,
                loss_masks=loss_masks,
            )

        if self.config.max_random_keyframes > 0 and self._sample_batch_condition(
            self.config.random_keyframe_conditioning_p,
            device,
        ):
            interior_frame_indices = self._sample_interior_latent_frame_indices(
                num_frames=num_frames,
                max_keyframes=self.config.max_random_keyframes,
                device=device,
            )
            for latent_frame_idx in interior_frame_indices:
                self._append_keyframe(
                    target_latents=target_latents[:, :, latent_frame_idx : latent_frame_idx + 1],
                    frame_idx=latent_frame_idx * VIDEO_SCALE_FACTORS.time,
                    fps=fps,
                    latent_parts=latent_parts,
                    positions_parts=positions_parts,
                    conditioning_masks=conditioning_masks,
                    loss_masks=loss_masks,
                )

        combined_latents = torch.cat(latent_parts, dim=1)
        conditioning_mask = torch.cat(conditioning_masks, dim=1)
        video_loss_mask = torch.cat(loss_masks, dim=1)
        positions = torch.cat(positions_parts, dim=2)
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())

        video_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=combined_latents,
            timesteps=timesteps,
            positions=positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE on target tokens only. Returns [B,]."""
        target_seq_len = inputs.video_targets.shape[1]
        target_pred = video_pred[:, :target_seq_len, :]
        target_loss_mask = inputs.video_loss_mask[:, :target_seq_len]

        loss = (target_pred - inputs.video_targets).pow(2)
        loss_mask = target_loss_mask.unsqueeze(-1).float()
        masked = loss.mul(loss_mask)
        return masked.mean(dim=[-2, -1]) / loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8)

    @staticmethod
    def _sample_batch_condition(probability: float, device: torch.device) -> bool:
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return bool((torch.rand((), device=device) < probability).item())

    @staticmethod
    def _sample_interior_latent_frame_indices(
        num_frames: int,
        max_keyframes: int,
        device: torch.device,
    ) -> list[int]:
        interior_count = max(0, num_frames - 2)
        if interior_count == 0 or max_keyframes == 0:
            return []

        keyframe_count = min(interior_count, max_keyframes)
        indices = torch.randperm(interior_count, device=device)[:keyframe_count] + 1
        return sorted(int(idx.item()) for idx in indices)

    def _append_keyframe(
        self,
        target_latents: Tensor,
        frame_idx: int,
        fps: float,
        latent_parts: list[Tensor],
        positions_parts: list[Tensor],
        conditioning_masks: list[Tensor],
        loss_masks: list[Tensor],
    ) -> None:
        keyframe_tokens = self._video_patchifier.patchify(target_latents)
        keyframe_positions = self._get_keyframe_positions(
            keyframes=target_latents,
            frame_idx=frame_idx,
            fps=fps,
        )

        batch_size, keyframe_seq_len, _channels = keyframe_tokens.shape
        keyframe_conditioning_mask = torch.ones(
            batch_size,
            keyframe_seq_len,
            dtype=torch.bool,
            device=keyframe_tokens.device,
        )
        keyframe_loss_mask = torch.zeros(
            batch_size,
            keyframe_seq_len,
            dtype=torch.bool,
            device=keyframe_tokens.device,
        )

        latent_parts.append(keyframe_tokens)
        positions_parts.append(keyframe_positions)
        conditioning_masks.append(keyframe_conditioning_mask)
        loss_masks.append(keyframe_loss_mask)

    def _get_keyframe_positions(
        self,
        keyframes: Tensor,
        frame_idx: int,
        fps: float,
    ) -> Tensor:
        latent_coords = self._video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(keyframes.shape),
            device=keyframes.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=frame_idx == 0,
        )
        positions[:, 0, ...] += frame_idx
        positions[:, 0, ..., 1:] = positions[:, 0, ..., :1] + 1
        positions = positions.to(dtype=torch.float32)
        positions[:, 0, ...] /= fps
        return positions.to(dtype=keyframes.dtype)
