import unittest

import torch

from ltx_trainer.training_strategies.keyframe_to_video import (
    KeyframeToVideoConfig,
    KeyframeToVideoStrategy,
)


class FixedTimestepSampler:
    def __init__(self, sigma: float = 0.5) -> None:
        self.sigma = sigma

    def sample_for(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.full((latents.shape[0], 1, 1), self.sigma, device=latents.device, dtype=latents.dtype)


class KeyframeToVideoStrategyTest(unittest.TestCase):
    def test_appends_clean_first_and_last_keyframe_tokens(self) -> None:
        torch.manual_seed(0)
        strategy = KeyframeToVideoStrategy(KeyframeToVideoConfig())
        batch = self._batch(batch_size=2, channels=4, frames=3, height=2, width=3)

        inputs = strategy.prepare_training_inputs(batch, FixedTimestepSampler(sigma=0.25))

        target_tokens = 3 * 2 * 3
        keyframe_tokens = 2 * 3
        first_keyframe_start = target_tokens
        last_keyframe_start = target_tokens + keyframe_tokens

        expected_first = strategy._video_patchifier.patchify(batch["latents"]["latents"][:, :, :1])
        expected_last = strategy._video_patchifier.patchify(batch["latents"]["latents"][:, :, -1:])

        self.assertEqual(inputs.video.latent.shape, (2, target_tokens + 2 * keyframe_tokens, 4))
        torch.testing.assert_close(
            inputs.video.latent[:, first_keyframe_start:last_keyframe_start],
            expected_first,
        )
        torch.testing.assert_close(
            inputs.video.latent[:, last_keyframe_start:],
            expected_last,
        )

        self.assertTrue(torch.all(inputs.video.timesteps[:, :target_tokens] == 0.25))
        self.assertTrue(torch.all(inputs.video.timesteps[:, target_tokens:] == 0.0))

        self.assertTrue(torch.all(inputs.video_loss_mask[:, :target_tokens]))
        self.assertFalse(torch.any(inputs.video_loss_mask[:, target_tokens:]))

        # A 3-frame latent sequence corresponds to pixel keyframes at frame 0 and frame 16.
        last_keyframe_positions = inputs.video.positions[:, :, last_keyframe_start:]
        torch.testing.assert_close(
            last_keyframe_positions[:, 0, :, 0],
            torch.full((2, keyframe_tokens), 16 / 24, dtype=last_keyframe_positions.dtype),
        )
        torch.testing.assert_close(
            last_keyframe_positions[:, 0, :, 1],
            torch.full((2, keyframe_tokens), 17 / 24, dtype=last_keyframe_positions.dtype),
        )

    def test_loss_ignores_appended_keyframe_tokens(self) -> None:
        torch.manual_seed(0)
        strategy = KeyframeToVideoStrategy(KeyframeToVideoConfig())
        batch = self._batch(batch_size=2, channels=4, frames=3, height=2, width=3)

        inputs = strategy.prepare_training_inputs(batch, FixedTimestepSampler(sigma=0.25))

        target_tokens = inputs.video_targets.shape[1]
        video_pred = torch.zeros_like(inputs.video.latent)
        video_pred[:, target_tokens:] = 1_000_000.0

        expected = inputs.video_targets.pow(2).mean(dim=[-2, -1])
        torch.testing.assert_close(strategy.compute_loss(video_pred, None, inputs), expected)

    @staticmethod
    def _batch(
        batch_size: int,
        channels: int,
        frames: int,
        height: int,
        width: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        values = torch.arange(batch_size * channels * frames * height * width, dtype=torch.float32)
        latents = values.reshape(batch_size, channels, frames, height, width)
        return {
            "latents": {
                "latents": latents,
                "num_frames": torch.full((batch_size,), frames),
                "height": torch.full((batch_size,), height),
                "width": torch.full((batch_size,), width),
                "fps": torch.full((batch_size,), 24.0),
            },
            "conditions": {
                "video_prompt_embeds": torch.randn(batch_size, 5, 8),
                "audio_prompt_embeds": torch.randn(batch_size, 5, 8),
                "prompt_attention_mask": torch.ones(batch_size, 5, dtype=torch.bool),
            },
        }


if __name__ == "__main__":
    unittest.main()
