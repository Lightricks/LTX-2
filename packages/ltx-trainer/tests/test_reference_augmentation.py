"""Reference augmentation exists to remove the copy shortcut, so what it must guarantee is
that the reference the model sees is no longer the reference it could reproduce verbatim —
while staying off entirely by default and at inference.
"""

import torch

from ltx_trainer.training_strategies.flexible import FlexibleStrategy, FlexibleStrategyConfig


def _condition(**kwargs) -> object:
    strategy = FlexibleStrategy(
        FlexibleStrategyConfig.model_validate(
            {
                "name": "flexible",
                "video": {
                    "is_generated": True,
                    "latents_dir": "latents",
                    "conditions": [{"type": "reference", "latents_dir": "ref", **kwargs}],
                },
            }
        )
    )
    return strategy.config.video.conditions[0]


def test_disabled_by_default() -> None:
    """Existing configs must be untouched — augmentation is opt-in."""
    latents = torch.randn(1, 8, 128)

    assert FlexibleStrategy._augment_reference(latents, _condition()) is latents


def test_noise_perturbs_the_reference() -> None:
    torch.manual_seed(0)
    latents = torch.randn(1, 8, 128)

    augmented = FlexibleStrategy._augment_reference(latents, _condition(augment_noise=0.3))

    assert not torch.allclose(augmented, latents)
    assert augmented.shape == latents.shape


def test_noise_scales_with_the_latents_own_spread() -> None:
    """The setting is a fraction of the reference's std, so it means the same thing whatever
    scale the VAE puts its latents on."""
    torch.manual_seed(0)
    small = torch.randn(1, 64, 128) * 0.1
    large = small * 100

    condition = _condition(augment_noise=0.5)
    small_delta = (FlexibleStrategy._augment_reference(small, condition) - small).std()
    large_delta = (FlexibleStrategy._augment_reference(large, condition) - large).std()

    assert torch.isclose(large_delta / small_delta, torch.tensor(100.0), rtol=0.1)


def test_noise_magnitude_follows_the_setting() -> None:
    torch.manual_seed(0)
    latents = torch.randn(1, 256, 128)

    delta = FlexibleStrategy._augment_reference(latents, _condition(augment_noise=0.25)) - latents

    assert torch.isclose(delta.std() / latents.std(), torch.tensor(0.25), rtol=0.1)


def test_each_call_draws_fresh_noise() -> None:
    """Reusing one perturbation would let the model learn the noise instead of seeing through it."""
    torch.manual_seed(0)
    latents = torch.randn(1, 8, 128)
    condition = _condition(augment_noise=0.3)

    first = FlexibleStrategy._augment_reference(latents, condition)
    second = FlexibleStrategy._augment_reference(latents, condition)

    assert not torch.allclose(first, second)
