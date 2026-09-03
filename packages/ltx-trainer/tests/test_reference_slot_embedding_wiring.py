"""The slot embedding is zero-initialised, which makes a wiring mistake silent.

If the module ever falls out of the autograd graph — detached tokens, a stale copy after a
``.to()``, an application path that bypasses it — it stays at its zero init for the entire run
and produces a checkpoint whose tag does nothing. Nothing else in training would complain: the
loss curve looks normal, because a zero tag is exactly the untagged baseline.

These tests pin the two properties that make the difference detectable.
"""

import pytest
import torch

from ltx_trainer.training_strategies.flexible import FlexibleStrategy, FlexibleStrategyConfig


def _strategy(**slot_kwargs) -> FlexibleStrategy:
    return FlexibleStrategy(
        FlexibleStrategyConfig.model_validate(
            {
                "name": "flexible",
                "reference_slot_embedding": {"num_frequencies": 16, "hidden_dim": 256},
                "video": {
                    "is_generated": True,
                    "latents_dir": "latents",
                    "conditions": [
                        {"type": "reference", "latents_dir": "ref0", "source_id": 1, **slot_kwargs},
                        {"type": "reference", "latents_dir": "ref1", "source_id": 2, **slot_kwargs},
                    ],
                },
            }
        )
    )


def test_slot_embedding_receives_gradient_through_the_training_path() -> None:
    """The whole point: applying the tag must connect its parameters to the loss."""
    strategy = _strategy(slot_embedding=True)
    module = strategy.reference_slot_embedding
    tokens = torch.randn(1, 12, 128)

    first = strategy._apply_slot_embedding(tokens, strategy.config.video.conditions[0])
    second = strategy._apply_slot_embedding(tokens, strategy.config.video.conditions[1])
    (first.pow(2).mean() + (second - 1).pow(2).mean()).backward()

    assert module.net[2].weight.grad is not None
    assert module.net[2].weight.grad.abs().sum() > 0


def test_disabled_condition_leaves_tokens_untouched() -> None:
    strategy = _strategy(slot_embedding=False)
    tokens = torch.randn(1, 12, 128)

    assert strategy._apply_slot_embedding(tokens, strategy.config.video.conditions[0]) is tokens


def test_missing_module_raises_rather_than_dropping_the_tag() -> None:
    """Applying a tag with no module must fail loudly, not train an untagged model quietly."""
    strategy = _strategy(slot_embedding=True)
    strategy.reference_slot_embedding = None

    with pytest.raises(RuntimeError, match="silently dropped"):
        strategy._apply_slot_embedding(torch.randn(1, 4, 128), strategy.config.video.conditions[0])


def test_checkpoint_metadata_describes_the_module() -> None:
    """An inference pipeline rebuilds the module from metadata, so it must be self-describing."""
    metadata = _strategy(slot_embedding=True).get_checkpoint_metadata()

    assert metadata["reference_slot_embedding_enabled"] == "True"
    assert metadata["reference_slot_embedding_type"] == "fourier_mlp"
    assert metadata["reference_slot_embedding_num_frequencies"] == 16
    assert metadata["reference_slot_indices"] == "1,2"


def test_slot_index_overrides_source_id() -> None:
    strategy = _strategy(slot_embedding=True, slot_index=7)

    assert strategy.get_checkpoint_metadata()["reference_slot_indices"] == "7"
