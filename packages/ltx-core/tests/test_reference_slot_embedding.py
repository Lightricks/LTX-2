"""Tests for the learned reference-slot embedding."""

import torch

from ltx_core.conditioning.reference_slot_embedding import ReferenceSlotEmbedding


def test_parameter_layout_matches_published_adapter() -> None:
    """Key names and shapes must match the LiconStudio MSR adapter so checkpoints interchange."""
    module = ReferenceSlotEmbedding()
    shapes = {name: tuple(tensor.shape) for name, tensor in module.state_dict().items()}

    assert shapes == {
        "frequencies": (16,),
        "net.0.weight": (256, 33),  # 33 = index + sin/cos of 16 frequencies
        "net.0.bias": (256,),
        "net.2.weight": (128, 256),
        "net.2.bias": (128,),
    }


def test_starts_as_a_no_op() -> None:
    """Enabling the tag must not perturb reference tokens before it has learned anything."""
    module = ReferenceSlotEmbedding()

    assert torch.count_nonzero(module(1)) == 0
    assert torch.count_nonzero(module(7)) == 0


def test_slots_separate_under_training() -> None:
    """Distinct indices must be able to produce distinct tags — the module's entire purpose."""
    torch.manual_seed(0)
    module = ReferenceSlotEmbedding()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)

    for _ in range(60):
        loss = (module(1) - 1).pow(2).mean() + (module(2) + 1).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert (module(1) - module(2)).abs().mean() > 0.5


def test_broadcasts_over_reference_tokens() -> None:
    """The tag is added to every token of one reference, so it must broadcast over [B, S, D]."""
    module = ReferenceSlotEmbedding()
    tokens = torch.randn(2, 5, 128)

    tagged = tokens + module(3)

    assert tagged.shape == tokens.shape


def test_accepts_batched_indices() -> None:
    """A tensor of indices keeps its leading shape and gains the token dimension."""
    module = ReferenceSlotEmbedding()

    assert module(torch.tensor([1.0, 2.0, 3.0])).shape == (3, 128)


def test_frequencies_travel_with_the_checkpoint() -> None:
    """Persistent buffer: loading a foreign adapter must bring its own frequency schedule."""
    module = ReferenceSlotEmbedding()
    foreign = {**module.state_dict(), "frequencies": torch.full((16,), 3.0)}

    module.load_state_dict(foreign)

    assert torch.equal(module.frequencies, torch.full((16,), 3.0))


def test_custom_hyperparameters_size_the_network() -> None:
    module = ReferenceSlotEmbedding(num_frequencies=8, hidden_dim=64)
    shapes = {name: tuple(tensor.shape) for name, tensor in module.state_dict().items()}

    assert shapes["frequencies"] == (8,)
    assert shapes["net.0.weight"] == (64, 17)
    assert shapes["net.2.weight"] == (128, 64)
