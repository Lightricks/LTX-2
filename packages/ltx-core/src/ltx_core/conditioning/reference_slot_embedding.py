"""Learned per-slot embedding for reference conditioning.

``layout`` and ``source_phase`` (see :mod:`ltx_core.conditioning.reference_layout`) make
reference tokens *separable* — they occupy distinct coordinates, or carry a distinct rotary
phase. Neither tells the model **which** reference is which in a way it can bind to a prompt: a
fixed rotary phase is not a feature the network has been pretrained to read, so a LoRA has to
learn to exploit it from scratch.

This module takes the other route. Each reference slot is embedded by Fourier-featurising its
integer index and passing that through a small MLP, producing a vector in token space that is
**added to that reference's latent tokens**. The tag is then an ordinary feature-space signal,
which the attention layers can use directly.

The two mechanisms are independent and compose: the phase separates positionally, the slot
embedding tags in feature space.

Fourier-featurising a scalar before a small MLP is the standard way to give a network a usable
handle on an integer or coordinate — the same construction as sinusoidal position encodings and
NeRF-style input encodings. Here the scalar is the slot index, and the MLP's output width is the
patchified token width so the result can simply be added.

Defaults (16 frequencies, hidden 256, output 128) and the parameter names ``frequencies`` /
``net.0`` / ``net.2`` follow the layout this technique is conventionally published with, which
keeps checkpoints portable across implementations that use it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

DEFAULT_NUM_FREQUENCIES = 16
DEFAULT_HIDDEN_DIM = 256


class ReferenceSlotEmbedding(nn.Module):
    """Map an integer reference-slot index to an additive token-space vector.

    Args:
        token_dim: Width of the patchified latent token the output is added to (128 for LTX
            video latents).
        num_frequencies: Number of Fourier frequencies used to featurise the slot index.
        hidden_dim: Width of the MLP's hidden layer.
    """

    def __init__(
        self,
        token_dim: int = 128,
        num_frequencies: int = DEFAULT_NUM_FREQUENCIES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.num_frequencies = num_frequencies

        # Persistent so the schedule travels with the checkpoint: loading a foreign adapter
        # overwrites these values rather than silently keeping ours.
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32))

        # Input is [index, sin(f * index)..., cos(f * index)...] = 1 + 2 * num_frequencies.
        self.net = nn.Sequential(
            nn.Linear(1 + 2 * num_frequencies, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

        # Start as a no-op: an untrained tag must not perturb the reference tokens, so a run
        # that enables this begins byte-identical to one that does not and departs only as the
        # embedding learns.
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, slot_index: int | float | Tensor) -> Tensor:
        """Return the additive embedding for ``slot_index``, shaped ``[token_dim]``.

        Accepts a scalar (one slot) or a tensor of indices, in which case the leading shape of
        the input is preserved and ``token_dim`` is appended.
        """
        if not isinstance(slot_index, Tensor):
            slot_index = torch.tensor(float(slot_index), device=self.frequencies.device)
        index = slot_index.to(self.frequencies.dtype).unsqueeze(-1)

        scaled = index * self.frequencies
        features = torch.cat([index, torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.net(features.to(self.net[0].weight.dtype))
