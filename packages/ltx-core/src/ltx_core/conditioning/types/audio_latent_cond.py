"""Audio conditioning by latent replacement."""

import torch

from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import LatentTools
from ltx_core.types import LatentState


class AudioConditionByLatent(ConditioningItem):
    """
    Conditions audio generation by injecting latents at the start.

    Replaces audio tokens in the latent state and sets denoise strength
    according to the strength parameter. Similar to VideoConditionByLatentIndex
    but for audio.
    """

    def __init__(self, latent: torch.Tensor, strength: float):
        """Initialize audio conditioning.

        Args:
            latent: Audio latent tensor, shape (batch, channels, frames, mel_bins).
            strength: Conditioning strength (0-1). 1.0 = fully conditioned (no denoising).
        """
        self.latent = latent
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        """Apply audio conditioning by replacing tokens and setting denoise mask.

        Args:
            latent_state: Current latent state to modify.
            latent_tools: Tools for patchifying/unpatchifying latents.

        Returns:
            Modified latent state with audio conditioning applied.
        """
        tokens = latent_tools.patchifier.patchify(self.latent)
        num_tokens = tokens.shape[1]

        latent_state = latent_state.clone()

        # Place audio at the start
        latent_state.latent[:, :num_tokens] = tokens
        latent_state.clean_latent[:, :num_tokens] = tokens
        # Set denoise mask: 0 = full denoising, 1 = no denoising
        # strength=1.0 means no denoising (preserve input)
        latent_state.denoise_mask[:, :num_tokens] = 1.0 - self.strength

        return latent_state
