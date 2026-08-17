import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.conditioning.reference_layout import apply_reference_layout
from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import apply_segment_phase_to_freqs_cis, segment_phase_rate_vector
from ltx_core.multigpu.transformer.sequence_parallel import (
    pad_modality_for_uniform_sharding,
    tile_modality_for_rank,
)
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape


def test_virtual_sidecar_places_reference_to_the_right() -> None:
    reference = torch.tensor([[[[0.0, 1.0]], [[0.0, 4.0]], [[0.0, 4.0]]]])
    target = torch.tensor([[[[0.0, 1.0], [1.0, 2.0]], [[0.0, 8.0], [0.0, 8.0]], [[0.0, 8.0], [8.0, 16.0]]]])

    positioned = apply_reference_layout(reference, target, layout="virtual_sidecar", sidecar_margin_pixels=2)

    assert positioned[0, 2, 0, 0].item() == 18.0
    assert positioned[0, 0, 0].tolist() == [0.0, 2.0]


def test_source_phase_is_noop_for_target_and_norm_preserving() -> None:
    cos_freq = torch.ones(1, 2, 3, 4)
    sin_freq = torch.zeros_like(cos_freq)
    segment_ids = torch.tensor([[0.0, 2.0, 0.0]])

    out_cos, out_sin = apply_segment_phase_to_freqs_cis(
        (cos_freq, sin_freq), segment_ids, segment_phase_rate_vector(4, 10000.0)
    )

    assert torch.equal(out_cos[:, :, [0, 2]], cos_freq[:, :, [0, 2]])
    assert out_sin[:, :, 1].abs().sum().item() > 0
    assert torch.allclose(out_cos.square() + out_sin.square(), torch.ones_like(out_cos))


def test_source_phase_survives_sequence_parallel_padding_and_tiling() -> None:
    modality = Modality(
        latent=torch.zeros(1, 3, 4),
        sigma=torch.zeros(1),
        timesteps=torch.zeros(1, 3),
        positions=torch.zeros(1, 3, 3, 2),
        context=torch.zeros(1, 1, 4),
        segment_ids=torch.tensor([[0.0, 2.0, 0.0]]),
    )

    padded, original_length = pad_modality_for_uniform_sharding(modality, world_size=2)
    rank_one, token_counts = tile_modality_for_rank(padded, rank=1, world_size=2)

    assert original_length == 3
    assert token_counts == [2, 2]
    assert padded.segment_ids is not None
    assert padded.segment_ids.tolist() == [[0.0, 2.0, 0.0, 0.0]]
    assert rank_one.segment_ids is not None
    assert rank_one.segment_ids.tolist() == [[0.0, 0.0]]


def test_reference_condition_extends_ltx25_token_fields() -> None:
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=1, channels=128, frames=2, height=2, width=2),
        fps=24.0,
    )
    state = tools.create_initial_state(torch.device("cpu"), torch.float32)
    reference = torch.randn(1, 128, 1, 2, 1)

    result = VideoConditionByReferenceLatent(
        reference,
        layout="virtual_sidecar",
        source_phase=True,
        source_id=2,
        phase_scale=0.5,
    ).apply_to(state, tools)

    target_len = state.latent.shape[1]
    assert result.segment_ids is not None
    assert torch.all(result.segment_ids[:, :target_len] == 0)
    assert torch.all(result.segment_ids[:, target_len:] == 1)
    assert result.keyframes_mask is not None
    assert result.keyframes_mask.shape[1] == result.latent.shape[1]
    assert result.positions[0, 2, target_len:].min() >= result.positions[0, 2, :target_len].max()
