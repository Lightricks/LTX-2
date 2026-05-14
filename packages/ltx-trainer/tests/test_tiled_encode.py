import pytest
import torch
import torch.nn as nn

from process_videos import encode_video, tiled_encode_video


class MockVAE(nn.Module):
    """Mean-pool VAE using Conv3d with weight 1/(3*32*32).

    Input:  [B, 3, F, H, W]
    Output: [B, 128, floor((F-1)/8)+1, H//32, W//32]
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=3,
            out_channels=128,
            kernel_size=(1, 32, 32),
            stride=(8, 32, 32),
            padding=0,
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / (3 * 32 * 32))

    def forward(self, x):
        return self.conv(x)


class CountingVAE(nn.Module):
    """Wraps MockVAE and counts forward calls."""

    def __init__(self):
        super().__init__()
        self.inner = MockVAE()
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return self.inner(x)


@pytest.fixture
def vae():
    return MockVAE()


@pytest.fixture
def video_896():
    gen = torch.Generator()
    gen.manual_seed(42)
    return torch.rand(1, 3, 9, 896, 896, generator=gen)


@pytest.fixture
def video_960():
    gen = torch.Generator()
    gen.manual_seed(42)
    return torch.rand(1, 3, 9, 960, 960, generator=gen)


@pytest.fixture
def video_small():
    gen = torch.Generator()
    gen.manual_seed(42)
    return torch.rand(1, 3, 9, 256, 256, generator=gen)


def test_output_shape(vae, video_896):
    out = tiled_encode_video(vae, video_896)
    assert out.shape == (1, 128, 2, 28, 28), f"unexpected shape {out.shape}"


def test_tile_batch_size_produces_identical_output(vae, video_896):
    out_seq = tiled_encode_video(vae, video_896, tile_batch_size=1)
    out_bat = tiled_encode_video(vae, video_896, tile_batch_size=4)
    max_diff = (out_seq - out_bat).abs().max().item()
    assert torch.allclose(out_seq, out_bat, atol=1e-6), (
        f"max abs diff between tile_batch_size=1 and tile_batch_size=4: {max_diff}"
    )


def test_tile_batch_size_zero_raises(vae, video_896):
    with pytest.raises(ValueError, match="tile_batch_size"):
        tiled_encode_video(vae, video_896, tile_batch_size=0)


def test_tile_batch_size_negative_raises(vae, video_896):
    with pytest.raises(ValueError, match="tile_batch_size"):
        tiled_encode_video(vae, video_896, tile_batch_size=-1)


def test_encode_video_threads_tile_batch_size(vae, video_896):
    out_tiled = tiled_encode_video(vae, video_896, tile_batch_size=4)
    out_encode = encode_video(vae, video_896, use_tiling=True, tile_batch_size=4)["latents"]
    max_diff = (out_tiled - out_encode).abs().max().item()
    assert torch.allclose(out_tiled, out_encode, atol=1e-6), (
        f"encode_video and tiled_encode_video differ; max abs diff: {max_diff}"
    )
    # Verify tile_batch_size is live through the full encode_video stack.
    out_encode_seq = encode_video(vae, video_896, use_tiling=True, tile_batch_size=1)["latents"]
    max_diff_seq = (out_encode - out_encode_seq).abs().max().item()
    assert torch.allclose(out_encode, out_encode_seq, atol=1e-6), (
        f"encode_video tile_batch_size=4 vs tile_batch_size=1 differ; max abs diff: {max_diff_seq}"
    )


def test_tile_batch_size_larger_than_tile_count(vae, video_896):
    """tile_batch_size=100 > number of tiles (4); exercises partial-last-sub-list path."""
    out_large = tiled_encode_video(vae, video_896, tile_batch_size=100)
    out_seq = tiled_encode_video(vae, video_896, tile_batch_size=1)
    assert out_large.shape == (1, 128, 2, 28, 28), f"unexpected shape {out_large.shape}"
    max_diff = (out_large - out_seq).abs().max().item()
    assert torch.allclose(out_large, out_seq, atol=1e-6), (
        f"tile_batch_size=100 vs tile_batch_size=1 max abs diff: {max_diff}"
    )


def test_fast_path_single_tile(vae, video_small):
    """Video fits in one tile; fast-path must not be broken by refactor."""
    out = tiled_encode_video(vae, video_small)
    assert out.shape == (1, 128, 2, 8, 8), f"unexpected shape {out.shape}"


def test_tile_batch_size_identical_output_mixed_shapes(vae, video_960):
    """video_960 produces multiple shape groups; batching must be parity-preserving."""
    out_seq = tiled_encode_video(vae, video_960, tile_batch_size=1)
    out_bat = tiled_encode_video(vae, video_960, tile_batch_size=4)
    max_diff = (out_seq - out_bat).abs().max().item()
    assert torch.allclose(out_seq, out_bat, atol=1e-6), (
        f"mixed-shape max abs diff between tile_batch_size=1 and tile_batch_size=4: {max_diff}"
    )


def test_tile_size_not_divisible_raises(vae, video_896):
    with pytest.raises(ValueError, match="tile_size"):
        tiled_encode_video(vae, video_896, tile_size=500)


def test_tile_overlap_not_divisible_raises(vae, video_896):
    with pytest.raises(ValueError, match="tile_overlap"):
        tiled_encode_video(vae, video_896, tile_overlap=100)


def test_tile_overlap_gte_tile_size_raises(vae, video_896):
    with pytest.raises(ValueError, match="tile_overlap"):
        tiled_encode_video(vae, video_896, tile_overlap=512, tile_size=512)


def test_mixed_shape_group_call_count(video_960):
    """Each distinct (tile_h, tile_w) shape group produces ceil(group_size / tile_batch_size) calls.

    video_960 at default tile_size=512, tile_overlap=128 produces 4 shape groups:
      (512,512)×4, (512,192)×2, (192,512)×2, (192,192)×1  →  9 tiles total.

    With tile_batch_size=4: ceil(4/4)+ceil(2/4)+ceil(2/4)+ceil(1/4) = 4 VAE calls.
    With tile_batch_size=1: one call per tile = 9 VAE calls.
    """
    counting_vae = CountingVAE()
    tiled_encode_video(counting_vae, video_960, tile_batch_size=4)
    assert counting_vae.call_count == 4, (
        f"expected 4 VAE calls with tile_batch_size=4, got {counting_vae.call_count}"
    )

    counting_seq = CountingVAE()
    tiled_encode_video(counting_seq, video_960, tile_batch_size=1)
    assert counting_seq.call_count == 9, (
        f"expected 9 VAE calls with tile_batch_size=1, got {counting_seq.call_count}"
    )
