"""Chunk operators: stream ``Chunk`` / ``PlaybackSegment`` through denoise, upsample, and decode."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import torch

from ltx_core.conditioning import ConditioningItem, VideoConditionByKeyframeIndex, VideoGeneratedKeyframeSlots
from ltx_core.conditioning.types.latent_cond import AudioConditionByLatentIndex, VideoConditionByLatentIndex
from ltx_core.model.video_vae.keyframes import DecodeKeyframes
from ltx_core.types import VIDEO_SCALE_FACTORS, Audio, AudioLatentShape
from ltx_pipelines.chunks.layout import ChunkLayout
from ltx_pipelines.chunks.state import (
    Chunk,
    GeneratedKeyframes,
    MakeAudioConditionings,
    MakeVideoConditionings,
    VideoConditioningsWithGeneratedKeyframes,
)
from ltx_pipelines.utils import SimpleDenoiser, assert_stage_supports_generated_keyframes
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DenoisingLoop,
    DiffusionStage,
    Noiser,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.helpers import decode_keyframes_from_slots
from ltx_pipelines.utils.media_io.types import PlaybackSegment
from ltx_pipelines.utils.types import Denoiser, ModalitySpec, VideoAudio

# Longer than the vocoder receptive field so independently decoded seams can
# overlap-add without exposing either chunk's boundary transient.
AUDIO_SEAM_CROSSFADE_MS = 40.0


def denoise_chunks(  # noqa: PLR0913
    chunks: Iterator[Chunk],
    stage: DiffusionStage,
    sigmas: torch.Tensor,
    noiser: Noiser,
    fps: float,
    *,
    noise_scale: float | None = None,
    audio_noise_scale: float | None = None,
    loop: DenoisingLoop | None = None,
    max_batch_size: int = 1,
    denoiser: Denoiser | None = None,
    freeze_video: bool = False,
    freeze_audio: bool = False,
) -> Iterator[Chunk]:
    """Denoise each ``Chunk`` through ``stage``, materializing conditionings and threading carry.
    ``denoiser`` (``SimpleDenoiser`` when omitted) is shared by every chunk; each chunk's positive
    context reaches it through the chunk's ``ModalitySpec``.
    """
    denoiser = SimpleDenoiser() if denoiser is None else denoiser
    scale = sigmas[0].item() if noise_scale is None else noise_scale
    audio_scale = scale if audio_noise_scale is None else audio_noise_scale
    carry_video: torch.Tensor | None = None
    carry_audio: torch.Tensor | None = None
    # Each denoise pass owns a separate carry at that stage's spatial resolution.
    stage_carry_keyframes: GeneratedKeyframes | None = None
    for chunk in chunks:
        incoming_keyframes = stage_carry_keyframes if stage_carry_keyframes is not None else chunk.incoming_keyframes
        video_cond = _append_incoming_keyframes(list(chunk.make_video_conditionings(chunk)), incoming_keyframes)
        audio_cond = list(chunk.make_audio_conditionings(chunk))
        if any(isinstance(item, VideoGeneratedKeyframeSlots) for item in video_cond):
            assert_stage_supports_generated_keyframes(stage)

        # A frozen modality is pinned to ``noise_scale=0.0``, matching every other freeze site in
        # the package. ``_build_state`` noises before it zeroes the denoise mask, so a frozen spec
        # carrying the schedule's scale would hand the transformer a latent that is mostly Gaussian
        # noise while ``sigma=0`` labels it clean conditioning.
        video = (
            ModalitySpec(
                latent=chunk.video,
                conditioning_fps=fps,
                context=chunk.video_context,
                conditionings=_append_carry(video_cond, carry_video, VideoConditionByLatentIndex),
                noise_scale=0.0 if freeze_video else scale,
                frozen=freeze_video,
            )
            if chunk.video is not None
            else None
        )
        audio = (
            ModalitySpec(
                latent=chunk.audio,
                conditioning_fps=fps,
                context=chunk.audio_context,
                conditionings=_append_carry(audio_cond, carry_audio, AudioConditionByLatentIndex),
                noise_scale=0.0 if freeze_audio else audio_scale,
                frozen=freeze_audio,
            )
            if chunk.audio is not None
            else None
        )

        video_state, audio_state = stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            modalities=VideoAudio(video=video, audio=audio),
            loop=loop,
            max_batch_size=max_batch_size,
        )
        video_latent = video_state.latent if video_state is not None else None
        audio_latent = audio_state.latent if audio_state is not None else None
        # A frozen modality is conditioning, not output: keep the latent that went in rather
        # than the loop's copy of it. ``euler_denoising_loop`` steps frozen states too, and
        # ``to_velocity`` rounds to the latent dtype on every step, so what comes back has
        # drifted a few bf16 ulps from what was frozen -- inaudible in a latent, but the
        # vocoder turns it into a visibly different waveform.
        if freeze_video:
            video_latent = chunk.video
        if freeze_audio:
            audio_latent = chunk.audio
        sidecar = None
        if video_state is not None and video_state.generated_keyframes is not None:
            layout = video_state.generated_keyframe_layout
            positions = layout.pixel_frame_indices if layout is not None else ()
            sidecar = GeneratedKeyframes(latent=video_state.generated_keyframes, pixel_frame_indices=positions)
        carry_video_frames = 0
        carry_audio_frames = 0
        if chunk.layout.next_video_carry_frames != 0:
            t = VIDEO_SCALE_FACTORS.time
            carry_video_frames = (chunk.layout.next_video_carry_frames - 1) // t + 1
            carry_audio_frames = max(
                1,
                AudioLatentShape.from_duration(
                    batch=1,
                    duration=float(chunk.layout.next_video_carry_frames) / fps,
                ).frames,
            )
        carry_video = _slice_latent_tail(video_latent, carry_video_frames)
        carry_audio = _slice_latent_tail(audio_latent, carry_audio_frames)
        stage_carry_keyframes = _keyframes_for_next_chunk(sidecar, chunk.layout)
        yield replace(
            chunk,
            video=video_latent,
            audio=audio_latent,
            generated_keyframes=sidecar,
            incoming_keyframes=incoming_keyframes,
        )


def spatially_upsample_chunks(chunks: Iterator[Chunk], upsampler: VideoUpsampler) -> Iterator[Chunk]:
    """Spatially upsample each chunk's video latent and generated-keyframe slots; audio and factories pass through."""
    for chunk in chunks:
        if chunk.video is None:
            yield chunk
            continue
        video = upsampler(chunk.video[:1])
        sidecar = chunk.generated_keyframes
        if sidecar is not None:
            sidecar = replace(sidecar, latent=upsampler(sidecar.latent[:1]))
        incoming = chunk.incoming_keyframes
        if incoming is not None:
            incoming = replace(incoming, latent=upsampler(incoming.latent[:1]))
        yield replace(chunk, video=video, generated_keyframes=sidecar, incoming_keyframes=incoming)


def replace_video_conditionings(
    chunks: Iterator[Chunk],
    make_video_conditionings: MakeVideoConditionings,
    *,
    preserve_generated_keyframes: bool = True,
) -> Iterator[Chunk]:
    """Replace pipeline-owned video conditionings, preserving planned keyframes by default."""
    for chunk in chunks:
        current = chunk.make_video_conditionings
        replacement = (
            current.replace_inner(make_video_conditionings)
            if preserve_generated_keyframes and isinstance(current, VideoConditioningsWithGeneratedKeyframes)
            else make_video_conditionings
        )
        yield replace(chunk, make_video_conditionings=replacement)


def replace_audio_conditionings(
    chunks: Iterator[Chunk],
    make_audio_conditionings: MakeAudioConditionings,
) -> Iterator[Chunk]:
    """Overwrite each chunk's audio conditioning factory; the video factory is unchanged."""
    for chunk in chunks:
        yield replace(chunk, make_audio_conditionings=make_audio_conditionings)


def decode_chunks(
    chunks: Iterator[Chunk],
    video_decoder: VideoDecoder,
    audio_decoder: AudioDecoder | None,
    fps: float,
    *,
    tiling_config: object | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
    keyframes: bool = False,
) -> Iterator[PlaybackSegment]:
    """Decode chunks and join their carried overlap.
    Each body streams immediately, while its outgoing video tail waits to blend
    with the next chunk's carried prefix. Audio retains matching seam samples
    and equal-power crossfades them across the same boundary.
    With ``keyframes=True``, incoming carried keyframes and the chunk's newly
    generated keyframes anchor its decode; their local indices are rebased to
    the global timeline. Empty decoder output is skipped. Without an
    ``audio_decoder``, chunks are video-only.
    """
    pending_tail: PlaybackSegment | None = None
    for chunk in chunks:
        if chunk.video is None:
            raise ValueError("decode_chunks requires chunk.video")
        layout = chunk.layout
        decoded_frames = _decode_chunk_video(
            chunk,
            video_decoder,
            tiling_config,
            generator,
            dtype=dtype,
            keyframes=keyframes,
        )
        if decoded_frames is None:
            continue

        frames = _drop_video_overlap(decoded_frames, layout.prev_video_carry_frames)
        audio, seam_fade_samples = _vocoded_chunk_audio(
            chunk,
            frames,
            audio_decoder,
            fps=fps,
            pending_tail=pending_tail,
        )
        pending_tail, audio = _apply_pending_audio_crossfade(pending_tail, audio, seam_fade_samples)

        if pending_tail is not None:
            yield _blend_pending_video_tail(pending_tail, decoded_frames, layout.prev_video_carry_frames)
            pending_tail = None

        body, pending_tail = _body_and_pending_tail(
            PlaybackSegment(video=frames, audio=audio),
            layout=layout,
            fps=fps,
        )
        yield body
    if pending_tail is not None:
        yield pending_tail


def _decode_chunk_video(
    chunk: Chunk,
    video_decoder: VideoDecoder,
    tiling_config: object | None,
    generator: torch.Generator | None,
    *,
    dtype: torch.dtype | None,
    keyframes: bool,
) -> torch.Tensor | None:
    decode_kwargs: dict[str, object] = {
        "dtype": dtype,
        "device_fn": lambda _batch, device=chunk.video.device: device,
    }
    if keyframes:
        decode_kf = _decode_generated_keyframes(chunk)
        if decode_kf is not None:
            decode_kwargs["keyframes"] = decode_kf
    decoded_batches = list(video_decoder(chunk.video, tiling_config, generator, **decode_kwargs))
    if not decoded_batches:
        return None
    if len(decoded_batches) == 1:
        return decoded_batches[0]
    return torch.cat(decoded_batches, dim=0)


def _vocoded_chunk_audio(
    chunk: Chunk,
    frames: torch.Tensor,
    audio_decoder: AudioDecoder | None,
    *,
    fps: float,
    pending_tail: PlaybackSegment | None,
) -> tuple[Audio | None, int]:
    """Vocode this chunk's audio aligned to ``frames``, plus extra prefix samples for seam crossfade."""
    if audio_decoder is None or chunk.audio is None:
        return None, 0
    decoded_audio = audio_decoder(chunk.audio)
    fade_samples = _audio_seam_crossfade_samples(decoded_audio.sampling_rate)
    pending_audio_samples = (
        0 if pending_tail is None or pending_tail.audio is None else pending_tail.audio.waveform.shape[-1]
    )
    extra_samples = min(fade_samples, pending_audio_samples)
    base_samples = min(
        decoded_audio.waveform.shape[-1],
        round(frames.shape[0] / fps * decoded_audio.sampling_rate),
    )
    waveform, kept_samples = _keep_vocoded_audio_tail(
        decoded_audio.waveform,
        kept_video_frames=frames.shape[0],
        fps=fps,
        sampling_rate=decoded_audio.sampling_rate,
        extra_samples=extra_samples,
    )
    audio = Audio(waveform=waveform, sampling_rate=decoded_audio.sampling_rate)
    seam_fade_samples = max(0, kept_samples - base_samples)
    return audio, seam_fade_samples


def _apply_pending_audio_crossfade(
    pending_tail: PlaybackSegment | None,
    audio: Audio | None,
    seam_fade_samples: int,
) -> tuple[PlaybackSegment | None, Audio | None]:
    if pending_tail is None or pending_tail.audio is None or audio is None:
        return pending_tail, audio
    pending_audio, audio = _crossfade_audio_seam(pending_tail.audio, audio, seam_fade_samples)
    return replace(pending_tail, audio=pending_audio), audio


def _blend_pending_video_tail(
    pending_tail: PlaybackSegment,
    decoded_frames: torch.Tensor,
    prev_video_carry_frames: int,
) -> PlaybackSegment:
    overlap_end = prev_video_carry_frames
    blend_start = overlap_end - pending_tail.video.shape[0]
    if blend_start < 0:
        raise ValueError(
            f"pending blend tail ({pending_tail.video.shape[0]} frames) exceeds current overlap ({overlap_end} frames)"
        )
    overlap = decoded_frames[blend_start:overlap_end]
    return replace(
        pending_tail,
        video=_crossfade_video_overlap(pending_tail.video, overlap),
    )


def _body_and_pending_tail(
    decoded: PlaybackSegment,
    *,
    layout: ChunkLayout,
    fps: float,
) -> tuple[PlaybackSegment, PlaybackSegment | None]:
    blend_frames = layout.next_video_blend_frames
    if blend_frames == 0:
        return decoded, None
    prefix_audio_samples = 0
    if decoded.audio is not None:
        tail_audio_samples = round(blend_frames / fps * decoded.audio.sampling_rate)
        prefix_audio_samples = max(0, decoded.audio.waveform.shape[-1] - tail_audio_samples)
    return _split_decoded_tail(
        decoded,
        tail_frames=blend_frames,
        prefix_audio_samples=prefix_audio_samples,
    )


def replace_chunks_audio(
    chunks: Iterator[PlaybackSegment],
    source: Audio,
    fps: float,
) -> Iterator[PlaybackSegment]:
    """Replace each chunk's audio with the equivalent slice of ``source``.
    Each chunk takes the next unused samples so the stitched waveform stays contiguous.
    A source shorter than the video emits what remains, then ``None``. Slices stay on
    ``source.waveform``'s device. Run after :func:`decode_chunks` once overlap frames
    are already dropped.
    """
    stitched_pixel_frames = 0
    emitted_audio_samples = 0
    sampling_rate = source.sampling_rate
    waveform = source.waveform
    for chunk in chunks:
        stitched_pixel_frames += chunk.video.shape[0]
        end_audio_sample = round(stitched_pixel_frames / fps * sampling_rate)
        remaining = waveform.shape[-1] - emitted_audio_samples
        chunk_audio_samples = max(0, min(remaining, end_audio_sample - emitted_audio_samples))
        if chunk_audio_samples == 0:
            yield replace(chunk, audio=None)
            continue
        sliced = waveform[..., emitted_audio_samples : emitted_audio_samples + chunk_audio_samples]
        emitted_audio_samples += chunk_audio_samples
        yield replace(chunk, audio=Audio(waveform=sliced, sampling_rate=sampling_rate))


def _decode_generated_keyframes(chunk: Chunk) -> DecodeKeyframes | None:
    """Build ``DecodeKeyframes`` from incoming shared anchors and this chunk's slots."""
    sidecars = [sidecar for sidecar in (chunk.incoming_keyframes, chunk.generated_keyframes) if sidecar is not None]
    if not sidecars:
        return None
    latent = torch.cat([sidecar.latent for sidecar in sidecars], dim=2)
    positions = tuple(position for sidecar in sidecars for position in sidecar.pixel_frame_indices)
    kf = decode_keyframes_from_slots(
        latent,
        positions,
        chunk.layout.pixel_frames,
    )
    if kf is None:
        return None
    start_pixel_frame = chunk.layout.start_pixel_frame
    if start_pixel_frame == 0:
        return kf
    return replace(
        kf,
        pixel_frame_indices=kf.pixel_frame_indices + start_pixel_frame,
        clip_start_frame=start_pixel_frame,
    )


def _append_carry(
    conditionings: list[ConditioningItem],
    carry: torch.Tensor | None,
    carry_condition_cls: type[VideoConditionByLatentIndex] | type[AudioConditionByLatentIndex],
) -> list[ConditioningItem]:
    if carry is None:
        return conditionings
    return [*conditionings, carry_condition_cls(latent=carry, strength=1.0, latent_idx=0)]


def _append_incoming_keyframes(
    conditionings: list[ConditioningItem],
    incoming: GeneratedKeyframes | None,
) -> list[ConditioningItem]:
    """Append prior overlap planes as fixed one-frame keyframe conditioning."""
    if incoming is None:
        return conditionings
    incoming_items = [
        VideoConditionByKeyframeIndex(
            keyframes=incoming.latent[:, :, index : index + 1],
            frame_idx=position,
            strength=1.0,
            num_pixel_frames=1,
        )
        for index, position in enumerate(incoming.pixel_frame_indices)
    ]
    return [*conditionings, *incoming_items]


def _keyframes_for_next_chunk(
    sidecar: GeneratedKeyframes | None,
    layout: ChunkLayout,
) -> GeneratedKeyframes | None:
    """Keep only generated planes in the outgoing overlap and rebase them for the next chunk."""
    next_carry = layout.next_video_carry_frames
    if sidecar is None or next_carry <= 0:
        return None
    next_chunk_start = layout.pixel_frames - next_carry
    keep = [
        (index, position)
        for index, position in enumerate(sidecar.pixel_frame_indices)
        if next_chunk_start <= position < layout.pixel_frames
    ]
    if not keep:
        return None
    return GeneratedKeyframes(
        latent=torch.cat([sidecar.latent[:, :, index : index + 1] for index, _ in keep], dim=2),
        pixel_frame_indices=tuple(position - next_chunk_start for _, position in keep),
    )


def _slice_latent_tail(latent: torch.Tensor | None, frames: int) -> torch.Tensor | None:
    """Last ``frames`` along time (dim 2), cloned. ``None`` if missing or ``frames <= 0``."""
    if latent is None or frames <= 0:
        return None
    return latent[:, :, -frames:].clone()


def _drop_video_overlap(frames: torch.Tensor, prev_video_carry_frames: int) -> torch.Tensor:
    """Drop leading ``prev_video_carry_frames`` decoded video frames (already emitted by the prior chunk)."""
    if prev_video_carry_frames <= 0:
        return frames
    if prev_video_carry_frames >= frames.shape[0]:
        raise ValueError(
            f"prev_video_carry_frames ({prev_video_carry_frames}) must be < decoded frame count ({frames.shape[0]})"
        )
    return frames[prev_video_carry_frames:]


def _split_decoded_tail(
    chunk: PlaybackSegment,
    *,
    tail_frames: int,
    prefix_audio_samples: int,
) -> tuple[PlaybackSegment, PlaybackSegment]:
    """Split a decoded chunk, cloning only the retained tail so it releases the full frame storage."""
    body_audio = None
    tail_audio = None
    if chunk.audio is not None:
        body_audio = replace(chunk.audio, waveform=chunk.audio.waveform[..., :prefix_audio_samples])
        tail_audio = replace(chunk.audio, waveform=chunk.audio.waveform[..., prefix_audio_samples:].clone())
    return (
        PlaybackSegment(video=chunk.video[:-tail_frames], audio=body_audio),
        PlaybackSegment(video=chunk.video[-tail_frames:].clone(), audio=tail_audio),
    )


def _crossfade_video_overlap(previous: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
    """Replace the matching tail of ``previous`` with a linear crossfade into ``overlap``."""
    count = min(previous.shape[0], overlap.shape[0])
    if count == 0:
        return previous
    weights = torch.arange(1, count + 1, device=previous.device, dtype=previous.dtype) / (count + 1)
    weights = weights.reshape(count, *([1] * (previous.ndim - 1)))
    blended = torch.lerp(previous[-count:], overlap[-count:], weights)
    return torch.cat([previous[:-count], blended], dim=0)


def _crossfade_audio_seam(left: Audio, right: Audio, fade: int) -> tuple[Audio, Audio]:
    """Equal-power crossfade ``right``'s leading overlap into ``left``'s tail."""
    if left.sampling_rate != right.sampling_rate:
        raise ValueError(
            f"chunk audio sampling rate {right.sampling_rate} does not match expected {left.sampling_rate}"
        )
    if left.waveform.shape[:-1] != right.waveform.shape[:-1]:
        raise ValueError(
            f"chunk audio shape {tuple(right.waveform.shape)} does not match pending tail "
            f"shape {tuple(left.waveform.shape)}"
        )
    fade = min(fade, left.waveform.shape[-1], right.waveform.shape[-1])
    if fade <= 0:
        return left, right
    t = torch.linspace(0, 1, fade, device=left.waveform.device, dtype=left.waveform.dtype)
    weight_shape = (*([1] * (left.waveform.ndim - 1)), fade)
    weight_out = torch.cos(t * (torch.pi / 2)).reshape(weight_shape)
    weight_in = torch.sin(t * (torch.pi / 2)).reshape(weight_shape)
    mixed = left.waveform[..., -fade:] * weight_out + right.waveform[..., :fade] * weight_in
    left_waveform = torch.cat([left.waveform[..., :-fade], mixed], dim=-1)
    right_waveform = right.waveform[..., fade:]
    return replace(left, waveform=left_waveform), replace(right, waveform=right_waveform)


def _audio_seam_crossfade_samples(sampling_rate: int) -> int:
    """Sample count for :data:`AUDIO_SEAM_CROSSFADE_MS`."""
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be > 0, got {sampling_rate}")
    return max(1, round(AUDIO_SEAM_CROSSFADE_MS / 1000.0 * sampling_rate))


def _keep_vocoded_audio_tail(
    waveform: torch.Tensor,
    *,
    kept_video_frames: int,
    fps: float,
    sampling_rate: int,
    extra_samples: int = 0,
) -> tuple[torch.Tensor, int]:
    """Keep this chunk's video-aligned audio tail plus optional seam overlap."""
    keep = round(kept_video_frames / fps * sampling_rate) + extra_samples
    keep = max(0, min(waveform.shape[-1], keep))
    if keep == 0:
        return waveform[..., :0], 0
    return waveform[..., -keep:], keep


def _keep_audio_tail_to_match_video(
    waveform: torch.Tensor,
    *,
    fps: float,
    stitched_pixel_frames: int,
    emitted_audio_samples: int,
    sampling_rate: int,
) -> tuple[torch.Tensor, int]:
    """Keep the waveform tail so cumulative samples match cumulative kept video.
    Causal VAE decoding and chunk overlap live at the front of each decode; the tail
    is what aligns stitched audio duration with stitched video.
    """
    end_audio_sample = round(stitched_pixel_frames / fps * sampling_rate)
    chunk_audio_samples = max(0, min(waveform.shape[-1], end_audio_sample - emitted_audio_samples))
    if chunk_audio_samples == 0:
        return waveform[..., :0], 0
    return waveform[..., -chunk_audio_samples:], chunk_audio_samples
