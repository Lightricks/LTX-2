from __future__ import annotations

import logging
from pathlib import Path

import torch

from ltx_core.duration_head import (
    DURATION_HEAD_KEY_OPS,
    DurationHead,
    DurationHeadConfigurator,
)
from ltx_core.loader import SDOps
from ltx_core.loader.mapping import build_mapping
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_pipelines.utils.helpers import seconds_to_clamped_num_frames
from ltx_pipelines.utils.types import AutoDuration

logger = logging.getLogger(__name__)

__all__ = ["DurationPredictor", "require_num_frames_source", "resolve_num_frames"]


class DurationPredictor:
    """Predicts shot duration (in frames) from ``PromptEncoder`` output.
    Unlike most blocks, the model is held directly rather than rebuilt on every call:
    DurationHead is a few MB, so there's no memory pressure motivating the
    build-on-call / free-on-exit pattern used for the large transformer/VAE blocks.
    Construct via :meth:`from_checkpoint`.
    """

    def __init__(self, head: DurationHead) -> None:
        """Construct from an already-built, already-loaded head."""
        self._head = head

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | None,
        dtype: torch.dtype,
        device: torch.device,
        model_configurator: type[ModelConfigurator[DurationHead]] = DurationHeadConfigurator,
        model_sd_ops: SDOps = DURATION_HEAD_KEY_OPS,
    ) -> "DurationPredictor | None":
        """Build a predictor from a checkpoint path, or ``None`` if unavailable.
        Returns ``None`` when ``checkpoint_path`` is ``None`` (split pack without a
        duration-head file), or when the file has no DurationHead weights (pre-2.5 /
        gemma3 monoliths). Checkpoint-key selection detects an entirely missing
        head before building it. A partially present head fails the loader's
        model-key check.
        The head uses a local builder registry; it is small enough that sharing
        its weights or shell across pipeline instances is unnecessary.
        """
        if checkpoint_path is None:
            return None
        path = Path(checkpoint_path)
        mapping = build_mapping((path,), (model_sd_ops,))
        if not mapping.tensor_mapping.destination_metadata:
            logger.info(
                "No DurationHead weights found in %s; auto-duration prediction unavailable.",
                checkpoint_path,
            )
            return None
        head = SimpleBuilder(model_configurator, (path,), (model_sd_ops,)).build(device=device, dtype=dtype).eval()
        return cls(head)

    def __call__(
        self,
        video_encoding: torch.Tensor | None,
        audio_encoding: torch.Tensor | None,
        *,
        frame_rate: float,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
    ) -> int:
        """Predict a frame count from caption connector tokens, snapped to the VAE's grid.
        ``min_seconds``/``max_seconds`` clamp the prediction so a misbehaving prediction can't
        request a degenerate or OOM-sized generation; the defaults are 1s and 20s. The result is
        a frame count snapped to the VAE's ``8k + 1`` causal temporal grid. Callers doing offline
        analysis of raw predictions (rather than feeding a real generation) can pass a much wider
        range to see the unclamped duration.
        """
        if video_encoding is None and audio_encoding is None:
            raise ValueError("DurationPredictor requires at least one of video_encoding / audio_encoding")
        seconds_pred = self._head(video_encoding, audio_encoding)
        if seconds_pred.shape != (1,):
            raise ValueError(
                f"DurationPredictor only supports a single-item batch, got prediction shape {tuple(seconds_pred.shape)}"
            )
        seconds = seconds_pred.item()
        min_frames = round(min_seconds * frame_rate)
        max_frames = round(max_seconds * frame_rate)
        num_frames = seconds_to_clamped_num_frames(
            seconds, frame_rate=frame_rate, min_frames=min_frames, max_frames=max_frames
        )
        if seconds > max_seconds or seconds < min_seconds:
            logger.warning(
                "DurationHead prediction clamped: raw %.2fs outside [%.2fs, %.2fs], using %.2fs (%d frames) @ %.2f fps",
                seconds,
                min_seconds,
                max_seconds,
                num_frames / frame_rate,
                num_frames,
                frame_rate,
            )
        else:
            logger.info("DurationHead predicted %.2fs (%d frames @ %.2f fps)", seconds, num_frames, frame_rate)
        return num_frames


def require_num_frames_source(num_frames: int | AutoDuration, duration_predictor: DurationPredictor | None) -> None:
    """Guard against an unsatisfiable auto-duration request.
    Call at the very top of a pipeline's ``__call__`` -- before prompt encoding or any other
    work -- so a checkpoint without DurationHead weights (anything predating 2.5) fails fast
    with a clear message instead of after paying for work whose result would be discarded.
    """
    if isinstance(num_frames, AutoDuration) and duration_predictor is None:
        raise ValueError(
            "num_frames was AutoDuration but this checkpoint has no DurationHead weights to "
            "auto-predict duration from (DurationHead ships from LTX 2.5 / gemma4 onward). "
            "Pass num_frames explicitly."
        )


def resolve_num_frames(
    num_frames: int | AutoDuration,
    duration_predictor: DurationPredictor | None,
    *,
    video_encoding: torch.Tensor | None,
    audio_encoding: torch.Tensor | None,
    frame_rate: float,
) -> int:
    """Resolve ``num_frames`` to a concrete frame count, predicting it if ``AutoDuration``.
    Call after prompt encoding (once ``video_encoding``/``audio_encoding`` exist) and after
    ``require_num_frames_source`` has already validated a predictor is available when needed.
    """
    if not isinstance(num_frames, AutoDuration):
        return num_frames
    return duration_predictor(
        video_encoding,
        audio_encoding,
        frame_rate=frame_rate,
        min_seconds=num_frames.min_seconds,
        max_seconds=num_frames.max_seconds,
    )
