"""Incremental asynchronous EXR sequence writing."""

from __future__ import annotations

import shutil
import tempfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

from ltx_pipelines.utils.media_io.color_config import EXRColorSpace, decode_hdr_video
from ltx_pipelines.utils.media_io.exr import exr_colorspace_tag, save_exr_tensor


class ExrSequenceWriter:
    """Write ordered HDR frame batches into one EXR sidecar directory."""

    def __init__(
        self,
        destination: Path,
        *,
        max_workers: int = 8,
    ) -> None:
        self._destination = destination
        self._staging = _make_staging_directory(destination)
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: deque[Future[None]] = deque()
        self._frame_index = 0
        self._finished = False

    def write(self, video: torch.Tensor, color_space: EXRColorSpace) -> None:
        """Convert one decoded ACEScct ``[F,H,W,3]`` batch to ``color_space`` and queue its frames."""
        if self._finished:
            raise RuntimeError("EXR writer is already finished")
        primaries = color_space.source_primaries
        colorspace_tag = exr_colorspace_tag(color_space)
        aces_container = color_space.is_aces_container
        for frame in _to_exr_space(video, color_space):
            frame_path = self._staging / f"frame_{self._frame_index:05d}.exr"
            self._pending.append(
                self._executor.submit(
                    save_exr_tensor,
                    frame.cpu().clone(),
                    str(frame_path),
                    True,
                    primaries,
                    colorspace_tag,
                    aces_container=aces_container,
                )
            )
            self._frame_index += 1

        while len(self._pending) > 2 * self._max_workers:
            self._pending.popleft().result()

    def finish(self) -> None:
        """Finish all writes and atomically publish the completed EXR sequence."""
        while self._pending:
            self._pending.popleft().result()
        self._executor.shutdown(wait=True)
        backup = self._staging.with_name(f"{self._staging.name}.previous")
        had_destination = self._destination.exists()
        if had_destination:
            self._destination.replace(backup)
        try:
            self._staging.replace(self._destination)
        except BaseException:
            if had_destination:
                backup.replace(self._destination)
            raise
        if had_destination:
            shutil.rmtree(backup)
        self._finished = True

    def abort(self) -> None:
        """Release worker threads and remove the unpublished staging directory."""
        self._executor.shutdown(wait=True)
        shutil.rmtree(self._staging, ignore_errors=True)


def _to_exr_space(video: torch.Tensor, color_space: EXRColorSpace) -> torch.Tensor:
    if color_space.is_log_working:
        return video.float()
    return decode_hdr_video(video, out_primaries=color_space.source_primaries)


def _make_staging_directory(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".partial",
            dir=destination.parent,
        )
    )
