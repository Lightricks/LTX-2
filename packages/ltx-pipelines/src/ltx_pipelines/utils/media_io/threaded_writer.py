"""Threaded wrapper around :class:`ContainerWriter` with fail-fast command delivery."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue

from ltx_core.types import Audio
from ltx_pipelines.utils.media_io.audio_stream import AudioStreamSpec
from ltx_pipelines.utils.media_io.types import VideoPayload
from ltx_pipelines.utils.media_io.video_stream import VideoStreamSpec
from ltx_pipelines.utils.media_io.writer import ContainerWriter

_QUEUE_WAIT_SECONDS = 0.05

ContainerWriterProvider = Callable[[], ContainerWriter]


@dataclass(frozen=True, slots=True)
class _WriteVideo:
    """Write one converted video payload."""

    payload: VideoPayload


@dataclass(frozen=True, slots=True)
class _WriteAudio:
    """Write one contiguous audio segment."""

    audio: Audio


@dataclass(frozen=True, slots=True)
class _Finish:
    """Drain all streams, write the trailer, and close the container."""


@dataclass(frozen=True, slots=True)
class _Abort:
    """Stop the writer and close its temporary container."""


_Command = _WriteVideo | _WriteAudio | _Finish | _Abort


class ThreadedContainerWriter:
    """Run a ContainerWriter on its own thread behind a bounded command queue.
    Producers may convert frames concurrently with encoding. Any worker failure
    is published through a ``Future`` and checked while bounded queue puts wait,
    so a failed encoder does not force a lazy source to be fully consumed.
    """

    def __init__(self, writer_provider: ContainerWriterProvider) -> None:
        self._writer_provider = writer_provider
        self._commands: Queue[_Command] = Queue(maxsize=1)
        self._result: Future[None] = Future()
        self._thread = threading.Thread(target=self._run, name="media-container-writer")
        self._thread.start()

    @classmethod
    def open(
        cls,
        output_path: Path,
        *,
        video_stream_spec: VideoStreamSpec,
        audio_stream_spec: AudioStreamSpec | None,
    ) -> ThreadedContainerWriter:
        """Start a worker thread that builds a :class:`ContainerWriter` for ``output_path``."""

        def factory() -> ContainerWriter:
            return ContainerWriter(
                output_path,
                video_stream_spec=video_stream_spec,
                audio_stream_spec=audio_stream_spec,
            )

        return cls(factory)

    def write_video(self, payload: VideoPayload) -> None:
        """Queue one video payload or raise the worker's first error."""
        self._enqueue(_WriteVideo(payload))

    def write_audio(self, audio: Audio) -> None:
        """Queue one audio segment or raise the worker's first error."""
        self._enqueue(_WriteAudio(audio))

    def finish(self) -> None:
        """Finish the container and propagate codec, mux, trailer, or close errors."""
        self._enqueue(_Finish())
        self._thread.join()
        self._result.result()

    def abort(self) -> None:
        """Stop the worker after a producer-side failure."""
        while self._thread.is_alive() and not self._result.done():
            try:
                self._commands.put(_Abort(), timeout=_QUEUE_WAIT_SECONDS)
                break
            except Full:
                continue
        self._thread.join()

    def _enqueue(self, command: _Command) -> None:
        while True:
            self._raise_if_unavailable()
            try:
                self._commands.put(command, timeout=_QUEUE_WAIT_SECONDS)
                return
            except Full:
                continue

    def _raise_if_unavailable(self) -> None:
        if not self._result.done():
            return
        self._result.result()
        raise RuntimeError("container writer is already closed")

    def _run(self) -> None:
        writer: ContainerWriter | None = None
        try:
            writer = self._writer_provider()
            while True:
                command = self._commands.get()
                match command:
                    case _WriteVideo(payload=payload):
                        writer.write_video(payload)
                    case _WriteAudio(audio=audio):
                        writer.write_audio(audio)
                    case _Abort():
                        writer.abort()
                        self._result.set_result(None)
                        return
                    case _Finish():
                        writer.finish()
                        self._result.set_result(None)
                        return
                    case _:
                        raise TypeError(f"unsupported writer command: {type(command).__name__}")
        except BaseException as error:
            if writer is not None:
                writer.abort()
            self._result.set_exception(error)
