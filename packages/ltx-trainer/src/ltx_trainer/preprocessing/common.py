"""Shared helpers used by every preprocessing step.
Path layout, atomic writes, Accelerate sharding, and progress UI — not media-domain
logic (that lives in :mod:`~ltx_trainer.preprocessing.media`).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from accelerate import PartialState
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Subset

from ltx_trainer import logger
from ltx_trainer.dataset_metadata import read_rows


def output_relative(path: Path, data_root: Path) -> Path:
    """Relative path used to name a sample's cached output, mirroring the input layout.
    Normally media lives under the dataset directory and this is just the path relative to it.
    If a media path is absolute or otherwise outside the dataset directory (e.g. a one-off
    metadata file that references media elsewhere), mirror its absolute structure under the
    output directory instead of raising, so out-of-tree media stays collision-free.
    """
    try:
        return path.relative_to(data_root)
    except ValueError:
        return Path(*path.parts[1:]) if path.is_absolute() else path


def load_paths_from_dataset(dataset_file: Path, column: str) -> list[Path]:
    """Load file paths from a dataset column, resolving relative to the dataset file's directory."""
    data_root = dataset_file.parent
    paths: list[Path] = []
    for row in read_rows(dataset_file):
        if column not in row:
            raise ValueError(f"Column '{column}' not found in {dataset_file.name}")
        paths.append(data_root / Path(str(row[column]).strip()))
    return paths


def load_source_and_naming_paths(
    dataset_file: Path,
    source_column: str,
    naming_column: str | None,
) -> tuple[list[Path], list[Path]]:
    """Return ``(source_paths, naming_paths)`` for a preprocessing step.
    ``source_paths`` are the files to process; ``naming_paths`` decide the output ``.pt`` layout
    and default to the source paths unless ``naming_column`` selects a different key column
    (used so reference/audio latents land under their target sample's path).
    """
    source_paths = load_paths_from_dataset(dataset_file, source_column)
    resolved_naming_column = naming_column or source_column
    if resolved_naming_column == source_column:
        return source_paths, source_paths
    return source_paths, load_paths_from_dataset(dataset_file, resolved_naming_column)


def atomic_save(data: Any, out: Path) -> None:  # noqa: ANN401
    """Save to ``out`` atomically via per-PID temp file + replace.
    Crash mid-write leaves an orphan ``.tmp.<pid>`` file that the skip logic
    ignores. The per-PID suffix makes concurrent writes from multiple ranks
    collision-free.
    """
    tmp = out.with_suffix(f"{out.suffix}.tmp.{os.getpid()}")
    torch.save(data, tmp)
    tmp.replace(out)


def build_sharded_dataloader(
    dataset: Sequence[Any],
    *,
    batch_size: int,
    num_workers: int,
    is_done: Callable[[int], bool],
    overwrite: bool,
) -> DataLoader | None:
    """Return a DataLoader over this rank's interleaved shard of ``dataset``.
    When ``overwrite`` is False, items whose outputs already exist (per
    ``is_done``) are filtered out. Returns ``None`` if this rank has nothing
    to do, so the caller can early-return without loading any models.
    """
    state = PartialState()
    todo = [i for i in range(state.process_index, len(dataset), state.num_processes) if overwrite or not is_done(i)]
    if not todo:
        logger.info(f"Rank {state.process_index}/{state.num_processes}: nothing to do")
        return None
    logger.info(f"Rank {state.process_index}/{state.num_processes}: processing {len(todo):,} of {len(dataset):,} items")
    return DataLoader(Subset(dataset, todo), batch_size=batch_size, shuffle=False, num_workers=num_workers)


@contextmanager
def preprocessing_progress(description: str, total: int) -> Iterator[Callable[[], None]]:
    """Standard preprocessing progress bar; yields an ``advance()`` callback to call per item."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(description, total=total)
        yield lambda: progress.advance(task)
