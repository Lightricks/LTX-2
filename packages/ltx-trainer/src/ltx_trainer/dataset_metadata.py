"""Read and write dataset metadata files (CSV / JSON / JSONL) as lists of row dicts.
One place that knows the on-disk metadata formats, shared by preprocessing and the dataset-authoring
scripts (captioning, reference generation) so format branching lives here and nowhere else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_rows(dataset_file: str | Path) -> list[dict[str, Any]]:
    """Read a CSV/JSON/JSONL metadata file into a list of per-sample row dicts."""
    path = Path(dataset_file)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _normalize_csv_rows(pd.read_csv(path).to_dict(orient="records"))
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        return _validate_rows(data, path)
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return _validate_rows(rows, path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def write_rows(dataset_file: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write row dicts back to a CSV/JSON/JSONL metadata file (format from the suffix)."""
    path = Path(dataset_file)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        pd.DataFrame(rows).to_csv(path, index=False)
        return
    if suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return
    if suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def column_names(dataset_file: str | Path) -> set[str]:
    """Read the set of column names from a dataset file."""
    path = Path(dataset_file)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return set(pd.read_csv(path, nrows=0).columns)
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        return set().union(*(row.keys() for row in _validate_rows(data, path)))
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return set().union(*(row.keys() for row in _validate_rows(rows, path)))
    return set()


def _validate_rows(rows: list[Any], path: Path) -> list[dict[str, Any]]:
    """Ensure every JSON or JSONL row is an object."""
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Dataset rows in {path} must be objects")
    return rows


def _normalize_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert pandas missing values to the JSON-compatible None representation."""
    return [
        {key: None if pd.api.types.is_scalar(value) and pd.isna(value) else value for key, value in row.items()}
        for row in rows
    ]
