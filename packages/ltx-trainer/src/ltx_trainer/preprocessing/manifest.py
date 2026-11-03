"""Kernel-owned preprocessing manifest persistence and compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ltx_trainer import RANK, logger
from ltx_trainer.preprocessing.types import JsonValue

PREPROCESS_MANIFEST_FILE = "preprocess_manifest.json"
PREPROCESS_MANIFEST_FORMAT_VERSION = 1


def write_preprocess_manifest(root: Path, fingerprint: dict[str, JsonValue]) -> None:
    """Atomically persist a preprocessing fingerprint under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / PREPROCESS_MANIFEST_FILE
    temporary_path = root / f".{PREPROCESS_MANIFEST_FILE}.tmp"
    payload = {
        "format_version": PREPROCESS_MANIFEST_FORMAT_VERSION,
        "fingerprint": fingerprint,
    }
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(manifest_path)


def read_preprocess_manifest(root: Path) -> dict[str, JsonValue] | None:
    """Read and validate a preprocessing manifest, or return ``None`` when absent."""
    manifest_path = root / PREPROCESS_MANIFEST_FILE
    if not manifest_path.exists():
        return None

    try:
        payload: Any = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"Failed to read preprocessing manifest {manifest_path}: {exc}. Re-run preprocessing with --overwrite."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid preprocessing manifest {manifest_path}: expected a JSON object.")
    version = payload.get("format_version")
    if version != PREPROCESS_MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported preprocessing manifest format_version {version!r} in {manifest_path}; "
            f"expected {PREPROCESS_MANIFEST_FORMAT_VERSION}. Re-run preprocessing with --overwrite."
        )
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, dict) or any(not isinstance(key, str) for key in fingerprint):
        raise ValueError(f"Invalid preprocessing manifest {manifest_path}: 'fingerprint' must be a JSON object.")
    return fingerprint


def assert_preprocess_manifest_matches(
    root: Path,
    expected: dict[str, JsonValue],
    *,
    allow_stale: bool = False,
    allow_missing: bool = True,
) -> None:
    """Require the saved fingerprint to match, grandfathering manifest-less trees."""
    saved = read_preprocess_manifest(root)
    if saved is None:
        if not allow_missing:
            raise ValueError(
                f"Cannot verify preprocessed data under {root}: {PREPROCESS_MANIFEST_FILE} is missing. "
                "Re-run preprocessing into a fresh output root or with --overwrite."
            )
        _warn(f"No {PREPROCESS_MANIFEST_FILE} found under {root}. Continuing without verification.")
        return

    differences = _fingerprint_differences(saved, expected)
    if not differences:
        return

    message = _mismatch_message(root, differences)
    if allow_stale:
        _warn(f"⚠️ {message}\nContinuing because allow_stale_preprocessed=true.")
        return
    raise ValueError(
        f"{message}\nRe-run preprocessing with --overwrite, or set "
        "allow_stale_preprocessed=true to intentionally bypass this check."
    )


def assert_preprocess_manifest_can_be_updated(
    root: Path,
    expected: dict[str, JsonValue],
    *,
    overwrite: bool,
) -> None:
    """Prevent an incremental run from blessing artifacts made by another config."""
    if overwrite or not root.exists() or not any(root.iterdir()):
        return

    saved = read_preprocess_manifest(root)
    if saved is None:
        raise ValueError(
            f"Cannot incrementally preprocess the populated directory {root} because it has no "
            f"{PREPROCESS_MANIFEST_FILE}. Re-run with --overwrite."
        )

    differences = _fingerprint_differences(saved, expected)
    if differences:
        raise ValueError(f"{_mismatch_message(root, differences)}\nRe-run preprocessing with --overwrite.")


def _fingerprint_differences(
    saved: dict[str, JsonValue],
    expected: dict[str, JsonValue],
    *,
    prefix: str = "",
) -> list[str]:
    differences: list[str] = []
    missing = object()
    for key in sorted(saved.keys() | expected.keys()):
        path = f"{prefix}.{key}" if prefix else key
        saved_value = saved.get(key, missing)
        expected_value = expected.get(key, missing)
        if isinstance(saved_value, dict) and isinstance(expected_value, dict):
            differences.extend(_fingerprint_differences(saved_value, expected_value, prefix=path))
        elif saved_value != expected_value:
            differences.append(
                f"  {path}: preprocessed={_display_value(saved_value, missing)}, "
                f"current={_display_value(expected_value, missing)}"
            )
    return differences


def _display_value(value: object, missing: object) -> str:
    return "<missing>" if value is missing else json.dumps(value, sort_keys=True)


def _mismatch_message(root: Path, differences: list[str]) -> str:
    return f"Preprocessed data under {root} does not match the current configuration:\n" + "\n".join(differences)


def _warn(message: str) -> None:
    if RANK == 0:
        logger.warning(message)
