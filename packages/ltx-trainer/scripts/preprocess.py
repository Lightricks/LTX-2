#!/usr/bin/env python

"""
Preprocess a training dataset using the strategy-declared pipeline from a YAML config.
Basic usage:
    uv run python scripts/preprocess.py configs/t2v_lora.yaml [--overwrite] [--device cuda]
    uv run python scripts/preprocess.py configs/t2v_lora.yaml --check
For multi-GPU preprocessing, invoke under ``accelerate launch`` — each process handles an
interleaved shard of the dataset. Sharding is implemented inside the encoder modules via
``accelerate.PartialState`` (caption/video/audio encoding); mask steps run on every rank.
Strategies may opt into the kernel-owned manifest protocol. The CLI checks compatibility
before encoding and writes the manifest from the main process after all ranks finish.
"""

from pathlib import Path

import typer
import yaml
from accelerate import PartialState
from rich.console import Console

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.preprocessing.manifest import (
    assert_preprocess_manifest_can_be_updated,
    assert_preprocess_manifest_matches,
    write_preprocess_manifest,
)
from ltx_trainer.preprocessing.types import PreprocessingFingerprint
from ltx_trainer.training_strategies import TrainingStrategy, get_training_strategy

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Preprocess a training dataset using strategy-owned preprocessing steps from a YAML config.",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to YAML training configuration file"),
    check_compatibility: bool = typer.Option(
        False,
        "--check",
        help="Check whether existing preprocessed artifacts match this config, without encoding",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-compute every item even if its output already exists",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Device to use for preprocessing",
    ),
) -> None:
    """Run preprocessing for the dataset described in the training config."""
    if check_compatibility and overwrite:
        raise typer.BadParameter("❌ --check cannot be combined with --overwrite")

    config_mode = "train" if check_compatibility else "preprocess"
    trainer_config = _load_config(Path(config_path), mode=config_mode)
    strategy = get_training_strategy(trainer_config.training_strategy)
    fingerprint = strategy.preprocessing_fingerprint()

    if check_compatibility:
        _check_compatibility(fingerprint, strategy_name=strategy.config.name)
        return

    _run_preprocessing(strategy, fingerprint, device=device, overwrite=overwrite)


def _load_config(config_path: Path, *, mode: str) -> LtxTrainerConfig:
    if not config_path.exists():
        raise typer.BadParameter(f"❌ Configuration file does not exist: {config_path}")

    try:
        config_data = yaml.safe_load(config_path.read_text())
        return LtxTrainerConfig.from_dict(config_data, mode=mode)
    except Exception as exc:
        raise typer.BadParameter(f"❌ Invalid configuration data: {exc}") from exc


def _check_compatibility(fingerprint: PreprocessingFingerprint | None, *, strategy_name: str) -> None:
    if fingerprint is None:
        raise typer.BadParameter(
            f"❌ Strategy {strategy_name!r} does not provide a preprocessing fingerprint; "
            "compatibility cannot be checked."
        )

    try:
        assert_preprocess_manifest_matches(
            fingerprint.artifact_root,
            fingerprint.fields,
            allow_missing=False,
        )
    except ValueError as exc:
        raise typer.BadParameter(f"❌ {exc}") from exc
    console.print(f"✅ Preprocessed data is compatible with this config: {fingerprint.artifact_root}")


def _run_preprocessing(
    strategy: TrainingStrategy,
    fingerprint: PreprocessingFingerprint | None,
    *,
    device: str,
    overwrite: bool,
) -> None:
    try:
        pipeline = strategy.build_preprocessing_pipeline()
    except NotImplementedError as exc:
        raise typer.BadParameter(f"❌ {exc}") from exc

    if fingerprint is not None:
        output_root = pipeline.output_root.expanduser().resolve()
        if fingerprint.artifact_root != output_root:
            raise RuntimeError(
                "Strategy preprocessing fingerprint and pipeline disagree on the artifact root: "
                f"{fingerprint.artifact_root} != {output_root}"
            )
        assert_preprocess_manifest_can_be_updated(
            fingerprint.artifact_root,
            fingerprint.fields,
            overwrite=overwrite,
        )

    pipeline.run(device=device, overwrite=overwrite)
    state = PartialState()
    state.wait_for_everyone()
    if fingerprint is not None and state.is_main_process:
        write_preprocess_manifest(fingerprint.artifact_root, fingerprint.fields)
    state.wait_for_everyone()


if __name__ == "__main__":
    app()
