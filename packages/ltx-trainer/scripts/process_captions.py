#!/usr/bin/env python

"""
Compute text embeddings for video generation training.
This module provides functionality for processing text captions, including:
- Loading captions from various file formats (CSV, JSON, JSONL)
- Cleaning and preprocessing text (removing LLM prefixes, adding ID tokens)
- CaptionsDataset for caption-only preprocessing workflows
Can be used as a standalone script:
    python scripts/process_captions.py dataset.json --output-dir /path/to/output \
        --model-source /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma
"""

import json
import os
import gc
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import typer
from rich.console import Console
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
from torch.utils.data import DataLoader, Dataset
from transformers.utils.logging import disable_progress_bar

from ltx_trainer import logger
from ltx_trainer.model_loader import load_text_encoder

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

disable_progress_bar()

# Common phrases that LLMs often add to captions that we might want to remove
COMMON_BEGINNING_PHRASES: tuple[str, ...] = (
    "This video",
    "The video",
    "This clip",
    "The clip",
    "The animation",
    "This image",
    "The image",
    "This picture",
    "The picture",
)

COMMON_CONTINUATION_WORDS: tuple[str, ...] = (
    "shows",
    "depicts",
    "features",
    "captures",
    "highlights",
    "introduces",
    "presents",
)

COMMON_LLM_START_PHRASES: tuple[str, ...] = (
    "In the video,",
    "In this video,",
    "In this video clip,",
    "In the clip,",
    "Caption:",
    *(
        f"{beginning} {continuation}"
        for beginning in COMMON_BEGINNING_PHRASES
        for continuation in COMMON_CONTINUATION_WORDS
    ),
)

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process text captions and save embeddings for video generation training.",
)


class CaptionsDataset(Dataset):
    """
    Dataset for processing text captions only.
    This dataset is designed for caption preprocessing workflows where you only need
    to process text without loading videos. Useful for:
    - Precomputing text embeddings
    - Caption cleaning and preprocessing
    - Text-only preprocessing pipelines
    """

    def __init__(
        self,
        dataset_file: str | Path,
        caption_column: str,
        media_column: str = "media_path",
        lora_trigger: str | None = None,
        remove_llm_prefixes: bool = False,
    ) -> None:
        """
        Initialize the captions dataset.
        Args:
            dataset_file: Path to CSV/JSON/JSONL metadata file
            caption_column: Column name for captions in the metadata file
            media_column: Column name for media paths (used for output naming)
            lora_trigger: Optional trigger word to prepend to each caption
            remove_llm_prefixes: Whether to remove common LLM-generated prefixes
        """
        super().__init__()

        self.dataset_file = Path(dataset_file)
        self.caption_column = caption_column
        self.media_column = media_column
        self.lora_trigger = f"{lora_trigger.strip()} " if lora_trigger else ""

        # Load captions with their corresponding output embedding paths
        self.caption_data = self._load_caption_data()

        # Convert to lists for indexing
        self.output_paths = list(self.caption_data.keys())
        self.prompts = list(self.caption_data.values())

        # Clean LLM start phrases if requested
        if remove_llm_prefixes:
            self._clean_llm_prefixes()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single caption with optional trigger word prepended and output path."""
        prompt = self.lora_trigger + self.prompts[index]
        return {
            "prompt": prompt,
            "output_path": self.output_paths[index],
            "index": index,
        }

    def _load_caption_data(self) -> dict[str, str]:
        """Load captions and compute their output embedding paths."""
        if self.dataset_file.suffix == ".csv":
            return self._load_caption_data_from_csv()
        elif self.dataset_file.suffix == ".json":
            return self._load_caption_data_from_json()
        elif self.dataset_file.suffix == ".jsonl":
            return self._load_caption_data_from_jsonl()
        else:
            raise ValueError("Expected `dataset_file` to be a path to a CSV, JSON, or JSONL file.")

    def _load_caption_data_from_csv(self) -> dict[str, str]:
        """Load captions from a CSV file and compute output embedding paths."""
        df = pd.read_csv(self.dataset_file)

        if self.caption_column not in df.columns:
            raise ValueError(f"Column '{self.caption_column}' not found in CSV file")
        if self.media_column not in df.columns:
            raise ValueError(f"Column '{self.media_column}' not found in CSV file")

        caption_data = {}
        for _, row in df.iterrows():
            media_path = Path(row[self.media_column].strip())
            # Convert media path to embedding output path (same structure, .pt extension)
            output_path = str(media_path.with_suffix(".pt"))
            caption_data[output_path] = row[self.caption_column]

        return caption_data

    def _load_caption_data_from_json(self) -> dict[str, str]:
        """Load captions from a JSON file and compute output embedding paths."""
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")

        caption_data = {}
        for entry in data:
            if self.caption_column not in entry:
                raise ValueError(f"Key '{self.caption_column}' not found in JSON entry: {entry}")
            if self.media_column not in entry:
                raise ValueError(f"Key '{self.media_column}' not found in JSON entry: {entry}")

            media_path = Path(entry[self.media_column].strip())
            # Convert media path to embedding output path (same structure, .pt extension)
            output_path = str(media_path.with_suffix(".pt"))
            caption_data[output_path] = entry[self.caption_column]

        return caption_data

    def _load_caption_data_from_jsonl(self) -> dict[str, str]:
        """Load captions from a JSONL file and compute output embedding paths."""
        caption_data = {}
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                if self.caption_column not in entry:
                    raise ValueError(f"Key '{self.caption_column}' not found in JSONL entry: {entry}")
                if self.media_column not in entry:
                    raise ValueError(f"Key '{self.media_column}' not found in JSONL entry: {entry}")

                media_path = Path(entry[self.media_column].strip())
                # Convert media path to embedding output path (same structure, .pt extension)
                output_path = str(media_path.with_suffix(".pt"))
                caption_data[output_path] = entry[self.caption_column]

        return caption_data

    def _clean_llm_prefixes(self) -> None:
        """Remove common LLM-generated prefixes from captions."""
        for i in range(len(self.prompts)):
            self.prompts[i] = self.prompts[i].strip()
            for phrase in COMMON_LLM_START_PHRASES:
                if self.prompts[i].startswith(phrase):
                    self.prompts[i] = self.prompts[i].removeprefix(phrase).strip()
                    break


def compute_captions_embeddings(  # noqa: PLR0913
    dataset_file: str | Path,
    output_dir: str,
    model_path: str,
    text_encoder_path: str,
    caption_column: str = "caption",
    media_column: str = "media_path",
    lora_trigger: str | None = None,
    remove_llm_prefixes: bool = False,
    batch_size: int = 8,
    device: str = "cuda",
    load_in_8bit: bool = False,
) -> None:
    """
    Process captions and save text embeddings.
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL) containing captions and media paths
        output_dir: Directory to save embeddings
        model_path: Path to LTX-2 checkpoint (.safetensors)
        text_encoder_path: Path to Gemma text encoder directory
        caption_column: Column name containing captions in the metadata file
        media_column: Column name containing media paths (used for output naming)
        lora_trigger: Optional trigger word to prepend to each caption
        remove_llm_prefixes: Whether to remove common LLM-generated prefixes
        batch_size: Batch size for processing
        device: Device to use for computation
        load_in_8bit: Whether to load the Gemma text encoder in 8-bit precision
    """

    console = Console()

    # Create dataset
    dataset = CaptionsDataset(
        dataset_file=dataset_file,
        caption_column=caption_column,
        media_column=media_column,
        lora_trigger=lora_trigger,
        remove_llm_prefixes=remove_llm_prefixes,
    )
    logger.info(f"Loaded {len(dataset):,} captions")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load text encoder
    with console.status("[bold]Loading Gemma text encoder...", spinner="dots"):
        text_encoder = load_text_encoder(
            model_path,
            text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
        )

    logger.info("Text encoder loaded successfully")

    # TODO(batch-tokenization): The current Gemma tokenizer doesn't support batched tokenization.
    if batch_size > 1:
        logger.warning(
            "Batch size greater than 1 is not currently supported with the Gemma tokenizer. "
            "Overriding batch_size to 1. This will be fixed in a future update."
        )
        batch_size = 1

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Process batches
    total_batches = len(dataloader)
    logger.info(f"Processing captions in {total_batches:,} batches...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing captions", total=len(dataloader))
        for batch in dataloader:
            # Encode prompts using _preprocess_text (returns embeddings before connector)
            # This is what we want to save - the connector is applied during training
            with torch.inference_mode():
                # TODO(batch-tokenization): When tokenizer supports batching, encode all prompts at once:
                #   prompt_embeds, prompt_attention_mask = text_encoder._preprocess_text(batch["prompt"]) # noqa: ERA001
                # For now, process one at a time:
                for i in range(len(batch["prompt"])):
                    prompt_embeds, prompt_attention_mask = text_encoder._preprocess_text(
                        batch["prompt"][i], padding_side="left"
                    )

                    output_rel_path = Path(batch["output_path"][i])

                    # Create output directory maintaining structure
                    output_dir_path = output_path / output_rel_path.parent
                    output_dir_path.mkdir(parents=True, exist_ok=True)

                    embedding_data = {
                        "prompt_embeds": prompt_embeds[0].cpu().contiguous(),
                        "prompt_attention_mask": prompt_attention_mask[0].cpu().contiguous(),
                    }

                    output_file = output_path / output_rel_path
                    torch.save(embedding_data, output_file)

            progress.advance(task)

    logger.info(f"Processed {len(dataset):,} captions. Embeddings saved to {output_path}")

    if device.startswith("cuda"):
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()


@app.command()
def main(  # noqa: PLR0913
    dataset_file: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing captions and media paths",
    ),
    output_dir: str = typer.Option(
        ...,
        help="Output directory to save text embeddings",
    ),
    model_path: str = typer.Option(
        ...,
        help="Path to LTX-2 checkpoint (.safetensors file)",
    ),
    text_encoder_path: str = typer.Option(
        ...,
        help="Path to Gemma text encoder directory",
    ),
    caption_column: str = typer.Option(
        default="caption",
        help="Column name containing captions in the dataset JSON/JSONL/CSV file",
    ),
    media_column: str = typer.Option(
        default="media_path",
        help="Column name in the dataset JSON/JSONL/CSV file containing media paths "
        "(used for output file naming and folder structure)",
    ),
    batch_size: int = typer.Option(
        default=8,
        help="Batch size for processing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    lora_trigger: str | None = typer.Option(
        default=None,
        help="Optional trigger word to prepend to each caption (activates the LoRA during inference)",
    ),
    remove_llm_prefixes: bool = typer.Option(
        default=False,
        help="Remove common LLM-generated prefixes from captions",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the Gemma text encoder in 8-bit precision to save GPU memory (requires bitsandbytes)",
    ),
) -> None:
    """Process text captions and save embeddings for video generation training.
    This script processes captions from metadata files and saves text embeddings
    that can be used for training video generation models. The output embeddings
    will maintain the same folder structure and naming as the corresponding media files.
    Note: This script is designed for LTX-2 models which use the Gemma text encoder.
    Examples:
        # Process captions with LTX-2 model
        python scripts/process_captions.py dataset.json --output-dir ./embeddings \\
            --model-path /path/to/ltx2_checkpoint.safetensors \\
            --text-encoder-path /path/to/gemma
        # Add a trigger word for LoRA training
        python scripts/process_captions.py dataset.json --output-dir ./embeddings \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --lora-trigger "mytoken"
        # Remove LLM-generated prefixes from captions
        python scripts/process_captions.py dataset.json --output-dir ./embeddings \\
            --model-path /path/to/ltx2.safetensors --text-encoder-path /path/to/gemma \\
            --remove-llm-prefixes
    """

    # Validate dataset file
    if not Path(dataset_file).is_file():
        raise typer.BadParameter(f"Dataset file not found: {dataset_file}")

    if lora_trigger:
        logger.info(f'LoRA trigger word "{lora_trigger}" will be prepended to all captions')

    # Process embeddings
    compute_captions_embeddings(
        dataset_file=dataset_file,
        output_dir=output_dir,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        caption_column=caption_column,
        media_column=media_column,
        lora_trigger=lora_trigger,
        remove_llm_prefixes=remove_llm_prefixes,
        batch_size=batch_size,
        device=device,
        load_in_8bit=load_text_encoder_in_8bit,
    )


if __name__ == "__main__":
    app()
