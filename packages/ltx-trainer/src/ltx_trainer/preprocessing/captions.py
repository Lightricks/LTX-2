#!/usr/bin/env python

"""
Compute text embeddings for video generation training.
Loads captions from a CSV/JSON/JSONL metadata file, encodes them with the Gemma text encoder,
and saves per-sample embeddings under ``conditions/`` (keyed to mirror the media latent layout).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from transformers.utils.logging import disable_progress_bar

from ltx_trainer import logger
from ltx_trainer.dataset_metadata import read_rows
from ltx_trainer.model_loader import embedding_weight_paths, load_embeddings_processor, load_text_encoder
from ltx_trainer.preprocessing.common import (
    atomic_save,
    build_sharded_dataloader,
    output_relative,
    preprocessing_progress,
)
from ltx_trainer.preprocessing.types import PreprocessContext

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

disable_progress_bar()


def compute_captions_embeddings(  # noqa: PLR0913
    dataset_file: str | Path,
    output_dir: str,
    model_path: str,
    text_encoder_path: str,
    caption_column: str,
    media_column: str,
    lora_trigger: str | None = None,
    batch_size: int = 8,
    device: str = "cuda",
    load_in_8bit: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Process captions and save text embeddings.
    Under ``accelerate launch``, each process handles an interleaved shard of
    the dataset (rank/world read from ``accelerate.PartialState``). Already-
    computed ``.pt`` outputs are skipped unless ``overwrite=True``; writes are
    atomic so an interrupted run is safe to resume.
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL) containing captions and media paths
        output_dir: Directory to save embeddings
        model_path: Path to LTX-2 checkpoint (.safetensors)
        text_encoder_path: Path to a Gemma directory or packed text-encoder safetensors
        caption_column: Column name containing captions in the metadata file
        media_column: Column name containing media paths (used for output naming)
        lora_trigger: Optional trigger word to prepend to each caption
        batch_size: Batch size for processing
        device: Device to use for computation
        load_in_8bit: Whether to load the Gemma text encoder in 8-bit precision
        overwrite: Re-encode every item even if its output exists. Use when rerunning with
            changed parameters (different text encoder, lora_trigger, etc.) so stale
            outputs are replaced.
    """
    console = Console()

    items = _load_caption_items(dataset_file, caption_column, media_column, lora_trigger)
    logger.info(f"Loaded {len(items):,} captions")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # TODO(batch-tokenization): The current Gemma tokenizer doesn't support batched tokenization.
    if batch_size > 1:
        logger.warning(
            "Batch size greater than 1 is not currently supported with the Gemma tokenizer. "
            "Overriding batch_size to 1. This will be fixed in a future update."
        )
        batch_size = 1

    dataloader = build_sharded_dataloader(
        items,
        batch_size=batch_size,
        num_workers=2,
        is_done=lambda idx: (output_path / items[idx]["output_path"]).is_file(),
        overwrite=overwrite,
    )
    if dataloader is None:
        return

    # Load text encoder and embeddings processor
    with console.status("[bold]Loading Gemma text encoder...", spinner="dots"):
        text_encoder = load_text_encoder(
            text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
        )
        embeddings_processor = load_embeddings_processor(
            embedding_weight_paths(model_path, text_encoder_path),
            gemma_model_path=text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
        )

    logger.info("Text encoder and embeddings processor loaded successfully")
    logger.info(f"Processing captions in {len(dataloader):,} batches...")

    with preprocessing_progress("Processing captions", total=len(dataloader)) as advance:
        for batch in dataloader:
            # Encode prompts using text_encoder.encode() + feature_extractor
            # (returns video/audio features before connector).
            # The connector is applied during training via embeddings_processor
            with torch.inference_mode():
                # TODO(batch-tokenization): When tokenizer supports batching, encode all prompts at once.
                # For now, process one at a time:
                for i in range(len(batch["prompt"])):
                    encoded = text_encoder.encode([batch["prompt"][i]])
                    hidden_states, prompt_attention_mask = encoded[0]
                    video_prompt_embeds, audio_prompt_embeds = embeddings_processor.feature_extractor(
                        hidden_states, prompt_attention_mask, "left"
                    )

                    output_rel_path = Path(batch["output_path"][i])

                    # Create output directory maintaining structure
                    output_dir_path = output_path / output_rel_path.parent
                    output_dir_path.mkdir(parents=True, exist_ok=True)

                    embedding_data = {
                        "video_prompt_embeds": video_prompt_embeds[0].cpu().contiguous(),
                        "prompt_attention_mask": prompt_attention_mask[0].cpu().contiguous(),
                    }
                    if audio_prompt_embeds is not None:
                        embedding_data["audio_prompt_embeds"] = audio_prompt_embeds[0].cpu().contiguous()

                    output_file = output_path / output_rel_path
                    atomic_save(embedding_data, output_file)

            advance()

    logger.info(f"Processed {len(dataloader.dataset):,} captions -> {output_path}")  # type: ignore[arg-type]


def _load_caption_items(
    dataset_file: str | Path,
    caption_column: str,
    media_column: str,
    lora_trigger: str | None,
) -> list[dict[str, str]]:
    """Build one ``{"prompt", "output_path"}`` item per dataset row.
    ``output_path`` is the media path made relative to the dataset dir (with a ``.pt`` suffix),
    so caption embeddings land next to their matching video/audio latents. An optional
    ``lora_trigger`` is prepended to every prompt.
    """
    data_root = Path(dataset_file).parent
    trigger = f"{lora_trigger.strip()} " if lora_trigger else ""
    items: list[dict[str, str]] = []
    for entry in read_rows(dataset_file):
        for column in (caption_column, media_column):
            if column not in entry:
                raise ValueError(f"Column '{column}' not found in {Path(dataset_file).name}")
        media_path = Path(str(entry[media_column]).strip())
        output_path = str(output_relative(data_root / media_path, data_root).with_suffix(".pt"))
        prompt = trigger + entry[caption_column]
        items.append({"prompt": prompt, "output_path": output_path})
    return items


@dataclass(kw_only=True)
class CaptionEncodeStep:
    """Encode captions with the text encoder into ``conditions/``."""

    caption_column: str
    media_column: str
    model_path: str
    text_encoder_path: str
    output_subdir: str = "conditions"
    lora_trigger: str | None = None
    batch_size: int = 1
    load_in_8bit: bool = False

    def run(self, ctx: PreprocessContext) -> None:
        compute_captions_embeddings(
            dataset_file=ctx.dataset_file,
            output_dir=str(ctx.output_root / self.output_subdir),
            model_path=self.model_path,
            text_encoder_path=self.text_encoder_path,
            caption_column=self.caption_column,
            media_column=self.media_column,
            lora_trigger=self.lora_trigger,
            batch_size=self.batch_size,
            device=ctx.device,
            load_in_8bit=self.load_in_8bit,
            overwrite=ctx.overwrite,
        )
