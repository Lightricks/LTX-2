# Implementing Custom Training Strategies

This guide explains how to implement your own training strategy for specialized use cases like audio-only training,
video inpainting, or other custom training recipes.

## 📋 Overview

The trainer uses the **Strategy Pattern** to separate training logic from the core training loop. Each strategy defines:

1. **What data is needed** - Which preprocessed data directories to load
2. **How to prepare inputs** - Transform batch data into model inputs
3. **How to compute loss** - Calculate the training objective

This architecture lets you implement new training modes without modifying the core trainer code.

### When You Need a Custom Strategy

Consider implementing a custom strategy when you need:

- **Different input modalities** (e.g., audio-only, audio-to-video conditioning)
- **Additional conditioning signals** (e.g., masks for inpainting, depth maps)
- **Custom loss computation** (e.g., weighted losses, auxiliary losses)
- **Different noise application patterns** (e.g., partial masking)

## 🏗️ Architecture Overview

### How Strategies Fit Into the Trainer

The trainer delegates all training-mode-specific logic to the strategy:

1. **Initialization** — The trainer calls `get_data_sources()` to determine which preprocessed data directories to load
2. **Each training step:**
    - Calls `prepare_training_inputs()` to transform the raw batch into model-ready inputs
    - Runs the transformer forward pass
    - Calls `compute_loss()` to compute the training objective

The trainer handles everything else: optimization, checkpointing, validation, and distributed training.

### Key Components

| Component                                                                               | Purpose                                                      |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [`TrainingStrategyConfigBase`](../src/ltx_trainer/training_strategies/base_strategy.py) | Base class for strategy configuration (Pydantic model)       |
| [`TrainingStrategy`](../src/ltx_trainer/training_strategies/base_strategy.py)           | Abstract base class defining the strategy interface          |
| [`ModelInputs`](../src/ltx_trainer/training_strategies/base_strategy.py)                | Dataclass containing prepared inputs for the transformer     |
| [`Modality`](../../ltx-core/src/ltx_core/model/transformer/modality.py)                 | ltx-core dataclass representing video or audio modality data |

## 📝 Step-by-Step Implementation

### Step 1: Plan Your Strategy

Before writing code, answer these questions:

1. **What additional data does your strategy need?**
    - Example: Inpainting needs mask latents alongside video latents
    - Example: Audio-to-video needs reference audio embeddings

2. **What does conditioning look like?**
    - Which tokens should be noised vs. kept clean?
    - How should conditioning tokens be structured (e.g., first frame, reference video, mask)?

3. **How should loss be computed?**
    - Which tokens contribute to the loss?
    - Are there multiple loss terms to combine?

### Step 2: Extend Data Preprocessing (If Needed)

If your strategy requires additional preprocessed data beyond video latents, audio latents, and text embeddings, you'll
need to extend the preprocessing pipeline.

#### Option A: Modify `process_dataset.py`

For integrated preprocessing, add new arguments and processing steps to the main script. For example, to add mask
preprocessing:

```python
# In process_dataset.py, add a new argument
@app.command()
def main(
        # ... existing arguments ...
        mask_column: str | None = typer.Option(
            default=None,
            help="Column name containing mask video paths (for inpainting)",
        ),
) -> None:
    # ... existing processing ...

    # Process masks if provided
    if mask_column:
        logger.info("Processing mask videos for inpainting training...")
        mask_latents_dir = output_base / "mask_latents"

        compute_latents(
            dataset_file=dataset_path,
            video_column=mask_column,
            resolution_buckets=parsed_resolution_buckets,
            output_dir=str(mask_latents_dir),
            model_path=model_path,
            # ... other args ...
        )
```

#### Option B: Create a Standalone Script

For complex preprocessing that doesn't fit naturally into the existing pipeline, create a dedicated script
(e.g., `scripts/process_masks.py`). Use [`scripts/compute_reference.py`](../scripts/compute_reference.py) as a
template - it shows how to process paired data and update the dataset JSON.

#### Expected Output Structure

Your preprocessing should create a directory structure that the strategy can reference:

```
preprocessed_data_root/
├── latents/           # Video latents (standard)
├── conditions/        # Text embeddings (standard)
├── audio_latents/     # Audio latents (if with_audio)
├── mask_latents/      # Your custom data directory
└── reference_latents/ # Reference videos (for IC-LoRA)
```

### Step 3: Create the Strategy Configuration

Create a new file for your strategy (e.g., `src/ltx_trainer/training_strategies/inpainting.py`):

```python
"""Inpainting training strategy.

This strategy implements video inpainting training where:
- Mask latents indicate which regions to inpaint
- Loss is computed only on masked (inpainted) regions
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class InpaintingConfig(TrainingStrategyConfigBase):
    """Configuration for inpainting training strategy."""

    # The 'name' field acts as a discriminator for the config union
    name: Literal["inpainting"] = "inpainting"

    mask_latents_dir: str = Field(
        default="mask_latents",
        description="Directory name for mask latents",
    )

    # Add any strategy-specific parameters
    mask_threshold: float = Field(
        default=0.5,
        description="Threshold for binary mask conversion",
        ge=0.0,
        le=1.0,
    )
```

**Key points:**

- Inherit from `TrainingStrategyConfigBase`
- Use `Literal["your_strategy_name"]` for the `name` field - this enables automatic strategy selection
- Use Pydantic `Field` for validation and documentation

### Step 4: Implement the Strategy Class

```python
class InpaintingStrategy(TrainingStrategy):
    """Inpainting training strategy.

    Trains the model to fill in masked regions of videos while
    keeping unmasked regions as conditioning.
    """

    config: InpaintingConfig

    def __init__(self, config: InpaintingConfig):
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        """Whether this strategy requires audio components."""
        return False  # Set to True if your strategy needs audio

    def get_data_sources(self) -> dict[str, str]:
        """Define which data directories to load.

        Returns a mapping of directory names to batch keys.
        The trainer will load .pt files from each directory and
        make them available in the batch under the specified key.
        """
        return {
            "latents": "latents",  # -> batch["latents"]
            "conditions": "conditions",  # -> batch["conditions"]
            self.config.mask_latents_dir: "masks",  # -> batch["masks"]
        }

    def prepare_training_inputs(
            self,
            batch: dict[str, Any],
            timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Transform batch data into model inputs.

        This is where the core training logic lives:
        1. Extract and patchify latents
        2. Sample noise and apply it appropriately
        3. Create conditioning masks
        4. Build Modality objects for the transformer
        """
        # Get video latents [B, C, F, H, W]
        latents_data = batch["latents"]
        video_latents = latents_data["latents"]

        # Get dimensions
        num_frames = latents_data["num_frames"][0].item()
        height = latents_data["height"][0].item()
        width = latents_data["width"][0].item()

        # Patchify: [B, C, F, H, W] -> [B, seq_len, C]
        video_latents = self._video_patchifier.patchify(video_latents)

        batch_size, seq_len, _ = video_latents.shape
        device = video_latents.device
        dtype = video_latents.dtype

        # Get mask latents and process them
        mask_data = batch["masks"]
        mask_latents = mask_data["latents"]
        mask_latents = self._video_patchifier.patchify(mask_latents)

        # Create binary mask: True = inpaint this region, False = keep original
        inpaint_mask = mask_latents.mean(dim=-1) > self.config.mask_threshold

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(video_latents)
        noise = torch.randn_like(video_latents)

        # Apply noise only to inpaint regions
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas_expanded) * video_latents + sigmas_expanded * noise

        # Keep original latents for non-inpaint regions (conditioning)
        inpaint_mask_expanded = inpaint_mask.unsqueeze(-1)
        noisy_latents = torch.where(inpaint_mask_expanded, noisy_latents, video_latents)

        # Create per-token timesteps
        # Conditioning tokens (non-inpaint) get timestep=0
        # Inpaint tokens get the sampled sigma
        timesteps = self._create_per_token_timesteps(~inpaint_mask, sigmas.squeeze())

        # Compute targets (velocity prediction: noise - clean)
        targets = noise - video_latents

        # Get text embeddings
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Generate position embeddings
        positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=24.0,  # Or get from latents_data
            device=device,
            dtype=dtype,
        )

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=noisy_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Loss mask: only compute loss on inpaint regions
        loss_mask = inpaint_mask

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=targets,
            audio_targets=None,
            video_loss_mask=loss_mask,
            audio_loss_mask=None,
        )

    def compute_loss(
            self,
            video_pred: Tensor,
            audio_pred: Tensor | None,
            inputs: ModelInputs,
    ) -> Tensor:
        """Compute training loss on inpaint regions only. Returns [B,]."""
        # MSE loss
        loss = (video_pred - inputs.video_targets).pow(2)

        # Apply loss mask and reduce to per-element [B,]
        loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        masked = loss.mul(loss_mask)
        return masked.mean(dim=[-2, -1]) / loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8)
```

### Step 5: Register the Strategy

You need to register your strategy in two places:

**1. Update [`src/ltx_trainer/training_strategies/__init__.py`](../src/ltx_trainer/training_strategies/__init__.py):**

```python
# Add import for your strategy
from ltx_trainer.training_strategies.inpainting import InpaintingConfig, InpaintingStrategy

# Add to the TrainingStrategyConfig type alias
TrainingStrategyConfig = TextToVideoConfig | VideoToVideoConfig | InpaintingConfig

# Add to __all__
__all__ = [
    # ... existing exports ...
    "InpaintingConfig",
    "InpaintingStrategy",
]


# Add case in get_training_strategy()
def get_training_strategy(config: TrainingStrategyConfig) -> TrainingStrategy:
    match config:
        # ... existing cases ...
        case InpaintingConfig():
            strategy = InpaintingStrategy(config)
```

**2. Update [`src/ltx_trainer/config.py`](../src/ltx_trainer/config.py):**

```python
# Add import
from ltx_trainer.training_strategies.inpainting import InpaintingConfig

# Add to the TrainingStrategyConfig union with a Tag matching your strategy name
TrainingStrategyConfig = Annotated[
    Annotated[TextToVideoConfig, Tag("text_to_video")]
    | Annotated[VideoToVideoConfig, Tag("video_to_video")]
    | Annotated[InpaintingConfig, Tag("inpainting")],  # Add your config
    Discriminator(_get_strategy_discriminator),
]
```

### Step 6: Create a Configuration File

Create an example config in `configs/`:

```yaml
# configs/ltx2_inpainting_lora.yaml

model:
  model_path: "/path/to/ltx2.safetensors"
  text_encoder_path: "/path/to/gemma"
  training_mode: "lora"

training_strategy:
  name: "inpainting"  # Must match your Literal type
  mask_latents_dir: "mask_latents"
  mask_threshold: 0.5

lora:
  rank: 32
  alpha: 32
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"

data:
  preprocessed_data_root: "/path/to/preprocessed/dataset"

optimization:
  learning_rate: 1e-4
  steps: 2000
  batch_size: 1

# ... other config sections ...
```

## 🔧 Helper Methods Reference

The base `TrainingStrategy` class provides these helper methods:

| Method                                       | Purpose                                         |
|----------------------------------------------|-------------------------------------------------|
| `_video_patchifier.patchify(latents)`        | Convert `[B, C, F, H, W]` → `[B, seq_len, C]`   |
| `_audio_patchifier.patchify(latents)`        | Convert `[B, C, T, F]` → `[B, T, C*F]`          |
| `_get_video_positions(...)`                  | Generate position embeddings for video          |
| `_get_audio_positions(...)`                  | Generate position embeddings for audio          |
| `_create_per_token_timesteps(mask, sigma)`   | Create timesteps with 0 for conditioning tokens |
| `_create_first_frame_conditioning_mask(...)` | Create mask for first-frame conditioning        |

## 📊 Understanding ModelInputs

The `ModelInputs` dataclass contains everything needed for the forward pass and loss computation:

```python
@dataclass
class ModelInputs:
    video: Modality  # Video modality data
    audio: Modality | None  # Audio modality (None if video-only)

    video_targets: Tensor  # Target values for loss (velocity)
    audio_targets: Tensor | None

    video_loss_mask: Tensor  # Boolean: True = compute loss for this token
    audio_loss_mask: Tensor | None

    ref_seq_len: int | None = None  # For IC-LoRA: reference sequence length
```

## 📊 Understanding Modality

The `Modality` dataclass (from ltx-core) represents a single modality's data:

```python
@dataclass(frozen=True)
class Modality:
    enabled: bool  # Whether this modality is active
    latent: Tensor  # [B, seq_len, C] - the latent tokens
    timesteps: Tensor  # [B, seq_len] - per-token timesteps (sigmas)
    positions: Tensor  # [B, dims, seq_len, 2] - position bounds
    context: Tensor  # [B, ctx_len, C] - text embeddings
    context_mask: Tensor  # [B, ctx_len] - attention mask for context
```

> [!NOTE]
> **Per-token timesteps:** Each token in the sequence has its own timestep. Conditioning tokens—those that should remain
> un-noised—must have `timestep=0`. This is how the model distinguishes clean reference tokens from tokens to denoise. Use
`_create_per_token_timesteps(conditioning_mask, sigma)` to set this up correctly.

> [!NOTE]
> `Modality` is immutable (frozen dataclass). Use `dataclasses.replace()` to create modified copies.

## ✅ Testing Your Strategy

1. **Verify your training configuration is valid:**
   ```bash
   uv run python -c "
   from ltx_trainer.config import LtxTrainerConfig
   import yaml

   with open('configs/ltx2_inpainting_lora.yaml') as f:
       config = LtxTrainerConfig(**yaml.safe_load(f))
   print(f'Strategy: {config.training_strategy.name}')
   "
   ```

2. **Test strategy instantiation:**
   ```bash
   uv run python -c "
   from ltx_trainer.training_strategies import get_training_strategy
   from ltx_trainer.training_strategies.inpainting import InpaintingConfig

   config = InpaintingConfig()
   strategy = get_training_strategy(config)
   print(f'Data sources: {strategy.get_data_sources()}')
   "
   ```

3. **Run a short training test:**
   ```bash
   uv run python scripts/train.py configs/ltx2_inpainting_lora.yaml
   ```

## 💡 Tips and Best Practices

### Debugging

- Set `data.num_dataloader_workers: 0` to get clearer error messages
- Use a small dataset and few steps for initial testing
- Check tensor shapes at each step with print statements

## 🔗 Related Documentation

- [Training Modes](training-modes.md) - Overview of built-in training modes
- [Configuration Reference](configuration-reference.md) - All configuration options
- [Dataset Preparation](dataset-preparation.md) - Preprocessing workflow
- [ltx-core Documentation](../../ltx-core/README.md) - Core model components

## 📚 Reference: Existing Strategies

Study these implementations for guidance:

| Strategy                                                                           | Complexity | Key Features                                   |
|------------------------------------------------------------------------------------|------------|------------------------------------------------|
| [`TextToVideoStrategy`](../src/ltx_trainer/training_strategies/text_to_video.py)   | Simple     | First-frame conditioning, optional audio       |
| [`VideoToVideoStrategy`](../src/ltx_trainer/training_strategies/video_to_video.py) | Medium     | Reference video concatenation, split loss mask |
