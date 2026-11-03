"""JIT migration from the legacy flat trainer config layout to the strategy-owned schema.
Pure helpers only — no I/O. ``migrate_legacy_config`` is idempotent: already-migrated
configs are returned unchanged (deep-copied).
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

LEGACY_CONFIG_DEPRECATION_MESSAGE = (
    "Training config uses a deprecated layout (top-level model/lora/data/flow_matching, "
    "optimization learning-rate fields, or validation content at the root). It was "
    "auto-migrated to the strategy-owned layout. Update the YAML to the current schema; "
    "see docs/configuration-reference.md."
)

HUB_UPLOAD_REMOVED_MESSAGE = (
    "Training config had hub.push_to_hub enabled but Hugging Face Hub upload is no longer "
    "supported; hub settings were removed. Publish checkpoints with an external workflow."
)

_LEGACY_TOP_LEVEL_KEYS = frozenset({"model", "lora", "data", "flow_matching", "acceleration", "hub"})

_LEGACY_OPTIMIZATION_KEYS = frozenset(
    {
        "learning_rate",
        "optimizer_type",
        "scheduler_type",
        "scheduler_params",
        "enable_gradient_checkpointing",
    }
)

_VALIDATION_CADENCE_KEYS = frozenset({"interval", "skip_initial_validation"})

_LEGACY_STRATEGY_NAMES = frozenset({"text_to_video", "video_to_video"})

_LEGACY_STRATEGY_FIELD_KEYS = frozenset(
    {
        "first_frame_conditioning_p",
        "with_audio",
        "audio_latents_dir",
        "reference_latents_dir",
    }
)

_LEGACY_VALIDATION_FORMAT_KEYS = frozenset({"prompts", "images", "reference_videos", "include_reference_in_output"})


def migrate_legacy_config(raw: dict[str, Any], *, warn: bool = True) -> dict[str, Any]:
    """Migrate a legacy config dict to the new schema. Pure, idempotent, and deep-copies input.
    If already migrated, returns a deep copy (still rewriting deprecated scale-factor aliases).
    If warn is True, emits a deprecation warning when a full legacy layout is detected.
    """
    if not _is_legacy_config(raw):
        config = copy.deepcopy(raw)
        _rename_scale_factor_aliases(config)
        _migrate_validation_samples_in_config(config)
        return config

    if warn:
        warnings.warn(LEGACY_CONFIG_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

    config = copy.deepcopy(raw)
    _validate_legacy_block_types(config)

    # The old top-level config supplied TextToVideoConfig when the block was omitted.
    config.setdefault("training_strategy", {"name": "text_to_video"})
    strategy = _ensure_dict(config, "training_strategy")
    optimization = _ensure_dict(config, "optimization")
    checkpoints = _ensure_dict(config, "checkpoints")
    validation = _ensure_dict(config, "validation")
    strategy_validation = _ensure_dict(strategy, "validation")
    strategy_model = _ensure_dict(strategy, "model")
    strategy_optimizer = _ensure_dict(strategy, "optimizer")
    strategy_timestep_sampling = _ensure_dict(strategy, "timestep_sampling")

    _migrate_model_block(config, strategy, strategy_model, checkpoints)
    _migrate_lora_and_data(config, strategy)
    _migrate_flow_matching(config, strategy_timestep_sampling)
    _migrate_acceleration(config, strategy, strategy_model, optimization)
    _migrate_optimization_strategy_fields(optimization, strategy_optimizer)
    _migrate_reference_scale_factors(validation, strategy)
    _migrate_validation_content(validation, strategy_validation)
    _migrate_validation_samples_format(strategy_validation)
    _migrate_legacy_strategy_fields(strategy)
    _migrate_hub_block(config)
    _rename_scale_factor_aliases(config)
    _strip_reference_condition_scale_fields(strategy.get("validation"), strategy)

    return config


def _is_legacy_config(raw: dict[str, Any]) -> bool:
    """Return True if the dict uses any legacy config conventions needing migration."""
    if _LEGACY_TOP_LEVEL_KEYS & raw.keys():
        return True

    optimization = raw.get("optimization")
    if isinstance(optimization, dict) and _LEGACY_OPTIMIZATION_KEYS & optimization.keys():
        return True

    validation = raw.get("validation")
    if isinstance(validation, dict) and set(validation.keys()) - _VALIDATION_CADENCE_KEYS:
        return True

    strategy = raw.get("training_strategy")
    if not isinstance(strategy, dict):
        return False

    strategy_validation = strategy.get("validation")
    return (
        strategy.get("name") in _LEGACY_STRATEGY_NAMES
        or bool(_LEGACY_STRATEGY_FIELD_KEYS & strategy.keys())
        or (isinstance(strategy_validation, dict) and _uses_legacy_validation_format(strategy_validation))
    )


def _uses_legacy_validation_format(validation: dict[str, Any]) -> bool:
    """Return True when validation still uses deprecated prompts-based layout."""
    return bool(validation.get("prompts")) and not validation.get("samples")


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """Ensure parent[key] is a dict and return it; assign empty dict if not present or not a dict."""
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _merge_into(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Set any keys in 'source' not yet in 'target'; doesn't overwrite existing."""
    for key, value in source.items():
        target.setdefault(key, value)


def _promote_reference_scale(data: dict[str, Any], key: str, value: object, source: str) -> None:
    """Set a reference scale only when it agrees with its existing canonical value."""
    if key in data and data[key] != value:
        raise ValueError(f"Conflicting {key} values: {data[key]!r} is already set, but {source} specifies {value!r}.")
    data.setdefault(key, value)


def _validate_legacy_block_types(config: dict[str, Any]) -> None:
    """Reject malformed legacy blocks before migration could discard their values."""
    for key in ("model", "flow_matching", "acceleration", "hub"):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"Legacy {key!r} block must be a dict, got {type(config[key]).__name__}.")


def _migrate_lora_and_data(config: dict[str, Any], strategy: dict[str, Any]) -> None:
    """Move lora and data keys from top-level config into the strategy block, if present."""
    if "lora" in config:
        if "lora" in strategy:
            raise ValueError("Cannot migrate top-level 'lora': training_strategy.lora is already set.")
        strategy["lora"] = config.pop("lora")
    if "data" in config:
        if "data" in strategy:
            raise ValueError("Cannot migrate top-level 'data': training_strategy.data is already set.")
        strategy["data"] = config.pop("data")


def _migrate_acceleration(
    config: dict[str, Any],
    strategy: dict[str, Any],
    strategy_model: dict[str, Any],
    optimization: dict[str, Any],
) -> None:
    """Move acceleration fields to the correct blocks; raise error if unsupported legacy fields remain."""
    acceleration = config.pop("acceleration", None)
    if not isinstance(acceleration, dict):
        return

    mixed_precision_mode = acceleration.pop("mixed_precision_mode", None)
    if mixed_precision_mode is not None:
        optimization.setdefault("mixed_precision_mode", mixed_precision_mode)

    offload = acceleration.pop("offload_optimizer_during_validation", None)
    if offload is not None:
        optimization.setdefault("offload_optimizer_during_validation", offload)

    quantization = acceleration.pop("quantization", None)
    if quantization is not None and "quantization" not in strategy:
        strategy["quantization"] = quantization

    load_text_encoder_in_8bit = acceleration.pop("load_text_encoder_in_8bit", None)
    if load_text_encoder_in_8bit is not None:
        strategy_model.setdefault("load_text_encoder_in_8bit", load_text_encoder_in_8bit)

    if acceleration:
        raise ValueError(
            "Unsupported fields in legacy acceleration block: "
            f"{sorted(acceleration.keys())}. Remove or migrate them manually."
        )


def _migrate_optimization_strategy_fields(
    optimization: dict[str, Any],
    strategy_optimizer: dict[str, Any],
) -> None:
    """Move certain optimizer fields from optimization to strategy.optimizer, if found."""
    field_map = {
        "learning_rate": "learning_rate",
        "optimizer_type": "optimizer_type",
        "scheduler_type": "scheduler_type",
        "scheduler_params": "scheduler_params",
        "enable_gradient_checkpointing": "gradient_checkpointing",
    }
    for legacy_key, new_key in field_map.items():
        if legacy_key not in optimization:
            continue
        strategy_optimizer.setdefault(new_key, optimization.pop(legacy_key))


def _migrate_validation_content(
    validation: dict[str, Any],
    strategy_validation: dict[str, Any],
) -> None:
    """Move validation blocks (except cadence settings) into strategy.validation."""
    content_keys = [key for key in list(validation.keys()) if key not in _VALIDATION_CADENCE_KEYS]
    for key in content_keys:
        strategy_validation.setdefault(key, validation.pop(key))


def _migrate_validation_samples_in_config(config: dict[str, Any]) -> None:
    """Convert deprecated validation prompts layout under ``training_strategy.validation``."""
    strategy = config.get("training_strategy")
    if not isinstance(strategy, dict):
        return
    validation = strategy.get("validation")
    if isinstance(validation, dict):
        _migrate_validation_samples_format(validation)


def _migrate_validation_samples_format(validation: dict[str, Any]) -> None:
    """Convert deprecated ``prompts``/``images``/``reference_videos`` to ``samples``."""
    prompts = validation.get("prompts")
    if not prompts:
        _strip_legacy_validation_format_keys(validation)
        return

    if validation.get("samples"):
        _strip_legacy_validation_format_keys(validation)
        return

    images = validation.pop("images", None)
    reference_videos = validation.pop("reference_videos", None)
    include_reference_in_output = validation.pop("include_reference_in_output", False)
    prompts = validation.pop("prompts")

    if images is not None and len(images) != len(prompts):
        raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")
    if reference_videos is not None and len(reference_videos) != len(prompts):
        raise ValueError(
            f"Number of reference videos ({len(reference_videos)}) must match number of prompts ({len(prompts)})"
        )

    samples: list[dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        conditions: list[dict[str, Any]] = []
        if images and i < len(images):
            conditions.append({"type": "first_frame", "image_or_video": images[i]})
        if reference_videos and i < len(reference_videos):
            conditions.append(
                {
                    "type": "reference",
                    "video": reference_videos[i],
                    "include_in_output": include_reference_in_output,
                }
            )
        samples.append({"prompt": prompt, "conditions": conditions})
    validation["samples"] = samples


def _strip_legacy_validation_format_keys(validation: dict[str, Any]) -> None:
    """Remove deprecated validation-format keys so ``extra='forbid'`` parsing succeeds."""
    for key in _LEGACY_VALIDATION_FORMAT_KEYS:
        validation.pop(key, None)


def _ensure_condition(conditions: list[Any], condition_type: str, default: dict[str, Any]) -> None:
    """Append a condition unless one with the same type is already present."""
    for condition in conditions:
        if isinstance(condition, dict) and condition.get("type") == condition_type:
            for key, value in default.items():
                condition.setdefault(key, value)
            return
    conditions.append(default)


def _migrate_hub_block(config: dict[str, Any]) -> None:
    """Drop the removed hub block, warning when upload had been enabled."""
    hub = config.pop("hub", None)
    if not isinstance(hub, dict):
        return
    if hub.get("push_to_hub"):
        hub_model_id = hub.get("hub_model_id")
        detail = f" (hub_model_id={hub_model_id!r})" if hub_model_id else ""
        warnings.warn(f"{HUB_UPLOAD_REMOVED_MESSAGE}{detail}", UserWarning, stacklevel=3)


def _promote_condition_scale(
    condition: dict[str, Any],
    strategy: dict[str, Any],
    legacy_key: str,
    data_key: str,
) -> None:
    """Promote one legacy condition scale field, creating strategy.data when needed."""
    if legacy_key not in condition:
        return
    value = condition.pop(legacy_key)
    _promote_reference_scale(
        _ensure_dict(strategy, "data"),
        data_key,
        value,
        f"reference condition {legacy_key}",
    )


def _rename_scale_factor_aliases(config: dict[str, Any]) -> None:
    """Rewrite the supported deprecated ``reference_downscale_factor`` data alias."""
    strategy = config.get("training_strategy")
    if not isinstance(strategy, dict):
        return
    data = strategy.get("data")
    if isinstance(data, dict) and "reference_downscale_factor" in data:
        _promote_reference_scale(
            data,
            "reference_spatial_scale_factor",
            data.pop("reference_downscale_factor"),
            "training_strategy.data.reference_downscale_factor",
        )


def _migrate_model_block(
    config: dict[str, Any],
    strategy: dict[str, Any],
    strategy_model: dict[str, Any],
    checkpoints: dict[str, Any],
) -> None:
    """Move model block fields to proper locations in the current schema."""
    model = config.pop("model", None)
    if not isinstance(model, dict):
        return

    load_checkpoint = model.pop("load_checkpoint", None)
    if load_checkpoint is not None and "load_checkpoint" not in checkpoints:
        checkpoints["load_checkpoint"] = load_checkpoint

    training_mode = model.pop("training_mode", None)
    if training_mode is not None and "training_mode" not in strategy:
        strategy["training_mode"] = training_mode

    _merge_into(strategy_model, model)


def _migrate_flow_matching(config: dict[str, Any], strategy_timestep_sampling: dict[str, Any]) -> None:
    """Move flow_matching fields into the strategy's timestep_sampling block."""
    flow_matching = config.pop("flow_matching", None)
    if isinstance(flow_matching, dict):
        _merge_into(strategy_timestep_sampling, flow_matching)


def _migrate_reference_scale_factors(validation: dict[str, Any], strategy: dict[str, Any]) -> None:
    """Route legacy ``validation.reference_*_factor`` onto ``strategy.data``.
    Reference scale factors moved from validation to the strategy's ``data`` block (the encode
    parameters the validation runner now inherits). Pop them before validation content is bulk-moved
    so they land on ``strategy.data`` (their canonical home) rather than ``strategy.validation``,
    where the strategy fills them from ``data`` at runtime.
    Accepts both the current ``reference_spatial_scale_factor`` name and the legacy
    ``reference_downscale_factor`` alias; always writes the current name onto ``data``.
    """
    scales = [
        (
            "reference_spatial_scale_factor",
            validation.pop("reference_spatial_scale_factor", None),
            "validation.reference_spatial_scale_factor",
        ),
        (
            "reference_spatial_scale_factor",
            validation.pop("reference_downscale_factor", None),
            "validation.reference_downscale_factor",
        ),
        (
            "reference_temporal_scale_factor",
            validation.pop("reference_temporal_scale_factor", None),
            "validation.reference_temporal_scale_factor",
        ),
    ]
    if all(value is None for _, value, _ in scales):
        return
    data = _ensure_dict(strategy, "data")
    for key, value, source in scales:
        if value is None:
            continue
        _promote_reference_scale(
            data,
            key,
            value,
            source,
        )


def _strip_reference_condition_scale_fields(node: Any, strategy: dict[str, Any]) -> None:  # noqa: ANN401
    """Move legacy per-condition reference scales to strategy data, rejecting conflicts."""
    if isinstance(node, dict):
        if node.get("type") == "reference":
            _promote_condition_scale(
                node,
                strategy,
                "spatial_scale_factor",
                "reference_spatial_scale_factor",
            )
            _promote_condition_scale(
                node,
                strategy,
                "downscale_factor",
                "reference_spatial_scale_factor",
            )
            _promote_condition_scale(
                node,
                strategy,
                "temporal_scale_factor",
                "reference_temporal_scale_factor",
            )
        for value in node.values():
            _strip_reference_condition_scale_fields(value, strategy)
    elif isinstance(node, list):
        for item in node:
            _strip_reference_condition_scale_fields(item, strategy)


def _migrate_legacy_strategy_fields(strategy: dict[str, Any]) -> None:
    """Map removed text_to_video / video_to_video strategy fields onto flexible modality config."""
    name = strategy.get("name")
    has_legacy_fields = bool(_LEGACY_STRATEGY_FIELD_KEYS & strategy.keys())
    if name not in _LEGACY_STRATEGY_NAMES and not has_legacy_fields:
        return

    first_frame_p = strategy.pop("first_frame_conditioning_p", None)
    if first_frame_p is None and name in _LEGACY_STRATEGY_NAMES:
        first_frame_p = 0.1
    with_audio = strategy.pop("with_audio", None)
    audio_latents_dir = strategy.pop("audio_latents_dir", None)
    reference_latents_dir = strategy.pop("reference_latents_dir", None)

    is_video_to_video = name == "video_to_video" or reference_latents_dir is not None

    video = _ensure_dict(strategy, "video")
    video.setdefault("is_generated", True)
    video.setdefault("latents_dir", "latents")
    conditions = video.setdefault("conditions", [])
    if not isinstance(conditions, list):
        conditions = []
        video["conditions"] = conditions

    if is_video_to_video:
        _ensure_condition(
            conditions,
            "reference",
            {
                "type": "reference",
                "latents_dir": reference_latents_dir or "reference_latents",
                "probability": 1.0,
            },
        )

    if first_frame_p is not None and first_frame_p > 0:
        _ensure_condition(
            conditions,
            "first_frame",
            {"type": "first_frame", "probability": first_frame_p},
        )

    if with_audio:
        strategy.setdefault("share_sigmas", True)
        audio = _ensure_dict(strategy, "audio")
        audio.setdefault("is_generated", True)
        audio.setdefault("latents_dir", audio_latents_dir or "audio_latents")

    if name in _LEGACY_STRATEGY_NAMES:
        strategy["name"] = "flexible"
