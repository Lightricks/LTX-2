import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.register_buffer("offset", torch.ones(2))
        self.register_buffer("optional_cache", None, persistent=False)


class TinyModelConfigurator:
    @classmethod
    def from_metadata(cls, metadata: dict) -> TinyModel:
        _ = metadata
        return TinyModel()


def _write_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file(tensors, str(path), metadata={"config": json.dumps({})})


def _builder(path: Path, registry: ModelRegistry | None = None) -> SingleGPUModelBuilder[TinyModel]:
    return SingleGPUModelBuilder(
        model_class_configurator=TinyModelConfigurator,
        model_path=str(path),
        registry=registry,
    )


@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
def test_build_loads_complete_safetensors_checkpoint(tmp_path: Path, dtype: torch.dtype | None) -> None:
    checkpoint = tmp_path / "complete.safetensors"
    _write_checkpoint(
        checkpoint,
        {"weight": torch.tensor([2.0, 3.0]), "offset": torch.tensor([4.0, 5.0])},
    )

    model = _builder(checkpoint).build(device=torch.device("cpu"), dtype=dtype)

    assert torch.equal(model.weight, torch.tensor([2.0, 3.0]))
    assert torch.equal(model.offset, torch.tensor([4.0, 5.0]))
    assert all(parameter.device.type != "meta" for parameter in model.parameters())
    assert all(buffer.device.type != "meta" for buffer in model.buffers())
    assert model.weight.dtype == (dtype or torch.float32)
    assert model.offset.dtype == (dtype or torch.float32)
    assert model.optional_cache is None


def test_explicit_meta_shell_construction_remains_available(tmp_path: Path) -> None:
    model = _builder(tmp_path / "unused.safetensors").meta_model({}, ())
    assert model.weight.is_meta
    assert model.offset.is_meta


@pytest.mark.parametrize(
    ("tensors", "missing_name"),
    [
        ({"offset": torch.tensor([4.0, 5.0])}, "weight"),
        ({"weight": torch.tensor([2.0, 3.0])}, "offset"),
    ],
)
def test_build_rejects_safetensors_checkpoint_missing_model_state(
    tmp_path: Path, tensors: dict[str, torch.Tensor], missing_name: str
) -> None:
    checkpoint = tmp_path / f"missing-{missing_name}.safetensors"
    _write_checkpoint(checkpoint, tensors)

    with pytest.raises(RuntimeError, match=rf"uninitialized parameters or buffers: \['{missing_name}'\]"):
        _builder(checkpoint).build(device=torch.device("cpu"))


def test_failed_build_can_retry_with_registry_cached_shell(tmp_path: Path) -> None:
    checkpoint = tmp_path / "retry.safetensors"
    _write_checkpoint(checkpoint, {"weight": torch.tensor([2.0, 3.0])})
    registry = ModelRegistry(cache_models=True, cache_weights=False)
    builder = _builder(checkpoint, registry)

    with pytest.raises(RuntimeError, match="uninitialized parameters or buffers"):
        builder.build(device=torch.device("cpu"))

    _write_checkpoint(
        checkpoint,
        {"weight": torch.tensor([2.0, 3.0]), "offset": torch.tensor([4.0, 5.0])},
    )
    model = builder.build(device=torch.device("cpu"))

    assert torch.equal(model.weight, torch.tensor([2.0, 3.0]))
    assert torch.equal(model.offset, torch.tensor([4.0, 5.0]))
