import importlib.util
from pathlib import Path

import pytest
import torch

_RANGE_MAP = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ltx_pipelines"
    / "utils"
    / "media_io"
    / "range_map.py"
)


def _load_range_map() -> object:
    spec = importlib.util.spec_from_file_location("ltx_range_map", _RANGE_MAP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


range_map = _load_range_map()


def test_to_vae_range_maps_endpoints() -> None:
    assert torch.allclose(
        range_map.to_vae_range(torch.tensor([0.0, 0.5, 1.0])),
        torch.tensor([-1.0, 0.0, 1.0]),
    )


def test_to_vae_range_rejects_out_of_range_host_values() -> None:
    for bad in (-0.1, 1.1, float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="values in"):
            range_map.to_vae_range(torch.tensor([0.5, bad]))


@pytest.mark.parametrize("position", [0, 1, 2])
def test_to_vae_range_rejects_nan_host_values(position: int) -> None:
    values = [0.25, 0.5, 0.75]
    values[position] = float("nan")
    with pytest.raises(ValueError, match="values in"):
        range_map.to_vae_range(torch.tensor(values))


def test_to_vae_range_rejects_all_nan_host_tensor() -> None:
    with pytest.raises(ValueError, match="values in"):
        range_map.to_vae_range(torch.full((2, 2), float("nan")))
