"""Safetensors I/O and LoRA fusion for block streaming."""

from __future__ import annotations

import safetensors
import torch

from ltx_core.loader.sd_ops import SDOps


class DiskTensorReader:
    """Key-based tensor accessor over one or more safetensors files."""

    def __init__(self, paths: list[str]) -> None:
        self._handles: list[safetensors.safe_open] = []
        self._key_to_handle_idx: dict[str, int] = {}
        for path in paths:
            handle = safetensors.safe_open(path, framework="pt", device="cpu")
            handle_idx = len(self._handles)
            self._handles.append(handle)
            for sft_key in handle.keys():  # noqa: SIM118
                self._key_to_handle_idx[sft_key] = handle_idx

    def keys(self) -> list[str]:
        return list(self._key_to_handle_idx.keys())

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._handles[self._key_to_handle_idx[key]].get_tensor(key)

    def close(self) -> None:
        self._handles.clear()
        self._key_to_handle_idx.clear()


class DiskBlockReader:
    """Reads one block at a time from safetensors into provided buffers.
    Maps block indices to safetensors keys via a pre-computed key map.
    """

    def __init__(
        self,
        reader: DiskTensorReader,
        block_key_map: dict[int, list[tuple[str, str]]],
        dtype: torch.dtype,
    ) -> None:
        self._reader = reader
        self._block_key_map = block_key_map
        self._dtype = dtype

    def read_into(self, target: dict[str, torch.Tensor], block_idx: int) -> None:
        for sft_key, param_name in self._block_key_map[block_idx]:
            tensor = self._reader.get_tensor(sft_key)
            if tensor.dtype != self._dtype:
                tensor = tensor.to(self._dtype)
            target[param_name].copy_(tensor)

    def cleanup(self) -> None:
        self._reader.close()


class LoraSource:
    """Pinned-memory cache of LoRA A/B matrices for on-the-fly fusion.
    At init, loads all matched A/B pairs into pinned CPU memory.
    :meth:`get_delta` computes ``(B * strength) @ A`` on the given device.
    """

    def __init__(self, path: str, sd_ops: SDOps | None, strength: float) -> None:
        self.strength = strength

        # param_prefix -> (pinned_a, pinned_b)
        self._pinned_ab: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        a_keys: dict[str, str] = {}
        b_keys: dict[str, str] = {}
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            # First pass: build key map.
            for sft_key in handle.keys():  # noqa: SIM118
                model_key = sd_ops.apply_to_key(sft_key) if sd_ops is not None else sft_key
                if model_key is None:
                    continue
                if model_key.endswith(".lora_A.weight"):
                    a_keys[model_key[: -len(".lora_A.weight")]] = sft_key
                elif model_key.endswith(".lora_B.weight"):
                    b_keys[model_key[: -len(".lora_B.weight")]] = sft_key

            # Second pass: load and pin matched A+B pairs (orphans silently skipped).
            for prefix in a_keys.keys() & b_keys.keys():
                self._pinned_ab[prefix] = (
                    handle.get_tensor(a_keys[prefix]).pin_memory(),
                    handle.get_tensor(b_keys[prefix]).pin_memory(),
                )

    def get_delta(self, param_prefix: str, device: torch.device | None = None) -> torch.Tensor | None:
        """Return ``(B * strength) @ A`` for *param_prefix*, or ``None``."""
        pair = self._pinned_ab.get(param_prefix)
        if pair is None:
            return None
        a, b = pair
        if device is not None and device.type == "cuda":
            a = a.to(device=device)
            b = b.to(device=device)
        delta = torch.matmul(b * self.strength, a)
        return delta

    def cleanup(self) -> None:
        self._pinned_ab.clear()
