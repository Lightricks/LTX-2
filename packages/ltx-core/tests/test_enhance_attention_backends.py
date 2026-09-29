"""Prompt enhancement must not sample through the cuDNN SDPA backend, whose output varies per call."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from ltx_core.text_encoders.gemma.encoders import base_encoder
from ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder


class _Inputs(dict):
    def __getattr__(self, name: str) -> torch.Tensor:
        return self[name]

    def to(self, _device: object) -> "_Inputs":
        return self


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(
            pad_token_id=0,
            apply_chat_template=lambda *_a, **_k: "text",
            decode=lambda ids, **_k: " ".join(str(int(i)) for i in ids),
        )

    def __call__(self, **_kwargs: object) -> _Inputs:
        ids = torch.ones(1, 8, dtype=torch.long)
        return _Inputs(input_ids=ids, attention_mask=torch.ones_like(ids))


class _Model:
    def __init__(self, device_type: str) -> None:
        self.device = SimpleNamespace(type=device_type)
        self.config = SimpleNamespace(model_type="gemma3")
        self.backends_seen: dict[str, bool] = {}

    def generate(self, input_ids: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        self.backends_seen = {
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        }
        return torch.cat([input_ids, torch.full((1, 2), 7)], dim=1)


def _encoder(device_type: str) -> tuple[LTXGemmaTextEncoder, _Model]:
    model = _Model(device_type)
    encoder = LTXGemmaTextEncoder.__new__(LTXGemmaTextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.__dict__.update(model=model, processor=_Processor(), tokenizer=None, _dtype=torch.bfloat16)
    return encoder, model


def test_cuda_enhancement_excludes_cudnn_sdpa(monkeypatch: object) -> None:
    monkeypatch.setattr(base_encoder.torch.random, "fork_rng", lambda **_k: nullcontext())
    encoder, model = _encoder("cuda")
    assert encoder._enhance([{"role": "user", "content": "a cat"}], seed=3) == "7 7"
    assert model.backends_seen == {"cudnn": False, "flash": True, "mem_efficient": True, "math": True}
    assert torch.backends.cuda.cudnn_sdp_enabled(), "the backend restriction must not leak past generation"


def test_cpu_enhancement_leaves_sdpa_backends_alone() -> None:
    encoder, model = _encoder("cpu")
    encoder._enhance([{"role": "user", "content": "a cat"}], seed=3)
    assert model.backends_seen["cudnn"] == torch.backends.cuda.cudnn_sdp_enabled()
