"""Caption latents stored trimmed to their real tokens and padded back on load give embeddings bit-equal to the
1024-token file (whose pad rows hold the projection bias, not zeros), in a mixed-length batch."""

import torch
from torch.utils.data import default_collate

from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_trainer.datasets import pad_prompt_latents, trim_prompt_latents

N, DIM = 1024, 16


def _stored(n_real: int) -> dict:
    """A 1024-token caption latent: left-padded, pad rows = a constant (the aggregate_embed bias)."""
    mask = torch.zeros(N, dtype=torch.int64)
    mask[-n_real:] = 1
    v = torch.randn(N, DIM, dtype=torch.bfloat16)
    a = torch.randn(N, DIM, dtype=torch.bfloat16)
    v[:-n_real], a[:-n_real] = 0.37, -1.25
    return {"video_prompt_embeds": v, "audio_prompt_embeds": a, "prompt_attention_mask": mask}


def _connector() -> Embeddings1DConnector:
    return Embeddings1DConnector(attention_head_dim=8, num_attention_heads=2, num_layers=1)


def _embed(proc: EmbeddingsProcessor, b: dict) -> tuple:
    mask = convert_to_additive_mask(b["prompt_attention_mask"], b["video_prompt_embeds"].dtype)
    return proc.create_embeddings(b["video_prompt_embeds"], b["audio_prompt_embeds"], mask)


def test_trimmed_padded_matches_stored() -> None:
    torch.manual_seed(0)
    proc = EmbeddingsProcessor(video_connector=_connector(), audio_connector=_connector()).to(torch.bfloat16).eval()
    full = [_stored(223), _stored(700), _stored(256)]
    trimmed = [trim_prompt_latents(d) for d in full]
    # a multiple of 128 keeps one pad row, so code without pad_prompt_latents trips the connector's % 128 assert
    assert [t["prompt_attention_mask"].shape[-1] for t in trimmed] == [223, 700, 257]
    padded = [pad_prompt_latents(t) for t in trimmed]
    for d, p in zip(full, padded, strict=True):
        assert torch.equal(p["prompt_attention_mask"], d["prompt_attention_mask"])
    assert pad_prompt_latents(full[0]) is full[0]  # a 1024-token file is untouched
    with torch.inference_mode():
        want, got = _embed(proc, default_collate(full)), _embed(proc, default_collate(padded))
    for w, g in zip(want, got, strict=True):
        assert torch.equal(w, g)
