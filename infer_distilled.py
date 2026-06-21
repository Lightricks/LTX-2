#!/usr/bin/env python3
"""Standalone distilled LTX-2.3 inference using ltx-pipelines.

Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (official recommendation).
Real-time VRAM reporting for each component load/free.
Monkey-patches encode_prompts for aggressive GPU cleanup between stages.

Usage:
    python infer_distilled.py \\
        --distilled-checkpoint-path /path/to/distilled-fp8.safetensors \\
        --gemma-root /path/to/gemma \\
        --spatial-upsampler-path /path/to/upscaler.safetensors \\
        --prompt "A cat running" --output /tmp/output.mp4
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video


# ── VRAM helpers ──────────────────────────────────────────────────

def _vram_info() -> tuple[int, int]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return free, total


def _vram_status() -> str:
    free, total = _vram_info()
    used = total - free
    return f"{used / 1024**3:.1f} / {total / 1024**3:.1f} GB"


def _vram_used_gb() -> float:
    free, total = _vram_info()
    return (total - free) / 1024**3


def _wrap_model_method(ledger, name: str, label: str):
    orig = getattr(ledger, name)

    def wrapped(*args, **kwargs):
        free_before = _vram_info()[0]
        result = orig(*args, **kwargs)
        torch.cuda.synchronize()
        free_after, total = _vram_info()
        delta = (free_before - free_after) / 1024**3
        used_now = (total - free_after) / 1024**3
        if delta > 0.1:
            print(f"  [load]  {label:30s}  +{delta:.1f} GB  →  {used_now:.1f} / {total / 1024**3:.1f} GB used")
        return result

    setattr(ledger, name, wrapped)


# ── CLI ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LTX-2.3 Distilled Inference")
    p.add_argument("--distilled-checkpoint-path", type=str, required=True)
    p.add_argument("--gemma-root", type=str, required=True)
    p.add_argument("--spatial-upsampler-path", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--frame-rate", type=float, default=24.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("LTX-2.3 Distilled Inference (via ltx-pipelines)")
    print("=" * 70)
    print(f"Prompt:      {args.prompt}")
    print(f"Resolution:  {args.width}x{args.height}")
    print(f"Frames:      {args.num_frames} @ {args.frame_rate} fps")
    print(f"Seed:        {args.seed}")
    print(f"GPU VRAM:    {_vram_status()} used")
    print("=" * 70)

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        gemma_root=args.gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=[],
    )

    ledger = pipeline.model_ledger

    _wrap_model_method(ledger, "text_encoder", "text_encoder (BF16)")
    _wrap_model_method(ledger, "gemma_embeddings_processor", "embeddings_processor")
    _wrap_model_method(ledger, "video_encoder", "video_encoder")
    _wrap_model_method(ledger, "transformer", "transformer (FP8)")
    _wrap_model_method(ledger, "spatial_upsampler", "spatial_upsampler")
    _wrap_model_method(ledger, "video_decoder", "video_decoder")
    _wrap_model_method(ledger, "audio_decoder", "audio_decoder")
    _wrap_model_method(ledger, "vocoder", "vocoder")

    from ltx_pipelines.utils import helpers as _helpers
    import ltx_pipelines.distilled as _distilled

    # The distilled FP8 checkpoint has FP8 transformer weights. ModelLedger
    # passes dtype=torch.bfloat16 which upcasts FP8→BF16 (doubles memory).
    # Keep FP8 dtype — Blackwell (sm_120) supports FP8 matmul natively.
    _orig_transformer = ledger.transformer

    def _transformer_fp8():
        free_before = _vram_info()[0]
        from ltx_core.model.transformer import X0Model
        from ltx_core.quantization.fp8_cast import _amend_forward_with_upcast
        x0 = X0Model(
            ledger.transformer_builder.build(device=ledger._target_device(), dtype=None)
        ).to(ledger.device).eval()
        # FP8 weights need on-the-fly upcast: replace Linear.forward
        # to upcast weight to input dtype before F.linear.
        _amend_forward_with_upcast(x0.velocity_model)
        torch.cuda.synchronize()
        free_after, total = _vram_info()
        delta = (free_before - free_after) / 1024**3
        used_now = (total - free_after) / 1024**3
        print(f"  [load]  {'transformer (FP8)':30s}  +{delta:.1f} GB  →  {used_now:.1f} / {total / 1024**3:.1f} GB used")
        return x0

    ledger.transformer = _transformer_fp8  # type: ignore[method-assign]

    # Monkey-patch encode_prompts: same logic, but with aggressive GPU cleanup.
    # Must patch BOTH helpers (for other callers) AND distilled (for DistilledPipeline).
    # 'from helpers import encode_prompts' creates a local binding that our
    # helpers monkey-patch doesn't override.
    _orig_encode_prompts = _helpers.encode_prompts

    def _encode_prompts_clean(*args, **kwargs):
        import gc as _gc
        # Wrap in no_grad: output tensors won't hold grad_fn references
        # to model parameters → model can actually be freed by del.
        with torch.no_grad():
            result = _orig_encode_prompts(*args, **kwargs)
        for _ in range(3):
            _gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Diagnose: find all CUDA tensors still alive
        cuda_tensors = []
        for obj in _gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    cuda_tensors.append((obj.dtype, tuple(obj.shape), obj.element_size() * obj.numel()))
            except Exception:
                pass

        total = sum(sz for _, _, sz in cuda_tensors)
        print(f"\n  [diag] {len(cuda_tensors)} CUDA tensors alive, total {total/1024**3:.2f} GB")
        # Group by shape pattern
        from collections import Counter
        shape_counts = Counter(shape for _, shape, _ in cuda_tensors)
        for shape, cnt in shape_counts.most_common(10):
            total_sz = sum(sz for s, sz in [(s, z) for _, s, z in cuda_tensors if s == shape])
            print(f"    {cnt:4d} × {str(shape):40s}  {total_sz/1024**2:.1f} MB")

        free, total = _vram_info()
        print(f"  [free]  encode_prompts done  →  {(total-free)/1024**3:.1f} / {total/1024**3:.1f} GB used\n")
        return result

    _helpers.encode_prompts = _encode_prompts_clean  # type: ignore[attr-defined]
    _distilled.encode_prompts = _encode_prompts_clean  # type: ignore[attr-defined]

    # VAE decoder output has a negative bias on this PyTorch/CUDA version.
    # Shift raw output by +0.65 before the standard +1/2 → [0,1] mapping.
    # Without this, decoded video is very dark (mean ~47 instead of ~128).
    import ltx_core.model.video_vae.video_vae as _vvae_mod
    _orig_decode_video_fn = _vvae_mod.decode_video

    def _decode_video_fixed(latent, video_decoder, tiling_config=None, generator=None):
        from einops import rearrange
        def convert(frames):
            frames = (((frames + 1.65) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            return rearrange(frames[0], "c f h w -> f h w c")
        if tiling_config is not None:
            for frames in video_decoder.tiled_decode(latent, tiling_config, generator=generator):
                yield convert(frames)
        else:
            yield convert(video_decoder(latent, generator=generator))

    _vvae_mod.decode_video = _decode_video_fixed

    # Video encoder is loaded for conditioning but stays on GPU during denoising.
    # Move to CPU after conditioning, back to GPU before upsampling in stage 2.
    _orig_combined_cond = _helpers.combined_image_conditionings

    def _combined_cond_offload(*args, **kwargs):
        result = _orig_combined_cond(*args, **kwargs)
        ve = kwargs.get("video_encoder") or (args[2] if len(args) > 2 else None)
        if ve is not None:
            ve.to("cpu")
            torch.cuda.empty_cache()
            print("  [free]  video_encoder → CPU")
        return result

    _helpers.combined_image_conditionings = _combined_cond_offload  # type: ignore[attr-defined]
    _distilled.combined_image_conditionings = _combined_cond_offload  # type: ignore[attr-defined]

    _orig_upsample_video = _distilled.upsample_video

    def _upsample_video_reload(*args, **kwargs):
        ve = kwargs.get("video_encoder") or (args[1] if len(args) > 1 else None)
        if ve is not None:
            ve.to("cuda")
            print("  [load]  video_encoder → GPU (for upsampling)")
        return _orig_upsample_video(*args, **kwargs)

    _distilled.upsample_video = _upsample_video_reload  # type: ignore[attr-defined]

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    print("\n▸ Generating (2-stage: low-res → upscale → refine)...\n")
    with torch.no_grad():
        video, audio = pipeline(
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=[],
            tiling_config=tiling_config,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Aggressive cleanup: transformer del + empty_cache might not be enough
    import gc as _gc
    for _ in range(5):
        _gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"\n  GPU VRAM after cleanup: {_vram_status()} used")
    print(f"\n▸ Saving to {args.output}...")
    with torch.no_grad():
        encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=video_chunks_number,
    )

    print(f"\n  GPU VRAM: {_vram_status()} used")
    print(f"✓ Done → {args.output}")


if __name__ == "__main__":
    main()
