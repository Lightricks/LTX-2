from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import LatentTools
from ltx_core.types import (
    Audio,
    AudioLatentShape,
    LatentState,
    SpatioTemporalScaleFactors,
    VideoPixelShape,
)
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import QuantizationAction
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, detect_params
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    encode_prompts,
    get_device,
    multi_modal_guider_denoising_func,
    noise_audio_state,
    noise_video_state,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import (
    decode_audio_from_file,
    encode_video,
    get_videostream_metadata,
    load_video_conditioning,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


def _encode_video_for_retake(
    video_encoder: torch.nn.Module,
    video_path: str,
    output_shape: VideoPixelShape,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Load video and encode to latents."""
    pixel_video = load_video_conditioning(
        video_path=video_path,
        height=output_shape.height,
        width=output_shape.width,
        frame_cap=output_shape.frames,
        dtype=dtype,
        device=device,
    )  # (1, C, F, H, W)
    return video_encoder(pixel_video)


def _encode_audio_for_retake(
    audio_encoder: torch.nn.Module,
    waveform: torch.Tensor,
    waveform_sr: int,
    output_shape: VideoPixelShape,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode audio to latents and trim/pad to match output_shape."""
    waveform_batch = waveform.unsqueeze(0) if waveform.dim() == 2 else waveform
    initial_audio_latent = vae_encode_audio(
        Audio(waveform=waveform_batch.to(dtype), sampling_rate=waveform_sr), audio_encoder, None
    )
    expected_audio_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
    expected_frames = expected_audio_shape.frames
    actual_frames = initial_audio_latent.shape[2]
    if actual_frames > expected_frames:
        initial_audio_latent = initial_audio_latent[:, :, :expected_frames, :]
    elif actual_frames < expected_frames:
        pad = torch.zeros(
            initial_audio_latent.shape[0],
            initial_audio_latent.shape[1],
            expected_frames - actual_frames,
            initial_audio_latent.shape[3],
            device=initial_audio_latent.device,
            dtype=initial_audio_latent.dtype,
        )
        initial_audio_latent = torch.cat([initial_audio_latent, pad], dim=2)
    return initial_audio_latent


# ---------------------------------------------------------------------------
# Custom conditioning item: temporal region mask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalRegionMask:
    """Conditioning item that sets ``denoise_mask = 0`` outside a time range
    and ``1`` inside, so only the specified temporal region is regenerated.
    Uses ``start_time`` and ``end_time`` in seconds. Works in *patchified*
    (token) space using the patchifier's ``get_patch_grid_bounds``: for video
    coords are latent frame indices (converted from seconds via ``fps``), for
    audio coords are already in seconds.
    """

    start_time: float  # seconds, inclusive
    end_time: float  # seconds, exclusive
    fps: float

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        coords = latent_tools.patchifier.get_patch_grid_bounds(
            latent_tools.target_shape, device=latent_state.denoise_mask.device
        )
        # coords: [B, 3, N, 2] (video) or [B, 1, N, 2] (audio); temporal dim is index 0
        if coords.shape[1] == 1:
            # Audio: patchifier returns seconds
            t_start = coords[:, 0, :, 0]  # [B, N]
            t_end = coords[:, 0, :, 1]  # [B, N]
            in_region = (t_end > self.start_time) & (t_start < self.end_time)
        else:
            # Video: get pixel bounds per patch, find patches for start/end frame, read latent from coords.
            scale_factors = getattr(latent_tools, "scale_factors", SpatioTemporalScaleFactors.default())
            pixel_bounds = get_pixel_coords(coords, scale_factors, causal_fix=getattr(latent_tools, "causal_fix", True))
            timestamp_bounds = pixel_bounds[0, 0] / self.fps
            t_start, t_end = timestamp_bounds.unbind(dim=-1)
            in_region = (t_end > self.start_time) & (t_start < self.end_time)
        state = latent_state.clone()
        mask_val = in_region.to(state.denoise_mask.dtype)
        if state.denoise_mask.dim() == 3:
            mask_val = mask_val.unsqueeze(-1)
        state.denoise_mask.copy_(mask_val)
        return state


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RetakePipeline:
    """Regenerate a time region (retake) of an existing video.
    Given a source video file and a time window ``[start_time, end_time]``
    (in seconds), this pipeline keeps the video/audio outside that window
    unchanged and *regenerates* the content inside the window from a text
    prompt using the LTX-2 diffusion model.
    Parameters
    ----------
    checkpoint_path : str
        Path to the LTX-2 model checkpoint.
    gemma_root : str
        Root directory containing Gemma text-encoder weights.
    loras : list[LoraPathStrengthAndSDOps]
        Optional LoRA configs applied to the transformer.
    device : torch.device
        Target device (default: CUDA if available).
    quantization : QuantizationPolicy | None
        Optional quantization policy for the transformer.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
        )
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    # --------------------------------------------------------------------- #
    #  Public entry point                                                     #
    # --------------------------------------------------------------------- #

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913, PLR0915
        self,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        *,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = False,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        """Regenerate ``[start_time, end_time]`` of the source video (retake).
        Parameters
        ----------
        video_path : str
            Path to the source video file (must contain video; audio is optional).
        prompt : str
            Text prompt describing the *regenerated* section.
        start_time, end_time : float
            Time window (in seconds) of the section to regenerate.
        seed : int
            Random seed for reproducibility.
        negative_prompt : str
            Negative prompt for CFG guidance (ignored in distilled mode).
        num_inference_steps : int
            Number of Euler denoising steps (ignored in distilled mode which
            uses a fixed 8-step schedule).
        video_guider_params, audio_guider_params : MultiModalGuiderParams | None
            Guidance parameters for video and audio modalities.  Ignored in
            distilled mode.
        regenerate_video : bool
            If ``True`` (default), preserve video outside ``[start_time, end_time]``
            and only regenerate the masked region.  If ``False``, fully regenerate
            all video frames (the encoded video is still used as the initial latent
            but with ``denoise_mask = 1`` everywhere).
        regenerate_audio : bool
            If True, regenerate audio in the [start_time, end_time] window; if False,
            audio is preserved as-is (no regeneration).
        enhance_prompt : bool
            Whether to enhance the prompt via the text encoder.
        distilled : bool
            If ``True``, use the distilled sigma schedule
            (``DISTILLED_SIGMA_VALUES``) and a simple (non-guided) denoising
            function.  The model checkpoint must be the distilled variant.
        Returns
        -------
        tuple[Iterator[torch.Tensor], torch.Tensor]
            ``(video_frames_iterator, audio_waveform)``
        """
        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        effective_seed = torch.randint(0, 2**31, (1,), device=self.device).item() if seed < 0 else seed
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = self.dtype

        video_encoder = self.model_ledger.video_encoder()

        # Use av to get metadata
        fps, num_pixel_frames, src_width, src_height = get_videostream_metadata(video_path)

        output_shape = VideoPixelShape(
            batch=1,
            frames=num_pixel_frames,
            width=src_width,
            height=src_height,
            fps=fps,
        )
        initial_video_latent = _encode_video_for_retake(
            video_encoder=video_encoder,
            video_path=video_path,
            output_shape=output_shape,
            dtype=dtype,
            device=self.device,
        )
        video_conditionings: list[ConditioningItem] = [
            TemporalRegionMask(
                start_time=start_time if regenerate_video else 0.0,
                end_time=end_time if regenerate_video else 0.0,
                fps=fps,
            )
        ]
        del video_encoder
        cleanup_memory()

        initial_audio_latent: torch.Tensor | None = None
        audio_conditionings: list[ConditioningItem] = []

        audio_in = decode_audio_from_file(video_path, self.device)
        audio_encoder = self.model_ledger.audio_encoder()

        if audio_in is not None:
            waveform = audio_in.waveform.squeeze(0)
            waveform_sr = audio_in.sampling_rate
        else:
            waveform, waveform_sr = None, None
        if waveform is not None:
            initial_audio_latent = _encode_audio_for_retake(
                audio_encoder=audio_encoder,
                waveform=waveform,
                waveform_sr=waveform_sr,
                output_shape=output_shape,
                dtype=dtype,
            )
            audio_conditionings = [
                TemporalRegionMask(
                    start_time=start_time if regenerate_audio else 0.0,
                    end_time=end_time if regenerate_audio else 0.0,
                    fps=fps,
                )
            ]

        del audio_encoder
        cleanup_memory()

        prompts_to_encode = [prompt] if distilled else [prompt, negative_prompt]
        contexts = encode_prompts(
            prompts_to_encode,
            self.model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_seed=effective_seed,
        )

        v_context_p, a_context_p = contexts[0].video_encoding, contexts[0].audio_encoding
        if not distilled:
            v_context_n, a_context_n = contexts[1].video_encoding, contexts[1].audio_encoding

        transformer = self.model_ledger.transformer()

        sigmas = (
            torch.tensor(DISTILLED_SIGMA_VALUES) if distilled else LTX2Scheduler().execute(steps=num_inference_steps)
        ).to(dtype=torch.float32, device=self.device)
        if distilled:
            denoise_fn = simple_denoising_func(
                video_context=v_context_p,
                audio_context=a_context_p,
                transformer=transformer,
            )
        else:
            video_guider = MultiModalGuider(
                params=video_guider_params,
                negative_context=v_context_n,
            )
            audio_guider = MultiModalGuider(
                params=audio_guider_params,
                negative_context=a_context_n,
            )
            denoise_fn = multi_modal_guider_denoising_func(
                video_guider,
                audio_guider,
                v_context=v_context_p,
                a_context=a_context_p,
                transformer=transformer,
            )

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=denoise_fn,
            )

        # Build noised states with the encoded latents as initial values and
        # the temporal masks applied via conditionings.
        video_state, video_tools = noise_video_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=video_conditionings,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_latent=initial_video_latent,
        )
        audio_state, audio_tools = noise_audio_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=audio_conditionings,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_latent=initial_audio_latent,
        )

        video_state, audio_state = denoising_loop(sigmas, video_state, audio_state, stepper)

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        decoded_video = vae_decode_video(
            video_state.latent, self.model_ledger.video_decoder(), tiling_config, generator
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.model_ledger.audio_decoder(), self.model_ledger.vocoder()
        )

        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    """CLI entry point for retake (regenerate a time region)."""
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Retake: regenerate a time region of a video with LTX-2.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to the source video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the regenerated region.")
    parser.add_argument("--start-time", type=float, required=True, help="Start time of the region to regenerate (s).")
    parser.add_argument("--end-time", type=float, required=True, help="End time of the region to regenerate (s).")
    parser.add_argument("--output-path", type=str, required=True, help="Path for the output video.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the LTX-2 checkpoint.")
    parser.add_argument("--gemma-root", type=str, required=True, help="Path to Gemma text encoder weights.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Use -1 for a random seed.")
    parser.add_argument("--loras", nargs="*", default=[], help="LoRA paths (optional).")
    parser.add_argument(
        "--quantization",
        dest="quantization",
        action=QuantizationAction,
        nargs="+",
        metavar=("POLICY", "AMAX_PATH"),
        default=None,
        help="Quantization policy: fp8-cast or fp8-scaled-mm [AMAX_PATH].",
    )
    args = parser.parse_args()

    if args.start_time >= args.end_time:
        raise ValueError("start_time must be less than end_time")

    # Validate frame count (8k+1) and resolution (multiples of 32) at CLI stage
    video_scale = SpatioTemporalScaleFactors.default()
    fps, num_frames, width, height = get_videostream_metadata(args.video_path)
    if (num_frames - 1) % video_scale.time != 0:
        snapped = ((num_frames - 1) // video_scale.time) * video_scale.time + 1
        raise ValueError(
            f"Video frame count must satisfy 8k+1 (e.g. 97, 193). Got {num_frames}; use a video with {snapped} frames."
        )
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError(f"Video width and height must be multiples of 32. Got {width}x{height}.")

    pipeline = RetakePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.loras) if args.loras else (),
        quantization=args.quantization,
    )
    params = detect_params(args.checkpoint_path)
    tiling_config = TilingConfig.default()
    video_iter, audio = pipeline(
        video_path=args.video_path,
        prompt=args.prompt,
        start_time=args.start_time,
        end_time=args.end_time,
        seed=args.seed,
        video_guider_params=params.video_guider_params,
        audio_guider_params=params.audio_guider_params,
        tiling_config=tiling_config,
    )
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    encode_video(
        video=video_iter,
        fps=int(fps),
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
