import logging
from collections.abc import Iterator

import torch
import torchaudio

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.types import AudioConditionByLatent
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import AudioProcessor, decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_arg_parser
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_replacing_latent,
    multi_modal_guider_denoising_func,
    simple_denoising_func,
    video_conditionings_by_replacing_latent,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at the target resolution with CFG guidance, then
    Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output. Supports optional image conditioning via the images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: str = device,
        fp8transformer: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            fp8transformer=fp8transformer,
        )

        self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
            loras=distilled_lora,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def _build_audio_conditionings_from_waveform(
        self,
        input_waveform: torch.Tensor,
        input_sample_rate: int,
        num_frames: int,
        fps: float,
        strength: float,
    ) -> list[AudioConditionByLatent] | None:
        """Convert input waveform to audio conditioning for the diffusion process.

        Args:
            input_waveform: Audio waveform tensor, shape (samples,), (C, samples), or (B, C, samples).
            input_sample_rate: Sample rate of input waveform.
            num_frames: Number of video frames (for duration alignment).
            fps: Video frame rate.
            strength: Conditioning strength (0-1). 1.0 = fully preserve input audio.

        Returns:
            List containing AudioConditionByLatent, or None if strength <= 0.
        """
        strength = float(strength)
        if strength <= 0.0:
            return None

        # Normalize waveform shape to (B, C, T)
        waveform = input_waveform
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # (T,) -> (1, 1, T)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # (C, T) -> (1, C, T)
        elif waveform.ndim != 3:
            raise ValueError(f"input_waveform must be 1D/2D/3D, got shape {tuple(waveform.shape)}")

        # Get audio encoder and extract config
        audio_encoder = self.stage_1_model_ledger.audio_encoder()
        target_sr = int(getattr(audio_encoder, "sample_rate", 16000))
        target_channels = int(getattr(audio_encoder, "in_channels", 1))
        mel_bins = int(getattr(audio_encoder, "mel_bins", 64))
        mel_hop = int(getattr(audio_encoder, "mel_hop_length", 160))
        n_fft = int(getattr(audio_encoder, "n_fft", 1024))

        # Match channels
        if waveform.shape[1] != target_channels:
            if waveform.shape[1] == 1 and target_channels > 1:
                waveform = waveform.repeat(1, target_channels, 1)
            elif target_channels == 1:
                waveform = waveform.mean(dim=1, keepdim=True)
            else:
                waveform = waveform[:, :target_channels, :]

        # Resample to target sample rate
        waveform = waveform.to(device="cpu", dtype=torch.float32)
        if int(input_sample_rate) != target_sr:
            waveform = torchaudio.functional.resample(waveform, int(input_sample_rate), target_sr)

        # Waveform -> Mel spectrogram
        audio_processor = AudioProcessor(
            sample_rate=target_sr,
            mel_bins=mel_bins,
            mel_hop_length=mel_hop,
            n_fft=n_fft,
        ).to(waveform.device)
        mel = audio_processor.waveform_to_mel(waveform, target_sr)

        # Mel -> Audio latent via encoder
        audio_params = next(audio_encoder.parameters(), None)
        enc_device = audio_params.device if audio_params is not None else self.device
        enc_dtype = audio_params.dtype if audio_params is not None else self.dtype

        mel = mel.to(device=enc_device, dtype=enc_dtype)
        with torch.inference_mode():
            audio_latent = audio_encoder(mel)

        # For video extension: DON'T pad to full output duration
        # Only condition on the input audio portion, let the model generate continuation
        # If input audio is longer than output video, trim it
        audio_downsample = getattr(
            getattr(audio_encoder, "patchifier", None), "audio_latent_downsample_factor", 4
        )
        target_shape = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(
                batch=audio_latent.shape[0],
                frames=int(num_frames),
                width=1,
                height=1,
                fps=float(fps),
            ),
            channels=audio_latent.shape[1],
            mel_bins=audio_latent.shape[3],
            sample_rate=target_sr,
            hop_length=mel_hop,
            audio_latent_downsample_factor=audio_downsample,
        )
        max_frames = int(target_shape.frames)

        # Only trim if audio is longer than output video - do NOT pad
        # This way, conditioning only applies to input audio duration
        if audio_latent.shape[2] > max_frames:
            audio_latent = audio_latent[:, :, :max_frames, :]

        audio_latent = audio_latent.to(device=self.device, dtype=self.dtype)
        return [AudioConditionByLatent(audio_latent, strength)]

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        image_crf: float | None = None,
        skip_cleanup: bool = False,
        video_extend_path: str | None = None,
        video_extend_strength: float = 1.0,
        video_extend_context_seconds: float | None = None,
        input_waveform: torch.Tensor | None = None,
        input_waveform_sample_rate: int | None = None,
        audio_strength: float = 1.0,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        # Build audio conditionings from input waveform if provided
        audio_conditionings = None
        if input_waveform is not None:
            if input_waveform_sample_rate is None:
                raise ValueError("input_waveform_sample_rate must be provided when input_waveform is set.")
            audio_conditionings = self._build_audio_conditionings_from_waveform(
                input_waveform=input_waveform,
                input_sample_rate=input_waveform_sample_rate,
                num_frames=num_frames,
                fps=frame_rate,
                strength=audio_strength,
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        text_encoder = self.stage_1_model_ledger.text_encoder()
        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder, prompt, images[0][0] if len(images) > 0 else None, seed=seed
            )
        context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        if not skip_cleanup:
            torch.cuda.synchronize()
            del text_encoder
            cleanup_memory()

        # Stage 1: Initial low resolution video generation.
        video_encoder = self.stage_1_model_ledger.video_encoder()
        transformer = self.stage_1_model_ledger.transformer()
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            image_crf=image_crf,
        )

        # Add video extension conditioning if provided
        if video_extend_path:
            video_conds = video_conditionings_by_replacing_latent(
                video_path=video_extend_path,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                strength=video_extend_strength,
                context_seconds=video_extend_context_seconds,
                frame_rate=frame_rate,
            )
            stage_1_conditionings.extend(video_conds)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            audio_conditionings=audio_conditionings,
        )

        if not skip_cleanup:
            torch.cuda.synchronize()
            del transformer
            cleanup_memory()

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )

        if not skip_cleanup:
            torch.cuda.synchronize()
            cleanup_memory()

        transformer = self.stage_2_model_ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            image_crf=image_crf,
        )

        # Add video extension conditioning for stage 2 as well
        if video_extend_path:
            video_conds = video_conditionings_by_replacing_latent(
                video_path=video_extend_path,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                strength=video_extend_strength,
                context_seconds=video_extend_context_seconds,
                frame_rate=frame_rate,
            )
            stage_2_conditionings.extend(video_conds)

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
            audio_conditionings=audio_conditionings,
        )

        if not skip_cleanup:
            torch.cuda.synchronize()
            del transformer
            del video_encoder
            cleanup_memory()

        decoded_video = vae_decode_video(
            video_state.latent, self.stage_2_model_ledger.video_decoder(), tiling_config, generator
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.stage_2_model_ledger.audio_decoder(), self.stage_2_model_ledger.vocoder()
        )

        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
