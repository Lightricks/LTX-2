"""Two-stage lip-dubbing pipeline with IC-LoRA and appended audio reference conditioning."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.conditioning import AudioConditionByReferenceLatent
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.video_vae import TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, SpatioTemporalScaleFactors, VideoPixelShape
from ltx_pipelines.iclora_utils import (
    append_ic_lora_reference_video_conditionings,
    read_lora_reference_downscale_factor,
)
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    detect_checkpoint_path,
    lipdub_arg_parser,
)
from ltx_pipelines.utils.blocks import (
    AudioConditioner,
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS, detect_params
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings, get_device
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video, get_videostream_metadata
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


def _snap_frames_to_8k1(frames: int) -> int:
    """Round ``frames`` down to the nearest ``8k+1`` (the model's required frame count)."""
    time_scale = SpatioTemporalScaleFactors.default().time
    return ((frames - 1) // time_scale) * time_scale + 1


class LipDubPipeline:
    """Two-stage lip-dubbing with IC-LoRA video reference and appended audio reference tokens."""

    def __init__(
        self,
        distilled_checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        ic_lora: LoraPathStrengthAndSDOps,
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
        offload_mode: OffloadMode = OffloadMode.NONE,
    ) -> None:
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.ic_lora = ic_lora
        loras = (ic_lora,)

        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
        )
        self.image_conditioner = ImageConditioner(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_conditioner = AudioConditioner(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
        )
        self.stage = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=loras,
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
            offload_mode=offload_mode,
        )
        self.upsampler = VideoUpsampler(
            distilled_checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.reference_downscale_factor = read_lora_reference_downscale_factor(ic_lora.path)

    def _create_stage_conditionings(
        self,
        images: list[ImageConditioningInput],
        reference_video_path: str,
        reference_strength: float,
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        encode_tiling: TilingConfig | None,
    ) -> list:
        conditionings = combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )
        append_ic_lora_reference_video_conditionings(
            conditionings,
            [(reference_video_path, reference_strength)],
            height=height,
            width=width,
            num_frames=num_frames,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
            reference_downscale_factor=self.reference_downscale_factor,
            conditioning_attention_strength=1.0,
            conditioning_attention_mask=None,
            tiling_config=encode_tiling,
        )
        return conditionings

    def _encode_reference_audio_vae_latent(self, video_path: str) -> torch.Tensor:
        audio = decode_audio_from_file(video_path, self.device)
        if audio is None:
            msg = f"No audio stream found in {video_path}"
            raise ValueError(msg)
        return self.audio_conditioner(lambda enc: vae_encode_audio(audio, enc, None))

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        images: list[ImageConditioningInput],
        reference_video_path: str,
        reference_strength: float = 1.0,
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        meta = get_videostream_metadata(reference_video_path)
        num_frames = _snap_frames_to_8k1(meta.frames)
        frame_rate = float(meta.fps)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        encode_tiling = TilingConfig.default()

        def build_image_conditionings(output_shape: VideoPixelShape) -> list:
            return self.image_conditioner(
                lambda enc: self._create_stage_conditionings(
                    images=images,
                    reference_video_path=reference_video_path,
                    reference_strength=reference_strength,
                    height=output_shape.height,
                    width=output_shape.width,
                    num_frames=num_frames,
                    video_encoder=enc,
                    encode_tiling=encode_tiling,
                )
            )

        def build_audio_ref_conditioning(audio_latent: torch.Tensor) -> AudioConditionByReferenceLatent:
            ref_patch, ref_pos = patchify_lipdub_audio_reference_latent(
                audio_latent,
                negative_positions=True,
                device=self.device,
            )
            return AudioConditionByReferenceLatent(ref_patch, ref_pos, strength=1.0)

        stage_1_conditionings = build_image_conditionings(stage_1_output_shape)

        ref_vae = self._encode_reference_audio_vae_latent(reference_video_path)
        audio_conditionings = [build_audio_ref_conditioning(ref_vae)]

        stage_1_sigmas_tensor = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas_tensor,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_1_conditionings,
            ),
            audio=ModalitySpec(
                context=audio_context,
                conditionings=audio_conditionings,
            ),
        )

        s1_audio_latent = audio_state.latent.clone()

        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_sigmas_tensor = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = build_image_conditionings(stage_2_output_shape)

        stage_2_audio_conditionings = [build_audio_ref_conditioning(s1_audio_latent)]

        video_state, _audio_unused = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas_tensor,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas_tensor[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                conditionings=stage_2_audio_conditionings,
                frozen=True,
                noise_scale=0.0,
                initial_latent=s1_audio_latent,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(s1_audio_latent)
        return decoded_video, decoded_audio


def patchify_lipdub_audio_reference_latent(
    vae_latents: torch.Tensor,
    *,
    negative_positions: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchify audio VAE latents and build RoPE positions (optional negative shift for reference)."""
    patchifier = AudioPatchifier(patch_size=1)
    patchified = patchifier.patchify(vae_latents)
    b, c, _t, mel_bins = vae_latents.shape
    seq_len = patchified.shape[1]
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=AudioLatentShape(batch=b, channels=c, frames=seq_len, mel_bins=mel_bins),
        device=device,
    )
    positions = latent_coords.to(dtype=torch.float32)
    if negative_positions:
        aud_dur = positions[:, :, -1, 1].max().item()
        positions = positions - aud_dur - 0.04
    return patchified, positions


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = lipdub_arg_parser(params=params)
    args = parser.parse_args()

    if not args.lora or len(args.lora) != 1:
        raise ValueError("LipDub requires exactly one --lora (the lip-dub IC-LoRA).")

    pipeline = LipDubPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        ic_lora=args.lora[0],
        quantization=args.quantization,
        torch_compile=args.compile,
        offload_mode=args.offload_mode,
    )
    tiling_config = TilingConfig.default()
    src = get_videostream_metadata(args.reference_video)
    video_chunks_number = get_video_chunks_number(_snap_frames_to_8k1(src.frames), tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        images=[],
        reference_video_path=args.reference_video,
        reference_strength=args.reference_strength,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )
    encode_video(
        video=video,
        fps=int(src.fps),
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
