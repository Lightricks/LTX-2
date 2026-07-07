import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.utils.constants import (
    DEFAULT_IMAGE_CRF,
    DEFAULT_LORA_STRENGTH,
    DEFAULT_NEGATIVE_PROMPT,
    LTX_2_3_HQ_PARAMS,
    LTX_2_3_PARAMS,
    PipelineParams,
    detect_params,
)
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode


class ImageConditioningInput(NamedTuple):
    path: str
    frame_idx: int
    strength: float
    crf: int = DEFAULT_IMAGE_CRF


class VideoConditioningAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        path, strength_str = values
        resolved_path = resolve_existing_path(path)
        strength = float(strength_str)
        current = getattr(namespace, self.dest) or []
        current.append((resolved_path, strength))
        setattr(namespace, self.dest, current)


class VideoMaskConditioningAction(argparse.Action):
    """Parse ``--conditioning-attention-mask PATH STRENGTH``.
    Stores a ``(mask_path, strength)`` tuple on the namespace.  The mask video
    should be grayscale with pixel values in [0, 1] controlling per-region
    conditioning attention strength.  The scalar *STRENGTH* is multiplied with
    the spatial mask before it is applied.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) != 2:
            msg = f"{option_string} requires exactly 2 arguments (MASK_PATH STRENGTH), got {len(values)}"
            raise argparse.ArgumentError(self, msg)

        mask_path = resolve_existing_path(values[0])
        strength = float(values[1])
        setattr(namespace, self.dest, (mask_path, strength))


class ImageAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) not in (3, 4):
            msg = f"{option_string} requires 3 or 4 arguments (PATH FRAME_IDX STRENGTH [CRF]), got {len(values)}"
            raise argparse.ArgumentError(self, msg)

        conditioning = ImageConditioningInput(
            path=resolve_existing_path(values[0]),
            frame_idx=int(values[1]),
            strength=float(values[2]),
            crf=int(values[3]) if len(values) > 3 else DEFAULT_IMAGE_CRF,
        )
        current = getattr(namespace, self.dest) or []
        current.append(conditioning)
        setattr(namespace, self.dest, current)


class LoraAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) > 2:
            msg = f"{option_string} accepts at most 2 arguments (PATH and optional STRENGTH), got {len(values)} values"
            raise argparse.ArgumentError(self, msg)

        path = values[0]
        strength_str = values[1] if len(values) > 1 else str(DEFAULT_LORA_STRENGTH)

        resolved_path = resolve_existing_path(path)
        strength = float(strength_str)

        current = getattr(namespace, self.dest) or []
        current.append(LoraPathStrengthAndSDOps(resolved_path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
        setattr(namespace, self.dest, current)


class CompileAction(argparse.Action):
    """Parse ``--compile [KEY=VALUE ...]`` into a :class:`CompilationConfig`.
    The flag is absent           -> ``args.compile`` stays at its default (``None``).
    The flag is passed alone     -> ``CompilationConfig()`` (vanilla torch defaults).
    The flag is passed with args -> ``CompilationConfig`` with the given fields overridden.
    Errors (unknown key, malformed value, duplicate key, empty value) raise
    :class:`argparse.ArgumentError` so argparse formats them as friendly CLI
    messages rather than uncaught tracebacks.
    """

    _ALLOWED_KEYS = frozenset({"mode", "backend", "fullgraph", "dynamic", "inductor_config", "dynamo_config"})

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        overrides: dict[str, object] = {}
        for item in values:
            if "=" not in item:
                raise argparse.ArgumentError(self, f"expects KEY=VALUE pairs, got: {item!r}")
            key, _, raw = item.partition("=")
            key = key.strip()
            if key not in self._ALLOWED_KEYS:
                raise argparse.ArgumentError(
                    self,
                    f"{key!r} is not a CompilationConfig field; valid keys: {sorted(self._ALLOWED_KEYS)}",
                )
            if key in overrides:
                raise argparse.ArgumentError(self, f"{key} given more than once")
            if key == "mode":
                overrides[key] = self._parse_mode(raw)
            elif key == "backend":
                overrides[key] = self._parse_non_empty(key, raw)
            elif key == "fullgraph":
                overrides[key] = self._parse_bool(key, raw)
            elif key == "dynamic":
                overrides[key] = self._parse_dynamic(raw)
            elif key in ("inductor_config", "dynamo_config"):
                overrides[key] = self._parse_json_dict(key, raw)
        setattr(namespace, self.dest, CompilationConfig(**overrides))

    def _parse_mode(self, raw: str) -> str | None:
        stripped = raw.strip()
        if not stripped:
            raise argparse.ArgumentError(self, "mode=... value cannot be empty (use mode=none to clear)")
        if stripped.lower() == "none":
            return None
        return stripped

    def _parse_non_empty(self, key: str, raw: str) -> str:
        stripped = raw.strip()
        if not stripped:
            raise argparse.ArgumentError(self, f"{key}=... value cannot be empty")
        return stripped

    def _parse_bool(self, key: str, raw: str) -> bool:
        normalized = raw.strip().lower()
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
        raise argparse.ArgumentError(self, f"{key}=... must be true or false; got {raw!r}")

    def _parse_dynamic(self, raw: str) -> bool | None:
        normalized = raw.strip().lower()
        if normalized in ("auto", "none"):
            return None
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
        raise argparse.ArgumentError(self, f"dynamic=... must be auto/true/false; got {raw!r}")

    def _parse_json_dict(self, key: str, raw: str) -> dict[str, Any]:
        # Inline JSON object starts with '{'; otherwise treat the value as a path to a JSON file.
        stripped = raw.strip()
        if not stripped:
            raise argparse.ArgumentError(self, f"{key}=... value cannot be empty")
        if stripped.startswith("{"):
            source = stripped
        else:
            path = Path(stripped).expanduser()
            if not path.is_file():
                raise argparse.ArgumentError(
                    self, f"{key}=... must be a JSON object or a path to a JSON file; got {raw!r}"
                )
            source = path.read_text()
        try:
            value = json.loads(source)
        except json.JSONDecodeError as e:
            raise argparse.ArgumentError(self, f"{key}=... must be a JSON object; got {raw!r} ({e.msg})") from None
        if not isinstance(value, dict):
            raise argparse.ArgumentError(self, f"{key}=... must decode to a JSON object; got {type(value).__name__}")
        return value


def resolve_path(path: str) -> str:
    return str(Path(path).expanduser().resolve().as_posix())


def resolve_existing_path(path: str) -> str:
    """Resolve *path* and verify it exists."""
    resolved = resolve_path(path)
    if not Path(resolved).exists():
        raise argparse.ArgumentError(None, f"Path not found: {resolved}")
    return resolved


QUANTIZATION_POLICIES = tuple(k.value for k in QuantizationKind)


def _resolve_quantization(namespace: argparse.Namespace) -> None:
    # Resolution is deferred until after parse_args because fp8-scaled-mm needs the
    # checkpoint path, which isn't on the namespace when the --quantization argument
    # is parsed.
    name = getattr(namespace, "quantization", None)
    if name is None or isinstance(name, QuantizationPolicy):
        return
    try:
        kind = QuantizationKind(name)
    except ValueError:
        return
    ckpt = getattr(namespace, "checkpoint_path", None) or getattr(namespace, "distilled_checkpoint_path", None)
    if ckpt is None:
        raise SystemExit(f"--quantization {kind.value} requires --checkpoint-path (or --distilled-checkpoint-path).")
    namespace.quantization = kind.to_policy(checkpoint_path=ckpt)


class _PipelineArgumentParser(argparse.ArgumentParser):
    def parse_args(  # type: ignore[override]
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        ns = super().parse_args(args, namespace)
        _resolve_quantization(ns)
        return ns


def detect_checkpoint_path(distilled: bool = False) -> str:
    """Pre-parse argv to extract the checkpoint path before building the full parser."""
    pre = argparse.ArgumentParser(add_help=False)
    flag = "--distilled-checkpoint-path" if distilled else "--checkpoint-path"
    pre.add_argument(flag, type=resolve_existing_path, required=True)
    known, _ = pre.parse_known_args()
    return known.distilled_checkpoint_path if distilled else known.checkpoint_path


def help_requested() -> bool:
    """Whether ``-h``/``--help`` appears on the command line."""
    return "-h" in sys.argv or "--help" in sys.argv


def resolve_cli_params(distilled: bool = False) -> PipelineParams:
    """Return the model params a pipeline CLI uses to build its argument parser.
    Reads the model version from the checkpoint named on the command line so the
    parser's defaults match the target model.
    Args:
        distilled: Whether the pipeline takes a distilled checkpoint
            (``--distilled-checkpoint-path``) rather than a full one (``--checkpoint-path``).
    """
    if help_requested():
        return LTX_2_3_PARAMS
    return detect_params(detect_checkpoint_path(distilled=distilled))


def basic_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
    distilled: bool = False,
) -> argparse.ArgumentParser:
    parser = _PipelineArgumentParser()
    if distilled:
        parser.add_argument(
            "--distilled-checkpoint-path",
            type=resolve_existing_path,
            required=True,
            help="Path to LTX-2 distilled model checkpoint (.safetensors file).",
        )
    else:
        parser.add_argument(
            "--checkpoint-path",
            type=resolve_existing_path,
            required=True,
            help="Path to LTX-2 model checkpoint (.safetensors file).",
        )
        parser.add_argument(
            "--num-inference-steps",
            type=int,
            default=params.num_inference_steps,
            help=(
                f"Number of denoising steps in the diffusion sampling process. "
                f"Higher values improve quality but increase generation time (default: {params.num_inference_steps})."
            ),
        )
    parser.add_argument(
        "--gemma-root",
        type=resolve_existing_path,
        required=True,
        help="Path to the root directory containing the Gemma text encoder model files.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired video content to be generated by the model.",
    )
    parser.add_argument(
        "--output-path",
        type=resolve_path,
        required=True,
        help="Path to the output video file (MP4 format).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=params.seed,
        help=f"Random seed for reproducible generation (default: {params.seed}).",
    )
    parser.add_argument(
        "--lora",
        dest="lora",
        action=LoraAction,
        nargs="+",  # Accept 1-2 arguments per use (path and optional strength); validation is handled in LoraAction
        metavar=("PATH", "STRENGTH"),
        default=[],
        help=(
            "LoRA (Low-Rank Adaptation) model: path to model file and optional strength "
            f"(default strength: {DEFAULT_LORA_STRENGTH}). Can be specified multiple times. "
            "Example: --lora path/to/lora1.safetensors 0.8 --lora path/to/lora2.safetensors"
        ),
    )

    parser.add_argument("--enhance-prompt", action="store_true")

    def _positive_int(value: str) -> int:
        try:
            int_value = int(value)
            if int_value < 1:
                raise argparse.ArgumentTypeError("must be >= 1")
            return int_value
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"must be an integer, got {value}") from e

    # Weight offloading
    parser.add_argument(
        "--offload",
        dest="offload_mode",
        type=OffloadMode,
        default=OffloadMode.NONE,
        choices=list(OffloadMode),
        help=(
            "Weight offloading strategy. "
            "'none' keeps all weights on GPU (default). "
            "'cpu' pins weights in CPU RAM, streams to GPU per layer. "
            "'disk' reads weights from disk on demand (lowest memory). "
            "Example: --offload cpu"
        ),
    )

    parser.add_argument(
        "--max-batch-size",
        type=_positive_int,
        default=1,
        metavar="N",
        help=(
            "Maximum batch size per transformer forward pass. "
            "Guided denoisers batch up to 4 guidance passes into a single call. "
            "Default 1 runs passes sequentially. Set to 4 to batch all passes "
            "together, which reduces layer-streaming PCIe transfers. "
            "Example: --max-batch-size 4"
        ),
    )

    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_POLICIES,
        default=None,
        help=(
            f"Quantization policy: {', '.join(QUANTIZATION_POLICIES)}. "
            "fp8-cast uses FP8 casting with upcasting during inference. "
            "fp8-scaled-mm uses FP8 scaled matrix multiplication; the layer set is auto-discovered "
            "from the checkpoint's .weight_scale tensors. "
            "Example: --quantization fp8-cast or --quantization fp8-scaled-mm"
        ),
    )
    parser.add_argument(
        "--compile",
        nargs="*",
        action=CompileAction,
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Enable torch.compile for transformer blocks. Pass alone for defaults, "
            "or with KEY=VALUE overrides for any CompilationConfig field. "
            "Keys: mode, backend, fullgraph, dynamic, inductor_config, dynamo_config. "
            "inductor_config/dynamo_config take JSON objects (inline or a path to a .json file) "
            "that fully replace the defaults. "
            "Examples: --compile  or  --compile mode=reduce-overhead  or  "
            "--compile mode=reduce-overhead fullgraph=true backend=eager  or  "
            "--compile inductor_config='{\"max_autotune\": true}'"
        ),
    )
    return parser


def new_video_gen_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
    distilled: bool = False,
) -> argparse.ArgumentParser:
    parser = basic_arg_parser(params=params, distilled=distilled)
    parser.add_argument(
        "--height",
        type=int,
        default=params.stage_1_height,
        help=f"Video height in pixels, divisible by 32 (default: {params.stage_1_height}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=params.stage_1_width,
        help=f"Width of the generated video in pixels, should be divisible by 32 (default: {params.stage_1_width}).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=params.num_frames,
        help=f"Number of frames to generate in the output video sequence, num-frames = (8 x K) + 1, "
        f"where k is a non-negative integer (default: {params.num_frames}).",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=params.frame_rate,
        help=f"Frame rate of the generated video (fps) (default: {params.frame_rate}).",
    )
    parser.add_argument(
        "--image",
        dest="images",
        action=ImageAction,
        nargs="+",
        metavar="ARG",
        default=[],
        help=(
            "Image conditioning input: PATH FRAME_IDX STRENGTH [CRF]. "
            "PATH is the image file, FRAME_IDX is the target frame index, "
            "STRENGTH is the conditioning strength (all three required). "
            f"CRF is the optional H.264 compression quality (0=lossless, default: {DEFAULT_IMAGE_CRF}). "
            "Can be specified multiple times. Example: --image path/to/image1.jpg 0 0.8 "
            "--image path/to/image2.jpg 160 0.9 0"
        ),
    )

    return parser


def video_editing_arg_parser(
    distilled: bool = True,
) -> argparse.ArgumentParser:
    """Base argument parser for video-editing pipelines (retake, extension, inpainting, sticker movement).
    Uses the same actions and conventions as basic_arg_parser but only the args needed for editing
    (no height/width/num-frames; resolution comes from input video). Default is distilled checkpoint only.
    """
    parser = basic_arg_parser(distilled=distilled)
    parser.add_argument("--video-path", type=resolve_existing_path, required=True, help="Path to the source video.")
    parser.add_argument("--start-time", type=float, required=True, help="Start time of the region to regenerate (s).")
    parser.add_argument("--end-time", type=float, required=True, help="End time of the region to regenerate (s).")
    return parser


def lipdub_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
) -> argparse.ArgumentParser:
    """Argument parser for the lip-dub pipeline.
    Frame count and frame rate are derived from the reference video at runtime (the frame count
    is silently snapped down to the nearest 8k+1), so this parser intentionally omits
    --num-frames, --frame-rate, and --image. Distilled checkpoint only.
    """
    parser = basic_arg_parser(params=params, distilled=True)
    parser.add_argument(
        "--height",
        type=int,
        default=params.stage_2_height,
        help=(
            f"Height of the generated video in pixels, should be divisible by 64 (default: {params.stage_2_height})."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=params.stage_2_width,
        help=f"Width of the generated video in pixels, should be divisible by 64 (default: {params.stage_2_width}).",
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=resolve_path,
        required=True,
        help=(
            "Path to the spatial upsampler model used to increase the resolution "
            "of the generated video in the latent space."
        ),
    )
    parser.add_argument(
        "--reference-video",
        type=resolve_path,
        required=True,
        help="Reference video file (video + audio track used for IC-LoRA and audio identity).",
    )
    parser.add_argument(
        "--reference-strength",
        type=float,
        default=1.0,
        help="Strength for IC-LoRA video reference conditioning (default: 1.0).",
    )
    return parser


def default_1_stage_arg_parser(params: PipelineParams = LTX_2_3_PARAMS) -> argparse.ArgumentParser:
    video_guider = params.video_guider_params
    audio_guider = params.audio_guider_params
    parser = new_video_gen_arg_parser(params=params)
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help=(
            "Negative prompt describing what should not appear in the generated video, "
            "used to guide the diffusion process away from unwanted content. "
            "Default: a comprehensive negative prompt covering common artifacts and quality issues."
        ),
    )
    parser.add_argument(
        "--video-cfg-guidance-scale",
        type=float,
        default=video_guider.cfg_scale,
        help=(
            f"Classifier-free guidance (CFG) scale controlling how strongly "
            f"the model adheres to the video prompt. Higher values increase prompt "
            f"adherence but may reduce diversity. 1.0 means no effect "
            f"(default: {video_guider.cfg_scale})."
        ),
    )
    parser.add_argument(
        "--video-stg-guidance-scale",
        type=float,
        default=video_guider.stg_scale,
        help=(
            f"STG (Spatio-Temporal Guidance) scale controlling how strongly "
            f"the model reacts to the perturbation of the video modality. Higher values increase "
            f"the effect but may reduce quality. 0.0 means no effect "
            f"(default: {video_guider.stg_scale})."
        ),
    )
    parser.add_argument(
        "--video-rescale-scale",
        type=float,
        default=video_guider.rescale_scale,
        help=(
            f"Rescale scale controlling how strongly "
            f"the model rescales the video modality after applying other guidance. Higher values tend to decrease "
            f"oversaturation effects. 0.0 means no effect (default: {video_guider.rescale_scale})."
        ),
    )
    parser.add_argument(
        "--video-stg-blocks",
        type=int,
        nargs="*",
        default=video_guider.stg_blocks,
        help=(f"Which transformer blocks to perturb for STG. Default: {video_guider.stg_blocks}."),
    )
    parser.add_argument(
        "--a2v-guidance-scale",
        type=float,
        default=video_guider.modality_scale,
        help=(
            f"A2V (Audio-to-Video) guidance scale controlling how strongly "
            f"the model reacts to the perturbation of the audio-to-video cross-attention. Higher values may increase "
            f"lipsync quality. 1.0 means no effect (default: {video_guider.modality_scale})."
        ),
    )
    parser.add_argument(
        "--video-skip-step",
        type=int,
        default=video_guider.skip_step,
        help=(
            "Video skip step N controls periodic skipping during the video diffusion process: "
            "only steps where step_index %% (N + 1) == 0 are processed, all others are skipped "
            f"(e.g., 0 = no skipping; 1 = skip every other step; 2 = skip 2 of every 3 steps; "
            f"default: {video_guider.skip_step})."
        ),
    )
    parser.add_argument(
        "--audio-cfg-guidance-scale",
        type=float,
        default=audio_guider.cfg_scale,
        help=(
            f"Audio CFG (Classifier-free guidance) scale controlling how strongly "
            f"the model adheres to the audio prompt. Higher values increase prompt "
            f"adherence but may reduce diversity. 1.0 means no effect "
            f"(default: {audio_guider.cfg_scale})."
        ),
    )
    parser.add_argument(
        "--audio-stg-guidance-scale",
        type=float,
        default=audio_guider.stg_scale,
        help=(
            f"Audio STG (Spatio-Temporal Guidance) scale controlling how strongly "
            f"the model reacts to the perturbation of the audio modality. Higher values increase "
            f"the effect but may reduce quality. 0.0 means no effect "
            f"(default: {audio_guider.stg_scale})."
        ),
    )
    parser.add_argument(
        "--audio-rescale-scale",
        type=float,
        default=audio_guider.rescale_scale,
        help=(
            f"Audio rescale scale controlling how strongly "
            f"the model rescales the audio modality after applying other guidance. "
            f"Experimental. 0.0 means no effect (default: {audio_guider.rescale_scale})."
        ),
    )
    parser.add_argument(
        "--audio-stg-blocks",
        type=int,
        nargs="*",
        default=audio_guider.stg_blocks,
        help=(f"Which transformer blocks to perturb for Audio STG. Default: {audio_guider.stg_blocks}."),
    )
    parser.add_argument(
        "--v2a-guidance-scale",
        type=float,
        default=audio_guider.modality_scale,
        help=(
            f"V2A (Video-to-Audio) guidance scale controlling how strongly "
            f"the model reacts to the perturbation of the video-to-audio cross-attention. Higher values may increase "
            f"lipsync quality. 1.0 means no effect (default: {audio_guider.modality_scale})."
        ),
    )
    parser.add_argument(
        "--audio-skip-step",
        type=int,
        default=audio_guider.skip_step,
        help=(
            "Audio skip step N controls periodic skipping during the audio diffusion process: "
            "only steps where step_index %% (N + 1) == 0 are processed, all others are skipped "
            f"(e.g., 0 = no skipping; 1 = skip every other step; 2 = skip 2 of every 3 steps; "
            f"default: {audio_guider.skip_step})."
        ),
    )
    return parser


def default_1_stage_t2a_arg_parser(params: PipelineParams = LTX_2_3_PARAMS) -> argparse.ArgumentParser:
    """Argument parser for single-stage text-to-audio pipelines (audio-only)."""
    audio_guider = params.audio_guider_params
    parser = basic_arg_parser(params=params)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=params.num_frames,
        help="Number of frames used to derive audio duration (num-frames / frame-rate).",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=params.frame_rate,
        help="Frame rate used with --num-frames to derive the audio duration.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt to steer audio generation away from artifacts.",
    )
    parser.add_argument(
        "--audio-cfg-guidance-scale",
        type=float,
        default=audio_guider.cfg_scale,
        help=f"Audio CFG scale (default: {audio_guider.cfg_scale}).",
    )
    parser.add_argument(
        "--audio-stg-guidance-scale",
        type=float,
        default=audio_guider.stg_scale,
        help=f"Audio STG scale (default: {audio_guider.stg_scale}).",
    )
    parser.add_argument(
        "--audio-rescale-scale",
        type=float,
        default=audio_guider.rescale_scale,
        help=f"Audio rescale scale (default: {audio_guider.rescale_scale}).",
    )
    parser.add_argument(
        "--audio-stg-blocks",
        type=int,
        nargs="*",
        default=audio_guider.stg_blocks,
        help=f"Blocks to perturb for Audio STG (default: {audio_guider.stg_blocks}).",
    )
    parser.add_argument(
        "--audio-skip-step",
        type=int,
        default=audio_guider.skip_step,
        help=f"Audio skip step (default: {audio_guider.skip_step}).",
    )
    return parser


def default_2_stage_arg_parser(params: PipelineParams = LTX_2_3_PARAMS) -> argparse.ArgumentParser:
    parser = default_1_stage_arg_parser(params=params)
    parser.set_defaults(height=params.stage_2_height, width=params.stage_2_width)
    # Update help text to reflect 2-stage defaults
    for action in parser._actions:
        if "--height" in action.option_strings:
            action.help = (
                f"Height of the generated video in pixels, should be divisible by 64 "
                f"(default: {params.stage_2_height})."
            )
        if "--width" in action.option_strings:
            action.help = (
                f"Width of the generated video in pixels, should be divisible by 64 (default: {params.stage_2_width})."
            )
    parser.add_argument(
        "--distilled-lora",
        dest="distilled_lora",
        action=LoraAction,
        nargs="+",  # Accept 1-2 arguments per use (path and optional strength); validation is handled in LoraAction
        metavar=("PATH", "STRENGTH"),
        required=True,
        help=(
            "Distilled LoRA (Low-Rank Adaptation) model used in the second stage (upscaling and refinement): "
            f"path to model file and optional strength (default strength: {DEFAULT_LORA_STRENGTH}). "
            "The second stage upsamples the video by 2x resolution and refines it using a distilled "
            "denoising schedule (fewer steps, no CFG). The distilled LoRA is specifically trained "
            "for this refinement process to improve quality at higher resolutions. "
            "Example: --distilled-lora path/to/distilled_lora.safetensors 0.8"
        ),
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=resolve_existing_path,
        required=True,
        help=(
            "Path to the spatial upsampler model used to increase the resolution "
            "of the generated video in the latent space."
        ),
    )
    return parser


def hq_2_stage_arg_parser(params: PipelineParams = LTX_2_3_HQ_PARAMS) -> argparse.ArgumentParser:
    parser = default_2_stage_arg_parser(params=params)
    parser.add_argument(
        "--distilled-lora-strength-stage-1",
        type=float,
        default=0.25,
        help=(f"Strength of the distilled LoRA used in the first stage (default: {0.25})."),
    )
    parser.add_argument(
        "--distilled-lora-strength-stage-2",
        type=float,
        default=0.5,
        help=(f"Strength of the distilled LoRA used in the second stage (default: {0.5})."),
    )
    return parser


def default_2_stage_distilled_arg_parser(params: PipelineParams = LTX_2_3_PARAMS) -> argparse.ArgumentParser:
    parser = new_video_gen_arg_parser(params=params, distilled=True)
    parser.set_defaults(height=params.stage_2_height, width=params.stage_2_width)
    # Update help text to reflect 2-stage defaults
    for action in parser._actions:
        if "--height" in action.option_strings:
            action.help = (
                f"Height of the generated video in pixels, should be divisible by 64 "
                f"(default: {params.stage_2_height})."
            )
        if "--width" in action.option_strings:
            action.help = (
                f"Width of the generated video in pixels, should be divisible by 64 (default: {params.stage_2_width})."
            )
    parser.add_argument(
        "--spatial-upsampler-path",
        type=resolve_existing_path,
        required=True,
        help=(
            "Path to the spatial upsampler model used to increase the resolution "
            "of the generated video in the latent space."
        ),
    )
    return parser
