import argparse
import enum
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.chunks import ChunkConfig
from ltx_pipelines.marking import AudioMarker, VideoMarker, create_markers
from ltx_pipelines.utils.constants import (
    DEFAULT_LORA_STRENGTH,
    DEFAULT_NEGATIVE_PROMPT,
    LTX_2_3_HQ_PARAMS,
    LTX_2_3_PARAMS,
    LTX_2_4_PARAMS,
    VIDEO_SCALE_FACTORS,
    PipelineParams,
    detect_params,
)
from ltx_pipelines.utils.media_io.color_config import EXRColorSpace
from ltx_pipelines.utils.media_io.exr import is_exr_dir
from ltx_pipelines.utils.media_io.inputs import EXRVideoInput, VideoInput
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import AutoDuration, ImageConditioningInput, OffloadMode

logger = logging.getLogger(__name__)


class HDRICLoraInputColorSpace(str, enum.Enum):
    """Supported HDR IC-LoRA input colorspaces."""

    SRGB_GAMMA = "srgb_gamma"
    SRGB = "srgb"
    ACESCG = "acescg"
    ACESCCT = "acescct"


_VIDEO_GAMMA: dict[HDRICLoraInputColorSpace, bool] = {
    HDRICLoraInputColorSpace.SRGB_GAMMA: True,
    HDRICLoraInputColorSpace.SRGB: False,
}

_EXR_SPACES: dict[HDRICLoraInputColorSpace, EXRColorSpace] = {
    HDRICLoraInputColorSpace.SRGB: EXRColorSpace.SRGB_LINEAR,
    HDRICLoraInputColorSpace.ACESCG: EXRColorSpace.ACESCG,
    HDRICLoraInputColorSpace.ACESCCT: EXRColorSpace.ACESCCT,
}

_VIDEO_SUFFIXES = frozenset({".mp4", ".mov"})


def add_hdr_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--hdr`` colour-space flag. Behaviour is implemented in ``media_io``."""
    names = [m.name for m in EXRColorSpace]

    def _parse_hdr(value: str) -> EXRColorSpace:
        key = value.upper()
        try:
            return EXRColorSpace[key]
        except KeyError as exc:
            raise argparse.ArgumentTypeError(f"invalid --hdr {value!r}; choose from {', '.join(names)}") from exc

    parser.add_argument(
        "--hdr",
        type=_parse_hdr,
        choices=[EXRColorSpace[n] for n in names],
        default=None,
        metavar="{" + ",".join(names) + "}",
        help=(
            "HDR colour space for EXR conditioning / HDR encode (EXR half + BT.2020/HLG). "
            "Required when any EXR still or folder is passed. Default: SDR."
        ),
    )
    return parser


def add_marker_root_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--marker-root-path``. Requires the ``videoseal`` and ``audioseal`` extras."""
    parser.add_argument(
        "--marker-root-path",
        dest="marker_root_path",
        type=resolve_existing_dir,
        default=None,
        help=(
            "Path to a local path/to/marking pack directory (VideoSeal checkpoint/card/"
            "attenuation and AudioSeal generator/detector). Stamps SDR pixels and stereo PCM. "
            "HDR encode skips video and still stamps audio. "
            "Requires the videoseal and audioseal extras. "
            "Default: no marking."
        ),
    )
    return parser


def markers_from_args(
    args: object,
    device: torch.device,
    color_space: EXRColorSpace | None = None,
) -> tuple[VideoMarker | None, AudioMarker | None]:
    """Load resident markers for this CLI, or ``(None, None)`` when the flag is omitted.
    When ``--marker-root-path`` is set, both extras and pack files are checked
    immediately. HDR (``color_space`` set) drops the video marker, stamps audio,
    and logs that video marking is unavailable with HDR.
    """
    path = getattr(args, "marker_root_path", None)
    if path is None:
        return None, None
    video_marker, audio_marker = create_markers(path, device)
    if color_space is not None:
        logger.warning("video marking is unavailable with HDR")
        return None, audio_marker
    return video_marker, audio_marker


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


class AutoDurationAction(argparse.Action):
    """Parse ``--auto-duration MIN_SECONDS MAX_SECONDS`` into an :class:`AutoDuration`."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        min_seconds, max_seconds = float(values[0]), float(values[1])
        if min_seconds > max_seconds:
            msg = f"{option_string} MIN_SECONDS ({min_seconds}) must be <= MAX_SECONDS ({max_seconds})"
            raise argparse.ArgumentError(self, msg)
        setattr(namespace, self.dest, AutoDuration(min_seconds=min_seconds, max_seconds=max_seconds))


def _append_image_conditioning(namespace: argparse.Namespace, dest: str, item: ImageConditioningInput) -> None:
    current = getattr(namespace, dest) or []
    current.append(item)
    setattr(namespace, dest, current)


_IMAGE_BINDING_KEYWORDS = frozenset({"auto", "replace", "guide"})


class ImageAction(argparse.Action):
    """Parse ``--image PATH FRAME_IDX STRENGTH [BINDING] [CRF]``.
    ``BINDING`` is ``auto`` (default when omitted), ``replace``, or ``guide``. With ``auto``, frame 0
    replaces the latent and later frames guide as keyframes. When only one trailing token is given,
    a keyword selects ``BINDING`` and an integer selects ``CRF``. An omitted CRF stays ``None`` for
    the pipeline to resolve from its checkpoint.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,  # noqa: ARG002
        namespace: argparse.Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if len(values) not in (3, 4, 5):
            msg = (
                f"{option_string} requires 3 to 5 arguments "
                f"(PATH FRAME_IDX STRENGTH [BINDING] [CRF]), got {len(values)}"
            )
            raise argparse.ArgumentError(self, msg)

        path = resolve_existing_path(values[0])
        frame_idx = int(values[1])
        strength = float(values[2])
        binding_mode = "auto"
        crf: int | None = None

        if len(values) == 4:
            if values[3] in _IMAGE_BINDING_KEYWORDS:
                binding_mode = values[3]
            else:
                crf = int(values[3])
        elif len(values) == 5:
            if values[3] not in _IMAGE_BINDING_KEYWORDS:
                msg = f"{option_string} BINDING must be auto, replace, or guide, got {values[3]!r}"
                raise argparse.ArgumentError(self, msg)
            binding_mode = values[3]
            crf = int(values[4])

        match binding_mode:
            case "auto":
                image = ImageConditioningInput.at(path, frame_idx, strength, crf)
            case "replace":
                image = ImageConditioningInput.replace(path, frame_idx, strength, crf)
            case "guide":
                image = ImageConditioningInput.guide(path, frame_idx, strength, crf)
            case _:
                msg = f"{option_string} BINDING must be auto, replace, or guide, got {binding_mode!r}"
                raise argparse.ArgumentError(self, msg)

        _append_image_conditioning(namespace, self.dest, image)


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

    _ALLOWED_KEYS = frozenset(
        {
            "mode",
            "backend",
            "fullgraph",
            "dynamic",
            "inductor_config",
            "dynamo_config",
            "seq_dim_dynamic",
            "recompile_perturbed_block",
            "capture",
            "max_video_tokens",
            "max_audio_tokens",
        }
    )

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
            elif key in ("fullgraph", "seq_dim_dynamic", "recompile_perturbed_block", "capture"):
                overrides[key] = self._parse_bool(key, raw)
            elif key == "dynamic":
                overrides[key] = self._parse_dynamic(raw)
            elif key in ("inductor_config", "dynamo_config"):
                overrides[key] = self._parse_json_dict(key, raw)
            elif key in ("max_video_tokens", "max_audio_tokens"):
                overrides[key] = self._parse_positive_int(key, raw)
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

    def _parse_positive_int(self, key: str, raw: str) -> int:
        stripped = raw.strip()
        if not stripped.isdigit() or int(stripped) == 0:
            raise argparse.ArgumentError(self, f"{key}=... must be a positive integer; got {raw!r}")
        return int(stripped)

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


def resolve_existing_dir(path: str) -> str:
    """Resolve *path* and verify it is an existing directory."""
    resolved = resolve_existing_path(path)
    if not Path(resolved).is_dir():
        raise argparse.ArgumentError(None, f"Not a directory: {resolved}")
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
    # Always require a checkpoint path for a clear CLI error, even for kinds
    # (e.g. nvfp4-cast) whose policy builder does not consume it.
    # Split-checkpoint runs may only have --transformer-path.
    ckpt = (
        getattr(namespace, "checkpoint_path", None)
        or getattr(namespace, "distilled_checkpoint_path", None)
        or getattr(namespace, "transformer_path", None)
    )
    if ckpt is None:
        raise SystemExit(
            f"--quantization {kind.value} requires --checkpoint-path, "
            "--distilled-checkpoint-path, or --transformer-path."
        )
    namespace.quantization = kind.to_policy(checkpoint_path=ckpt)


def _resolve_model_paths(namespace: argparse.Namespace) -> None:
    """Attach ``namespace.model_paths`` — the post-parse path contract for pipelines.
    Raw checkpoint / split flags remain on the namespace for parser tests and rare
    pre-resolve helpers; pipeline ``main()`` and constructors should read only
    ``args.model_paths``.
    """
    # Deferred: model_paths imports argparse helpers from this module at module scope.
    from ltx_pipelines.utils.model_paths import model_paths_from_namespace  # noqa: PLC0415

    namespace.model_paths = model_paths_from_namespace(namespace)


def _resolve_num_frames(namespace: argparse.Namespace) -> None:
    """Collapse ``--num-frames``/``--auto-duration`` into the single value a pipeline's
    ``num_frames`` parameter expects: an explicit ``--num-frames`` wins (with a warning if
    ``--auto-duration`` was also given); otherwise ``--auto-duration`` if given, else the
    ``AutoDuration()`` default. No-op for parsers without ``supports_auto_duration=True`` (neither
    attribute exists on the namespace).
    """
    if not hasattr(namespace, "num_frames") or not hasattr(namespace, "auto_duration"):
        return
    if namespace.num_frames is not None and namespace.auto_duration is not None:
        logger.warning(
            "Both --num-frames and --auto-duration were given; using --num-frames=%d and ignoring --auto-duration.",
            namespace.num_frames,
        )
    if namespace.num_frames is None:
        namespace.num_frames = namespace.auto_duration if namespace.auto_duration is not None else AutoDuration()


def _verify_media_path_args(namespace: argparse.Namespace) -> None:
    """Validate HDR/SDR path shapes and couple retake ``--frame-rate`` to EXR folders.
    Rules:
    * ``--image``: a still file (``.exr`` or PNG/JPEG). Not a directory.
    * ``--video-conditioning`` / ``--video-path``: a video file or an EXR-frame folder.
    * Any EXR still/folder requires ``--hdr {SRGB_LINEAR,ACESCG,ACESCCT}``.
    * ``--image`` + ``--video-conditioning`` (and multiples of each) must be all EXR or all
      non-EXR — no mix. ``--conditioning-attention-mask`` is always SDR and is not checked.
    * Retake ``--frame-rate``: required for EXR folders, forbidden for video files.
    * Dub-It ``--reference-video``: video file only (fps + audio from the container). EXR
      folders / single ``.exr`` files are rejected.
    """
    modality: list[tuple[str, str, bool]] = []  # (flag, path, is_exr)

    for img in getattr(namespace, "images", None) or []:
        path = getattr(img, "path", img)
        if Path(path).is_dir():
            raise SystemExit(
                f"--image '{path}' is a directory; pass a still file (.exr or PNG/JPEG). "
                "For an EXR sequence use --video-conditioning or --video-path."
            )
        modality.append(("--image", path, _is_exr_file(path)))

    for item in getattr(namespace, "video_conditioning", None) or []:
        path = item[0] if isinstance(item, (tuple, list)) else item
        _verify_sequence_path("--video-conditioning", path)
        modality.append(("--video-conditioning", path, is_exr_dir(path)))

    if len({is_exr for _, _, is_exr in modality}) > 1:
        detail = ", ".join(f"{flag} '{path}' ({'EXR' if is_exr else 'SDR'})" for flag, path, is_exr in modality)
        raise SystemExit(
            "--image / --video-conditioning must be all EXR or all SDR; "
            f"got mixed: {detail}. (--conditioning-attention-mask is always SDR and is ignored here.)"
        )

    reference_video = getattr(namespace, "reference_video", None)
    if reference_video is not None:
        if _is_exr_file(reference_video) or is_exr_dir(reference_video):
            raise SystemExit(
                f"--reference-video '{reference_video}' must be a video file with an audio track; "
                "EXR is not supported (fps and audio identity come from the container)."
            )
        if Path(reference_video).is_dir():
            raise SystemExit(
                f"--reference-video '{reference_video}' is a directory; pass a video file "
                "(fps and audio identity come from the container)."
            )

    video_path = getattr(namespace, "video_path", None)
    if video_path is not None:
        _verify_sequence_path("--video-path", video_path)
        if is_exr_dir(video_path):
            modality.append(("--video-path", video_path, True))
        if hasattr(namespace, "frame_rate"):
            _verify_sequence_frame_rate("--video-path", video_path, namespace.frame_rate)

    if any(is_exr for _, _, is_exr in modality) and getattr(namespace, "hdr", None) is None:
        raise SystemExit("EXR input requires --hdr {SRGB_LINEAR,ACESCG,ACESCCT} to declare the source colour space.")


class _PipelineArgumentParser(argparse.ArgumentParser):
    def parse_args(  # type: ignore[override]
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        ns = super().parse_args(args, namespace)
        _resolve_quantization(ns)
        _resolve_model_paths(ns)
        _resolve_num_frames(ns)
        _verify_media_path_args(ns)
        _resolve_hdr_ic_lora_input(ns)
        return ns


def _is_exr_file(path: str | Path) -> bool:
    return Path(path).is_file() and str(path).lower().endswith(".exr")


def _verify_sequence_path(flag: str, path: str) -> None:
    """Video/sequence flags accept a video file or a directory of ``*.exr`` frames."""
    p = Path(path)
    if _is_exr_file(path):
        raise SystemExit(f"{flag} '{path}' is a single .exr file; pass a directory of *.exr frames.")
    if p.is_dir() and not is_exr_dir(path):
        raise SystemExit(f"{flag} '{path}' is a directory without *.exr frames.")


def _verify_sequence_frame_rate(flag: str, path: str, frame_rate: float | None) -> None:
    """Require ``--frame-rate`` for EXR-frame folders; forbid it for video files."""
    if is_exr_dir(path):
        if frame_rate is None:
            raise SystemExit(
                f"{flag} '{path}' is an EXR-frame folder; --frame-rate is required "
                "(EXR sequences have no container fps)."
            )
    elif frame_rate is not None:
        raise SystemExit(
            f"--frame-rate is only valid when {flag} is an EXR-frame folder; for video files the container fps is used."
        )


def _resolve_hdr_ic_lora_input(namespace: argparse.Namespace) -> None:
    """Map ``--input`` / ``--input-colorspace`` / ``--frame-rate`` to VideoInput or EXRVideoInput."""
    has_input = hasattr(namespace, "input")
    has_colorspace = hasattr(namespace, "input_colorspace")
    if has_input != has_colorspace:
        raise SystemExit("--input and --input-colorspace are required together")
    if not has_colorspace:
        return
    input_path = namespace.input
    if input_path is None:
        raise SystemExit("HDR IC-LoRA requires --input")

    _verify_sequence_path("--input", input_path)

    try:
        cs = HDRICLoraInputColorSpace(namespace.input_colorspace)
    except ValueError:
        raise SystemExit(
            f"invalid --input-colorspace {namespace.input_colorspace!r}; "
            f"expected one of {', '.join(m.value for m in HDRICLoraInputColorSpace)}"
        ) from None

    path = Path(input_path)
    fps = getattr(namespace, "frame_rate", None)

    if not is_exr_dir(path):
        if path.suffix.lower() not in _VIDEO_SUFFIXES:
            raise SystemExit(f"expected an MP4/MOV file or a directory of *.exr frames; got {path}")
        if cs not in _VIDEO_GAMMA:
            raise SystemExit(f"--input-colorspace {cs.value} expects a directory of *.exr frames; got {path}")
        if fps is not None:
            raise SystemExit("--frame-rate is only valid for an EXR-frame folder")
        namespace.input = VideoInput(path, gamma_encoded=_VIDEO_GAMMA[cs])
        return

    if cs not in _EXR_SPACES:
        raise SystemExit(f"--input-colorspace {cs.value} expects an MP4/MOV file; got EXR frames at {path}")
    if fps is None:
        raise SystemExit("EXR-frame folder requires --frame-rate")
    namespace.input = EXRVideoInput(path, color_space=_EXR_SPACES[cs], frame_rate=fps)


def detect_checkpoint_path(distilled: bool = False) -> str:
    """Pre-parse argv to extract the checkpoint path before building the full parser.
    Prefers monolith ``--checkpoint-path`` / ``--distilled-checkpoint-path``, then
    falls back to ``--transformer-path`` for split layouts.
    """
    pre = argparse.ArgumentParser(add_help=False)
    flag = "--distilled-checkpoint-path" if distilled else "--checkpoint-path"
    pre.add_argument(flag, type=resolve_existing_path, required=False, default=None)
    pre.add_argument("--transformer-path", type=resolve_existing_path, required=False, default=None)
    known, _ = pre.parse_known_args()
    monolith = known.distilled_checkpoint_path if distilled else known.checkpoint_path
    path = monolith or known.transformer_path
    if path is None:
        raise SystemExit(
            f"Missing {flag} (monolith) or --transformer-path (split). "
            "Pass one of them before building the argument parser."
        )
    return path


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


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    params: PipelineParams,
    distilled: bool = False,
    text_encoder: bool = True,
    audio_vae: bool = True,
    duration_head: bool = True,
) -> argparse.ArgumentParser:
    """Add the model-component path flags.
    One flag group per component, so a pipeline declares only the components it loads and
    ``model_paths_from_namespace`` sees exactly those. Turning a component off matters beyond
    tidiness for ``text_encoder``: that switch covers ``--gemma-root`` *and*
    ``--text-encoder-path``, the monolith and split halves of one component, and
    ``model_paths_from_namespace`` requires ``--gemma-root`` in monolith mode only when the
    parser declares it. A CLI with precomputed embeddings must therefore leave it out rather
    than accept it and ignore it.
    """
    if distilled:
        parser.add_argument(
            "--distilled-checkpoint-path",
            type=resolve_existing_path,
            required=False,
            default=None,
            help=(
                "Path to LTX-2 distilled monolith checkpoint (.safetensors). "
                "Required in monolith mode; omit when using split --transformer-path etc."
            ),
        )
    else:
        parser.add_argument(
            "--checkpoint-path",
            type=resolve_existing_path,
            required=False,
            default=None,
            help=(
                "Path to LTX-2 monolith checkpoint (.safetensors). "
                "Required in monolith mode; omit when using split --transformer-path etc."
            ),
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
    if text_encoder:
        parser.add_argument(
            "--gemma-root",
            type=resolve_existing_path,
            required=False,
            default=None,
            help=(
                "Path to the Gemma text encoder HF directory (monolith mode). "
                "Omit when using --text-encoder-path (split single-file TE)."
            ),
        )
    parser.add_argument(
        "--transformer-path",
        type=resolve_existing_path,
        default=None,
        help=(
            "Split layout: transformer safetensors (DiT + connectors). "
            "Provide the subset of split flags this pipeline needs; unused may be omitted."
        ),
    )
    if text_encoder:
        parser.add_argument(
            "--text-encoder-path",
            type=resolve_existing_path,
            default=None,
            help="Split layout: Gemma + text_embedding_projection single-file safetensors.",
        )
    parser.add_argument(
        "--video-vae-path",
        type=resolve_existing_path,
        default=None,
        help=(
            "Video VAE safetensors (encoder + decoder). Split: the vae/ component. "
            "Monolith: optional override of the VAE inside the fat checkpoint "
            "(e.g. a distilled DiffVAE). Decoder kind is selected from file metadata."
        ),
    )
    if audio_vae:
        parser.add_argument(
            "--audio-vae-path",
            type=resolve_existing_path,
            default=None,
            help="Split layout: audio VAE + vocoder safetensors.",
        )
    if duration_head:
        parser.add_argument(
            "--duration-head-path",
            type=resolve_existing_path,
            default=None,
            help="Split layout: duration head safetensors (omit if unused / unavailable).",
        )
    return parser


def add_seed_arg(
    parser: argparse.ArgumentParser,
    *,
    params: PipelineParams,
) -> argparse.ArgumentParser:
    """Add ``--seed``."""
    parser.add_argument(
        "--seed",
        type=int,
        default=params.seed,
        help=f"Random seed for reproducible generation (default: {params.seed}).",
    )
    return parser


def add_offload_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--offload``, the weight offloading strategy."""
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
    return parser


def add_diffvae_optimization_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--diffvae-optimization``, the DiffVAE decode preset."""
    parser.add_argument(
        "--diffvae-optimization",
        type=DiffVAEMode,
        default=DiffVAEMode.CHUNKED_EAGER,
        choices=list(DiffVAEMode),
        help=(
            "DiffVAE decode optimization preset. "
            "'chunked_eager' (default): deferred stage-4, W-chunks=4, cutlass-fna. "
            "'chunked_compile': same chunking with torch.compile (det stages off). "
            "'combined_compile': combined context, full compile (highest VRAM, fastest warm). "
            "'blackwell_dsl': deferred stage-4 + CuTe DSL NA/fused stage-5 (datacenter Blackwell). "
            "Ignored for convolutional VAEs. "
            "Example: --diffvae-optimization combined_compile"
        ),
    )
    return parser


def basic_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
    distilled: bool = False,
) -> argparse.ArgumentParser:
    parser = _PipelineArgumentParser()
    add_model_args(parser, params=params, distilled=distilled)
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
    add_seed_arg(parser, params=params)
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
    parser.add_argument(
        "--enhance-static-cache",
        action="store_true",
        help=(
            "Use HF static KV-cache for prompt enhancement (opt-in). Helps multi-prompt enhance "
            "latency after warmup; does not change encode."
        ),
    )
    parser.add_argument(
        "--prompt-enhancer-gemma-root",
        type=resolve_existing_path,
        default=None,
        help=(
            "Optional path to a generative Gemma instruct checkpoint used only for prompt "
            "enhancement. Required when --enhance-prompt is set and the encode TE "
            "(--gemma-root or --text-encoder-path) is not gemma3 "
            "(e.g. gemma4_unified encode + gemma4 E2B-it enhance). Ignored for gemma3 encode roots."
        ),
    )

    def _positive_int(value: str) -> int:
        try:
            int_value = int(value)
            if int_value < 1:
                raise argparse.ArgumentTypeError("must be >= 1")
            return int_value
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"must be an integer, got {value}") from e

    add_offload_arg(parser)

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
            "nvfp4-cast online-quantizes BF16 Linear weights to NVFP4 (Blackwell, ltx-kernels). "
            "nvfp4-prequant loads a pre-quantized NVFP4 checkpoint (pair with a BF16 VAE). "
            "Example: --quantization fp8-cast, --quantization fp8-scaled-mm, "
            "or --quantization nvfp4-prequant"
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
            "Keys: mode, backend, fullgraph, dynamic, inductor_config, dynamo_config, "
            "seq_dim_dynamic, recompile_perturbed_block, capture. "
            "inductor_config/dynamo_config take JSON objects (inline or a path to a .json file) "
            "that fully replace the defaults. "
            "Examples: --compile  or  --compile mode=reduce-overhead  or  "
            "--compile mode=reduce-overhead fullgraph=true backend=eager  or  "
            "--compile inductor_config='{\"max_autotune\": true}'"
        ),
    )
    add_diffvae_optimization_arg(parser)
    return parser


def _add_num_frames_args(
    parser: argparse.ArgumentParser, params: PipelineParams, supports_auto_duration: bool, noun: str
) -> None:
    """Add ``--num-frames`` (and, when optional, ``--auto-duration``) to *parser*.
    Shared by every pipeline CLI that exposes a frame-count knob (video-generating and
    audio-only alike) so the ``--num-frames``/``--auto-duration`` precedence rule lives in one
    place. ``supports_auto_duration`` makes ``--num-frames`` default to ``None`` and adds
    ``--auto-duration``, for pipelines that auto-predict duration from the caption via
    DurationHead when ``--num-frames`` is omitted. DurationHead ships from LTX-2.5
    (gemma4) onward, so the help text calls that out rather than silently requiring a
    2.5+ checkpoint.
    Resolving the parsed ``args.num_frames``/``args.auto_duration`` pair into the single value a
    pipeline's ``num_frames`` parameter expects happens automatically in
    ``_PipelineArgumentParser.parse_args`` (see ``_resolve_num_frames``) -- callers never need to
    do it themselves.
    """
    if supports_auto_duration:
        parser.add_argument(
            "--num-frames",
            type=int,
            default=None,
            help=(
                f"Number of frames {noun}, num_frames = 8 * k + 1, where k is a non-negative "
                "integer. Omit (along with --auto-duration) to auto-predict duration from the "
                "caption via DurationHead using its default range, or pass --auto-duration to "
                "control that range explicitly (only available for LTX-2.5 / gemma4 onward)."
            ),
        )
        parser.add_argument(
            "--auto-duration",
            dest="auto_duration",
            action=AutoDurationAction,
            nargs=2,
            metavar=("MIN_SECONDS", "MAX_SECONDS"),
            default=None,
            help=(
                "Auto-predict duration from the caption via DurationHead, clamped to "
                f"[MIN_SECONDS, MAX_SECONDS] (default when neither this nor --num-frames is given: "
                f"[{AutoDuration().min_seconds}, {AutoDuration().max_seconds}]). Ignored with a warning "
                "if --num-frames is also given."
            ),
        )
    else:
        parser.add_argument(
            "--num-frames",
            type=int,
            default=params.num_frames,
            help=f"Number of frames {noun}, num_frames = 8 * k + 1, "
            f"where k is a non-negative integer (default: {params.num_frames}).",
        )


def new_video_gen_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
    distilled: bool = False,
    supports_auto_duration: bool = False,
) -> argparse.ArgumentParser:
    """Build the shared video-generation argument parser.
    See :func:`_add_num_frames_args` for the ``supports_auto_duration``/``--auto-duration`` contract.
    """
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
    _add_num_frames_args(parser, params, supports_auto_duration, noun="to generate in the output video sequence")
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
            "Image conditioning: PATH FRAME_IDX STRENGTH [BINDING] [CRF]. "
            "PATH is a still file — PNG/JPEG (SDR) or scene-linear .exr (HDR). "
            "BINDING is auto (default when omitted), replace, or guide. With auto, FRAME_IDX 0 "
            "replaces the latent and later indices guide as keyframes. replace pins a still on a "
            f"causal latent start (0 or 1 + {VIDEO_SCALE_FACTORS.time} * k pixel frames). guide adds "
            "a keyframe at any pixel frame. When only one trailing token is given, a binding keyword "
            "selects BINDING and an integer selects CRF. STRENGTH is the conditioning strength. CRF "
            "optionally sets H.264 compression quality for SDR stills (0=lossless); when omitted, "
            "the checkpoint's default is used. Examples: --image path/to/start.jpg 0 1.0  or  "
            "--image path/to/end.jpg 160 1.0 guide  or  --image path/to/middle.jpg 65 1.0 replace  or  "
            "--image path/to/start.jpg 0 1.0 7  or  --image path/to/plate.exr 0 1.0 --hdr SRGB_LINEAR."
        ),
    )
    add_hdr_args(parser)
    add_marker_root_args(parser)
    return parser


def add_generated_keyframes_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--num-generated-keyframes`` to a pipeline that supports generated keyframe slots.
    Opt-in per CLI rather than added to the shared video parser: only pipelines that actually
    forward the value to their first diffusion stage should advertise the flag, otherwise it would
    parse and be silently ignored. Requires a checkpoint whose transformer config sets
    ``use_keyframes_abs_pos_embedding``; ``DiffusionStage`` raises otherwise.
    """
    parser.add_argument(
        "--num-generated-keyframes",
        type=int,
        default=0,
        help=(
            "Number of extra generated keyframes to place at evenly spaced interior frame "
            "positions (default: 0, off). Each keyframe relaxes the effective temporal "
            "compression at its position but costs a full latent frame of tokens while yielding "
            "one pixel frame, so N keyframes add roughly N / num_latent_frames to the sequence "
            "length. Requires a generated-keyframe checkpoint."
        ),
    )
    return parser


def add_keyframe_decode_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--decode-with-keyframes`` to a pipeline that supports generated keyframe slots."""
    parser.add_argument(
        "--decode-with-keyframes",
        action="store_true",
        default=False,
        help=(
            "Decode the output video through the keyframe-aware DiffVAE path, using the "
            "generated keyframe slots as anchors. Requires --num-generated-keyframes > 0 and "
            "a keyframe-trained video VAE. Conv VAEs log a warning and decode without anchors."
        ),
    )
    return parser


def chunk_config_from_args(args: argparse.Namespace) -> ChunkConfig | None:
    """Build ``ChunkConfig`` from CLI args, or ``None`` when chunking is disabled."""
    chunked = getattr(args, "chunked", False)
    chunk_pixel_frames = getattr(args, "chunk_pixel_frames", None)
    chunk_carry_frames = getattr(args, "chunk_carry_frames", None)
    chunk_blend_frames = getattr(args, "chunk_blend_frames", None)
    if not chunked and chunk_pixel_frames is None and chunk_carry_frames is None and chunk_blend_frames is None:
        return None
    kwargs: dict[str, int] = {}
    if chunk_pixel_frames is not None:
        kwargs["chunk_pixel_frames"] = chunk_pixel_frames
    if chunk_carry_frames is not None:
        kwargs["next_video_carry_frames"] = chunk_carry_frames
    if chunk_blend_frames is not None:
        kwargs["overlap_blend_frames"] = chunk_blend_frames
    return ChunkConfig(**kwargs)


def add_chunk_layout_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add chunking flags for long-video mode.
    With ``--chunked`` or any layout flag, ``stream_chunks`` uses a windowed layout.
    Omitted sizing values use ``ChunkConfig`` defaults (97 pixel frames, 25-pixel carry,
    crossfade over the full carry). With no chunk flags, the clip runs as one window.
    """
    parser.add_argument(
        "--chunked",
        action="store_true",
        help=(
            "Generate long clips in overlapping temporal windows using the default chunk "
            "layout (97-frame windows, 25-frame carry, crossfade over the full overlap). "
            "Override the layout with --chunk-pixel-frames, --chunk-carry-frames, "
            "and --chunk-blend-frames."
        ),
    )
    parser.add_argument(
        "--chunk-pixel-frames",
        type=int,
        default=None,
        help=(
            "Pixel frames in each chunked window (default: 97 when --chunk-carry-frames is set). "
            "Must be a causal-grid length such as 97 (8k+1). The stitched clip floors --num-frames "
            "to the causal grid when it is off-grid (for example 240 becomes 233); smaller windows "
            "produce more chunks."
        ),
    )
    parser.add_argument(
        "--chunk-carry-frames",
        type=int,
        default=None,
        help=(
            "Pixel frames each chunk passes to the next chunk for overlap continuity "
            "(default: 25 when --chunk-pixel-frames is set). Must be on the causal grid (8k+1), "
            "at least 17 when chunking, and less than the chunk window."
        ),
    )
    parser.add_argument(
        "--chunk-blend-frames",
        type=int,
        default=None,
        help=(
            "Decoded video frames crossfaded at each chunk seam (default: the full carry). "
            "Must be between 0 and --chunk-carry-frames; 0 uses a hard cut."
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
    parser.add_argument(
        "--video-path",
        type=resolve_existing_path,
        required=True,
        help=(
            "Path to the source video file, or a directory of scene-linear *.exr "
            "frames (HDR retake). EXR folders require --frame-rate (no container fps)."
        ),
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help=(
            "Frame rate (fps). Required when --video-path is an EXR-frame folder; "
            "must not be set for video files (container fps is used)."
        ),
    )
    parser.add_argument("--start-time", type=float, required=True, help="Start time of the region to regenerate (s).")
    parser.add_argument("--end-time", type=float, required=True, help="End time of the region to regenerate (s).")
    add_hdr_args(parser)
    add_marker_root_args(parser)
    return parser


def hdr_ic_lora_arg_parser(params: PipelineParams = LTX_2_4_PARAMS) -> argparse.ArgumentParser:
    """Argument parser for ACEScct SDR→HDR IC-LoRA (one-stage).
    Media checks (``--input`` / ``--input-colorspace`` / ``--frame-rate``) run inside
    :class:`_PipelineArgumentParser.parse_args`.
    """

    def _parse_exr_colorspace(value: str) -> EXRColorSpace:
        key = value.upper()
        try:
            return EXRColorSpace[key]
        except KeyError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid --exr-colorspace {value!r}; choose from {', '.join(cs.value for cs in EXRColorSpace)}"
            ) from exc

    parser = _PipelineArgumentParser(
        description="ACEScct SDR→HDR IC-LoRA (one-stage). CLI: IDT + EXR/HLG via encode_video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=resolve_existing_path,
        required=True,
        help=(
            "One input: MP4/MOV when --input-colorspace is srgb_gamma or srgb, or a directory of "
            "*.exr frames when srgb, acescg, or acescct. "
            "Output length matches the source; any frame count is padded to 8k+1 internally."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=resolve_path,
        required=True,
        help="Path to the HLG master MP4 (EXR frames go beside it as <stem>_<exr-colorspace>_exr/).",
    )
    parser.add_argument(
        "--exr-colorspace",
        type=_parse_exr_colorspace,
        choices=list(EXRColorSpace),
        default=EXRColorSpace.ACESCG,
        metavar="{" + ",".join(cs.value for cs in EXRColorSpace) + "}",
        help=(
            "Colour space of the EXR sidecar beside --output-path. "
            "acescg / srgb_linear write scene-linear; acescct writes VAE log codes. "
            "HLG master is always BT.2020/HLG. Default: acescg."
        ),
    )
    parser.add_argument("--hdr-lora", type=resolve_existing_path, required=True)
    parser.add_argument(
        "--text-embeddings",
        type=resolve_existing_path,
        required=True,
        help=".safetensors with video_context.",
    )
    # Prompts arrive as precomputed embeddings and there is no audio or duration prediction,
    # so only the transformer and video VAE components are declared.
    add_model_args(
        parser,
        params=params,
        distilled=True,
        text_encoder=False,
        audio_vae=False,
        duration_head=False,
    )
    parser.add_argument(
        "--input-colorspace",
        type=HDRICLoraInputColorSpace,
        choices=list(HDRICLoraInputColorSpace),
        default=HDRICLoraInputColorSpace.SRGB_GAMMA,
        help=(
            "Source encoding: srgb_gamma=display MP4/MOV (EOTF→ACEScct); "
            "srgb=linear Rec.709 MP4/MOV or EXR directory; acescg/acescct=EXR directory."
        ),
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Required for EXR-frame directories; must not be set for MP4/MOV (container fps).",
    )
    add_seed_arg(parser, params=params)
    parser.add_argument(
        "--high-quality",
        action="store_true",
        help=(
            "High-quality HDR mode. Generates at 2x frame count internally "
            "and keeps every other frame for smoother output. ~2x slower."
        ),
    )
    add_offload_arg(parser)
    parser.add_argument("--no-quantization", action="store_true", help="Disable FP8 cast on the transformer.")
    add_diffvae_optimization_arg(parser)
    parser.add_argument(
        "--no-keyframes",
        action="store_true",
        help=(
            "Disable DFR-style seam keyframes. They are on by default: every "
            "resolve_canvas x8 border gets a generated HDR slot and a 1-frame SDR "
            "guide. Requires a transformer with use_keyframes_abs_pos_embedding "
            "and --video-vae-path pointing at a keyframe-trained video VAE."
        ),
    )
    parser.add_argument(
        "--keyframe-strength",
        type=float,
        default=0.95,
        help=(
            "DFR seam keyframe strength when keyframes are on (the default; "
            "ignored with --no-keyframes). Default: 0.95."
        ),
    )
    add_marker_root_args(parser)
    return parser


def dubit_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
) -> argparse.ArgumentParser:
    """Argument parser for the Dub-It pipeline.
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
        help=(
            "Reference video file (video + audio). Frame count/fps and audio identity come "
            "from this container; EXR folders are not supported."
        ),
    )
    parser.add_argument(
        "--reference-strength",
        type=float,
        default=1.0,
        help="Strength for IC-LoRA video reference conditioning (default: 1.0).",
    )
    add_marker_root_args(parser)
    return parser


def default_1_stage_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
    supports_auto_duration: bool = False,
) -> argparse.ArgumentParser:
    video_guider = params.video_guider_params
    audio_guider = params.audio_guider_params
    parser = new_video_gen_arg_parser(params=params, supports_auto_duration=supports_auto_duration)
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
    _add_num_frames_args(
        parser, params, supports_auto_duration=True, noun="used to derive audio duration (num-frames / frame-rate)"
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
    add_marker_root_args(parser)
    return parser


def default_2_stage_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
    supports_auto_duration: bool = False,
) -> argparse.ArgumentParser:
    parser = default_1_stage_arg_parser(params=params, supports_auto_duration=supports_auto_duration)
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


def hq_2_stage_arg_parser(
    params: PipelineParams = LTX_2_3_HQ_PARAMS,
    supports_auto_duration: bool = False,
) -> argparse.ArgumentParser:
    parser = default_2_stage_arg_parser(params=params, supports_auto_duration=supports_auto_duration)
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


def default_2_stage_distilled_arg_parser(
    params: PipelineParams = LTX_2_3_PARAMS,
    supports_auto_duration: bool = False,
) -> argparse.ArgumentParser:
    parser = new_video_gen_arg_parser(params=params, distilled=True, supports_auto_duration=supports_auto_duration)
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
