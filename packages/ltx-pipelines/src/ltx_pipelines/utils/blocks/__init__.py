"""Pipeline blocks — each block owns its model lifecycle.
Blocks build a model on each ``__call__``, use it, then free GPU memory.
This eliminates manual ``del model; cleanup_memory()`` in pipelines: each
block is self-contained, so no central model-coordinator object is needed.
"""

from ltx_core.components.noisers import Noiser
from ltx_core.loader.simple_builder import SimpleBuilder as Builder
from ltx_pipelines.utils.blocks.audio_conditioner import AudioConditioner
from ltx_pipelines.utils.blocks.audio_decoder import AudioDecoder
from ltx_pipelines.utils.blocks.diffusion_stage import (
    DenoisingLoop,
    DiffusionStage,
    ModelWrapper,
    RecordingDiffusionStage,
)
from ltx_pipelines.utils.blocks.duration_predictor import (
    DurationPredictor,
    require_num_frames_source,
    resolve_num_frames,
)
from ltx_pipelines.utils.blocks.image_conditioner import ImageConditioner
from ltx_pipelines.utils.blocks.prompt_encoder import PromptEncoder
from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
from ltx_pipelines.utils.blocks.video_upsampler import VideoUpsampler
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.helpers import cleanup_memory

__all__ = [
    "AudioConditioner",
    "AudioDecoder",
    "Builder",
    "DenoisingLoop",
    "DiffusionStage",
    "DurationPredictor",
    "ImageConditioner",
    "ModelWrapper",
    "Noiser",
    "PromptEncoder",
    "RecordingDiffusionStage",
    "VideoDecoder",
    "VideoUpsampler",
    "cleanup_memory",
    "detect_params",
    "require_num_frames_source",
    "resolve_num_frames",
]
