"""
Audio-visual media captioning using multimodal models.
This module provides captioning capabilities for videos with audio using:
- Qwen2.5-Omni: Local model supporting text, audio, image, and video inputs (default)
- Gemini Flash: Cloud-based API for audio-visual captioning
Requirements:
- Qwen2.5-Omni: transformers>=4.50, torch
- Gemini Flash: google-generativeai (uv pip install google-generativeai)
  Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable
"""

import itertools
import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import torch

from ltx_core.devices import resolve_device

# Instruction for audio-visual captioning (default) - includes speech transcription and sounds
DEFAULT_CAPTION_INSTRUCTION = """\
Analyze this media and provide a detailed caption in the following EXACT format. Fill in ALL sections:

[VISUAL]: <Detailed description of people, objects, actions, settings, colors, and movements>
[SPEECH]: <Word-for-word transcription of everything spoken.
           Listen carefully and transcribe the exact words. If no speech, write "None">
[SOUNDS]: <Description of music, ambient sounds, sound effects. If none, write "None">
[TEXT]: <Any on-screen text visible. If none, write "None">

You MUST fill in all four sections. For [SPEECH], transcribe the actual words spoken, not a summary."""

# Instruction for video-only captioning (no audio processing)
VIDEO_ONLY_CAPTION_INSTRUCTION = """\
Analyze this media and provide a detailed caption in the following EXACT format. Fill in ALL sections:

[VISUAL]: <Detailed description of people, objects, actions, settings, colors, and movements>
[TEXT]: <Any on-screen text visible. If none, write "None">

You MUST fill in both sections."""


class CaptionerType(str, Enum):
    """Enum for different types of media captioners."""

    QWEN_OMNI = "qwen_omni"  # Local Qwen2.5-Omni model (audio + video)
    GEMINI_FLASH = "gemini_flash"  # Gemini Flash API (audio + video)


def create_captioner(captioner_type: CaptionerType, **kwargs) -> "MediaCaptioningModel":
    """Factory function to create a media captioner.
    Args:
        captioner_type: The type of captioner to create
        **kwargs: Additional arguments to pass to the captioner constructor
    Returns:
        An instance of a MediaCaptioningModel
    """
    match captioner_type:
        case CaptionerType.QWEN_OMNI:
            return QwenOmniCaptioner(**kwargs)
        case CaptionerType.GEMINI_FLASH:
            return GeminiFlashCaptioner(**kwargs)
        case _:
            raise ValueError(f"Unsupported captioner type: {captioner_type}")


class MediaCaptioningModel(ABC):
    """Abstract base class for audio-visual media captioning models."""

    @abstractmethod
    def caption(self, path: str | Path, **kwargs) -> str:
        """Generate a caption for the given video or image.
        Args:
            path: Path to the video/image file to caption
        Returns:
            A string containing the generated caption
        """

    @property
    @abstractmethod
    def supports_audio(self) -> bool:
        """Whether this captioner supports audio input."""

    @staticmethod
    def _is_image_file(path: str | Path) -> bool:
        """Check if the file is an image based on extension."""
        return str(path).lower().endswith((".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp"))

    @staticmethod
    def _is_video_file(path: str | Path) -> bool:
        """Check if the file is a video based on extension."""
        return str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

    @staticmethod
    def _clean_raw_caption(caption: str) -> str:
        """Clean up the raw caption by removing common VLM patterns."""
        start = ["The", "This"]
        kind = ["video", "image", "scene", "animated sequence", "clip", "footage"]
        act = ["displays", "shows", "features", "depicts", "presents", "showcases", "captures", "contains"]

        for x, y, z in itertools.product(start, kind, act):
            caption = caption.replace(f"{x} {y} {z} ", "", 1)

        return caption


class QwenOmniCaptioner(MediaCaptioningModel):
    """Audio-visual captioning using Alibaba's Qwen2.5-Omni model.
    Qwen2.5-Omni is an end-to-end multimodal model that can perceive text, images, audio, and video.
    It uses a Thinker-Talker architecture where the Thinker generates text and the Talker can
    generate speech. For captioning, we use only the Thinker component for text generation.
    Key features:
    - Block-wise processing for streaming multimodal inputs
    - TMRoPE (Time-aligned Multimodal RoPE) for synchronizing video and audio timestamps
    - Can extract and process audio directly from video files
    See: https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_omni
    Model: Qwen/Qwen2.5-Omni-7B (7B parameters)
    """

    MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

    # Default system prompt required by Qwen2.5-Omni for proper audio processing
    DEFAULT_SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def __init__(
        self,
        device: str | torch.device | None = None,
        use_8bit: bool = False,
        instruction: str | None = None,
    ):
        """
        Initialize the Qwen2.5-Omni captioner.
        Args:
            device: Device to use for inference (e.g., 'auto', 'cuda', 'mps', 'cpu')
            use_8bit: Whether to use 8-bit quantization for reduced memory usage
            instruction: Custom instruction prompt. If None, uses the default instruction
        """
        self.device = resolve_device(device)
        self.instruction = instruction
        self._load_model(use_8bit=use_8bit)

    @property
    def supports_audio(self) -> bool:
        return True

    def caption(
        self,
        path: str | Path,
        fps: int = 1,
        include_audio: bool = True,
        clean_caption: bool = True,
    ) -> str:
        """Generate a caption for the given video or image.
        Args:
            path: Path to the video/image file to caption
            fps: Frames per second to sample from videos
            include_audio: Whether to include audio in the captioning (for videos)
            clean_caption: Whether to clean up the raw caption by removing common VLM patterns
        Returns:
            A string containing the generated caption
        """
        path = Path(path)
        is_image = self._is_image_file(path)
        is_video = self._is_video_file(path)

        # Determine if we should process audio
        use_audio = include_audio and is_video

        # Use custom instruction if provided, otherwise pick appropriate default
        if self.instruction is not None:
            instruction = self.instruction
        else:
            instruction = DEFAULT_CAPTION_INSTRUCTION if use_audio else VIDEO_ONLY_CAPTION_INSTRUCTION

        # Build the user content based on media type
        # Based on HuggingFace docs: https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_omni
        user_content = []

        if is_image:
            user_content.append({"type": "image", "image": str(path)})
        elif is_video:
            user_content.append({"type": "video", "video": str(path)})

        # Add the instruction text
        user_content.append({"type": "text", "text": instruction})

        # Build conversation - use the default system prompt required by Qwen2.5-Omni
        # Using a custom system prompt causes warnings and may affect audio processing
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.DEFAULT_SYSTEM_PROMPT}],
            },
            {"role": "user", "content": user_content},
        ]

        # Process inputs using the processor's apply_chat_template
        # For videos with audio, use load_audio_from_video=True and use_audio_in_video=True
        inputs = self.processor.apply_chat_template(
            messages,
            load_audio_from_video=use_audio,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            fps=fps,
            padding=True,
            use_audio_in_video=use_audio,
        ).to(self.model.device)

        # Generate caption (text only, using Thinker-only model)
        # Note: For Qwen2_5OmniThinkerForConditionalGeneration, use standard generate params
        # (not thinker_ prefixed ones, those are for the full Qwen2_5OmniForConditionalGeneration)
        input_len = inputs["input_ids"].shape[1]

        output_tokens = self.model.generate(
            **inputs,
            use_audio_in_video=use_audio,
            do_sample=False,
            max_new_tokens=1024,
        )

        # Extract only the generated tokens (exclude the input/prompt tokens)
        generated_tokens = output_tokens[:, input_len:]

        # Decode only the generated response
        caption_raw = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Remove hallucinated conversation turns (e.g., "Human\nHuman\n..." or "Human: ...")
        # This is a known issue with chat models continuing to generate fake turns
        # We look for patterns that are clearly hallucinated chat turns, not legitimate uses of "human"

        # Match "\nHuman" followed by ":", "\n", or end of string (chat turn patterns)
        # This won't match "A human walks..." or "...the human body..."
        caption_raw = re.split(r"\nHuman(?::|(?:\s*\n)|$)", caption_raw, maxsplit=1)[0]
        caption_raw = caption_raw.strip()

        # Clean up caption if requested
        return self._clean_raw_caption(caption_raw) if clean_caption else caption_raw

    def _load_model(self, use_8bit: bool) -> None:
        """Load the Qwen2.5-Omni model and processor.
        Uses the Thinker-only model (Qwen2_5OmniThinkerForConditionalGeneration) for text generation
        to save compute by not loading the audio generation components.
        """
        from transformers import (  # noqa: PLC0415
            BitsAndBytesConfig,
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        if use_8bit and self.device.type == "mps":
            raise ValueError("8-bit Qwen-Omni captioning uses bitsandbytes and is not supported on MPS.")

        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None

        # Use Thinker-only model for text generation (saves memory by not loading Talker)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            device_map={"": self.device},
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.MODEL_ID)


class GeminiFlashCaptioner(MediaCaptioningModel):
    """Audio-visual captioning using Google's Gemini Flash API.
    Gemini Flash is a cloud-based multimodal model that natively supports
    audio and video understanding. Requires a Google API key.
    Note: This captioner requires the `google-generativeai` package and a valid API key.
    Set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable, or pass the key directly.
    """

    MODEL_ID = "gemini-flash-lite-latest"

    def __init__(
        self,
        api_key: str | None = None,
        instruction: str | None = None,
    ):
        """Initialize the Gemini Flash captioner.
        Args:
            api_key: Google API key. If not provided, will look for
                     GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
            instruction: Custom instruction prompt. If None, uses the default instruction
        """
        self.instruction = instruction
        self._init_client(api_key)

    @property
    def supports_audio(self) -> bool:
        return True

    def caption(
        self,
        path: str | Path,
        fps: int = 3,  # noqa: ARG002 - kept for API compatibility
        include_audio: bool = True,
        clean_caption: bool = True,
    ) -> str:
        """Generate a caption for the given video or image.
        Args:
            path: Path to the video/image file to caption
            fps: Frames per second (not used for Gemini, kept for API compatibility)
            include_audio: Whether to include audio content in the caption
            clean_caption: Whether to clean up the raw caption
        Returns:
            A string containing the generated caption
        """
        import time  # noqa: PLC0415

        path = Path(path)
        is_video = self._is_video_file(path)
        use_audio = include_audio and is_video

        # Use custom instruction if provided, otherwise pick appropriate default
        if self.instruction is not None:
            instruction = self.instruction
        else:
            instruction = DEFAULT_CAPTION_INSTRUCTION if use_audio else VIDEO_ONLY_CAPTION_INSTRUCTION

        # Upload the file to Gemini
        uploaded_file = self._genai.upload_file(path)

        # Wait for processing to complete (videos need time to process)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = self._genai.get_file(uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"File processing failed: {uploaded_file.state.name}")

        # Generate caption
        response = self._model.generate_content([uploaded_file, instruction])

        caption_raw = response.text

        # Clean up the uploaded file
        self._genai.delete_file(uploaded_file.name)

        # Clean up caption if requested
        return self._clean_raw_caption(caption_raw) if clean_caption else caption_raw

    def _init_client(self, api_key: str | None) -> None:
        """Initialize the Gemini API client."""
        import os  # noqa: PLC0415

        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "The `google-generativeai` package is required for Gemini Flash captioning. "
                "Install it with: `uv pip install google-generativeai`"
            ) from e

        # Get API key from argument or environment
        # GEMINI_API_KEY is the recommended variable, GOOGLE_API_KEY also works
        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "Gemini API key is required. Provide it via the `api_key` argument "
                "or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        # Configure the genai library with the API key
        genai.configure(api_key=resolved_api_key)

        # Store reference to genai module for file operations
        self._genai = genai

        # Initialize the model
        self._model = genai.GenerativeModel(self.MODEL_ID)


def example() -> None:
    """Example usage of the captioning module."""
    import sys  # noqa: PLC0415

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <video_path> [captioner_type]")  # noqa: T201
        print("  captioner_type: qwen_omni (default) or gemini_flash")  # noqa: T201
        sys.exit(1)

    video_path = sys.argv[1]
    captioner_type = CaptionerType(sys.argv[2]) if len(sys.argv) > 2 else CaptionerType.QWEN_OMNI

    print(f"Using {captioner_type.value} captioner:")  # noqa: T201
    captioner = create_captioner(captioner_type)
    caption = captioner.caption(video_path)
    print(f"CAPTION: {caption}")  # noqa: T201


if __name__ == "__main__":
    example()
