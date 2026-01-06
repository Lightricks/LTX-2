"""
LTX-2 Video Generation UI (lt1.py)

Gradio-based web interface for LTX-2 video generation with:
- Two-stage pipeline (low-res + hi-res refinement)
- Joint audio-video generation
- Image conditioning (I2V)
- LoRA support
- Memory optimization (offload, FP8, block swap)
- Prompt enhancement with Gemma
"""

import os
import sys

# Try to use local patched gradio from modules/ if available (allows jobs to continue after browser disconnect)
_modules_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
if os.path.exists(_modules_path):
    sys.path.insert(0, _modules_path)

import gradio as gr
from gradio import themes
from gradio.themes.utils import colors
import time
import random
import subprocess
import re
from typing import Generator, List, Tuple, Optional, Dict
import threading
import json
from PIL import Image

# Global state
stop_event = threading.Event()
current_process = None
current_output_filename = None


# =============================================================================
# Progress Parsing
# =============================================================================

def parse_ltx_progress_line(line: str) -> Optional[str]:
    """Parse LTX-2 generation progress output."""
    line = line.strip()

    # LTX-specific progress messages
    if ">>> Loading text encoder" in line:
        return "Loading text encoder..."
    if ">>> Enhancing prompt" in line:
        return "Enhancing prompt with Gemma..."
    if ">>> Encoding prompts" in line:
        return "Encoding text prompts..."
    # One-stage pipeline
    if ">>> One-stage: Loading" in line:
        return "One-stage: Loading models..."
    if ">>> One-stage: Generating" in line:
        return "One-stage: Generating at full resolution..."
    if ">>> One-stage completed" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"One-stage completed ({match.group(1)}s)"
    # Refine-only pipeline
    if ">>> Refine-only mode" in line:
        return "Refine-only: Loading input video..."
    if ">>> Input video encoded" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"Video encoded ({match.group(1)}s)"
    # Two-stage pipeline
    if ">>> Stage 1: Loading" in line:
        return "Stage 1: Loading models..."
    if ">>> Stage 1: Generating" in line:
        return "Stage 1: Low-res generation..."
    if ">>> Stage 1 completed" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"Stage 1 completed ({match.group(1)}s)"
    if ">>> Upsampling" in line:
        return "Upsampling latents (2x)..."
    if ">>> Stage 2: Loading" in line:
        return "Stage 2: Loading refined model..."
    if ">>> Stage 2: Refining" in line:
        return "Stage 2: Hi-res refinement..."
    if ">>> Stage 2 completed" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"Stage 2 completed ({match.group(1)}s)"
    if ">>> Decoding video" in line:
        return "Decoding video from latents..."
    if ">>> Decoding audio" in line:
        return "Decoding audio..."
    if ">>> Encoding video" in line:
        return "Encoding video to MP4..."
    if ">>> Done!" in line:
        return "Generation complete!"
    if ">>> Total generation time" in line:
        match = re.search(r'(\d+\.?\d*)s', line)
        if match:
            return f"Total time: {match.group(1)}s"

    # TQDM progress bar parsing
    match = re.search(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[.*?<([\d:]+)', line)
    if match:
        percent = match.group(1)
        current = match.group(2)
        total = match.group(3)
        eta = match.group(4)
        return f"Denoising: {percent}% ({current}/{total}) ETA: {eta}"

    return None


# =============================================================================
# LoRA Discovery
# =============================================================================

def get_ltx_lora_options(lora_folder: str = "lora") -> List[str]:
    """
    Discover LTX LoRAs in the specified folder.
    Returns ['None'] + list of .safetensors files.
    """
    if not os.path.exists(lora_folder):
        return ["None"]

    lora_items = []
    for item in os.listdir(lora_folder):
        if item.endswith(".safetensors"):
            lora_items.append(item)

    lora_items.sort(key=str.lower)
    return ["None"] + lora_items


def refresh_lora_dropdown(lora_folder: str):
    """Refresh LoRA dropdown choices."""
    new_choices = get_ltx_lora_options(lora_folder)
    return gr.update(choices=new_choices, value="None")


# =============================================================================
# Validation
# =============================================================================

def validate_num_frames(num_frames: int) -> Optional[str]:
    """Validate num_frames is in 8*K + 1 format."""
    if (num_frames - 1) % 8 != 0:
        valid_examples = [8*k + 1 for k in range(1, 20)]
        return f"Error: num_frames must be 8*K + 1 (e.g., {', '.join(map(str, valid_examples[:5]))}...)"
    return None


def validate_resolution(width: int, height: int) -> Optional[str]:
    """Validate resolution is divisible by 64."""
    if width % 64 != 0 or height % 64 != 0:
        return f"Error: width ({width}) and height ({height}) must be divisible by 64"
    return None


def validate_model_paths(checkpoint_path: str, spatial_upsampler_path: str,
                         distilled_lora_path: str, gemma_root: str,
                         is_one_stage: bool = False,
                         is_refine_only: bool = False) -> Optional[str]:
    """Validate all required model paths exist."""
    # Core paths required for all pipelines
    paths = [
        ("LTX Checkpoint", checkpoint_path),
        ("Gemma Root", gemma_root),
    ]
    # Two-stage specific paths (spatial upsampler always needed for two-stage)
    if not is_one_stage and not is_refine_only:
        paths.extend([
            ("Spatial Upsampler", spatial_upsampler_path),
            ("Distilled LoRA", distilled_lora_path),
        ])
    # Refine-only: distilled LoRA is optional, but validate if provided
    if is_refine_only:
        # Only validate distilled LoRA if path is provided
        if distilled_lora_path and distilled_lora_path.strip():
            paths.append(("Distilled LoRA", distilled_lora_path))
    for name, path in paths:
        if not path or not path.strip():
            return f"Error: {name} path is required"
        if not os.path.exists(path):
            return f"Error: {name} not found: {path}"
    return None


# =============================================================================
# Image Dimension Helpers
# =============================================================================

def update_image_dimensions(image_path):
    """Update original dimensions when image is uploaded."""
    if image_path is None:
        return "", gr.update(), gr.update()
    try:
        img = Image.open(image_path)
        w, h = img.size
        original_dims_str = f"{w}x{h}"
        # Calculate dimensions snapped to nearest multiple of 64 while maintaining aspect ratio
        new_w = round(w / 64) * 64
        new_h = round(h / 64) * 64
        new_w = max(64, new_w)
        new_h = max(64, new_h)
        return original_dims_str, gr.update(value=new_w), gr.update(value=new_h)
    except Exception as e:
        print(f"Error reading image dimensions: {e}")
        return "", gr.update(), gr.update()


def update_resolution_from_scale(scale, original_dims):
    """Update dimensions based on scale percentage (divisible by 64)."""
    if not original_dims:
        return gr.update(), gr.update()
    try:
        scale = float(scale) if scale is not None else 100.0
        if scale <= 0:
            scale = 100.0

        orig_w, orig_h = map(int, original_dims.split('x'))
        scale_factor = scale / 100.0

        # Calculate and round to the nearest multiple of 64
        new_w = round((orig_w * scale_factor) / 64) * 64
        new_h = round((orig_h * scale_factor) / 64) * 64

        # Ensure minimum size (must be multiple of 64)
        new_w = max(64, new_w)
        new_h = max(64, new_h)

        return gr.update(value=new_w), gr.update(value=new_h)
    except Exception as e:
        print(f"Error updating from scale: {e}")
        return gr.update(), gr.update()


def calculate_width_from_height(height, original_dims):
    """Calculate width based on height maintaining aspect ratio (divisible by 64)."""
    if not original_dims or height is None:
        return gr.update()
    try:
        height = int(height)
        if height <= 0:
            return gr.update()
        height = (height // 64) * 64
        height = max(64, height)

        orig_w, orig_h = map(int, original_dims.split('x'))
        if orig_h == 0:
            return gr.update()
        aspect_ratio = orig_w / orig_h
        new_width = round((height * aspect_ratio) / 64) * 64
        return gr.update(value=max(64, new_width))

    except Exception as e:
        print(f"Error calculating width: {e}")
        return gr.update()


def calculate_height_from_width(width, original_dims):
    """Calculate height based on width maintaining aspect ratio (divisible by 64)."""
    if not original_dims or width is None:
        return gr.update()
    try:
        width = int(width)
        if width <= 0:
            return gr.update()
        width = (width // 64) * 64
        width = max(64, width)

        orig_w, orig_h = map(int, original_dims.split('x'))
        if orig_w == 0:
            return gr.update()
        aspect_ratio = orig_w / orig_h
        new_height = round((width / aspect_ratio) / 64) * 64
        return gr.update(value=max(64, new_height))

    except Exception as e:
        print(f"Error calculating height: {e}")
        return gr.update()


def get_video_info(video_path: str) -> dict:
    """Get video information using PyAV."""
    try:
        import av
        with av.open(video_path) as container:
            # Get first video stream
            video_stream = container.streams.video[0]

            width = video_stream.width
            height = video_stream.height

            # Calculate FPS from average_rate or guessed_rate
            if video_stream.average_rate:
                fps = float(video_stream.average_rate)
            elif video_stream.guessed_rate:
                fps = float(video_stream.guessed_rate)
            else:
                fps = 30.0  # fallback

            # Calculate duration
            if video_stream.duration and video_stream.time_base:
                duration = float(video_stream.duration * video_stream.time_base)
            elif container.duration:
                duration = container.duration / av.time_base
            else:
                duration = 0

            total_frames = int(video_stream.frames) if video_stream.frames else int(duration * fps)

            return {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration
            }
    except Exception as e:
        print(f"Error extracting video info: {e}")
        return {}


def update_video_dimensions(video_path):
    """Update dimensions and frame count when video is uploaded."""
    if video_path is None:
        return "", gr.update(), gr.update(), gr.update(), gr.update()
    try:
        info = get_video_info(video_path)
        if info:
            w, h = info.get("width", 0), info.get("height", 0)
            if w and h:
                original_dims_str = f"{w}x{h}"
                # Snap to nearest multiple of 64
                new_w = round(w / 64) * 64
                new_h = round(h / 64) * 64
                new_w = max(64, new_w)
                new_h = max(64, new_h)
                # Calculate frame count (snap to 8*K+1 format)
                if info.get("total_frames"):
                    num_frames = info["total_frames"]
                elif info.get("duration") and info.get("fps"):
                    num_frames = int(info["duration"] * info["fps"])
                else:
                    num_frames = 121
                # Snap to nearest 8*K+1
                k = max(1, round((num_frames - 1) / 8))
                num_frames = 8 * k + 1
                fps = info.get("fps", 24)
                return original_dims_str, gr.update(value=new_w), gr.update(value=new_h), gr.update(value=num_frames), gr.update(value=int(fps))
    except Exception as e:
        print(f"Error reading video dimensions: {e}")
    return "", gr.update(), gr.update(), gr.update(), gr.update()


def extract_video_metadata(video_path: str) -> dict:
    """Extract metadata from video file using PyAV."""
    try:
        import av
        with av.open(video_path) as container:
            # Get comment metadata from container
            comment = container.metadata.get('comment', '{}')
            return json.loads(comment)
    except Exception as e:
        print(f"Metadata extraction failed: {str(e)}")
        return {}


def add_metadata_to_video(video_path: str, parameters: dict) -> None:
    """Add generation parameters to video metadata using ffmpeg."""
    # Convert parameters to JSON string
    params_json = json.dumps(parameters, indent=2)

    # Temporary output path
    temp_path = video_path.replace(".mp4", "_temp.mp4")

    # FFmpeg command to add metadata without re-encoding
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-metadata', f'comment={params_json}',
        '-codec', 'copy',
        temp_path
    ]

    try:
        # Execute FFmpeg command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Replace original file with the metadata-enhanced version
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError as e:
        print(f"Failed to add metadata: {e.stderr.decode()}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Error: {str(e)}")


def extract_video_details(video_path: str) -> Tuple[dict, str]:
    """Extract both metadata and video information."""
    metadata = extract_video_metadata(video_path)
    video_details = get_video_info(video_path)

    # Combine metadata with video details
    for key, value in video_details.items():
        if key not in metadata:
            metadata[key] = value

    # Return both the updated metadata and a status message
    return metadata, "Video details extracted successfully"


# =============================================================================
# Video Generation
# =============================================================================

def generate_ltx_video(
    # Prompts
    prompt: str,
    negative_prompt: str,
    # Model paths
    checkpoint_path: str,
    gemma_root: str,
    spatial_upsampler_path: str,
    distilled_lora_path: str,
    distilled_lora_strength: float,
    # Generation parameters
    mode: str,
    pipeline: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    cfg_guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    # Image conditioning (for I2V)
    input_image: str,
    image_frame_idx: int,
    image_strength: float,
    # End image conditioning
    end_image: str,
    end_image_strength: float,
    # Video input (for V2V / refine)
    input_video: str,
    refine_strength: float,
    refine_steps: int,
    # Audio & prompt
    disable_audio: bool,
    enhance_prompt: bool,
    # Memory optimization
    offload: bool,
    enable_fp8: bool,
    enable_dit_block_swap: bool,
    dit_blocks_in_memory: int,
    enable_text_encoder_block_swap: bool,
    text_encoder_blocks_in_memory: int,
    enable_refiner_block_swap: bool,
    refiner_blocks_in_memory: int,
    # LoRA
    lora_folder: str,
    user_lora: str,
    user_lora_strength: float,
    # Output
    save_path: str,
    batch_size: int,
) -> Generator[Tuple[List[Tuple[str, str]], Optional[str], str, str], None, None]:
    """
    Generate video using LTX-2 pipeline.

    Yields: (gallery_items, preview_path, status_text, progress_text)
    """
    global stop_event, current_process, current_output_filename
    stop_event.clear()
    current_process = None
    current_output_filename = None

    # Validate inputs
    error = validate_num_frames(int(num_frames))
    if error:
        yield [], None, error, ""
        return

    error = validate_resolution(int(width), int(height))
    if error:
        yield [], None, error, ""
        return

    # Round num_frames to valid 8*K+1 format
    num_frames = int(num_frames)
    k = max(1, round((num_frames - 1) / 8))
    num_frames = 8 * k + 1

    is_one_stage = (pipeline == "one-stage")
    is_refine_only = (pipeline == "refine-only")
    error = validate_model_paths(checkpoint_path, spatial_upsampler_path,
                                  distilled_lora_path, gemma_root,
                                  is_one_stage=is_one_stage,
                                  is_refine_only=is_refine_only)
    if error:
        yield [], None, error, ""
        return

    # Check image for I2V mode
    if mode == "i2v" and not input_image:
        yield [], None, "Error: Input image required for I2V mode", ""
        return

    # Check video for V2V/refine-only mode
    if (mode == "v2v" or pipeline == "refine-only") and not input_video:
        yield [], None, "Error: Input video required for V2V/refine-only mode", ""
        return

    os.makedirs(save_path, exist_ok=True)
    all_generated_videos = []

    for i in range(int(batch_size)):
        if stop_event.is_set():
            current_process = None
            yield all_generated_videos, None, "Generation stopped by user.", ""
            return

        # Seed handling
        current_seed = seed
        if seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif int(batch_size) > 1:
            current_seed = seed + i

        status_text = f"Processing {i+1}/{batch_size} (Seed: {current_seed})"
        yield all_generated_videos.copy(), None, status_text, "Starting generation..."

        timestamp = int(time.time())
        run_id = f"{timestamp}_{random.randint(1000, 9999)}"
        output_filename = os.path.join(save_path, f"ltx_{mode}_{timestamp}_{current_seed}.mp4")
        current_output_filename = output_filename

        # Build command
        command = [
            sys.executable, "ltx_generate_video.py",
            "--checkpoint-path", checkpoint_path,
            "--gemma-root", gemma_root,
            "--prompt", str(prompt),
            "--negative-prompt", str(negative_prompt),
            "--num-frames", str(int(num_frames)),
            "--frame-rate", str(float(frame_rate)),
            "--width", str(int(width)),
            "--height", str(int(height)),
            "--cfg-guidance-scale", str(float(cfg_guidance_scale)),
            "--num-inference-steps", str(int(num_inference_steps)),
            "--seed", str(current_seed),
            "--output-path", output_filename,
        ]

        # Pipeline selection
        if is_one_stage:
            command.append("--one-stage")
        elif is_refine_only:
            command.append("--refine-only")
            # Refine-only: distilled LoRA is optional
            if distilled_lora_path and distilled_lora_path.strip() and os.path.exists(distilled_lora_path):
                command.extend(["--distilled-lora", distilled_lora_path, str(distilled_lora_strength)])
        else:
            # Two-stage specific: spatial upsampler and distilled LoRA
            command.extend(["--spatial-upsampler-path", spatial_upsampler_path])
            command.extend(["--distilled-lora", distilled_lora_path, str(distilled_lora_strength)])

        # Video input (V2V / refine)
        if input_video:
            command.extend(["--input-video", str(input_video)])
            command.extend(["--refine-strength", str(float(refine_strength))])
            command.extend(["--refine-steps", str(int(refine_steps))])

        # Image conditioning (I2V)
        if mode == "i2v" and input_image:
            command.extend(["--image", str(input_image), str(int(image_frame_idx)), str(float(image_strength))])

        # End image conditioning (place at last frame)
        if end_image:
            last_frame_idx = int(num_frames) - 1
            command.extend(["--image", str(end_image), str(last_frame_idx), str(float(end_image_strength))])

        # User LoRA
        if user_lora and user_lora != "None" and lora_folder:
            lora_path = os.path.join(lora_folder, user_lora)
            if os.path.exists(lora_path):
                command.extend(["--lora", lora_path, str(user_lora_strength)])

        # Flags
        if disable_audio:
            command.append("--disable-audio")
        if enhance_prompt:
            command.append("--enhance-prompt")
        if offload:
            command.append("--offload")
        if enable_fp8:
            command.append("--enable-fp8")
        # Block swap controls (separate for DiT, text encoder, and refiner)
        if enable_dit_block_swap:
            command.append("--enable-dit-block-swap")
            command.extend(["--dit-blocks-in-memory", str(int(dit_blocks_in_memory))])
        if enable_text_encoder_block_swap:
            command.append("--enable-text-encoder-block-swap")
            command.extend(["--text-encoder-blocks-in-memory", str(int(text_encoder_blocks_in_memory))])
        if enable_refiner_block_swap:
            command.append("--enable-refiner-block-swap")
            command.extend(["--refiner-blocks-in-memory", str(int(refiner_blocks_in_memory))])

        # Print command for debugging
        print("\n" + "=" * 80)
        print(f"LAUNCHING COMMAND (Batch {i+1}/{batch_size}):")
        print(" ".join(command))
        print("=" * 80 + "\n")

        try:
            start_time = time.perf_counter()

            # Write subprocess output to log file
            os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
            log_file_path = os.path.join(save_path, "logs", f"gen_{run_id}.log")
            log_file = open(log_file_path, "w", encoding="utf-8")

            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            current_process = process
            last_progress = ""
            last_log_position = 0

            while True:
                if stop_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    current_process = None
                    log_file.close()
                    yield all_generated_videos, None, "Generation stopped by user.", ""
                    return

                # Read new lines from log file
                try:
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        f.seek(last_log_position)
                        new_lines = f.readlines()
                        last_log_position = f.tell()
                        for line in new_lines:
                            line = line.strip()
                            if line:
                                print(line)
                                parsed = parse_ltx_progress_line(line)
                                if parsed:
                                    last_progress = parsed
                except:
                    pass

                yield all_generated_videos.copy(), None, status_text, last_progress

                time.sleep(0.5)

                if process.poll() is not None:
                    # Read remaining log content
                    try:
                        with open(log_file_path, "r", encoding="utf-8") as f:
                            f.seek(last_log_position)
                            for line in f.readlines():
                                line = line.strip()
                                if line:
                                    print(line)
                                    parsed = parse_ltx_progress_line(line)
                                    if parsed:
                                        last_progress = parsed
                    except:
                        pass
                    log_file.close()
                    break

            current_process = None
            return_code = process.returncode
            elapsed = time.perf_counter() - start_time

            if return_code == 0 and os.path.exists(output_filename):
                # Metadata is now saved by the backend (ltx_generate_video.py)
                # embedded directly in the video file's comment metadata

                label = f"Seed: {current_seed} | {elapsed:.1f}s"
                all_generated_videos.append((output_filename, label))

                status_text = f"Completed {i+1}/{batch_size}"
                yield all_generated_videos.copy(), None, status_text, f"Video saved: {os.path.basename(output_filename)}"
            else:
                error_msg = f"Generation failed (return code: {return_code})"
                yield all_generated_videos.copy(), None, error_msg, "Check logs for details"

        except Exception as e:
            current_process = None
            yield all_generated_videos, None, f"Error: {str(e)}", ""
            return

    final_status = f"Completed {batch_size} video(s)" if batch_size > 1 else "Generation complete!"
    yield all_generated_videos, None, final_status, "Done!"


def stop_generation():
    """Signal to stop the current generation."""
    global stop_event
    stop_event.set()
    return "Stopping generation..."


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface():
    with gr.Blocks(
        theme=themes.Default(
            primary_hue=colors.Color(
                name="ltx_blue",
                c50="#E6F4FF",
                c100="#BAE0FF",
                c200="#91CAFF",
                c300="#69B4FF",
                c400="#409EFF",
                c500="#1677FF",
                c600="#0958D9",
                c700="#003EB3",
                c800="#002C8C",
                c900="#001D66",
                c950="#000F33"
            )
        ),
        css="""
        .green-btn {
            background: linear-gradient(to bottom right, #2ecc71, #27ae60) !important;
            color: white !important;
            border: none !important;
        }
        .green-btn:hover {
            background: linear-gradient(to bottom right, #27ae60, #219651) !important;
        }
        """,
        title="LTX-2 Video Generator"
    ) as demo:

        gr.Markdown("# LTX-2 Video Generator")
        gr.Markdown("Two-stage pipeline with joint audio-video generation")

        with gr.Tabs():
            # =================================================================
            # Generation Tab
            # =================================================================
            with gr.Tab("Generation"):
                with gr.Row():
                    # Left column - Inputs
                    with gr.Column(scale=3):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the video you want to generate...",
                            lines=4,
                            value="A serene mountain lake at sunrise, with mist rising from the water and birds flying overhead."
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, distorted, artifacts, ugly, deformed",
                            lines=2
                        )

                        with gr.Row():
                            batch_size = gr.Number(label="Batch Count", value=1, minimum=1, step=1, scale=1)
                            seed = gr.Number(label="Seed (-1 = random)", value=-1, scale=1)
                            random_seed_btn = gr.Button("🎲", scale=0, min_width=40)

                    # Right column - Status
                    with gr.Column(scale=1):
                        status_text = gr.Textbox(label="Status", interactive=False, value="Ready")
                        progress_text = gr.Textbox(label="Progress", interactive=False, value="")

                with gr.Row():
                    generate_btn = gr.Button("🎬 Generate Video", variant="primary", elem_classes="green-btn")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop")

                with gr.Row():
                    # Left column - Parameters
                    with gr.Column():
                        # Generation Parameters
                        with gr.Accordion("Generation Parameters", open=True):
                            with gr.Row():
                                mode = gr.Dropdown(
                                    label="Mode",
                                    choices=["t2v", "i2v", "v2v"],
                                    value="t2v",
                                    info="t2v = text-to-video, i2v = image-to-video, v2v = video-to-video (refine)"
                                )
                                pipeline = gr.Dropdown(
                                    label="Pipeline",
                                    choices=["two-stage", "one-stage", "refine-only"],
                                    value="two-stage",
                                    info="two-stage = higher quality, one-stage = faster, refine-only = stage 2 only on input video"
                                )
                            # Hidden state for original image/video dimensions
                            original_dims = gr.State(value="")
                        # Image Conditioning (I2V)
                        with gr.Accordion("Image Conditioning (I2V)", open=True) as i2v_section:
                            input_image = gr.Image(label="Start Image", type="filepath")
                            with gr.Row():
                                image_frame_idx = gr.Number(label="Frame Index", value=0, minimum=0, info="Which frame to condition (0 = first)")
                                image_strength = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Strength")

                            with gr.Accordion("End Image (optional)", open=False):
                                end_image = gr.Image(label="End Image (for start-to-end video)", type="filepath")
                                gr.Markdown("Set an ending frame to generate video that transitions from start to end image.")
                                with gr.Row():
                                    end_image_strength = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="End Image Strength")
                            gr.Markdown("### Resolution Settings")
                            scale_slider = gr.Slider(
                                minimum=1, maximum=200, value=100, step=1,
                                label="Scale % (adjusts resolution while maintaining aspect ratio)",
                                info="Scale the input image dimensions. Works for I2V mode."
                            )
                            with gr.Row():
                                width = gr.Number(label="Width", value=1024, step=64, info="Must be divisible by 64")
                                calc_height_btn = gr.Button("→", size="sm", min_width=40)
                                calc_width_btn = gr.Button("←", size="sm", min_width=40)
                                height = gr.Number(label="Height", value=1024, step=64, info="Must be divisible by 64")
                            with gr.Row():
                                num_frames = gr.Number(label="Num Frames (8*K+1)", value=121, step=8, info="e.g., 121 = 5s @ 24fps")
                                frame_rate = gr.Slider(minimum=12, maximum=60, value=24, step=1, label="Frame Rate")
                            with gr.Row():
                                cfg_guidance_scale = gr.Slider(minimum=1.0, maximum=15.0, value=4.0, step=0.5, label="CFG Scale")
                                num_inference_steps = gr.Slider(minimum=1, maximum=60, value=40, step=1, label="Inference Steps")

                        # Video Input (V2V / Refine)
                        with gr.Accordion("Video Input (V2V / Refine)", open=False) as v2v_section:
                            input_video = gr.Video(label="Input Video (for refinement)", sources=["upload"])
                            gr.Markdown("""
                            **Video-to-Video Refinement:**
                            - Upload a video to refine it using stage 2 (with distilled LoRA)
                            - Use "refine-only" pipeline to skip stage 1 generation
                            - Resolution and frame count will be auto-detected from video
                            """)
                            with gr.Row():
                                refine_strength = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                                    label="Refine Strength",
                                    info="Amount of noise to add before refinement (0=none, 1=full denoise)"
                                )
                                refine_steps = gr.Slider(
                                    minimum=1, maximum=30, value=10, step=1,
                                    label="Refine Steps",
                                    info="Number of refinement denoising steps"
                                )


                    # Right column - Output
                    with gr.Column():
                        output_gallery = gr.Gallery(
                            label="Generated Videos",
                            columns=2, rows=2,
                            object_fit="contain",
                            height="auto",
                            allow_preview=True,
                            preview=True
                        )
                        # User LoRA
                        with gr.Accordion("User LoRA (Optional)", open=False):
                            lora_folder = gr.Textbox(label="LoRA Folder", value="lora")
                            lora_refresh_btn = gr.Button("🔄 Refresh", size="sm")
                            with gr.Row():
                                user_lora = gr.Dropdown(
                                    label="LoRA",
                                    choices=get_ltx_lora_options("lora"),
                                    value="None",
                                    scale=3
                                )
                                user_lora_strength = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                        # Memory Optimization
                        with gr.Accordion("Model settings", open=True):
                            with gr.Row():
                                offload = gr.Checkbox(label="CPU Offloading", value=False, info="Offload models to CPU when not in use")
                                enable_fp8 = gr.Checkbox(label="FP8 Mode", value=False, info="Reduce memory with FP8 transformer")
                            gr.Markdown("### Block Swapping")
                            with gr.Row():
                                enable_dit_block_swap = gr.Checkbox(label="DiT Block Swap", value=True, info="Main transformer (stage 1)")
                                dit_blocks_in_memory = gr.Slider(minimum=1, maximum=47, value=22, step=1, label="DiT Blocks in GPU", visible=True)
                            with gr.Row():
                                enable_text_encoder_block_swap = gr.Checkbox(label="Text Encoder Block Swap", value=True, info="Gemma text encoder")
                                text_encoder_blocks_in_memory = gr.Slider(minimum=1, maximum=47, value=6, step=1, label="Text Encoder Blocks in GPU", visible=True, info="Gemma-3-12B has 48 layers")
                            with gr.Row():
                                enable_refiner_block_swap = gr.Checkbox(label="Refiner Block Swap", value=True, info="Stage 2 refiner transformer")
                                refiner_blocks_in_memory = gr.Slider(minimum=1, maximum=47, value=22, step=1, label="Refiner Blocks in GPU", visible=True)
                            with gr.Row():
                                disable_audio = gr.Checkbox(label="Disable Audio", value=False, info="Generate video only (no audio)")
                                enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False, info="Use Gemma to improve prompt")
                            checkpoint_path = gr.Textbox(
                                label="LTX Checkpoint Path",
                                value="./weights/ltx-2-19b-dev.safetensors",
                                info="Path to LTX-2 model checkpoint"
                            )
                            gemma_root = gr.Textbox(
                                label="Gemma Root Path",
                                value="./gemma-3-12b-it-qat-q4_0-unquantized",
                                info="Path to Gemma text encoder"
                            )
                            spatial_upsampler_path = gr.Textbox(
                                label="Spatial Upsampler Path",
                                value="./weights/ltx-2-spatial-upscaler-x2-1.0.safetensors",
                                info="Path to 2x spatial upsampler"
                            )
                            with gr.Row():
                                distilled_lora_path = gr.Textbox(
                                    label="Distilled LoRA Path",
                                    value="./weights/ltx-2-19b-distilled-lora-384.safetensors",
                                    info="For stage 2 refinement",
                                    scale=3
                                )
                                distilled_lora_strength = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Strength", scale=1
                                )
                            save_path = gr.Textbox(label="Output Folder", value="outputs")                                       

            # =================================================================
            # Video Info Tab
            # =================================================================
            with gr.Tab("Video Info"):
                with gr.Row():
                    info_video_input = gr.Video(label="Upload Video", interactive=True)
                    info_metadata_output = gr.JSON(label="Generation Parameters")

                with gr.Row():
                    info_status = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    info_send_btn = gr.Button("Send to Generation", variant="primary")

            # =================================================================
            # Help Tab
            # =================================================================
            with gr.Tab("Help"):
                gr.Markdown("""
                ## LTX-2 Video Generation Help

                ### Required Model Files
                **For two-stage pipeline** (default, higher quality):
                1. **LTX Checkpoint** - Main 19B model (.safetensors)
                2. **Gemma Root** - Text encoder directory
                3. **Spatial Upsampler** - For 2x resolution upscaling
                4. **Distilled LoRA** - For stage 2 refinement

                **For one-stage pipeline** (faster):
                1. **LTX Checkpoint** - Main 19B model (.safetensors)
                2. **Gemma Root** - Text encoder directory

                ### Pipeline Options
                - **Two-stage** (default): Generates at half resolution, upsamples 2x, then refines with distilled LoRA. Higher quality output.
                - **One-stage**: Generates directly at full resolution in a single pass. Faster but may have less detail.
                - **Refine-only**: Runs stage 2 refinement on an input video. Use this to improve a previously generated video.

                ### Generation Modes
                - **T2V (Text-to-Video)**: Generate video from text prompt only
                - **I2V (Image-to-Video)**: Start from an input image
                - **V2V (Video-to-Video)**: Refine an existing video

                ### Video Input (V2V / Refine)
                - Upload a video to refine it using stage 2
                - Use "refine-only" pipeline to skip stage 1 generation
                - **Refine Strength**: Amount of noise added before refinement (0=none, 1=full denoise)
                - Distilled LoRA is optional for refine-only mode

                ### Image Conditioning (I2V)
                - **Start Image**: Set the first frame of the video
                - **End Image** (optional): Set the last frame to create start-to-end transitions
                - **Scale Slider**: Automatically adjusts width/height based on input image aspect ratio
                - **→ / ← Buttons**: Calculate width from height or height from width while preserving aspect ratio

                ### Resolution Requirements
                - Width and height must be divisible by 64
                - Recommended: 1024x1024, 768x1024, 1024x768
                - When using I2V, the scale slider helps maintain the input image's aspect ratio

                ### Frame Count
                - Must be in format: 8*K + 1
                - Examples: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121
                - 121 frames @ 24fps = 5 seconds

                ### Memory Optimization
                - **CPU Offloading**: Reduces peak VRAM by moving models to CPU
                - **FP8 Mode**: Stores transformer in FP8 (lower precision)
                - **Block Swapping**: Keeps only N transformer blocks in GPU (~40% VRAM reduction)

                ### Audio Generation
                - LTX-2 generates synchronized audio by default
                - Disable audio if you only need video
                """)

        # =================================================================
        # Event Handlers
        # =================================================================

        # Random seed button
        random_seed_btn.click(
            fn=lambda: random.randint(0, 2**32 - 1),
            outputs=[seed]
        )

        # LoRA refresh
        lora_refresh_btn.click(
            fn=refresh_lora_dropdown,
            inputs=[lora_folder],
            outputs=[user_lora]
        )

        # Block swap visibility toggles
        enable_dit_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[enable_dit_block_swap],
            outputs=[dit_blocks_in_memory]
        )
        enable_text_encoder_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[enable_text_encoder_block_swap],
            outputs=[text_encoder_blocks_in_memory]
        )
        enable_refiner_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[enable_refiner_block_swap],
            outputs=[refiner_blocks_in_memory]
        )

        # Image dimension handlers
        input_image.change(
            fn=update_image_dimensions,
            inputs=[input_image],
            outputs=[original_dims, width, height]
        )

        scale_slider.change(
            fn=update_resolution_from_scale,
            inputs=[scale_slider, original_dims],
            outputs=[width, height]
        )

        calc_width_btn.click(
            fn=calculate_width_from_height,
            inputs=[height, original_dims],
            outputs=[width]
        )

        calc_height_btn.click(
            fn=calculate_height_from_width,
            inputs=[width, original_dims],
            outputs=[height]
        )

        # Mode change - show/hide I2V and V2V sections
        def update_sections_visibility(m):
            return gr.update(open=(m == "i2v")), gr.update(open=(m == "v2v"))

        mode.change(
            fn=update_sections_visibility,
            inputs=[mode],
            outputs=[i2v_section, v2v_section]
        )

        # Video input change - update dimensions and frame count
        input_video.change(
            fn=update_video_dimensions,
            inputs=[input_video],
            outputs=[original_dims, width, height, num_frames, frame_rate]
        )

        # Stop button
        stop_btn.click(
            fn=stop_generation,
            outputs=[status_text]
        )

        # Generate button
        generate_btn.click(
            fn=generate_ltx_video,
            inputs=[
                prompt, negative_prompt,
                checkpoint_path, gemma_root, spatial_upsampler_path,
                distilled_lora_path, distilled_lora_strength,
                mode, pipeline, width, height, num_frames, frame_rate,
                cfg_guidance_scale, num_inference_steps, seed,
                input_image, image_frame_idx, image_strength,
                end_image, end_image_strength,
                input_video, refine_strength, refine_steps,
                disable_audio, enhance_prompt,
                offload, enable_fp8,
                enable_dit_block_swap, dit_blocks_in_memory,
                enable_text_encoder_block_swap, text_encoder_blocks_in_memory,
                enable_refiner_block_swap, refiner_blocks_in_memory,
                lora_folder, user_lora, user_lora_strength,
                save_path, batch_size
            ],
            outputs=[output_gallery, gr.State(), status_text, progress_text]
        )

        # =================================================================
        # Video Info Tab Event Handlers
        # =================================================================
        info_video_input.upload(
            fn=extract_video_details,
            inputs=info_video_input,
            outputs=[info_metadata_output, info_status]
        )

        def send_to_generation_handler(metadata):
            """Send loaded metadata to generation tab parameters."""
            if not metadata:
                return [gr.update()] * 21 + ["No metadata loaded - upload a video first"]

            # Handle legacy metadata that used single enable_block_swap
            legacy_block_swap = metadata.get("enable_block_swap", True)

            # Return updates for all generation parameters
            return [
                gr.update(value=metadata.get("prompt", "")),  # prompt
                gr.update(value=metadata.get("negative_prompt", "")),  # negative_prompt
                gr.update(value=metadata.get("mode", "t2v")),  # mode
                gr.update(value=metadata.get("pipeline", "two-stage")),  # pipeline
                gr.update(value=metadata.get("width", 1024)),  # width
                gr.update(value=metadata.get("height", 1024)),  # height
                gr.update(value=metadata.get("num_frames", 121)),  # num_frames
                gr.update(value=metadata.get("frame_rate", 24)),  # frame_rate
                gr.update(value=metadata.get("cfg_guidance_scale", 4.0)),  # cfg_guidance_scale
                gr.update(value=metadata.get("num_inference_steps", 40)),  # num_inference_steps
                gr.update(value=metadata.get("seed", -1)),  # seed
                gr.update(value=metadata.get("image_strength", 0.9)),  # image_strength
                gr.update(value=metadata.get("end_image_strength", 0.9)),  # end_image_strength
                gr.update(value=metadata.get("refine_strength", 0.3)),  # refine_strength
                gr.update(value=metadata.get("refine_steps", 10)),  # refine_steps
                gr.update(value=metadata.get("offload", False)),  # offload
                gr.update(value=metadata.get("enable_fp8", False)),  # enable_fp8
                gr.update(value=metadata.get("enable_dit_block_swap", legacy_block_swap)),  # enable_dit_block_swap
                gr.update(value=metadata.get("enable_text_encoder_block_swap", legacy_block_swap)),  # enable_text_encoder_block_swap
                gr.update(value=metadata.get("enable_refiner_block_swap", legacy_block_swap)),  # enable_refiner_block_swap
                "Parameters sent to Generation tab"  # status
            ]

        info_send_btn.click(
            fn=send_to_generation_handler,
            inputs=[info_metadata_output],
            outputs=[
                prompt, negative_prompt, mode, pipeline,
                width, height, num_frames, frame_rate,
                cfg_guidance_scale, num_inference_steps, seed,
                image_strength, end_image_strength, refine_strength, refine_steps,
                offload, enable_fp8,
                enable_dit_block_swap, enable_text_encoder_block_swap, enable_refiner_block_swap,
                info_status
            ]
        )

        return demo


# =============================================================================
# Port Detection
# =============================================================================

def find_available_port(start_port: int = 7860, max_attempts: int = 100) -> int:
    """Find the first available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LTX-2 Video Generation UI")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port. Auto-detects available port if not specified.")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link.")
    args = parser.parse_args()

    # Determine port (auto-detect if not specified)
    if args.port is not None:
        port = args.port
    else:
        port = find_available_port(7860)

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"[lt1.py] Starting on port {port}, CUDA_VISIBLE_DEVICES={cuda_devices}")

    demo = create_interface()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=args.share,
        inbrowser=False
    )
