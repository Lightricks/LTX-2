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
                         distilled_lora_path: str, gemma_root: str) -> Optional[str]:
    """Validate all required model paths exist."""
    paths = [
        ("LTX Checkpoint", checkpoint_path),
        ("Spatial Upsampler", spatial_upsampler_path),
        ("Distilled LoRA", distilled_lora_path),
        ("Gemma Root", gemma_root),
    ]
    for name, path in paths:
        if not path or not path.strip():
            return f"Error: {name} path is required"
        if not os.path.exists(path):
            return f"Error: {name} not found: {path}"
    return None


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
    # Audio & prompt
    disable_audio: bool,
    enhance_prompt: bool,
    # Memory optimization
    offload: bool,
    enable_fp8: bool,
    enable_block_swap: bool,
    blocks_in_memory: int,
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

    error = validate_model_paths(checkpoint_path, spatial_upsampler_path,
                                  distilled_lora_path, gemma_root)
    if error:
        yield [], None, error, ""
        return

    # Check image for I2V mode
    if mode == "i2v" and not input_image:
        yield [], None, "Error: Input image required for I2V mode", ""
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
            "--spatial-upsampler-path", spatial_upsampler_path,
            "--distilled-lora", distilled_lora_path, str(distilled_lora_strength),
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

        # Image conditioning (I2V)
        if mode == "i2v" and input_image:
            command.extend(["--image", str(input_image), str(int(image_frame_idx)), str(float(image_strength))])

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
        if enable_block_swap:
            command.append("--enable-block-swap")
            command.extend(["--blocks-in-memory", str(int(blocks_in_memory))])

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
                # Save metadata
                params_for_meta = {
                    "model_type": "LTX-2",
                    "pipeline": "two_stage",
                    "mode": mode,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": int(width),
                    "height": int(height),
                    "num_frames": int(num_frames),
                    "frame_rate": float(frame_rate),
                    "cfg_guidance_scale": float(cfg_guidance_scale),
                    "num_inference_steps": int(num_inference_steps),
                    "seed": current_seed,
                    "audio_enabled": not disable_audio,
                    "prompt_enhanced": enhance_prompt,
                    "offload": offload,
                    "enable_fp8": enable_fp8,
                    "enable_block_swap": enable_block_swap,
                    "blocks_in_memory": int(blocks_in_memory) if enable_block_swap else None,
                    "distilled_lora_strength": float(distilled_lora_strength),
                    "user_lora": user_lora if user_lora != "None" else None,
                    "user_lora_strength": float(user_lora_strength) if user_lora != "None" else None,
                    "generation_time_seconds": round(elapsed, 2),
                }

                # Write metadata to JSON sidecar
                meta_path = output_filename.replace(".mp4", "_meta.json")
                try:
                    with open(meta_path, "w") as f:
                        json.dump(params_for_meta, f, indent=2)
                except:
                    pass

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
                        # Model Configuration
                        with gr.Accordion("Model Configuration", open=False):
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

                        # Generation Parameters
                        with gr.Accordion("Generation Parameters", open=True):
                            mode = gr.Dropdown(
                                label="Mode",
                                choices=["t2v", "i2v"],
                                value="t2v",
                                info="t2v = text-to-video, i2v = image-to-video"
                            )
                            with gr.Row():
                                width = gr.Number(label="Width", value=1024, step=64, info="Must be divisible by 64")
                                height = gr.Number(label="Height", value=1024, step=64, info="Must be divisible by 64")
                            with gr.Row():
                                num_frames = gr.Number(label="Num Frames (8*K+1)", value=121, step=8, info="e.g., 121 = 5s @ 24fps")
                                frame_rate = gr.Slider(minimum=12, maximum=60, value=24, step=1, label="Frame Rate")
                            with gr.Row():
                                cfg_guidance_scale = gr.Slider(minimum=1.0, maximum=15.0, value=4.0, step=0.5, label="CFG Scale")
                                num_inference_steps = gr.Slider(minimum=1, maximum=60, value=40, step=1, label="Inference Steps")

                        # Image Conditioning (I2V)
                        with gr.Accordion("Image Conditioning (I2V)", open=False) as i2v_section:
                            input_image = gr.Image(label="Input Image", type="filepath")
                            with gr.Row():
                                image_frame_idx = gr.Number(label="Frame Index", value=0, minimum=0, info="Which frame to condition")
                                image_strength = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Strength")

                        # Audio & Prompt Enhancement
                        with gr.Accordion("Audio & Prompt", open=False):
                            with gr.Row():
                                disable_audio = gr.Checkbox(label="Disable Audio", value=False, info="Generate video only (no audio)")
                                enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False, info="Use Gemma to improve prompt")

                        # Memory Optimization
                        with gr.Accordion("Memory Optimization", open=False):
                            with gr.Row():
                                offload = gr.Checkbox(label="CPU Offloading", value=False, info="Offload models to CPU when not in use")
                                enable_fp8 = gr.Checkbox(label="FP8 Mode", value=False, info="Reduce memory with FP8 transformer")
                            with gr.Row():
                                enable_block_swap = gr.Checkbox(label="Block Swapping", value=False, info="~40% VRAM reduction")
                                blocks_in_memory = gr.Slider(minimum=2, maximum=24, value=6, step=1, label="Blocks in GPU", visible=False)

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

                        # Output Settings
                        with gr.Accordion("Output Settings", open=False):
                            save_path = gr.Textbox(label="Output Folder", value="outputs")

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

            # =================================================================
            # Settings Tab
            # =================================================================
            with gr.Tab("Help"):
                gr.Markdown("""
                ## LTX-2 Video Generation Help

                ### Required Model Files
                You need 4 model files to run LTX-2:
                1. **LTX Checkpoint** - Main 19B model (.safetensors)
                2. **Gemma Root** - Text encoder directory
                3. **Spatial Upsampler** - For 2x resolution upscaling
                4. **Distilled LoRA** - For stage 2 refinement

                ### Generation Modes
                - **T2V (Text-to-Video)**: Generate video from text prompt only
                - **I2V (Image-to-Video)**: Start from an input image

                ### Resolution Requirements
                - Width and height must be divisible by 64
                - Recommended: 1024x1024, 768x1024, 1024x768

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

        # Block swap visibility toggle
        enable_block_swap.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[enable_block_swap],
            outputs=[blocks_in_memory]
        )

        # Mode change - show/hide I2V section
        mode.change(
            fn=lambda m: gr.update(open=(m == "i2v")),
            inputs=[mode],
            outputs=[i2v_section]
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
                mode, width, height, num_frames, frame_rate,
                cfg_guidance_scale, num_inference_steps, seed,
                input_image, image_frame_idx, image_strength,
                disable_audio, enhance_prompt,
                offload, enable_fp8, enable_block_swap, blocks_in_memory,
                lora_folder, user_lora, user_lora_strength,
                save_path, batch_size
            ],
            outputs=[output_gallery, gr.State(), status_text, progress_text]
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
