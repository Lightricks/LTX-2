""" Encode videos with LTX2 video encoder. """

from pathlib import Path

import torch
from torchcodec.decoders import VideoDecoder

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.video_vae import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER

INPUT_DATALOADER = Path("/home/ubuntu/Desktop/lehack/datasets/something_something_v2/train_dataloader.csv")
OUTPUT_DIR = Path("/home/ubuntu/Desktop/lehack/v_jepa_embedding_inversion_attack/embeddings/train/LTX2")

# Must follow 8n + 1
NUM_FRAMES: int = 33
# height and width must be divisible by 64
TARGET_H: int =  256
TARGET_W: int = 256


def main() -> None:
    encoder = SingleGPUModelBuilder(
        model_class_configurator=VideoEncoderConfigurator,
        model_path="models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER
    ).build(device=torch.device("cuda"), dtype=torch.bfloat16).eval()

    print("Model initialised, begin encoding")

    num_videos = 0
    with open(INPUT_DATALOADER, "r") as infile:
        for line in infile:
            path = line.split(" ")[0]
            path_name = path.split("/")[-1]
            try:
                # Decode video into video tensor
                decoder = VideoDecoder(str(path))
                video_tensor = decoder.get_frames_at(list(range(0, NUM_FRAMES))).data

                # Conform decoded video tensor to expected input shape
                video_tensor = torch.nn.functional.interpolate(
                    video_tensor,
                    size=(TARGET_W, TARGET_H),
                    mode="bilinear",
                    align_corners=False
                )
                video_tensor = video_tensor.to(
                    device="cuda", 
                    dtype=torch.bfloat16
                ) / 127.5 - 1.0
                video_tensor = (
                    video_tensor.permute(1, 0, 2, 3)
                                .unsqueeze(0)
                                .contiguous()
                )

                with torch.no_grad():
                    video_embedding = encoder(video_tensor)

                video_idx = path_name.removesuffix(".webm")
                torch.save(video_embedding, OUTPUT_DIR / f"{video_idx}.pt")
                num_videos += 1
            except FileNotFoundError:
                print(f"{OUTPUT_DIR} does not exist.")
                return
            except IndexError:
                print(f"{path_name} has less than {NUM_FRAMES} frames, skipping...")
            except Exception:
                print(f"Error occured when encoding {path_name}, skipping...")

    print(f"Encoded a total of {num_videos} video(s).")


if __name__ == "__main__":
    main()
