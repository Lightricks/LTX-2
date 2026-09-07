""" Decode videos with LTX2 video encoder. """

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchcodec.encoders import VideoEncoder
from tqdm import tqdm

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.video_vae import VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER

INPUT_DIR: Path = Path("/home/ubuntu/Desktop/lehack/v_jepa_embedding_inversion_attack/runs/diffusion/samples")
OUTPUT_DIR: Path = Path("/home/ubuntu/Desktop/lehack/v_jepa_embedding_inversion_attack/videos/validation")

FPS: float = 12.0


class DirectoryDataset(Dataset[tuple[torch.Tensor, str]]):
    """
    Attributes:
        directory_path (str): Absolute path to directory root.
        file_names (list[str]): Sorted list of file names found at directory root.
    """

    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.file_names = sorted(os.listdir(directory_path))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        file_name = self.file_names[idx]
        file_path = os.path.join(self.directory_path, self.file_names[idx])
        return torch.load(file_path).to(device="cpu", dtype=torch.bfloat16), file_name


def main() -> None:
    decoder = SingleGPUModelBuilder(
        model_class_configurator=VideoDecoderConfigurator,
        model_path="models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors",
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER
    ).build(device=torch.device("cuda"), dtype=torch.bfloat16).eval()

    dataset = DirectoryDataset(str(INPUT_DIR))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    num_videos = 0
    for batch_tensors, batch_filenames in tqdm(dataloader):
        batch_tensors = batch_tensors.to("cuda", dtype=torch.bfloat16, non_blocking=True)

        with torch.no_grad():
            batch_video_tensors: torch.Tensor = decoder(batch_tensors)

        for idx in range(batch_video_tensors.shape[0]):
            video_tensor = batch_video_tensors[idx]
            frames = video_tensor.squeeze(0).permute(1, 0, 2, 3)
            frames = ((frames.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu()
            encoder = VideoEncoder(frames, frame_rate=FPS)
            encoder.to_file(OUTPUT_DIR / f"{batch_filenames[idx].removesuffix('.pt')}.webm")
            num_videos += 1

        # Incase of OOM
        del batch_tensors, batch_video_tensors
        torch.cuda.empty_cache()

    print(f"Decoded {num_videos} video(s).")


if __name__ == "__main__":
    main()
