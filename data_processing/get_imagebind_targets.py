import sys
sys.path.append('VideoLLaMA') # for imports in VideoLLaMA

from video_llama.models.ImageBind.models import imagebind_model
from video_llama.models.ImageBind.models.imagebind_model import ModalityType
from video_llama.models.ImageBind import data
import torch
import numpy as np
from tqdm import tqdm
import os
import time

device = "cuda:0"

model, emb_size = imagebind_model.imagebind_huge()
model.load_state_dict(torch.load("../Video-LLaMA-2-7B-Finetuned/imagebind_huge.pth"))
model.to(device)

def process(input_dir, output_dir):
    os.makedirs(output_dir)

    video_count = 0
    for file in os.listdir(input_dir):
        if file.endswith(".mp4"):
            video_count += 1

    batch_size = 16
    for batch_start in tqdm(range(0, video_count, batch_size)):
        batch_end = min(batch_start + batch_size, video_count)
        video_paths = [f"{input_dir}/{x:03d}.mp4" for x in range(batch_start, batch_end)]
        inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(video_paths, device, clip_duration=2, clips_per_video=1)
        }
        with torch.no_grad():
            outputs = model(inputs)
        for i, video_idx in enumerate(range(batch_start, batch_end)):
            np.save(f"{output_dir}/{video_idx:03d}.npy", outputs["vision"][i].cpu().numpy())

process("data/videos_processed_dark/2022-07-14_09-59-00_S09_eye-tracking-video-worldGaze_frame", "data/imagebind_targets_dark/S09")
exit()

dirs = [
    ("data/videos_processed/2022-06-07_18-11-37_S00_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S00"),
    ("data/videos_processed/2022-06-13_21-48-24_S02_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S02_1"),
    ("data/videos_processed/2022-06-13_22-35-11_S02_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S02_2"),
    ("data/videos_processed/2022-06-13_23-22-44_S02_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S02_3"),
    ("data/videos_processed/2022-06-14_13-52-57_S03_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S03"),
    ("data/videos_processed/2022-06-14_16-38-43_S04_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S04"),
    ("data/videos_processed/2022-06-14_20-46-12_S05_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S05"),
    ("data/videos_processed/2022-07-12_15-08-08_S06_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S06"),
    ("data/videos_processed/2022-07-13_11-02-03_S07_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S07"),
    ("data/videos_processed/2022-07-13_14-15-26_S08_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S08"),
    ("data/videos_processed/2022-07-14_09-59-00_S09_eye-tracking-video-worldGaze_frame", "data/imagebind_targets/S09"),
]
for input_dir, output_dir in dirs:
    process(input_dir, output_dir)