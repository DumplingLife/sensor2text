from video_llama.models.ImageBind.models import imagebind_model
from video_llama.models.ImageBind.models.imagebind_model import ModalityType
from video_llama.models.ImageBind import data
import torch
import numpy as np
import os
import time

input_dir = "actionsense_data/videos_processed_2s/2022-06-13_23-22-44_S02_eye-tracking-video-worldGaze_frame"
output_dir = "actionsense_data/imagebind_targets_2s/S02_3"
# os.makedirs(output_dir)

device = "cuda:0"

model, emb_size = imagebind_model.imagebind_huge()
model.load_state_dict(torch.load("../Video-LLaMA-2-7B-Finetuned/imagebind_huge.pth"))
model.to(device)

# Define batch size
batch_size = 16

video_count = 0
for file in os.listdir(input_dir):
    if file.endswith(".mp4"):
        video_count += 1

# Process in batches
for batch_start in range(0, video_count, batch_size):
    batch_end = min(batch_start + batch_size, video_count)
    print(f"Processing batch: {batch_start} to {batch_end}")

    video_paths = [f"{input_dir}/video_{x:03d}.mp4" for x in range(batch_start, batch_end)]

    inputs = {
        ModalityType.VISION: data.load_and_transform_video_data(video_paths, device)
    }

    start_time = time.time()
    with torch.no_grad():
        embeddings = model(inputs)
    print(f"{time.time() - start_time}s taken")

    for i, video_idx in enumerate(range(batch_start, batch_end)):
        print(embeddings["vision"][i].shape)
        np.save(f"{output_dir}/{video_idx:03d}.npy", embeddings["vision"][i].cpu().numpy())