import sys
sys.path.append('VideoLLaMA')

from VideoLLaMA.video_llama.models.ImageBind.models import imagebind_model
from VideoLLaMA.video_llama.models.ImageBind.models.imagebind_model import ModalityType
from VideoLLaMA.video_llama.models.ImageBind import data
import torch
import numpy as np
import os
import time
import random

device = "cuda:0"

model, emb_size = imagebind_model.imagebind_huge()
model.load_state_dict(torch.load("../Video-LLaMA-2-7B-Finetuned/imagebind_huge.pth"))
model.to(device)

input_dir = "language_decoder/train2017"
output_dir = "language_decoder/train2017_embeds"

os.makedirs(output_dir)

# Define batch size
batch_size = 64

image_files = os.listdir(input_dir)[0:5000]
random.shuffle(image_files)
image_count = len(image_files)

# Process in batches
for batch_start in range(0, image_count, batch_size):
    batch_end = min(batch_start + batch_size, image_count)
    print(f"Processing batch: {batch_start} to {batch_end}")

    input_paths = [f"{input_dir}/{image_files[x]}" for x in range(batch_start, batch_end)]
    output_paths = [f"{output_dir}/{image_files[x].rsplit('.', 1)[0]}.npy" for x in range(batch_start, batch_end)]

    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(input_paths, device)
    }

    start_time = time.time()
    with torch.no_grad():
        embeddings = model(inputs)
    print(f"{time.time() - start_time}s taken")

    for output_path, embedding in zip(output_paths, embeddings["vision"]):
        np.save(output_path, embedding.cpu().numpy())