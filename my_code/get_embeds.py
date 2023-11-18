from video_llama.models.ImageBind.models import imagebind_model
from video_llama.models.ImageBind.models.imagebind_model import ModalityType
from video_llama.models.ImageBind import data
import torch
import numpy as np

device = "cuda:0"

model, emb_size = imagebind_model.imagebind_huge()
model.load_state_dict(torch.load("../Video-LLaMA-2-7B-Finetuned/imagebind_huge.pth"))
model.to(device)

# Define batch size
batch_size = 8

# Process in batches
for batch_start in range(0, 230, batch_size):
    batch_end = min(batch_start + batch_size, 230)
    print(f"Processing batch: {batch_start} to {batch_end}")

    video_dir = "actionsense_data/videos_processed/2022-06-07_18-11-37_S00_eye-tracking-video-worldGaze_frame"
    video_paths = [f"{video_dir}/video_{x:03d}.avi" for x in range(batch_start, batch_end)]

    inputs = {
        ModalityType.VISION: data.load_and_transform_video_data(video_paths, device)
    }

    with torch.no_grad():
        embeddings = model(inputs)
    print(embeddings["vision"].shape)
    exit()

    for i, video_idx in enumerate(range(batch_start, batch_end)):
        print(embeddings["vision"][i].shape)
        np.save(f"actionsense_data/S00_imagebind_embeds/{video_idx:03d}.npy", embeddings["vision"][i].cpu().numpy())