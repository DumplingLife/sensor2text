from video_llama.models.ImageBind.models import imagebind_model
from video_llama.models.ImageBind.models.imagebind_model import ModalityType
from video_llama.models.ImageBind import data
import torch
import numpy as np

device = "cuda:0"

model, emb_size = imagebind_model.imagebind_huge()
model.load_state_dict(torch.load("../Video-LLaMA-2-7B-Finetuned/imagebind_huge.pth"))
model.to(device)

for x in range(230):
    print(x)
    video_dir = "actionsense_data/videos_processed/2022-06-07_18-11-37_S00_eye-tracking-video-worldGaze_frame"
    video_paths = [f"{video_dir}/video_{x:03d}.avi"]

    inputs = {
        ModalityType.VISION: data.load_and_transform_video_data(video_paths, device)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    print(embeddings["vision"].shape)
    np.save(f"actionsense_data/S00_imagebind_embeds/{x:03d}.npy", embeddings["vision"][0].cpu().numpy())