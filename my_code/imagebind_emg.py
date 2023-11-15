from video_llama.models.ImageBind.models import imagebind_model
import torch

model, emb_size = imagebind_model.imagebind_huge()
model.load_state_dict(torch.load("../Video-LLaMA-2-7B-Finetuned/imagebind_huge.pth"))

print(model)