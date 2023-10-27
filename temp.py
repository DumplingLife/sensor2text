import torch
from video_stuff import VideoSide

device = "cuda:0"

model = VideoSide()
model.to(device)

batch_size = 8
num_channels = 3
time_length = 32
height = 224
width = 224

image = torch.randn(batch_size, num_channels, time_length, height, width)
image = image.to(device)
image = image.to(torch.float16)

inputs_llama, atts_llama = model.encode_videoQformer_visual(image)

print(inputs_llama)