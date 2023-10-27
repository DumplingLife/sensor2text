import torch
from video_stuff import VideoSide

model = VideoSide()

batch_size = 32
num_channels = 64
time_length = 20
height = 224
width = 224
num_audio_features = 20
device = "cuda:0"

image = torch.randn(batch_size, num_channels, time_length, height, width).to(device)

inputs_llama, atts_llama = model.encode_videoQformer_visual(image)

print(inputs_llama)