import torch
import torch.nn.functional as F
from torch import nn

def create(size, x):
    # Calculate the value to fill the non-diagonal elements
    fill_value = (1 - x) / (size - 1)

    # Create the tensor filled with the fill_value
    tensor = torch.full((size, size), fill_value)

    # Set the diagonal values to x
    tensor.fill_diagonal_(x)

    return tensor


def f(sim):
    temperature = 0.07
    sim /= temperature
    labels = torch.arange(sim.size(0))
    loss = F.cross_entropy(sim, labels)
    print(labels)
    return loss

# Define the softmax matrices as given in the expected output
vision_x_text = torch.tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
                              [3.3836e-05, 9.9994e-01, 2.4118e-05],
                              [4.7997e-05, 1.3496e-02, 9.8646e-01]])

audio_x_text = torch.tensor([[1., 0., 0.],
                             [0., 1., 0.],
                             [0., 0., 1.]])

vision_x_audio = torch.tensor([[0.8070, 0.1088, 0.0842],
                               [0.1036, 0.7884, 0.1079],
                               [0.0018, 0.0022, 0.9960]])

print(f(vision_x_text), f(audio_x_text), f(vision_x_audio), f(create(32, 0.1)))