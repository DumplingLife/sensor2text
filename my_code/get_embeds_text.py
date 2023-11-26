from video_llama.models.ImageBind.models import imagebind_model
from video_llama.models.ImageBind.models.imagebind_model import ModalityType
from video_llama.models.ImageBind import data
import torch
import numpy as np
import h5py

device = "cuda:0"

model, emb_size = imagebind_model.imagebind_huge()
model.load_state_dict(torch.load("../Video-LLaMA-2-7B-Finetuned/imagebind_huge.pth"))
model.to(device)

# Define batch size
batch_size = 16

with h5py.File("actionsense_data/text/S00_2s_withlabel.hdf5", 'r') as hdf_file:
    all_example_labels = hdf_file["example_labels"]
    example_labels = []
    for idx, label in enumerate(all_example_labels):
        if label: # if label isn't empty
            example_labels.append((idx, label.decode('utf-8')))

print("example_labels[0:5],", example_labels[0:5])

# Process in batches
for batch_start in range(0, len(example_labels), batch_size):
    batch_end = min(batch_start + batch_size, len(example_labels))
    print(f"Processing batch: {batch_start} to {batch_end}")

    inputs = {
        ModalityType.TEXT: data.load_and_transform_text([example_labels[i][1] for i in range(batch_start, batch_end)], device),
    }
    print(inputs[ModalityType.TEXT].shape)

    with torch.no_grad():
        embeddings = model(inputs)

    for i, example_labels_idx in enumerate(range(batch_start, batch_end)):
        print(embeddings["text"][i].shape)
        output_dir = "actionsense_data/S00_imagebind_embeds_text_2s"
        np.save(f"{output_dir}/{example_labels[example_labels_idx][0]:03d}.npy", embeddings["text"][i].cpu().numpy())