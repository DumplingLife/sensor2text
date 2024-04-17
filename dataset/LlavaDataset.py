import json
import torch
from torch.utils.data import Dataset
import os
import numpy as np

device = torch.device("cuda")

class LlavaDataset(Dataset):
    def __init__(self, json_file, embeds_dir):
        with open(json_file, 'r') as f:
            raw_data = json.load(f)

        self.llava_data = []
        for item in raw_data:
            embeds_path = f"{embeds_dir}/{item['image'].rsplit('.', 1)[0]}.npy"
            if not os.path.isfile(embeds_path):
                continue

            convo = item['conversations']
            formatted_convo = ["<s>###"]

            for i in range(0, len(convo), 2):
                assert convo[i]["from"] == "human"
                # Append user's query
                formatted_convo.extend(["User:", convo[i]["value"].replace("<image>","<ImageHere>"), "###"])
                
                if i < len(convo) - 1:
                    assistant_msg = convo[i + 1]["value"]
                    prompt = "\n".join(formatted_convo) + "\nAssistant:\n"
                    response = assistant_msg + "</s>"

                    # if lengths are too long, i get memory issues, so cap the length
                    max_len = 1800
                    if len(prompt) + len(response) <= max_len:
                        self.llava_data.append({
                            'embeds_path': embeds_path,
                            'prompt': prompt,
                            'response': response
                        })
                # Append assistant's response for the next queries
                formatted_convo.extend(["Assistant:", assistant_msg, "###"])
        
        print(f"{len(self.llava_data)} items found")

    def __len__(self):
        return len(self.llava_data)

    def __getitem__(self, idx):
        return (torch.from_numpy(np.array([np.load(self.llava_data[idx]['embeds_path'])])),
            self.llava_data[idx]['prompt'],
            self.llava_data[idx]['response']
            )