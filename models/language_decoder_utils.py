import torch
import os
import numpy as np

device = torch.device("cuda")

def load_qformer(model, qformer_path):
    loaded = torch.load(qformer_path)
    model.audio_Qformer.load_state_dict(loaded['audio_Qformer'])
    model.audio_query_tokens = loaded['audio_query_tokens']
    model.audio_position_embedding.load_state_dict(loaded['audio_position_embedding'])
    model.audio_llama_proj.load_state_dict(loaded['audio_llama_proj'])

def save_qformer(model, save_path):
    assert not os.path.isfile(save_path) # don't accidentally override
    torch.save({
        'audio_Qformer': model.audio_Qformer.state_dict(),
        'audio_query_tokens': model.audio_query_tokens,
        'audio_position_embedding': model.audio_position_embedding.state_dict(),
        'audio_llama_proj': model.audio_llama_proj.state_dict()
    }, save_path)

def get_imagebind_embeds(folder, a, b):
    imagebind_embeds = torch.zeros((1,0,1024)).to(device)
    for i in range(a, b+1):
        i_embeds = torch.from_numpy(np.load(f"{folder}/{i:03d}.npy").reshape(1, 1, 1024)).to(device)
        imagebind_embeds = torch.cat((imagebind_embeds, i_embeds), dim=1)
    return imagebind_embeds