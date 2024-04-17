import torch
import torch.nn as nn
import numpy as np
from transformers import LlamaTokenizer, BertConfig
import copy
from VideoLLaMA.video_llama.models.Qformer import BertConfig, BertLMHeadModel # duplicate BertConfig import, not sure why
from VideoLLaMA.video_llama.models.modeling_llama import LlamaForCausalLM
import random
from dataset.ImagebindEmbedsDataset import reworded_captions

device = torch.device("cuda")

num_audio_query_token = 8
audio_hidden_size = 1024

# [...]
# don't use model.train(), since we can't apply this to LLaMA
# the needed params are already marked require gradients
class VideoLlamaAL(nn.Module):
    # method copied from VideoLLAMA. It's called video Qformer, but it works for video and audio
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(self, llama_model=None):
        super().__init__()
        # skip VL loading stuff
        
        # tokenizer
        print('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained("../Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf", use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token 
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        # LLaMA
        if llama_model:
            print('using preloaded LLAMA Model')
            self.llama_model = llama_model
        else:
            print('Loading LLAMA Model')
            # assume low resource
            device_8bit = 0
            self.llama_model = LlamaForCausalLM.from_pretrained(
                "../Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf",
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
            print('Loading LLAMA Done')
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        # AL side
        # skip audio encoder (we don't need this; we will sub in embeddings directly)
        # Qformer
        self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(num_query_token = num_audio_query_token,\
            vision_width=audio_hidden_size, num_hidden_layers=2)
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        self.audio_llama_proj = nn.Linear(
            self.audio_Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.audio_position_embedding = nn.Embedding(8, audio_hidden_size)

        # assume audio_Qformer not frozen
        for name, param in self.audio_Qformer.named_parameters():
            param.requires_grad = True
        self.audio_query_tokens.requires_grad = True
        for name, param in self.audio_llama_proj.named_parameters():
            param.requires_grad = True
        for name, param in self.audio_position_embedding.named_parameters():
            param.requires_grad = True
    
    # audio_imagebind_finalout is 8 consecutive saved embeddings of 2s clips, to make 16s
    # it should have shape (batch_size, time_length, 1024), e.g. (32, 8, 1024)
    def encode_audioQformer(self, audio_imagebind_finalout):
        batch_size,time_length = audio_imagebind_finalout.size()[:2]

        position_ids = torch.arange(time_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        audio_position_embeddings = self.audio_position_embedding(position_ids)
        audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

        audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
        frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

        audio_query_output = self.audio_Qformer.bert(
            query_embeds=audio_query_tokens, #[32,768]
            encoder_hidden_states=audio_imagebind_finalout,
            encoder_attention_mask=frame_atts,
            return_dict=True,
            )
        audio_hidden = audio_query_output.last_hidden_state

        inputs_llama = self.audio_llama_proj(audio_hidden)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)

        return inputs_llama, atts_llama
    
    def get_inputs_embeds(self, imagebind_embeds, prompt, num_patch_tokens):
        input_ids = self.llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        input_ids = input_ids.repeat((imagebind_embeds.size()[0],1)).to(device)

        img_embeds, atts_img = self.encode_audioQformer(imagebind_embeds)
        # convering the <ImageHere> into the embs
        # supports multiple patch tokens with consecutive <ImageHere> tokens
        im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
        temp_input_ids = copy.deepcopy(input_ids)
        temp_input_ids[temp_input_ids == im_patch_token_id] = 0
        temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

        new_input_embeds=[]
        cur_image_idx = 0
        for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
            cur_image_features = img_embeds[cur_image_idx]

            if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens:
                raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
            masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
            mask_index_start = masked_indices[0]
            if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patch_tokens, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                raise ValueError("The image patch tokens should be consecutive.")
            
            cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patch_tokens:]), dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            
            cur_image_idx += 1
        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        return inputs_embeds, atts_img

    def get_random_prompt(self, num_patch_tokens, prompt_type):
        if prompt_type == "regular":
            start_stub = f"<s>###\nUser:\nOpen your eyes and imagine you see: {'<ImageHere>' * num_patch_tokens}. "
            end_stub = "\n###\nAssistant:\n"
            return start_stub + random.choice([
                "What does this video show?",
                "Tell me what happens in the video.",
                "Can you describe what happens in the video?",
                "Give a brief summary of the video content.",
                "What is the video about?",
                "Can you provide a quick overview of the video?",
                "Share a short description of the video.",
                "What is being depicted in the video?",
            ]) + end_stub

        if prompt_type == "no prompt wrapping":
            return f"<s>###\nUser:\n{'<ImageHere>' * num_patch_tokens}.\n###\nAssistant:\n"

        if prompt_type == "list labels":
            activity_labels = list(caption.replace("</s>", "") for caption in reworded_captions.values())
            # prompt = f"<s>User: Here is information regarding a scene: {'<ImageHere>' * num_patch_tokens}. What is happening here? Please choose one of the following responses to output:\n"
            prompt = f"<s>User: Open your eyes and imagine you see: {'<ImageHere>' * num_patch_tokens}. What is happening here? Please choose one of the following responses to output:\n"
            for label in activity_labels:
                prompt += f"- {label}\n"
            # prompt += "Again, using the information you were given, output exactly one of these labels, and nothing more. Do not explain your response."
            # bug: forgot to add this bottom part for the listlabel stuff
            # prompt += "\n###\nAssistant:\n"
            return prompt

    # similar to VideoLLAMA's forward, this only returns loss
    def forward(self, imagebind_embeds, targets_text, prompt=None, num_patch_tokens=8):
        if prompt is None:
            prompt = self.get_random_prompt(num_patch_tokens, "regular")
        batch_size = imagebind_embeds.size()[0]
        
        # many things are coded to only support batch_size of 1, like the prompt (only 1 prompt allowed)
        # my GPU can't support any more anyways, but if we want to increase this, it shouldn't be too hard
        assert batch_size == 1

        to_regress_tokens = self.llama_tokenizer(targets_text, return_tensors="pt", padding="longest", add_special_tokens=False).to(device)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds, atts_img = self.get_inputs_embeds(imagebind_embeds, prompt, num_patch_tokens)
        input_embeds_len = inputs_embeds.shape[1]

        # inputs_embeds shape: inputs + to_regress
        inputs_embeds = torch.cat([inputs_embeds, to_regress_embeds], dim=1)

        # targets seq length: inputs + to_regress
        empty_targets = torch.full([batch_size, input_embeds_len], -100, dtype=torch.long, device=device)
        targets = torch.cat([empty_targets, to_regress_tokens.input_ids], dim=1)

        # attention mask
        atts_img = atts_img[:, :1].expand(-1, input_embeds_len)
        attention_mask = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        return outputs
    
    def generate(self, imagebind_embeds, prompt=None, num_patch_tokens=8, prompt_type="regular"):
        if prompt is None:
            prompt = self.get_random_prompt(num_patch_tokens, prompt_type)
        inputs_embeds, _ = self.get_inputs_embeds(imagebind_embeds, prompt, num_patch_tokens)
        output_tokens = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=250,
        )
        return self.llama_tokenizer.decode(output_tokens[0], add_special_tokens=False)

    # for testing llama
    def generate_text_only(self, prompt):
        input_ids = self.llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        input_ids = input_ids.to(device)
        
        output_tokens = self.llama_model.generate(
            input_ids=input_ids,
            max_new_tokens=250,
        )
        
        return self.llama_tokenizer.decode(output_tokens[0], add_special_tokens=False)
