import os
import argparse

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model-type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--video-folder", type=str, required=True, help="path to folder containing .avi videos")
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

"""
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')
"""

# ========================================
#             Process Videos
# ========================================

def process_videos(video_folder, chat, args):
    video_folder = os.path.abspath(video_folder)
    chat_state = default_conversation.copy()
    img_list = []
    
    for video_file in os.listdir(video_folder):
        print(video_file)
        if video_file.endswith('.avi'):
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing video: {video_path}")
            chat.upload_video(video_path, chat_state, img_list)


# process_videos(args.video_folder, chat, args)
process_videos(args.video_folder, None, args)