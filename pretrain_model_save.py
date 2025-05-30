import torch
import re
import os

# def strip_dot_indices(checkpoint_in: str,
#                       checkpoint_out: str | None = None,
#                       resize_last_block=True,
#                       new_channels=512):
#     state_dict = torch.load(checkpoint_in)

#     pattern = re.compile(r"(.*?)(\.\d+)(\..*)")
#     new_state = {}
    
#     for k, v in state_dict.items():
#         m = pattern.fullmatch(k)
#         new_key = f"{m.group(1)}{m.group(2)[1:]}{m.group(3)}" if m else k
        
#         # 오직 norm5만 조정, transition3는 건드리지 않음
#         if resize_last_block and ('norm5' in new_key):
#             if v.shape[0] == 1024:
#                 new_state[new_key] = v[:new_channels]
#             else:
#                 new_state[new_key] = v
#         else:
#             new_state[new_key] = v

#     if checkpoint_out:
#         torch.save(new_state, checkpoint_out)
#         print(f"saved cleaned and reshaped state_dict to: {checkpoint_out}")
#     return new_state

import torch
import re

import torch
import re

def patch_mobilenet_keys(checkpoint_in: str, checkpoint_out: str):
    state_dict = torch.load(checkpoint_in)
    new_state = {}

    for k, v in state_dict.items():
        # conv.1 -> conv1
        k = re.sub(r'\.conv\.(\d+)', lambda m: f'.conv{m.group(1)}', k)
        k = re.sub(r'conv(\d)\.(\d)', lambda m: f'conv{m.group(1)}{m.group(2)}', k)

        # features. 추가
        if not k.startswith("features.features."):
            k = "features." + k  
        new_state[k] = v

    torch.save(new_state, checkpoint_out)
    print(f"Modified keys saved to: {checkpoint_out}")
    return new_state



if __name__ == "__main__":    
    pretrain_model_path = r'/home/seg_DenseASPP/pretrained/MobileNetV2/MobileNetV2_Original/mobilenet_v2_pretrained.pth'
    new_pretrain_model_path = r'/home/seg_DenseASPP/pretrained/MobileNetV2/MobileNetV2_modified/mobilenetV2_pretrained_modified.pth'

    if os.path.exists(new_pretrain_model_path):
        print(f'The file already exists at the path: {new_pretrain_model_path}')
    # else:
    #     strip_dot_indices(pretrain_model_path, new_pretrain_model_path,
    #                       resize_last_block=True, new_channels=512)
    else:
        patch_mobilenet_keys(pretrain_model_path, new_pretrain_model_path)
