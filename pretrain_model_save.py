import torch
import re
import os

def strip_dot_indices(checkpoint_in: str,
                      checkpoint_out: str | None = None,
                      resize_last_block=True,
                      new_channels=512):
    state_dict = torch.load(checkpoint_in)

    pattern = re.compile(r"(.*?)(\.\d+)(\..*)")
    new_state = {}
    
    for k, v in state_dict.items():
        m = pattern.fullmatch(k)
        new_key = f"{m.group(1)}{m.group(2)[1:]}{m.group(3)}" if m else k
        
        # 오직 norm5만 조정, transition3는 건드리지 않음
        if resize_last_block and ('norm5' in new_key):
            if v.shape[0] == 1024:
                new_state[new_key] = v[:new_channels]
            else:
                new_state[new_key] = v
        else:
            new_state[new_key] = v

    if checkpoint_out:
        torch.save(new_state, checkpoint_out)
        print(f"saved cleaned and reshaped state_dict to: {checkpoint_out}")
    return new_state

def strip_dot_indices(checkpoint_in: str,
                      checkpoint_out: str | None = None):
    state_dict = torch.load(checkpoint_in)

    # e.g. conv.0.weight → conv0.weight
    pattern = re.compile(r"(.*)\.(\d+)(\..*)")
    new_state = {}

    for k, v in state_dict.items():
        m = pattern.fullmatch(k)
        new_key = f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else k
        new_state[new_key] = v

    if checkpoint_out:
        torch.save(new_state, checkpoint_out)
        print(f"saved cleaned state_dict to: {checkpoint_out}")
    
    return new_state


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


def remove_module_prefix(checkpoint_in: str, checkpoint_out: str | None = None):
    state_dict = torch.load(checkpoint_in, map_location='cpu')
    new_state = {}

    for k, v in state_dict.items():
        # "module." prefix 제거
        new_key = k.replace("module.", "", 1)  # 가장 앞에 하나만
        new_state[new_key] = v

    if checkpoint_out:
        torch.save(new_state, checkpoint_out)
        print(f"Saved cleaned state_dict to: {checkpoint_out}")

    return new_state



if __name__ == "__main__":    
    pretrain_model_path = r'/home/seg_DenseASPP/pretrained/DenseNet161/DenseNet161_modi/DenseNet161_modi.pkl'
    new_pretrain_model_path = r'/home/seg_DenseASPP/pretrained/DenseNet161/DenseNet161_modi/DenseNet161_modi_no_module.pkl'

    if os.path.exists(new_pretrain_model_path):
        print(f'The file already exists at the path: {new_pretrain_model_path}')
    # else:
    #     strip_dot_indices(pretrain_model_path, new_pretrain_model_path,
    #                       resize_last_block=True, new_channels=512)
    else:
        remove_module_prefix(pretrain_model_path, new_pretrain_model_path)
