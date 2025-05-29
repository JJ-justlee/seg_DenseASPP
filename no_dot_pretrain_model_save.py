import torch
import re
import os

def strip_dot_indices(checkpoint_in: str,
                      checkpoint_out: str | None = None):
    """
    DenseNet-계열에서 'norm.1.weight' 같은 키를 'norm1.weight' 로 수정.
    
    ▸ checkpoint_in  : 원본 .pth / .pt 파일 경로
    ▸ checkpoint_out : 저장 경로(생략하면 덮어쓰지 않고 dict 반환)
    """
    state_dict = torch.load(checkpoint_in)

    pattern = re.compile(r"(.*?)(\.\d+)(\..*)")   # 그룹1  그룹2  그룹3
    new_state = {}
    for k, v in state_dict.items():
        m = pattern.fullmatch(k)
        if m:
            # '.1' → '1'  ('.2' → '2', …)
            new_key = f"{m.group(1)}{m.group(2)[1:]}{m.group(3)}"
            new_state[new_key] = v
        else:
            new_state[k] = v

    if checkpoint_out:
        torch.save(new_state, checkpoint_out)
        print(f"saved cleaned state_dict to: {checkpoint_out}")
    return new_state


if __name__ == "__main__":    
    pretrain_model_path = r'/home/seg_DenseASPP/pretrained/densenet121-a639ec97.pth'
    new_pretrain_model_path = r'/home/seg_DenseASPP/pretrained/densenet121_clean.pth'

    if os.path.exists(new_pretrain_model_path):
        print(f'The file already exists at the path:{new_pretrain_model_path}')
    else:
        strip_dot_indices(pretrain_model_path, new_pretrain_model_path)