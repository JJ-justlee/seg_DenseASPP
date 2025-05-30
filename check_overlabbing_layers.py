import torch
from Architectures.DenseASPP_modified import DenseASPP_modified
from Architectures.MobileNetDenseASPP import MobileNetDenseASPP

def load_partial_pretrained_weights(model, pretrained_path, show_missed=False):
    print(f"Loading partial pretrained weights from: {pretrained_path}")
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()

    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and v.shape == model_dict[k].shape}

    print(f'mapped {len(filtered_dict)}/{len(model_dict)} layers')
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    # 옵션: 매핑 안 된 첫 20개 키 출력
    if show_missed:
        missed = [k for k in pretrained_dict.keys() if k not in filtered_dict]
        print("매칭 실패 키:", missed)
    return model

def debug_key_diff(model, pretrained_path, n=40):
    state = torch.load(pretrained_path, map_location="cpu")
    model_keys = list(model.state_dict().keys())

    # 1) pretrained 키 앞 n개
    print(f"\n=== pretrained keys 앞 {40}개 ===")
    for k in list(state.keys())[:n]:
        print("  ", k)

    # 2) 모델 키 앞 n개
    print(f"\n=== model keys 앞 {40}개 ===")
    for k in model_keys[:n]:
        print("  ", k)

    # 3) 모델에는 없지만 pretrained에 있는 키 20개
    missing = [k for k in state if k not in model_keys]
    print("\n--- 모델에 없는(pretrained 전용) 키 ---")
    for k in missing: print("  ", k)

    # 4) 이름은 맞지만 shape이 안 맞는 키 10개
    shape_bad = [(k, state[k].shape, model.state_dict()[k].shape)
                 for k in state if k in model.state_dict() and
                 state[k].shape != model.state_dict()[k].shape]
    if shape_bad:
        print("\n--- shape 불일치 키 ---")
        for k,s1,s2 in shape_bad:
            print(f"  {k}: pretrained{tuple(s1)} vs model{tuple(s2)}")
    else:
        print("\nshape 불일치 없음")
        
        

model_cfg = {
    'bn_size': 4,
    'drop_rate': 0,
    'growth_rate': 32,
    'num_init_features': 64,
    'block_config': (6, 12, 24, 16),

    'dropout0': 0.1,
    'dropout1': 0.1,
    'd_feature0': 128,
    'd_feature1': 64,

    'pretrained_path': "/home/seg_DenseASPP/pretrained/MobileNetV2/MobileNetV2_modified/mobilenetV2_pretrained_modified.pth"
    }

if __name__ == "__main__":
    model = MobileNetDenseASPP(model_cfg, n_class=19, output_stride=8)
    # print("conv1 out :", model.features.denseblock1.denselayer1.conv1.out_channels)
    # print("norm1 dim :", model.features.denseblock1.denselayer1.norm1.num_features)
    # print(model.features.transition3.conv.out_channels)   # 1024
    # print(model.features.transition4.conv.out_channels)   # 1024
    model = load_partial_pretrained_weights(model, pretrained_path=model_cfg['pretrained_path'], show_missed=True)
    debug_key_diff(model, model_cfg['pretrained_path'])
# if __name__ == "__main__":
#     model = DenseASPP(model_cfg, n_class=19, output_stride=8)

#     model = model.cuda()
#     model.eval()

#     with torch.no_grad():
#         x = torch.randn(1, 3, 1024, 2048).cuda()   # dummy input (H=1024, W=2048)
#         feat = model.features(x)                   # <-- 여기!

#     print("feature map shape :", feat.shape)