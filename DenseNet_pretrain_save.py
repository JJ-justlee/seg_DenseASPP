import torch
from torchvision.models import densenet161

# DenseNet-161 사전학습된 모델 로드
model = densenet161(weights='IMAGENET1K_V1')  # torchvision ≥0.13
# model = densenet161(pretrained=True)       # torchvision <0.13 (deprecated)

# state_dict만 저장
torch.save(model.state_dict(), 'densenet161_imagenet_pretrained.pth')