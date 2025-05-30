import torch
from torchvision.models import mobilenet_v2

mobilenet = mobilenet_v2(weights='IMAGENET1K_V1')
torch.save(mobilenet.state_dict(), 'mobilenet_v2_pretrained.pth')