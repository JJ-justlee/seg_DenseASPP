import numpy as np


def colorize_mask(mask):
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    
    COLOR_MAP = [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), 
        (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        (255,  0,  0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for trainId, color in enumerate(COLOR_MAP):
        color_mask[mask == trainId] = color

    return color_mask