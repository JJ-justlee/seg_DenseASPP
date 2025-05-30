import os
import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import utility
import multiprocessing

from torch.utils.data import DataLoader
from Architectures.DenseASPP_modified import DenseASPP_modified
from Architectures.MobileNetDenseASPP import MobileNetDenseASPP
from utility.colorize import colorize_mask
from val_data_loader import CityScapesSegValDataset

from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

from Argument.Directory.Inference_Directories import Inference_Dics_args
from Argument.Parameter.Inference_Parameters import  Inference_Parameter_args

arg_Dic = Inference_Dics_args()

arg_parameter = Inference_Parameter_args()

color_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
color_map = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255,  0,  0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

def denorm(x):                       # x : (C,H,W) tensor on GPU/CPU
    mean = IMAGENET_MEAN.to(x.device)
    std  = IMAGENET_STD .to(x.device)
    return torch.clamp(x * std + mean, 0, 1)

def restore_original_size(result, mode='nearest'):
    # result shape: [N, H, W] → [N, 1, H, W]
    if result.dim() == 3:
        result = result.unsqueeze(1)  # Add channel dimension

    result = result.float()
    result = F.interpolate(result, size=(arg_parameter.raw_height, arg_parameter.raw_width), mode=mode)

    return result

def cover_colormap(image):
    dst = None
    if isinstance(image, memoryview):
        image = torch.from_numpy(np.array(image))  # Ensure it’s converted correctly to a tensor

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # If it’s a tensor, convert it to a numpy array

    # Assuming the image has shape [N, H, W]
    if image.ndim == 3:
        image = image[0]  # Select the first image if it's batched
    
    row, col = image.shape
    dst = np.zeros((row, col, 3), dtype=np.uint8)

    COLOR_MAP = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), 
                 (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), 
                 (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), 
                 (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), 
                 (0, 0, 230), (119, 11, 32)]

    # Map each class index to its color
    for i in range(19):
        dst[image == i] = COLOR_MAP[i]

    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    
    return dst


def test():
        # GPU 셋팅
    if torch.cuda.is_available():
        if arg_parameter.gpu is not None:
            device = torch.device(f'cuda:{arg_parameter.gpu}')
            print("Use GPU: {} for training".format(arg_parameter.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')

    if not osp.exists(arg_Dic.pred_dir):
        os.makedirs(arg_Dic.pred_dir)

    CityScapes_val_dataset = CityScapesSegValDataset(root_dir=arg_Dic.root_dir,
                                    input_height=arg_parameter.input_height, input_width=arg_parameter.input_width)

    val_dataloader = DataLoader(CityScapes_val_dataset,
                        batch_size=arg_parameter.batch_size,
                        shuffle=False,
                        num_workers=multiprocessing.cpu_count() // 2,
                        pin_memory=True,
                        persistent_workers=True)
    
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

    'pretrained_path': "/home/seg_DenseASPP/pretrained/densenet121_clean.pth"
    }

    model_dir = osp.join(arg_Dic.bestModel_dir, '0075_69.pth')
    #model_dir을 파이썬 객체로 복원
    checkpoint = torch.load(model_dir, map_location='cpu')

    # 학습된 모델 불러오기
    #model = models.segmentation.fcn_resnet50(num_classes=1)
    n_class = 19
    model = DenseASPP_modified(model_cfg, n_class, output_stride=8)
    # model = MobileNetDenseASPP(model_cfg, n_class, output_stride=8)
    
    #모델 구조에 입혀줌
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for step, (sample_image, sample_gt) in enumerate(val_dataloader):

            # sample_image = torch.tensor(sample_image, device=device, dtype=torch.float32)
            # sample_gt = torch.tensor(sample_gt, device=device, dtype=torch.float32)
            sample_image = sample_image.to(device, dtype=torch.float32)
            sample_gt = sample_gt.to(device, dtype=torch.long)
            sample_gt = torch.unsqueeze(sample_gt, dim=1)

            prediction = model(sample_image)
            # 원래 사이즈로 복구
            prediction = torch.softmax(prediction, dim=1)
            prediction_class = torch.argmax(prediction, dim=1)   

            #to make images bigger
            sample_image_resized = restore_original_size(sample_image)
            sample_image_resized = denorm(sample_image_resized)
            sample_gt_resized = restore_original_size(sample_gt)
            prediction_resized = restore_original_size(prediction_class)

            #do not need to use detach function in sapcific python version
            # segmentation 결과 (array)
            sample_image_cpu = sample_image_resized.cpu().detach().numpy()
            sample_gt_cpu = sample_gt_resized.cpu().detach().numpy()
            # 변수 prediction_cpu가 모델의 최종 결과.
            prediction_cpu = prediction_resized.cpu().detach().numpy()

            # 그림으로 그리기
            for num, (sp_image, sp_gt, sp_pred) in enumerate(zip(sample_image_cpu, sample_gt_cpu, prediction_cpu)):
                sp_image = np.transpose(sp_image, (1, 2, 0))
                sp_gt = np.transpose(sp_gt, (1, 2, 0))
                sp_pred = np.transpose(sp_pred, (1, 2, 0))
                
                gt_color = colorize_mask(np.squeeze(sp_gt))
                pred_color = colorize_mask(np.squeeze(sp_pred))

                plt.subplot(1, 3, 1)
                plt.imshow(sp_image)
                plt.title('input')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(gt_color)
                plt.title('gt')
                plt.axis('off')

                sp_gt_tensor = torch.tensor(np.squeeze(sp_gt)).to(device=device, dtype=torch.long)
                sp_pred_tensor = torch.tensor(np.squeeze(sp_pred)).to(device=device, dtype=torch.long)
            
                # ignore label 처리
                sp_gt_tensor = torch.where(sp_gt_tensor == 255, -1, sp_gt_tensor)
                gt_classes = torch.unique(sp_gt_tensor)
                gt_classes = gt_classes[gt_classes != -1]
                
                #시간복잡도(O(n))
                #it can be going down by using numpy
                IoU_list = []
                
                for gt_class in gt_classes:
                    pred_cls = (sp_pred_tensor == gt_class).bool()
                    gt_cls = (sp_gt_tensor == gt_class).bool()

                    intersection = (pred_cls & gt_cls).sum().float()
                    union = (pred_cls | gt_cls).sum().float()

                    IoU_per_image = intersection / (union + 1e-6)    

                    IoU_list.append(IoU_per_image)

                mIoU_per_image = (sum(IoU_list) / len(IoU_list))
                mIoU_per_image = mIoU_per_image * 100

                plt.subplot(1, 3, 3)
                plt.imshow(pred_color)
                plt.title(f'prediction(IoU: {mIoU_per_image:.4f}%)')
                plt.axis('off')

                plt.tight_layout(w_pad=1.0)
                plt.savefig(osp.join(arg_Dic.pred_dir, f'{step}-{num}.png'))
                plt.close()

if __name__ == "__main__":
    test()


"""

def change():
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    folders = sorted(os.listdir(IMG_PATH))
    for f in folders:
        folder_path = IMG_PATH + f + "/"
        save_path = SAVE_PATH + f + "/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        names = sorted(os.listdir(folder_path))
        for n in names:
            print(n)
            img = cv2.cvtColor(cv2.imread(folder_path + n, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            R, G, B = cv2.split(img)
            mask = numpy.zeros_like(R, dtype=numpy.uint8)

            for i in range(color_list.__len__()):
                tmp_mask = numpy.zeros_like(R, dtype=numpy.uint8)
                color = color_map[i]
                tmp_mask[R[:] == color[0]] += 1
                tmp_mask[G[:] == color[1]] += 1
                tmp_mask[B[:] == color[2]] += 1

                mask[tmp_mask[:] == 3] = color_list[i]
            cv2.imwrite(save_path + n, mask)
            cv2.waitKey(1)
"""