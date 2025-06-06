import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
import torchvision.transforms.functional as TF
import tempfile
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utility import colorize
from glob import glob
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def get_dataset_pair(directory, mode, kinds_list):
    image_dir = osp.join(directory, 'leftImg8bit_trainvaltest', 'leftImg8bit', mode)
    gt_dir = osp.join(directory, 'gtFine_trainvaltest', 'gtFine', mode)

    gt_list = []
    image_list = []    
    image_gt_pairs = []
    for folder_name in sorted(os.listdir(image_dir)):
        for image_name in sorted(os.listdir(osp.join(image_dir, folder_name))):
            common_name = image_name.split('_leftImg8bit')[0]
            
            gt_path = osp.join(gt_dir, folder_name, common_name+'_gtFine_labelIds.png')
            image_path = osp.join(image_dir, folder_name, image_name)

            if kinds_list == 'gt':
                gt_list.append(gt_path)
            elif kinds_list == 'image':
                image_list.append(image_path)
            elif kinds_list == 'gtImagePair':
                image_gt_pairs.append((image_path, gt_path))

    return image_gt_pairs, image_list, gt_list


class CityScapesSeg_dataset(Dataset):
    def __init__(self, root_dir, input_height, input_width, dataset):
        self.input_height = input_height
        self.input_width = input_width

        self.ignore_label = 255
        self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: self.ignore_label, 2: self.ignore_label,
                            3: self.ignore_label, 4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label,
                            7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label, 11: 2, 12: 3, 13: 4,
                            14: self.ignore_label, 15: self.ignore_label, 16: self.ignore_label, 17: 5,
                            18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                            26: 13, 27: 14, 28: 15, 29: self.ignore_label, 30: self.ignore_label,
                            31: 16, 32: 17, 33: 18}

        self.dataset = dataset

        if self.dataset == 'train':
            self.gtImagedataset, _, _ = get_dataset_pair(directory=root_dir, mode='train', kinds_list = 'gtImagePair')
            _, self.image_data, _ = get_dataset_pair(directory=root_dir, mode='train', kinds_list = 'image')
            _, _, self.gt_data = get_dataset_pair(directory=root_dir, mode='train', kinds_list = 'gt')

        elif self.dataset == 'val':        
            self.gtImagedataset, _, _ = get_dataset_pair(directory=root_dir, mode='val', kinds_list = 'gtImagePair')
            _, self.image_data, _ = get_dataset_pair(directory=root_dir, mode='val', kinds_list = 'image')
            _, _, self.gt_data = get_dataset_pair(directory=root_dir, mode='val', kinds_list = 'gt')

    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, index):
        image_path = self.image_data[index]
        gt_path = self.gt_data[index]
        
        open_image = Image.open(image_path).convert("RGB")
        open_gt = Image.open(gt_path)
        
        array_gt = np.array(open_gt)
        
        """ -------- Mapping id to train id --------"""
        gt_mapped = np.full_like(array_gt, self.ignore_label)
        for k, v in self.id_to_trainid.items():
            gt_mapped[array_gt == k] = v
        """----------------------------------------"""
        
        # self.transform = A.Compose([
        #     A.HorizontalFlip(p=0.5),
        #     # Albumentations RandomScale = “1 + scale_limit” 
        #     # 배울 그대로 받고 싶으면 Affine이나 RandomResizedCrop 사용해야함
        #     A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
        #     A.ColorJitter(brightness=0.1, p=1.0),
        #     A.RandomCrop(height=512, width=512, p=1.0),
        #     A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        #     ToTensorV2()
        # ])
        
        # augmented = self.transform(image=np.array(open_image), mask=gt_mapped)
        # image_t = augmented['image']       # Tensor
        # gt_t = augmented['mask'].long()  # Tensor (segmentation label)

        # sample = {'image': image_t, 'gt': gt_t}

        vis_gt = colorize.colorize_mask(gt_mapped)

        aug_image, aug_gt, aug_vis_gt = self.resize(image=open_image, gt=gt_mapped, vis_gt=vis_gt, size=(1024, 1024))
        aug_image, aug_gt, aug_vis_gt = self.random_scaling(image=aug_image, gt=aug_gt, vis_gt=aug_vis_gt)
        aug_image, aug_gt, aug_vis_gt = self.random_flipping_horizontally(image=aug_image, gt=aug_gt, vis_gt=aug_vis_gt)
        aug_image, aug_gt, aug_vis_gt = self.random_crop(image=aug_image, gt=aug_gt, vis_gt=aug_vis_gt)
        aug_image, aug_gt, aug_vis_gt = self.random_brightness_jittering_changed(image=aug_image, gt=aug_gt, vis_gt=aug_vis_gt)
 
        
        aug_image = np.array(aug_image, dtype=np.float32) / 255.0
        aug_gt = np.array(aug_gt, dtype=np.float32)
        aug_vis_gt = np.array(aug_vis_gt)

        if isinstance(aug_image, np.ndarray) and isinstance(aug_gt, np.ndarray) and isinstance(aug_vis_gt, np.ndarray):
            tensor_image = torch.from_numpy(aug_image.transpose(2, 0, 1)).float()
            tensor_image = (tensor_image - IMAGENET_MEAN) / IMAGENET_STD
            tensor_gt = torch.from_numpy(aug_gt).long()
            tensor_vis_gt = torch.from_numpy(aug_vis_gt)

        sample = {'image': tensor_image, 'gt': tensor_gt, 'vis_gt': tensor_vis_gt}
        
        # preprocessing_transforms = transforms.Compose([ToTensor()])
        # aug_image, aug_gt, aug_vis_gt = preprocessing_transforms(sample)

        return sample
    
    """DenseASPP"""
    # Random Scaling: [0.5 ~ 2.0]
    # Rndom Brightness [-10 ~ 10]
    # Random Horizontal Flip
    # Random Crop: 512x512

    def resize(self, image, gt, vis_gt, size):
        resized_image = image.resize(size, Image.BICUBIC) 

        if isinstance(gt, np.ndarray):
            gt = Image.fromarray(gt.astype(np.uint8))
        if isinstance(vis_gt, np.ndarray):
            vis_gt = Image.fromarray(vis_gt.astype(np.uint8))

        resized_gt = gt.resize(size,    Image.NEAREST)
        resized_vis_gt = vis_gt.resize(size,    Image.BICUBIC)

        return resized_image, resized_gt, resized_vis_gt


    def rotate_image(self, image, gt, vis_gt, angle):
        
        if isinstance(gt, np.ndarray):
            gt = Image.fromarray(gt)
        if isinstance(vis_gt, np.ndarray):
            vis_gt = Image.fromarray(vis_gt)
        
        angle = random.uniform(-angle, angle)
        image = TF.rotate(image, angle)
        gt = TF.rotate(gt, angle)
        vis_gt = TF.rotate(vis_gt, angle)

        return image, gt, vis_gt


    def flip(self, image, gt):
        hflip = random.random()
        vflip = random.random()
        if hflip > 0.5:
            image = (image[:, ::-1, :]).copy()
            gt = (gt[:, ::-1]).copy()

        if vflip > 0.5:
            image = (image[::-1, :, :]).copy()
            gt = (gt[::-1, :]).copy()

        return image, gt

    #DenseASPP aug
    def random_flipping_horizontally(self, image, gt, vis_gt):
        hflip = random.random()

        if isinstance(gt, np.ndarray):
            gt = Image.fromarray(gt)
        if isinstance(vis_gt, np.ndarray):
            vis_gt = Image.fromarray(vis_gt)

        if hflip > 0.5:
            image = TF.hflip(image)
            gt = TF.hflip(gt)
            vis_gt = TF.hflip(vis_gt)

        return image, gt, vis_gt

    # region - scaling
    #DenseASPP aug
    #range 0.5, 2
    def random_scaling(self, image, gt, vis_gt):
        scale = random.uniform(0.5, 2.0)
        height = int(self.input_height * scale)
        width = int(self.input_width * scale)
        
        # numpy to PIL
        if isinstance(gt, np.ndarray):
            gt = Image.fromarray(gt)
        if isinstance(vis_gt, np.ndarray):
            vis_gt = Image.fromarray(vis_gt)

        # Resize
        image = TF.resize(image, (height, width))  # bilinear
        gt = TF.resize(gt, (height, width), interpolation=TF.InterpolationMode.NEAREST)
        # vis_gt = TF.resize(vis_gt, (height, width), interpolation=TF.InterpolationMode.NEAREST)
        vis_gt = TF.resize(vis_gt, (height, width))

        return image, gt, vis_gt

    #DenseASPP aug
    #range -10, 10
    def random_brightness_jittering(self, image, gt, vis_gt):
        # image = TF.adjust_brightness(image, brightness_factor=random.uniform(-10, 10))
        image_np = np.array(image).astype(np.float32)
        shift = random.uniform(-10, 10)
        image_np += shift
        image_np = np.clip(image_np, 0, 255)
        image = Image.fromarray(image_np.astype(np.uint8))

        return image, gt, vis_gt

    def random_brightness_jittering_changed(self, image, gt, vis_gt):
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        brightness = random.uniform(-10, 10)
        brightness_image = TF.adjust_brightness(image, brightness_factor=1.0 + brightness/100.)
        
        return brightness_image, gt, vis_gt

    #DenseASPP aug
    #512, 512 image patches
    def random_crop_torchvision(self, image, gt, vis_gt):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(1024, 1024))

        if isinstance(gt, np.ndarray):
            gt = Image.fromarray(gt)
        if isinstance(vis_gt, np.ndarray):
            vis_gt = Image.fromarray(vis_gt)

        image = TF.crop(image, i, j, h, w)
        gt = TF.crop(gt, i, j, h, w)
        vis_gt = TF.crop(vis_gt, i, j, h, w)
        
        return image, gt, vis_gt
    
    def random_crop(self, image, gt, vis_gt,
                    size=(512, 512), max_tries=10, min_valid=0.5):

        # ── PIL → numpy 변환 (이미 numpy/torch 면 그대로 사용)
        img_np     = np.array(image)    if isinstance(image,  Image.Image) else image
        gt_np      = np.array(gt)       if isinstance(gt,     Image.Image) else gt
        vis_gt_np  = np.array(vis_gt)   if isinstance(vis_gt, Image.Image) else vis_gt

        H, W = gt_np.shape
        top, left = 0, 0
        for _ in range(max_tries):
            top  = np.random.randint(0, H - size[0] + 1)
            left = np.random.randint(0, W - size[1] + 1)

            crop_gt     = gt_np     [top:top+size[0], left:left+size[1]]
            crop_vis_gt = vis_gt_np [top:top+size[0], left:left+size[1]]
            valid_ratio = (crop_gt != 255).mean() # sum number of pixels without 255 / (height * width)

            if valid_ratio > min_valid:
                crop_img = img_np[top:top+size[0], left:left+size[1]]
                return crop_img, crop_gt, crop_vis_gt

        # fallback
        crop_img = img_np[top:top+size[0], left:left+size[1]]
        return crop_img, crop_gt, crop_vis_gt


# class ToTensor(object):
#     def __call__(self, sample):
#         image, gt = sample['image'], sample['gt']
#         image = np.array(image, dtype= np.float32)
#         gt    = np.array(gt,    dtype= np.int32)

#         if isinstance(image, np.ndarray) and isinstance(gt, np.ndarray):
#             image = torch.from_numpy(image.transpose(2, 0, 1))
#             gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()

#             return image, gt