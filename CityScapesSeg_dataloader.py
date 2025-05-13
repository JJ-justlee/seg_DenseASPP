import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
import torchvision.transforms.functional as TF
import tempfile

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utility import colorize
from glob import glob


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
        
        open_image = Image.open(image_path)
        open_gt = Image.open(gt_path)
        
        array_gt = np.array(open_gt)
        
        """ -------- Mapping id to train id --------"""
        gt_mapped = np.full_like(array_gt, self.ignore_label)
        for k, v in self.id_to_trainid.items():
            gt_mapped[array_gt == k] = v
        """----------------------------------------"""

        vis_gt = colorize.colorize_mask(gt_mapped)
        vis_gt = Image.fromarray(vis_gt)
        
        # aug_image = self.transform(open_image)
        # aug_gt = self.transform(gt_mapped)
        # aug_vis_gt = self.transform(vis_gt)
        
        # aug_image, aug_gt = self.resize(image=image, gt=gt, size=(self.input_width, self.input_height))
        # aug_image, aug_gt = self.rotate_image(image=aug_image, gt=aug_gt, angle=45)
        # aug_image         = np.array(aug_image, dtype= np.float32) / 255.
        # aug_gt            = np.array(aug_gt,    dtype= np.float32)
        # aug_image, aug_gt = self.flip(image=aug_image, gt=aug_gt)

        aug_image, aug_gt, aug_vis_gt = self.random_scaling(image=open_image, gt=gt_mapped, vis_gt=vis_gt)
        aug_image, aug_gt, aug_vis_gt = self.random_brightness_jittering(image=open_image, gt=gt_mapped, vis_gt=vis_gt)
        aug_image, aug_gt, aug_vis_gt = self.random_flipping_horizontally(image=open_image, gt=gt_mapped, vis_gt=vis_gt)
        aug_image, aug_gt, aug_vis_gt = self.random_crop(image=open_image, gt=gt_mapped, vis_gt=vis_gt)
        #crop 전에 filp 먼저 했을 수도 있음. 확인 필요
        
        aug_image = np.array(aug_image, dtype=np.float32) / 255.0
        aug_gt = np.array(aug_gt, dtype=np.uint8)
        aug_vis_gt = np.array(aug_vis_gt, dtype=np.uint8)

        tensor_image = torch.from_numpy(aug_image.transpose(2, 0, 1)).float()
        tensor_gt = torch.from_numpy(aug_gt).long()
        tensor_vis_gt = torch.from_numpy(aug_vis_gt.transpose(2, 0, 1)).float() / 255.0

        sample = {'image': tensor_image, 'gt': tensor_gt, 'vis_gt': tensor_vis_gt}
        
        # preprocessing_transforms = transforms.Compose([ToTensor()])
        # aug_image, aug_gt, aug_vis_gt = preprocessing_transforms(sample)

        return sample
    
    """DenseASPP"""
    # Random Scaling: [0.5 ~ 2.0]
    # Rndom Brightness [-10 ~ 10]
    # Random Horizontal Flip
    # Random Crop: 512x512

    def resize(self, image, gt, size):
        resized_image = image.resize(size, Image.BICUBIC)

        if isinstance(gt, np.ndarray):
            gt = Image.fromarray(gt.astype(np.uint8))

        resized_gt = gt.resize(size,    Image.NEAREST)

        return resized_image, resized_gt


    def rotate_image(self, image, gt, angle):
        angle = random.uniform(-angle, angle)
        image = TF.rotate(image, angle)
        gt = TF.rotate(gt, angle)

        return image, gt


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

    #DenseASPP aug
    #range 0.5, 2
    def random_scaling(self, image, gt, vis_gt):
        scale = random.uniform(0.5, 2.0)
        height = int(image.height * scale)
        width = int(image.width * scale)
        
        # numpy to PIL
        if isinstance(gt, np.ndarray):
            gt = Image.fromarray(gt)
        if isinstance(vis_gt, np.ndarray):
            vis_gt = Image.fromarray(vis_gt)

        # Resize
        image = TF.resize(image, (height, width))  # bilinear
        gt = TF.resize(gt, (height, width), interpolation=TF.InterpolationMode.NEAREST)
        vis_gt = TF.resize(vis_gt, (height, width), interpolation=TF.InterpolationMode.NEAREST)

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

    #DenseASPP aug
    #512, 512 image patches
    def random_crop(self, image, gt, vis_gt):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))

        if isinstance(gt, np.ndarray):
            gt = Image.fromarray(gt)
        if isinstance(vis_gt, np.ndarray):
            vis_gt = Image.fromarray(vis_gt)

        image = TF.crop(image, i, j, h, w)
        gt = TF.crop(gt, i, j, h, w)
        vis_gt = TF.crop(gt, i, j, h, w)
        
        return image, gt, vis_gt


class ToTensor(object):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = np.array(image, dtype= np.float32)
        gt    = np.array(gt,    dtype= np.int32)

        if isinstance(image, np.ndarray) and isinstance(gt, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))
            gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()

            return image, gt