import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
import torchvision.transforms.functional as TF
import tempfile
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

            """DenseASPP"""
            # Random Scaling: [0.5 ~ 2.0]
            # Rndom Brightness [-10 ~ 10]
            # Random Horizontal Flip
            # Random Crop: 512x512
            self.transform = A.Compose([A.RandomResizedCrop(size=(256, 256), scale=(0.5, 2.0)),
                                                A.RandomHorizontalFlip(p=0.5), 
                                                A.ColorJitter(brightness=0.1), 
                                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                                ToTensorV2()
                                                ],
                                                additional_targets={
                                                    'gt_mapped': 'mask',
                                                    'vis_gt': 'mask'
                                                    }
                                                )

        elif self.dataset == 'val':        
            self.gtImagedataset, _, _ = get_dataset_pair(directory=root_dir, mode='val', kinds_list = 'gtImagePair')
            _, self.image_data, _ = get_dataset_pair(directory=root_dir, mode='val', kinds_list = 'image')
            _, _, self.gt_data = get_dataset_pair(directory=root_dir, mode='val', kinds_list = 'gt')
        
            self.transform = A.Compose([A.RandomResizedCrop(size=(256, 256), scale=(0.5, 2.0)),
                                                A.RandomHorizontalFlip(p=0.5), 
                                                A.ColorJitter(brightness=0.1), 
                                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                                ToTensorV2()
                                                ],
                                                additional_targets={
                                                    'gt_mapped': 'mask',
                                                    'vis_gt': 'mask'
                                                    }
                                                )

    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, index):
        image_path = self.image_data[index]
        gt_path = self.gt_data[index]
        
        open_image = Image.open(image_path)
        open_gt = Image.open(gt_path)
        
        array_gt = np.array(open_gt)
        
        """ -------- Mapping id to train id --------"""
        mapped = np.full_like(array_gt, self.ignore_label)
        for k, v in self.id_to_trainid.items():
            mapped[array_gt == k] = v
        
        gt_mapped = Image.fromarray(mapped.astype(np.uint8))
        """----------------------------------------"""

        vis_gt = colorize.colorize_mask(mapped)
        vis_gt = Image.fromarray(vis_gt)
        transformed = self.transform(image=np.array(open_image), 
                                     gt_mapped=np.array(gt_mapped), 
                                     vis_gt=np.array(vis_gt)
                                     )
        
        aug_image = transformed["image"]
        aug_gt = transformed["gt_mapped"]
        aug_vis_gt = transformed['vis_gt']
        # aug_image = self.transform(open_image)
        # aug_gt = self.transform(gt_mapped)
        # aug_vis_gt = self.transform(vis_gt)
                
        # aug_image, aug_gt = self.resize(image=image, gt=gt, size=(self.input_width, self.input_height))
        # aug_image, aug_gt = self.rotate_image(image=aug_image, gt=aug_gt, angle=45)
        # aug_image         = np.array(aug_image, dtype= np.float32) / 255.
        # aug_gt            = np.array(aug_gt,    dtype= np.float32)
        # aug_image, aug_gt = self.flip(image=aug_image, gt=aug_gt)

        sample = {'image': aug_image, 'gt': aug_gt, 'vis_gt': aug_vis_gt}
        
        # preprocessing_transforms = transforms.Compose([ToTensor()])
        # aug_image, aug_gt = preprocessing_transforms(sample)

        return sample
    

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


class ToTensor(object):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = np.array(image, dtype= np.float32)
        gt    = np.array(gt,    dtype= np.int32)

        if isinstance(image, np.ndarray) and isinstance(gt, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))
            gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()

            return image, gt