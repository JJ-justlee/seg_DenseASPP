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



def get_dataset_pair(directory, mode):
    image_dir = osp.join(directory, 'leftImg8bit_trainvaltest', 'leftImg8bit', mode)
    gt_dir = osp.join(directory, 'gtFine_trainvaltest', 'gtFine', mode)
    
    image_gt_pairs = []
    for folder_name in os.listdir(image_dir):
        for image_name in os.listdir(osp.join(image_dir, folder_name)):
            common_name = image_name.split('_leftImg8bit')[0]
            
            gt_path = osp.join(gt_dir, folder_name, common_name+'_gtFine_labelIds.png')
            image_path = osp.join(image_dir, folder_name, image_name)

            image_gt_pairs.append((image_path, gt_path))
            
    return image_gt_pairs



class CityScapesSeg_dataset(Dataset):
    def __init__(self, root_dir, input_height, input_width):
        self.input_height = input_height
        self.input_width = input_width

        ignore_label = 255
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                            3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                            7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                            14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                            18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                            26: 13, 27: 14, 28: 15, 29: ignore_label, 30: ignore_label,
                            31: 16, 32: 17, 33: 18}

        self.train_dataset = get_dataset_pair(directory=root_dir, mode='train')
        self.val_dataset = get_dataset_pair(directory=root_dir, mode='val')
        
        """DenseASPP"""
        # Random Scaling: [0.5 ~ 2.0]
        # Rndom Brightness [-10 ~ 10]
        # Random Horizontal Flip
        # Random Crop: 512x512
        self.transform = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.5, 2.0)),
                                            transforms.RandomHorizontalFlip(p=0.5), 
                                            transforms.ColorJitter(brightness=0.1), 
                                            transforms.ToTensor()])
        
        self.val_tranform = transforms.Compose([transforms.Resize(size=256),
                                                transforms.ToTensor()])

    def __len__(self):
        return len(self.train_dataset)


    def __getitem__(self, index):
        train_image_path, train_gt_path   = self.train_dataset[index]
        val_image_path, val_gt_path = self.val_dataset[index]
        
        train_image = Image.open(train_image_path)
        train_gt = Image.open(train_gt_path)
        
        val_image = Image.open(val_image_path)
        val_gt = Image.open(val_gt_path)
        
        """ -------- Mapping id to train id --------"""
        train_gt = np.array(train_gt)
        for k, v in self.id_to_trainid.items():
            train_gt[train_gt == k] = v
        
        vis_train_gt = colorize.colorize_mask(train_gt)
        vis_train_gt = Image.fromarray(vis_train_gt)
        """----------------------------------------"""
        
        train_gt = Image.fromarray(train_gt)
        aug_image = self.transform(train_image)
        aug_gt = self.transform(train_gt)
        aug_vis_gt = self.transform(vis_train_gt)
        
        aug_val_image = self.val_tranform(val_image)
        aug_val_gt = self.val_tranform(val_gt)
        
        # aug_image, aug_gt = self.resize(image=image, gt=gt, size=(self.input_width, self.input_height))
        # aug_image, aug_gt = self.rotate_image(image=aug_image, gt=aug_gt, angle=45)
        # aug_image         = np.array(aug_image, dtype= np.float32) / 255.
        # aug_gt            = np.array(aug_gt,    dtype= np.float32)
        # aug_image, aug_gt = self.flip(image=aug_image, gt=aug_gt)

        sample = {'image': aug_image, 'gt': aug_gt, 'vis_gt': aug_vis_gt}
        val_sample = {'val_image': aug_val_image, 'val_gt': aug_val_gt}
        
        # preprocessing_transforms = transforms.Compose([ToTensor()])
        # aug_image, aug_gt = preprocessing_transforms(sample)

        return sample, val_sample
    

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
        # vflip = random.random()
        if hflip > 0.5:
            image = (image[:, ::-1, :]).copy()
            gt = (gt[:, ::-1]).copy()

        # if vflip > 0.5:
        #     image = (image[::-1, :, :]).copy()
        #     gt = (gt[::-1, :]).copy()

        return image, gt

  # #DenseASPP aug
  # def random_flipping_horizontally(self, image, gt):
  
  #   return image, gt

  # #DenseASPP aug
  # #range 0.5, 2
  # def random_scalig(self, image, gt):
  
  #   return image, gt

  # #DenseASPP aug
  # #range -10, 10
  # def random_brightness_jittering(self, image, gt):

  #   return image, gt
  
  # #DenseASPP aug
  # #512, 512 image patches
  # def random_crop(self, image, gt):

  #   return image, gt


class ToTensor(object):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = np.array(image, dtype= np.float32)
        gt    = np.array(gt,    dtype= np.int32)

        if isinstance(image, np.ndarray) and isinstance(gt, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))
            gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()

            return image, gt