import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utility import colorize

import tempfile
import shutil

class CityScapesSeg_dataset(Dataset):
  def __init__(self, root_dir, input_height, input_width):
    self.image_filenames = []
    self.gt_filenames = []

    # 매핑 정의
    self.ignore_label = 255
    self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: self.ignore_label, 2: self.ignore_label,
                    3: self.ignore_label, 4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label,
                    7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label, 11: 2, 12: 3, 13: 4,
                    14: self.ignore_label, 15: self.ignore_label, 16: self.ignore_label, 17: 5,
                    18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                    26: 13, 27: 14, 28: 15, 29: self.ignore_label, 30: self.ignore_label,
                    31: 16, 32: 17, 33: 18}

    gt_train_path = os.path.join(root_dir, 'gtFine_trainvaltest', 'gtFine', 'train')
    
    for firstGtfolder in os.listdir(gt_train_path):
        firstGtfolder_path = os.path.join(gt_train_path, firstGtfolder)
        if os.path.isdir(firstGtfolder_path):
          for firstGtfile in os.listdir(firstGtfolder_path):
            firstGtfile_path = os.path.join(firstGtfolder_path, firstGtfile)
            if firstGtfile.endswith('_gtFine_labelIds.png'):
                self.gt_filenames.append(firstGtfile_path)

    image_train_path = os.path.join(root_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'train')

    for firstImageFolder in os.listdir(image_train_path):
        firstImageFolder_path = os.path.join(image_train_path, firstImageFolder)
        if os.path.isdir(firstImageFolder_path):
            for firstImageFile in os.listdir(firstImageFolder_path):
              firstImageFile_path = os.path.join(firstImageFolder_path, firstImageFile)
              if firstImageFile.endswith('_leftImg8bit.png'):
                  self.image_filenames.append(firstImageFile_path)

    self.image_filenames = sorted(self.image_filenames)
    self.mask_filenames = sorted(self.gt_filenames)

    self.input_height = input_height
    self.input_width = input_width
    
    """DenseASPP"""
    # Random Scaling: [0.5 ~ 2.0]
    # Rndom Brightness [-10 ~ 10]
    # Random Horizontal Flip
    # Random Crop: 512x512
    self.transform = transforms.Compose([
      transforms.RandomResizedCrop(size=256, scale=(0.5, 2.0)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ColorJitter(brightness=0.1),
      transforms.ToTensor()])

    return 

  def __len__(self):
    return len(self.image_filenames)

  def __getitem__(self, index):
    img_ids = self.image_filenames[index]
    gt_ids = self.gt_filenames[index]

    image = Image.open(img_ids)
    gt = Image.open(gt_ids)

    gt = np.array(gt)

    # 매핑 적용
    gt_mapped = np.full_like(gt, self.ignore_label)
    for k, v in self.id_to_trainid.items():
        gt_mapped[gt == k] = v

    vis_mask = colorize.colorize_mask(np.array(gt_mapped))
    vis_mask = Image.fromarray(vis_mask)
    
    gt = Image.fromarray(gt)
    aug_image = self.transform(image)
    aug_gt = self.transform(gt)
    aug_vis_gt = self.transform(vis_mask)
    # aug_image, aug_gt = self.resize(image=image, gt=mask, size=(self.input_width, self.input_height))
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