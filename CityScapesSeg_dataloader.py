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
    self.image_dict = {}
    self.mask_dict = {}

    # gt_train_path = os.path.join(root_dir, 'gtFine_trainvaltest', 'gtFine', 'save_train_mapped')

    # for city_name in os.listdir(gt_train_path):
    #     city_path = os.path.join(gt_train_path, city_name)
    #     if os.path.isdir(city_path):
    #         for labelTrainIds in os.listdir(city_path):
    #           labelTrainIds_path = os.path.join(city_path, labelTrainIds)
    #           if labelTrainIds.endswith('labeltrainIds.png'):
    #             self.mask_filenames.append(labelTrainIds_path)

    # 매핑 정의
    ignore_label = 255
    id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                    26: 13, 27: 14, 28: 15, 29: ignore_label, 30: ignore_label,
                    31: 16, 32: 17, 33: 18}

    gt_train_path = os.path.join(root_dir, 'gtFine_trainvaltest', 'gtFine', 'train')
    
    for firstGtfolder in os.listdir(gt_train_path):
        firstGtfolder_path = os.path.join(gt_train_path, firstGtfolder)
        if os.path.isdir(firstGtfolder_path):
          for secondGtfile in os.listdir(firstGtfolder_path):
            if secondGtfile.endswith('labelIds.png'):                
                # 파일 불러오기
                label_path = os.path.join(firstGtfolder_path, secondGtfile)
                label_img = np.array(Image.open(label_path))

                # 매핑 적용
                mapped = np.full_like(label_img, ignore_label)
                for k, v in id_to_trainid.items():
                    mapped[label_img == k] = v

                # save_name = filename.replace('labelIds.png', f'label{split}Ids.png')
                # save_path = os.path.join(output_city_path, save_name)

                # 이미지로 저장 (uint8로 변환)
                gt_mapped = Image.fromarray(mapped.astype(np.uint8))

                key = secondGtfile.replace('_gtFine_labelIds.png', '')
                
                # self.mask_filenames.append((secondGtfile, gt_mapped))
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                  gt_mapped.save(tmp.name)
                  self.mask_dict[key] = tmp.name

    image_train_dir = os.path.join(root_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'train')

    for image_dir in os.listdir(image_train_dir):
        rgbImage_path = os.path.join(image_train_dir, image_dir)
        if os.path.isdir(rgbImage_path):
            for image_file in os.listdir(rgbImage_path):
              image_file_path = os.path.join(rgbImage_path, image_file)
              if image_file.endswith('leftImg8bit.png'):
                  key = image_file.replace('_leftImg8bit.png', '')
                  self.image_dict[key] = image_file_path
                  

    common_keys = sorted(set((self.image_dict.keys())) & set((self.mask_dict.keys())))
    self.image_filenames = [self.image_dict[k] for k in common_keys]
    self.mask_filenames = [self.mask_dict[k] for k in common_keys]
    self.mask_ids = common_keys

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

  def __len__(self):
    return len(self.mask_ids)

  def __getitem__(self, index):
    img_id   = self.mask_ids[index]

    image = Image.open(self.image_filenames[index])
    mask = Image.open(self.mask_filenames[index])

    vis_mask = colorize.colorize_mask(np.array(mask))
    vis_mask = Image.fromarray(vis_mask)
    
    aug_image = self.transform(image)
    aug_gt = self.transform(mask)
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