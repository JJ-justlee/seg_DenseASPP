import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.cm as cm

class CityScapesSegValDataset(Dataset):
  def __init__(self, root_dir, input_height, input_width):
    gt_test_path = os.path.join(root_dir, 'gtFine_trainvaltest', 'gtFine', 'save_val_mapped')
    # gt_train_path = os.path.join(root_dir, 'gtFine_trainvaltest', 'gtFine', 'train')

    self.image_filenames = []
    self.mask_filenames = []

    for city_name in os.listdir(gt_test_path):
        city_path = os.path.join(gt_test_path, city_name)
        if os.path.isdir(city_path):
            for labelvalIds in os.listdir(city_path):
              labelvalIds_path = os.path.join(city_path, labelvalIds)
              # if labelvalIds.endswith('labelvalIds.png'):
              if labelvalIds.endswith('labelvalIds.png'):
                self.mask_filenames.append(labelvalIds_path)

    image_train_dir = os.path.join(root_dir, "leftImg8bit_trainvaltest", 'leftImg8bit', 'val')

    for image_dir in os.listdir(image_train_dir):
        rgbImage_path = os.path.join(image_train_dir, image_dir)
        if os.path.isdir(rgbImage_path):
            for image_file in os.listdir(rgbImage_path):
              image_file_path = os.path.join(rgbImage_path, image_file)
              if image_file.endswith('leftImg8bit.png'):
                  self.image_filenames.append(image_file_path)

    self.image_filenames = sorted(self.image_filenames)
    self.mask_filenames = sorted(self.mask_filenames)
    self.ids = sorted([os.path.basename(f).split('_leftImg8bit.png')[0] for f in self.image_filenames])

    self.input_height = input_height
    self.input_width = input_width

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, index):
    img_id   = self.ids[index]

    image = Image.open(self.image_filenames[index])
    mask = Image.open(self.mask_filenames[index])

    aug_image, aug_gt = self.resize(image=image, gt=mask, size=(self.input_width, self.input_height))
    aug_image         = np.array(aug_image, dtype= np.float32) / 255.
    aug_gt            = np.array(aug_gt,    dtype= np.float32)

    sample = {'image': aug_image, 'gt': aug_gt}

    preprocessing_transforms = transforms.Compose([ToTensor()])
    aug_image, aug_gt = preprocessing_transforms(sample)

    return aug_image, aug_gt

  def resize(self, image, gt, size):
    resized_image = image.resize(size, Image.BICUBIC)

    if isinstance(gt, np.ndarray):
      gt = Image.fromarray(gt.astype(np.uint8))

    resized_gt = gt.resize(size,    Image.NEAREST)

    return resized_image, resized_gt

class ToTensor(object):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = np.array(image, dtype= np.float32)
        gt    = np.array(gt,    dtype= np.int32)

        if isinstance(image, np.ndarray) and isinstance(gt, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))
            gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()

            return image, gt