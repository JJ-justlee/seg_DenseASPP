import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CityScapesSegValDataset(Dataset):
  def __init__(self, root_dir, input_height, input_width):
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
    
    image_dir = osp.join(root_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'val')
    gt_dir = osp.join(root_dir, 'gtFine_trainvaltest', 'gtFine', 'val')

    self.gt_list = []
    self.image_list = []    
    for folder_name in sorted(os.listdir(image_dir)):
        for image_name in sorted(os.listdir(osp.join(image_dir, folder_name))):
            common_name = image_name.split('_leftImg8bit')[0]
            
            gt_path = osp.join(gt_dir, folder_name, common_name+'_gtFine_labelIds.png')
            image_path = osp.join(image_dir, folder_name, image_name)

            self.gt_list.append(gt_path)
            self.image_list.append(image_path)
    

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, index):
    image_path = self.image_list[index]
    gt_path = self.gt_list[index]
    
    open_image = Image.open(image_path)
    open_gt = Image.open(gt_path)

    array_gt = np.array(open_gt)
    
    """ -------- Mapping id to train id --------"""
    gt_mapped = np.full_like(array_gt, self.ignore_label)
    for k, v in self.id_to_trainid.items():
        gt_mapped[array_gt == k] = v
    """----------------------------------------"""

    aug_image, aug_gt = self.resize(image=open_image, gt=gt_mapped, size=(self.input_width, self.input_height))
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