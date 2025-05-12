import os
import numpy as np
from PIL import Image

import tempfile
import shutil

gt_dir = r'/home/seg_DenseASPP/CityScapesDataset/gtFine_trainvaltest/gtFine/val'
gt_data = []

for first_gt_folder in sorted(os.listdir(gt_dir)):
    first_gt_folder_path = os.path.join(gt_dir, first_gt_folder)
    for first_gt_file in sorted(os.listdir(first_gt_folder_path)):
        first_gt_file_path = os.path.join(first_gt_folder_path, first_gt_file)
        if first_gt_file.endswith('_gtFine_labelIds.png'):
            gt_data.append(first_gt_file_path)

image_dir = r'/home/seg_DenseASPP/CityScapesDataset/leftImg8bit_trainvaltest/leftImg8bit/val'
image_data = []

for first_image_folder in sorted(os.listdir(image_dir)):
    first_image_folder_path = os.path.join(image_dir, first_image_folder)
    for first_image_file in sorted(os.listdir(first_image_folder_path)):
        first_image_file_path = os.path.join(first_image_folder_path, first_image_file)
        image_data.append(first_image_file_path)

print(f"GT 파일 개수: {len(gt_data)}")
print(f"이미지 파일 개수: {len(image_data)}")

gt_filenames = sorted([os.path.basename(f).split('_gtFine_labelIds.png') for f in gt_data])
image_filenames = sorted([os.path.basename(f).split('_leftImg8bit.png') for f in image_data])

# gt_filenames = sorted([os.path.basename(f).removesuffix('_gtFine_labelIds.png') for f in gt_data])
# image_filenames = sorted([os.path.basename(f).removesuffix('_leftImg8bit.png') for f in image_data])

non_same_gt = []
non_same_image = []

same_gt = []
same_image = []

if gt_filenames == image_filenames:
    print("파일 이름이 모두 일치합니다!")
else:
    print("파일 이름이 일치하지 않습니다.")

# 서로 다른 파일 이름 저장
for gt_name, img_name in zip(gt_filenames, image_filenames):
    if gt_name != img_name:
        # print(f"불일치: GT='{gt_name}' vs Image='{img_name}'")
        non_same_gt.append(gt_name)
        non_same_image.append(img_name)
    else:
        # print(f"일치: GT='{gt_name}' vs Image='{img_name}'")
        same_gt.append(gt_name)
        same_image.append(img_name)

print(f"불일치한 GT 파일 개수: {len(non_same_gt)}")
print(f"불일치한 이미지 파일 개수: {len(non_same_image)}")
print(f"일치한 GT 파일 개수: {len(same_gt)}")
print(f"일치한 이미지 파일 개수: {len(same_image)}")