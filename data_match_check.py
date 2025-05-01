import os
import numpy as np
from PIL import Image

import tempfile
import shutil

# gt_dir = r'/home/addinedu/Documents/GitHub/seg_DenseASPP/CityScapesDataset/gtFine_trainvaltest/gtFine/save_train_mapped'
# gt_data = []

# for first_gt_folder in sorted(os.listdir(gt_dir)):
#     first_gt_folder_path = os.path.join(gt_dir, first_gt_folder)
#     for second_gt_folder in sorted(os.listdir(first_gt_folder_path)):
#         second_gt_folder_path = os.path.join(first_gt_folder_path, second_gt_folder)
#         gt_data.append(second_gt_folder_path)

# 매핑 정의
ignore_label = 255
id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                26: 13, 27: 14, 28: 15, 29: ignore_label, 30: ignore_label,
                31: 16, 32: 17, 33: 18}

gt_train_path = r'/home/addinedu/Documents/GitHub/seg_DenseASPP/CityScapesDataset/gtFine_trainvaltest/gtFine/train'
mask_filenames = []

for firstGtfolder in sorted(os.listdir(gt_train_path)):
    firstGtfolder_path = os.path.join(gt_train_path, firstGtfolder)
    if os.path.isdir(firstGtfolder_path):
        for secondGtFile in sorted(os.listdir(firstGtfolder_path)):
            if secondGtFile.endswith('labelIds.png'):
                secondGtFolder_path = os.path.join(firstGtfolder_path, secondGtFile)
                gt_name = os.path.splitext(os.path.basename(secondGtFolder_path))[0]
                
                # 파일 불러오기
                label_path = os.path.join(firstGtfolder_path, secondGtFile)
                label_img = np.array(Image.open(label_path))

                # 매핑 적용
                mapped = np.full_like(label_img, ignore_label)
                for k, v in id_to_trainid.items():
                    mapped[label_img == k] = v

                # save_name = filename.replace('labelIds.png', f'label{split}Ids.png')
                # save_path = os.path.join(output_city_path, save_name)

                # 이미지로 저장 (uint8로 변환)
                gt_mapped = Image.fromarray(mapped.astype(np.uint8))

                # # self.mask_filenames.append((secondGtfile, gt_mapped))
                # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                #     gt_mapped.save(tmp.name)
                #     temp_path = tmp.name
                
                # gt_name = gt_name
                # gt_name_path = os.path.join(firstGtfolder_path, gt_name)
                # shutil.move(temp_path, gt_name_path)
                # mask_filenames.append(gt_name_path)

image_dir = r'/home/addinedu/Documents/GitHub/seg_DenseASPP/CityScapesDataset/leftImg8bit_trainvaltest/leftImg8bit/train'
image_data = []

for first_image_folder in sorted(os.listdir(image_dir)):
    first_image_folder_path = os.path.join(image_dir, first_image_folder)
    for second_image_folder in sorted(os.listdir(first_image_folder_path)):
        second_image_folder_path = os.path.join(first_image_folder_path, second_image_folder)
        image_data.append(second_image_folder)

print(f"GT 파일 개수: {len(mask_filenames)}")
print(f"이미지 파일 개수: {len(image_data)}")

def clean_image_filename(filename):
    base = os.path.splitext(filename)[0]
    if base.endswith('_leftImg8bit'):
        # base = base.replace('_leftImg8bit', '')
        pass
    return base

def clean_gt_filename(filename):
    base = os.path.splitext(filename)[0]
    if base.endswith('_gtFine_labelIds'):
        # base = base.replace('_gtFine_labelIds', '')
        pass
    return base

gt_filenames = [clean_gt_filename(os.path.basename(path)) for path in mask_filenames]
image_filenames = [clean_image_filename(os.path.basename(path)) for path in image_data]

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
            same_gt.append(gt_name)
            same_image.append(img_name)
            print(f"일치: GT='{gt_name}' vs Image='{img_name}'")

print(f"불일치한 GT 파일 개수: {len(non_same_gt)}")
print(f"불일치한 이미지 파일 개수: {len(non_same_image)}")
print(f"일치한 GT 파일 개수: {len(same_gt)}")
print(f"일치한 이미지 파일 개수: {len(same_image)}")