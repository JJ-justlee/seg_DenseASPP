import os
from PIL import Image
import numpy as np

# 매핑 정의
ignore_label = 255
id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                 3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                 7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                 14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                 26: 13, 27: 14, 28: 15, 29: ignore_label, 30: ignore_label,
                 31: 16, 32: 17, 33: 18}

# 폴더 경로 설정
base_path = '/home/addinedu/Documents/GitHub/seg_DenseASPP/CityScapesDataset/gtFine_trainvaltest/gtFine'
splits = ['train', 'val', 'test']  # 세 가지 데이터셋 분할

for split in splits:
  output_split_dir = os.path.join(base_path, f'save_{split}_mapped')
  os.makedirs(output_split_dir, exist_ok=True)

for split in splits:
    split_path = os.path.join(base_path, split)
    output_split_path = os.path.join(base_path, f'save_{split}_mapped')
    for city in os.listdir(split_path):
        city_path = os.path.join(split_path, city)
        output_city_path = os.path.join(output_split_path, city)
        os.makedirs(output_city_path, exist_ok=True)
        for filename in os.listdir(city_path):
            if filename.endswith('labelIds.png'):
                # 파일 불러오기
                label_path = os.path.join(city_path, filename)
                label_img = np.array(Image.open(label_path))

                # 매핑 적용
                mapped = np.full_like(label_img, ignore_label)
                for k, v in id_to_trainid.items():
                    mapped[label_img == k] = v

                save_name = filename.replace('labelIds.png', f'label{split}Ids.png')
                save_path = os.path.join(output_city_path, save_name)

                # 이미지로 저장 (uint8로 변환)
                Image.fromarray(mapped.astype(np.uint8)).save(save_path)