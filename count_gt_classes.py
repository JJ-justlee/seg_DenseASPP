import os
from PIL import Image
import numpy as np

mapped_dir = r'/home/addinedu/Documents/seg_DenseASPP_CityScapes/CityScapesDataset/gtFine_trainvaltest/gtFine/save_train_mapped'

all_unique_values = set()

for city in os.listdir(mapped_dir):
    city_path = os.path.join(mapped_dir, city)
    for filename in os.listdir(city_path):
        if filename.endswith('labelTrainIds.png'):
            file_path = os.path.join(city_path, filename)
            gt_image = np.array(Image.open(file_path))
            all_unique_values.update(np.unique(gt_image))

print(f"Total classes: {len(all_unique_values)}")
print(f"\nUnique class IDs: {sorted([int(x) for x in all_unique_values])}")