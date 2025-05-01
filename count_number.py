import os

gt_path = r'/home/addinedu/Documents/GitHub/seg_DenseASPP/CityScapesDataset/gtFine_trainvaltest/gtFine/save_train_mapped'
image_path = r'/home/addinedu/Documents/GitHub/seg_DenseASPP/CityScapesDataset/leftImg8bit_trainvaltest/leftImg8bit/train'

list_gt = []
list_image = []

for gt_firstfile in os.listdir(gt_path):
    gt_firstfile_path = os.path.join(gt_path, gt_firstfile)
    for gt_secondfile in os.listdir(gt_firstfile_path):
        if gt_secondfile.endswith('png'):
            list_gt.append(os.path.join(gt_firstfile, gt_secondfile))

print(len(list_gt))

for image_firstfile in os.listdir(image_path):
    image_firstfile_path = os.path.join(image_path, image_firstfile)
    for image_secondfile in os.listdir(image_firstfile_path):
        if image_secondfile.endswith('png'):
            list_image.append(os.path.join(image_firstfile_path, image_secondfile)) 

print(len(list_image))