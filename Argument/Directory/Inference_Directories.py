import argparse
import os.path as osp

def Inference_Dics_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Dictionaries for Inference')

    base_dir = '/home/addinedu/Documents/GitHub/seg_DenseASPP'
    # Directory
    parser.add_argument('--root_dir', type=str, help='dataset directory',
                        default=osp.join(base_dir, 'CityScapesDataset'))
    parser.add_argument('--save_dir', type=str, help='model directory for saveing',
                        default=osp.join(base_dir, 'experiments', 'DenseAsppWithCitySacapes', 'models', 'secondTraining'))
    parser.add_argument('--pred_dir', type=str, help='prediction image directory',
                        default=osp.join(base_dir, 'experiments', 'DenseAsppWithCitySacapes', 'prediction', 'DenseASPP_aug3'))
    
    return parser.parse_args()