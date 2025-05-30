import argparse
import os.path as osp

def Inference_Dics_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Dictionaries for Inference')

    base_dir = '/home/seg_DenseASPP'
    # Directory
    parser.add_argument('--root_dir', type=str, help='dataset directory',
                        default=osp.join(base_dir, 'CityScapesDataset'))
    
    parser.add_argument('--exp_dir', type=str, help='result save directory',
                        default=osp.join(base_dir, 'experiments'))
    parser.add_argument('--model_name', type=str,   help='Customized DenseASPP', default='DenseASPP_pretrain')    

    args = parser.parse_args()
    
    parser.add_argument('--bestModel_dir', type=str, help='model directory for saving',
                        default=osp.join(args.exp_dir, args.model_name, 'bestModel'))
    parser.add_argument('--pred_dir', type=str, help='prediction image directory',
                        default=osp.join(args.exp_dir, args.model_name, 'prediction'))
    
    return parser.parse_args()