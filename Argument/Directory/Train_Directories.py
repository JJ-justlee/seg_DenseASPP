import argparse
import os.path as osp

def Train_Dics_args():
    #put a base path below
    base_dir = '/home/seg_DenseASPP'

    parser = argparse.ArgumentParser(description='Semantic Segmentation Directories for Train')

    parser.add_argument('--mode', type=str, help='train or test', default='test')

    # Directory
    parser.add_argument('--model_name', type=str,   help='Customized DenseASPP', default='conventional_denseASPP_lr')
    parser.add_argument('--root_dir', type=str, help='dataset directory',
                        default=osp.join(base_dir, 'CityScapesDataset'))
    parser.add_argument('--exp_dir', type=str, help='result save directory',
                        default=osp.join(base_dir, 'experiments'))
    parser.add_argument('--model_dir', type=str, help='model directory for saving',
                        default=osp.join(base_dir, 'experiments', 'conventional_denseASPP_lr', 'models'))
    parser.add_argument('--higher_model_dir', type=str, help='model directory for saving',
                        default=osp.join(base_dir, 'experiments', 'conventional_denseASPP_lr', 'models_higher_mIoU'))
    parser.add_argument('--bestModel_dir', type=str, help='model directory for saving',
                        default=osp.join(base_dir, 'experiments', 'conventional_denseASPP_lr', 'models_bestModel'))
    parser.add_argument('--log_dir', type=str, help='log directory for tensorboard',
                        default=osp.join(base_dir, 'experiments', 'conventional_denseASPP_lr', 'logs'))
    parser.add_argument('--pred_dir', type=str, help='prediction image directory',
                        default=osp.join(base_dir, 'experiments', 'conventional_denseASPP_lr', 'prediction'))
    return parser.parse_args()