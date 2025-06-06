import argparse
import os.path as osp

def Train_Dics_args():
    #put a base path below
    base_dir = '/home/seg_DenseASPP'

    parser = argparse.ArgumentParser(description='Semantic Segmentation Directories for Train')

    parser.add_argument('--mode', type=str, help='train mode', default='train')
    
    # Directory
    parser.add_argument('--root_dir', type=str, help='dataset directory',
                        default=osp.join(base_dir, 'CityScapesDataset'))

    parser.add_argument('--exp_dir', type=str, help='result save directory',
                        default=osp.join(base_dir, 'experiments'))
    parser.add_argument('--model_name', type=str,   help='Customized DenseASPP', default='DenseASPP_changed_brightness_aug_seq_add_resize')    

    args = parser.parse_args()

    parser.add_argument('--model_dir', type=str, help='model directory for saving',
                        default=osp.join(args.exp_dir, args.model_name, 'models'))
    parser.add_argument('--train_higher_model_dir', type=str, help='model directory for saving',
                        default=osp.join(args.exp_dir, args.model_name, 'train_higherModel'))
    parser.add_argument('--train_bestModel_dir', type=str, help='model directory for saving',
                        default=osp.join(args.exp_dir, args.model_name, 'train_bestModel'))
    parser.add_argument('--val_higher_model_dir', type=str, help='model directory for saving',
                        default=osp.join(args.exp_dir, args.model_name, 'val_higherModel'))
    parser.add_argument('--val_bestModel_dir', type=str, help='model directory for saving',
                        default=osp.join(args.exp_dir, args.model_name, 'val_bestModel'))
    parser.add_argument('--log_dir', type=str, help='log directory for tensorboard',
                        default=osp.join(args.exp_dir, args.model_name, 'log'))
    
    return parser.parse_args()