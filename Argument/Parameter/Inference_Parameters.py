import argparse

def Inference_Parameter_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation parameters for Inference')

    parser.add_argument('--ckpt_name', type=str, help='saved weight file name', default='0099.pth')
    parser.add_argument('--raw_height', type=int, help='image raw height size', default=1080)
    parser.add_argument('--raw_width', type=int, help='image raw width size', default=1920)
    parser.add_argument('--input_height', type=int, help='model input height size ', default=256)
    parser.add_argument('--input_width', type=int, help='model input width size ', default=256)
    parser.add_argument('--batch_size', type=int, help='input batch size for training ', default=8)
    parser.add_argument('--gpu', type=int, help='GPU id to use', default=0)
    
    return parser.parse_args()