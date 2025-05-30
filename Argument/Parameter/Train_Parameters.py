import argparse

def Train_Parameters_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation parameters for Train')

    parser.add_argument('--input_height', type=int, help='model input height size ', default=1024)
    parser.add_argument('--input_width', type=int, help='model input width size ', default=2048)
    parser.add_argument('--batch_size', type=int, help='input batch size for training ', default=2)
    # parser.add_argument('--learning_rate', type=int, help='learning rate ', default=1e-3) #0.001
    parser.add_argument('--learning_rate', type=int, help='learning rate ', default=3e-4) #0.0003
    # parser.add_argument('--num_epochs', type=int, help='epoch number for training', default=100)
    parser.add_argument('--weight_decay', type=float, help='Weight decay (L2 regularization) to prevent overfitting', default=0.00001)
    parser.add_argument('--num_epochs', type=int, help='epoch number for training', default=80)
    parser.add_argument('--gpu', type=int, help='GPU id to use', default=0)

    return parser.parse_args()