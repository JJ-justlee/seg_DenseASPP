import os
import os.path as osp
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import time
import utility
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DenseASPP import DenseASPP
from datetime import timedelta
from CityScapesSeg_dataloader import CityScapesSeg_dataset
from utility import colorize

#put a base path below
base_dir = '/home/dev/DATASET'
parser = argparse.ArgumentParser(description='Simple Semantic Segmentation Train')

parser.add_argument('--mode', type=str, help='train or test', default='test')

# Directory
parser.add_argument('--model_name', type=str,   help='Customized DenseASPP', default='conventional_denseASPP')
parser.add_argument('--root_dir', type=str, help='dataset directory',
                    default=osp.join(base_dir, 'CityScapesDataset'))
parser.add_argument('--exp_dir', type=str, help='result save directory',
                    default=osp.join(base_dir, 'experiments'))
parser.add_argument('--model_dir', type=str, help='model directory for saving',
                    default=osp.join(base_dir, 'experiments', 'conventional_denseASPP', 'models1'))
parser.add_argument('--higher_model_dir', type=str, help='model directory for saving',
                    default=osp.join(base_dir, 'experiments', 'conventional_denseASPP', 'models_higher_mIoU'))
parser.add_argument('--bestModel_dir', type=str, help='model directory for saving',
                    default=osp.join(base_dir, 'experiments', 'conventional_denseASPP', 'models_bestModel'))
parser.add_argument('--log_dir', type=str, help='log directory for tensorboard',
                    default=osp.join(base_dir, 'experiments', 'conventional_denseASPP', 'logs'))
parser.add_argument('--pred_dir', type=str, help='prediction image directory',
                    default=osp.join(base_dir, 'experiments', 'conventional_denseASPP', 'prediction'))


# Inference
parser.add_argument('--ckpt_name', type=str, help='saved weight file name', default='0099.pth')

# Parameter
parser.add_argument('--input_height', type=int, help='model input height size ', default=256)
parser.add_argument('--input_width', type=int, help='model input width size ', default=256)
parser.add_argument('--batch_size', type=int, help='input batch size for training ', default=8)
parser.add_argument('--learning_rate', type=int, help='learning rate ', default=1e-3)
parser.add_argument('--num_epochs', type=int, help='epoch number for training', default=100)
parser.add_argument('--gpu', type=int, help='GPU id to use', default=0)

#args = parser.parse_args()
args, unknown = parser.parse_known_args()

model_cfg = {
    'bn_size': 4,
    'drop_rate': 0,
    'growth_rate': 32,
    'num_init_features': 64,
    'block_config': (6, 12, 24, 16),

    'dropout0': 0.1,
    'dropout1': 0.1,
    'd_feature0': 128,
    'd_feature1': 64,

    'pretrained_path': "./pretrained/densenet121.pth"
    }

def main():
    # 경로 셋팅
    if not osp.exists(osp.join(args.exp_dir, args.model_name)):
        os.makedirs(osp.join(args.exp_dir, args.model_name))

    if not osp.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    if not osp.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    if not osp.exists(args.higher_model_dir):
        os.makedirs(args.higher_model_dir)
        
    if not osp.exists(args.bestModel_dir):
        os.makedirs(args.bestModel_dir)

    # GPU 셋팅
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
            print("Use GPU: {} for training".format(args.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')

    # 텐서보드 셋팅
    writer = SummaryWriter(args.log_dir)

    CityScapes_dataset = CityScapesSeg_dataset(root_dir=args.root_dir,
                                    input_height=args.input_height, input_width=args.input_width)

    #root_dir(json_path와 image_dir변수에 들어있는)는 위 coco_dataset변수에 선언된 값, 즉 root_dir에 담긴 args.root_dir로 경로가 선언된다
    dataloader = DataLoader(CityScapes_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True)

    # 뉴럴네트워크 로드
    #model = models.segmentation.fcn_resnet50(weights_backbone=True, num_classes=1)
    model = DenseASPP(model_cfg, n_class=19, output_stride=8)
    model = model.cuda()

    # 최적화
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()

    save_IoU = []

    # 학습
    for epoch in range(args.num_epochs): #100개의 epochs 모든 input date들이 모델에 학습이 되고 output으로 나오는 시각
        #시간을 체크해주는 코드 
        start = time.time()
        
        epoch_IoU = []
        
        for step, (batch_image, _) in enumerate(dataloader):
            sample_image = batch_image['image']
            sample_gt = batch_image['gt']
            sample_vis_gt = batch_image['vis_gt']
            
            #리스트, 튜플의 인덱스와 원소를 함께 출력하기 위해 enumerate()를 사용
            #unpacking을 통해 따로 출력 step, (sample_image, sample_gt) step와 (sample_image, sample_gt)을 따로 둠 > unpacking
            optimizer.zero_grad()

            #device=device는 텐서가 CPU/GPU에 맞게 자동으로 할당됨
            #(,,C)는 4개의 차원인데 이걸 rank가 4라고 표현을 함 > tensor
            # sample_image = torch.tensor(sample_image, device=device, dtype=torch.float32)
            # sample_gt = torch.tensor(sample_gt, device=device, dtype=torch.float32)
            sample_image = sample_image.to(device)
            sample_gt = sample_gt.to(device)

            sample_gt = sample_gt.long()
            sample_gt = sample_gt.squeeze()
            output = model(sample_image) #  (B, 19, H, W) -> (B, H, W)
            # 0번째 인덱스: Road
            # 1번째 인덱스: Person
            loss = criterion(output, sample_gt)
            loss.backward()

            optimizer.step()
            
            output = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1)  # 클래스 인덱스 추출 (B, H, W)

            intersection = (predicted_class == sample_gt).sum()  # 정답과 예측 비교
            union = ((predicted_class > 0) | (sample_gt > 0)).sum()  # 0보다 큰 값 비교
            iou = intersection / union if union > 0 else torch.tensor(0.0)
            epoch_IoU.append(iou.item())  # Store IoU for the batch

        for _, val_batch_image in dataloader:
            sample_val_image = val_batch_image['val_image']
            sample_val_gt = val_batch_image['val_gt']

            with torch.no_grad():
                val_logit = model(sample_val_image)
                val_prediction = torch.softmax(val_logit, dim=1)
                predicted_class = torch.argmax(val_prediction, dim=1)
                
                
        # Calculate average IoU for the epoch
        avg_IoU = sum(epoch_IoU) / len(epoch_IoU)
        save_IoU.append(avg_IoU)

        #한 epoch이 끝나고 나면 시간 출력 
        t_elapsed = timedelta(seconds=time.time() - start)
        training_time_left = ((args.num_epochs - (epoch + 1)) * t_elapsed.total_seconds()) / (3600)
        print(f"Name: {args.model_name} | Epoch: {'[':>4}{epoch + 1:>4}/{args.num_epochs}] | time left: {training_time_left:.2f} hours | loss: {loss:.4f} | train mIoU: {iou:.4f}")
        torch.save(model.state_dict(), osp.join(args.model_dir, f"{epoch:04d}.pth"))

        if len(save_IoU) >= 2:
            if save_IoU[-1] > max(save_IoU[:-1]):
                torch.save(model.state_dict(), osp.join(args.higher_model_dir, f"{epoch:04d}.pth"))

        #일 단위
        # training_time_left = ((total_epochs - (epoch + 1)) * t_elapsed.total_seconds()) / (3600*24)
        # print(f"Training time left: {training_time_left:.2f} days")
        
        # 텐서보드
        idx_random = torch.randint(0, sample_image.size(0), (1,)).item()

        writer.add_image('Input/Image', sample_image[idx_random], global_step=epoch)
        writer.add_image('Input/Gt', sample_vis_gt[idx_random], global_step=epoch)

        # print(predicted_class.shape)
        predicted_class = torch.unsqueeze(predicted_class, dim=1)
        writer.add_image('Results/Prediction', predicted_class[idx_random], global_step=epoch)

        writer.add_scalar('Results/Loss', loss, global_step=epoch)
        writer.add_scalar('Results/Accuracy', iou, global_step=epoch)
        return model, save_IoU, epoch

def save_best_model(model, save_IoU, epoch):
    if max(save_IoU):
        best_epoch = save_IoU.index(max(save_IoU))
        torch.save(model.state_dict(), osp.join(args.bestModel_dir, f"{epoch:04d}.pth"))

if __name__ == "__main__":
    model, save_IoU, epoch = main()
    save_best_model(model, save_IoU, epoch)
