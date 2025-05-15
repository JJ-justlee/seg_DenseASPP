import os
import os.path as osp
import argparse
import torch
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DenseASPP import DenseASPP
from datetime import timedelta
from CityScapesSeg_dataloader import CityScapesSeg_dataset
from utility.colorize import colorize_mask
import numpy as np
from PIL import Image

from Argument.Parameter.Train_Parameters import Train_Parameters_args

from Argument.Directory.Train_Directories import Train_Dics_args

from schedule_learning_rate import schedule_learning_rate

arg_Dic = Train_Dics_args()

arg_parameter = Train_Parameters_args()

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
    if not osp.exists(osp.join(arg_Dic.exp_dir, arg_Dic.model_name)):
        os.makedirs(osp.join(arg_Dic.exp_dir, arg_Dic.model_name))

    if not osp.exists(arg_Dic.model_dir):
        os.makedirs(arg_Dic.model_dir)
        
    if not osp.exists(arg_Dic.log_dir):
        os.makedirs(arg_Dic.log_dir)
        
    if not osp.exists(arg_Dic.higher_model_dir):
        os.makedirs(arg_Dic.higher_model_dir)
        
    if not osp.exists(arg_Dic.bestModel_dir):
        os.makedirs(arg_Dic.bestModel_dir)

    # GPU 셋팅
    if torch.cuda.is_available():
        if arg_parameter.gpu is not None:
            device = torch.device(f'cuda:{arg_parameter.gpu}')
            print("Use GPU: {} for training".format(arg_parameter.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')

    # 텐서보드 셋팅
    writer = SummaryWriter(arg_Dic.log_dir)

    CityScapes_train_dataset = CityScapesSeg_dataset(root_dir=arg_Dic.root_dir,
                                    input_height=arg_parameter.input_height, input_width=arg_parameter.input_width, dataset = 'train')
    
    train_dataloader = DataLoader(CityScapes_train_dataset,
                            batch_size=arg_parameter.batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)

    CityScapes_val_dataset = CityScapesSeg_dataset(root_dir=arg_Dic.root_dir,
                                    input_height=arg_parameter.input_height, input_width=arg_parameter.input_width, dataset = 'val')

    val_dataloader = DataLoader(CityScapes_val_dataset,
                        batch_size=arg_parameter.batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

    # 뉴럴네트워크 로드
    #model = models.segmentation.fcn_resnet50(weights_backbone=True, num_classes=1)
    model = DenseASPP(model_cfg, n_class=19, output_stride=8)
    model = model.cuda()

    # 최적화
    optimizer = torch.optim.Adam(model.parameters(), lr=arg_parameter.learning_rate, weight_decay=arg_parameter.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
    
    n_class = 19

    # 학습
    for epoch in range(arg_parameter.num_epochs): #100개의 epochs 모든 input date들이 모델에 학습이 되고 output으로 나오는 시각
        #시간을 체크해주는 코드
        start = time.time()
        
        iou_per_class = [0.0 for _ in range(n_class)]
        total_per_class = [0 for _ in range(n_class)]

        schedule_learning_rate(epoch, optimizer)

        for step, batch_image in enumerate(train_dataloader):
            sample_image = batch_image['image']
            sample_gt = batch_image['gt']
            sample_vis_gt = batch_image['vis_gt']
            
            if step > 1: break
            
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
            
            output = torch.softmax(output, dim=1) #(8, 19, 256, 256)
            predicted_class = torch.argmax(output, dim=1)  # 클래스 인덱스 추출 (B, H, W)

        for batch_image in val_dataloader:
            sample_val_image = batch_image['image'].to(device)
            sample_val_gt = batch_image['gt'].to(device)

            with torch.no_grad():
                val_logit = model(sample_val_image)
                val_prediction = torch.softmax(val_logit, dim=1)
                val_predicted_class = torch.argmax(val_prediction, dim=1)

            for cls in range(n_class):
                pred_cls = (val_predicted_class == cls)
                gt_cls = (sample_val_gt == cls)

                intersection = (pred_cls & gt_cls).sum().float()
                union = (pred_cls | gt_cls).sum().float()

                if union > 0:
                    iou = intersection / union
                    #클래스에 IoU값을 더함
                    iou_per_class[cls] += iou
                    #클래스가 등장한 횟수 더함
                    total_per_class[cls] += 1
        
        mIoU_list = []
        final_iou_list = []
        for cls in range(n_class):
            if total_per_class[cls] > 0:
                final_iou_list.append(iou_per_class[cls] / total_per_class[cls])

        mIoU = (sum(final_iou_list) / len(final_iou_list)) * 100
        mIoU_list.append(mIoU)


        #한 epoch이 끝나고 나면 시간 출력 
        t_elapsed = timedelta(seconds=time.time() - start)
        training_time_left = ((arg_parameter.num_epochs - (epoch + 1)) * t_elapsed.total_seconds()) / (3600)
        print(f"Name: {arg_Dic.model_name} | Epoch: {'[':>4}{epoch + 1:>4}/{arg_parameter.num_epochs}] | time left: {training_time_left:.2f} hours | loss: {loss:.4f} | mIoU: {int(mIoU):d}")
        torch.save(model.state_dict(), osp.join(arg_Dic.model_dir, f"{epoch + 1:04d}_{int(mIoU):d}.pth"))

        if len(mIoU_list) >= 2 and mIoU_list[-1] > max(mIoU_list[:-1]):
            betterIoU = mIoU_list[-1]
            better_IoU = mIoU_list.index(betterIoU)
            torch.save(model.state_dict(), osp.join(arg_Dic.higher_model_dir, f"{better_IoU + 1:04d}_{int(betterIoU):d}.pth"))
        else:
            torch.save(model.state_dict(), osp.join(arg_Dic.higher_model_dir, f"{epoch + 1:04d}_{int(mIoU):d}.pth"))

        #일 단위
        # training_time_left = ((total_epochs - (epoch + 1)) * t_elapsed.total_seconds()) / (3600*24)
        # print(f"Training time left: {training_time_left:.2f} days")
        
        # 텐서보드
        idx_random = torch.randint(0, sample_image.size(0), (1,)).item()
        # print(idx_random)
        # print(sample_image.shape)
        writer.add_image('Input/Image', sample_image[idx_random], global_step=epoch)
        
        # sample_vis_gt = sample_vis_gt[0] #배치 제거
        # print(sample_vis_gt.shape)
        sample_vis_gt = sample_vis_gt.permute(0, 3, 1, 2) #(B, H, W, C) > (B, C, H, W)
        # print(sample_vis_gt.shape)
        writer.add_image('Input/Gt', sample_vis_gt[idx_random], global_step=epoch)

        # 예측 클래스에서 하나 선택
        predicted_class = predicted_class[idx_random] # shape: (H, W)
        print(type(predicted_class))
        predicted_class = predicted_class.detach().cpu().numpy()
        predicted_class = colorize_mask(predicted_class)  # shape: (H, W, 3), dtype: uint8
        predicted_class = torch.tensor(predicted_class).permute(2, 0, 1)  # (3, H, W)
        writer.add_image('Results/trained_result', predicted_class, global_step=epoch)

        writer.add_scalar('Results/Loss', loss, global_step=epoch)
        writer.add_scalar('Results/Accuracy', mIoU, global_step=epoch)
    
    return model, mIoU_list

def save_best_model(model, mIoU_list):
    if len(mIoU_list) == arg_parameter.num_epochs:
        max_IoU = max(mIoU_list)
        best_epoch = mIoU_list.index(max_IoU)
        torch.save(model.state_dict(), osp.join(arg_Dic.bestModel_dir, f"{best_epoch}_{int(max_IoU):d}.pth"))

if __name__ == "__main__":
    model, mIoU_list = main()
    save_best_model(model, mIoU_list)