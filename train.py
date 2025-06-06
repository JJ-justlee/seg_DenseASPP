import os
import os.path as osp
import torch
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import timedelta
import multiprocessing
from ptflops import get_model_complexity_info #flops 측정
import importlib
import numpy as np
import random

from CityScapesSeg_dataloader import CityScapesSeg_dataset
from utility.colorize import colorize_mask

import Architectures.DenseASPP as DenseASPP
import Architectures.DenseASPP_modified as DenseASPP_modified
import Architectures.MobileNetDenseASPP as MobileNetDenseASPP

importlib.reload(DenseASPP)
importlib.reload(DenseASPP_modified)
importlib.reload(MobileNetDenseASPP)

DenseASPP = DenseASPP.DenseASPP
DenseASPP_modified = DenseASPP_modified.DenseASPP_modified
MobileNetDenseASPP = MobileNetDenseASPP.MobileNetDenseASPP

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

from Argument.Parameter.Train_Parameters import Train_Parameters_args

from Argument.Directory.Train_Directories import Train_Dics_args

from schedule_learning_rate import schedule_learning_rate

from cfg.DenseNet121 import Model_CFG as model_cfg
from loss_function.Focal_loss import Focal_loss


arg_Dic = Train_Dics_args()

arg_parameter = Train_Parameters_args()
  
  
def check_FLOPs_and_Parameters(model):
    model.eval()
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, verbose=False, print_per_layer_stat=False)
        
        print(f'FLOPs: {macs}')
        print(f'Parameters: {params}')
    
    save_path = "/home/seg_DenseASPP/Params_and_FLOPs/DenseASPP_pretrain.txt"
        
    if os.path.exists(save_path):
        print(f'FLOPs and prarmeter file already exist at {save_path}')
        pass
    else:
        with open(save_path, "w") as f:
            f.write(f"Model: DenseASPP_pretrain\n")
            f.write(f"Input Size: (3, 512, 512)\n")
            f.write(f"FLOPs: {macs}\n")
            f.write(f"Params: {params}\n")

        print(f"FLOPs and Params saved to {save_path}")

def load_partial_pretrained_weights(model, pretrained_path, show_missed=False):
    print(f"Loading partial pretrained weights from: {pretrained_path}")
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()

    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and v.shape == model_dict[k].shape}

    print(f'mapped {len(filtered_dict)}/{len(model_dict)} layers')
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    # 옵션: 매핑 안 된 첫 20개 키 출력
    if show_missed:
        missed = [k for k in pretrained_dict.keys() if k not in filtered_dict]
        print("매칭 실패 키:", missed[:40])
    return model

def denorm(x):                       # x : (C,H,W) tensor on GPU/CPU
    mean = IMAGENET_MEAN.to(x.device)
    std  = IMAGENET_STD .to(x.device)
    return torch.clamp(x * std + mean, 0, 1)

def set_seed(seed: int = 42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # Numpy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # GPU seed (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU seed

    # CuDNN 설정 (가능한 한 deterministic하게)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def main():
    set_seed()
    
    # 경로 셋팅
    if not osp.exists(osp.join(arg_Dic.exp_dir, arg_Dic.model_name)):
        os.makedirs(osp.join(arg_Dic.exp_dir, arg_Dic.model_name))

    if not osp.exists(arg_Dic.model_dir):
        os.makedirs(arg_Dic.model_dir)
        
    if not osp.exists(arg_Dic.log_dir):
        os.makedirs(arg_Dic.log_dir)
        
    if not osp.exists(arg_Dic.train_higher_model_dir):
        os.makedirs(arg_Dic.train_higher_model_dir)

    if not osp.exists(arg_Dic.val_higher_model_dir):
        os.makedirs(arg_Dic.val_higher_model_dir)
        
    if not osp.exists(arg_Dic.train_bestModel_dir):
        os.makedirs(arg_Dic.train_bestModel_dir)

    if not osp.exists(arg_Dic.val_bestModel_dir):
        os.makedirs(arg_Dic.val_bestModel_dir)

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
                            num_workers=multiprocessing.cpu_count() // 2,
                            pin_memory=True,
                            persistent_workers=True)

    CityScapes_val_dataset = CityScapesSeg_dataset(root_dir=arg_Dic.root_dir,
                                    input_height=arg_parameter.input_height, input_width=arg_parameter.input_width, dataset = 'val')

    val_dataloader = DataLoader(CityScapes_val_dataset,
                        batch_size=arg_parameter.batch_size,
                        shuffle=False,
                        num_workers=multiprocessing.cpu_count() // 2,
                        pin_memory=True,
                        persistent_workers=True)

    # 뉴럴네트워크 로드
    #model = models.segmentation.fcn_resnet50(weights_backbone=True, num_classes=1)
    model = DenseASPP(model_cfg, n_class=19, output_stride=8)
    model = load_partial_pretrained_weights(model, pretrained_path=model_cfg['pretrained_path'], show_missed=False)
    model = model.cuda()
    check_FLOPs_and_Parameters(model)

    # 최적화
    optimizer = torch.optim.Adam(model.parameters(), lr=arg_parameter.learning_rate, weight_decay=arg_parameter.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
    
    n_class = 19
    train_mIoU_list = []
    val_mIoU_list = []

    # 학습
    for epoch in range(arg_parameter.num_epochs): #100개의 epochs 모든 input date들이 모델에 학습이 되고 output으로 나오는 시각
        #시간을 체크해주는 코드
        start = time.time()
        model.train()

        lr = schedule_learning_rate(epoch, optimizer)
        if epoch in [0, 10, 40, 79]:
            print(f"epoch {epoch:02d}  lr = {optimizer.param_groups[0]['lr']:.6f}")

        total_loss = 0

        train_inter_total = torch.zeros(n_class, dtype=torch.float64, device=device)
        train_union_total = torch.zeros(n_class, dtype=torch.float64, device=device)

        #리스트, 튜플의 인덱스와 원소를 함께 출력하기 위해 enumerate()를 사용
        #unpacking을 통해 따로 출력 step, (sample_image, sample_gt) step와 (sample_image, sample_gt)을 따로 둠 > unpacking
        for step, batch_image in enumerate(train_dataloader):
            # print(f"Epoch: {epoch:>3}/{arg_parameter.num_epochs} | step: {step:>3}/{len(train_dataloader)} | lr: {optimizer.param_groups[0]['lr']:.6f}")
            sample_image = batch_image['image'].to(device)
            sample_gt = batch_image['gt'].to(device).squeeze(1).long()
            # sample_vis_gt = batch_image['vis_gt']
            
            # if step > 1: break
            
            optimizer.zero_grad()

            output_logit = model(sample_image) #  (B, 19, H, W) -> (B, H, W)

            loss = criterion(output_logit, sample_gt)
            # loss = Focal_loss(output, sample_gt, gamma= 2, alpha= 0.25)
            
            total_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
        
            # train mIoU
            with torch.no_grad():
                predicted_class = torch.argmax(output_logit, dim=1)
                train_valid_pixel = (sample_gt != 255)
            
                for cls in range(n_class):
                    Intersection = ((predicted_class == cls) & (sample_gt == cls) & train_valid_pixel).sum()
                    Union = (((predicted_class == cls) | (sample_gt == cls)) & train_valid_pixel).sum()
                    
                    train_inter_total[cls] += Intersection
                    train_union_total[cls] += Union

        epoch_ave_loss = total_loss / len(train_dataloader)

        train_iou_per_class = train_inter_total / torch.clamp(train_union_total, min=1)    # 0으로 나눔 방지
        train_valid_cls     = train_union_total > 0                                  # 데이터셋에 존재하는 클래스
        train_mIoU          = train_iou_per_class[train_valid_cls].mean().item() * 100
        train_mIoU_list.append(train_mIoU)
                
        # val mIoU
        val_inter_total = torch.zeros(n_class, dtype=torch.float64, device=device)   # 교집합 합계
        val_union_total = torch.zeros(n_class, dtype=torch.float64, device=device)   # 합집합 합계            

        with torch.no_grad():
            for batch in val_dataloader:
                sample_val_image = batch['image'].to(device)
                sample_val_gt    = batch['gt'].to(device)          # (B,H,W)  0‥18, 255
                # sample_val_vis_gt = batch['vis_gt'].to(device)

                val_logit   = model(sample_val_image)              # (B,19,H,W)
                val_predicted_class    = torch.argmax(val_logit, dim=1)       # (B,H,W)
                valid_pixel  = (sample_val_gt != 255)
                
                for cls in range(n_class):
                    val_pred_c = (val_predicted_class == cls) & valid_pixel
                    val_gt_c   = (sample_val_gt == cls) & valid_pixel

                    val_inter_total[cls] += (val_pred_c & val_gt_c).sum()
                    val_union_total[cls] += (val_pred_c | val_gt_c).sum()

        val_iou_per_class = val_inter_total / torch.clamp(val_union_total, min=1)    # 0으로 나눔 방지
        val_valid_cls     = val_union_total > 0                                  # 데이터셋에 존재하는 클래스
        val_mIoU          = val_iou_per_class[val_valid_cls].mean().item() * 100
        val_mIoU_list.append(val_mIoU)

        #한 epoch이 끝나고 나면 시간 출력 
        t_elapsed = timedelta(seconds=time.time() - start)
        training_time_left = ((arg_parameter.num_epochs - (epoch + 1)) * t_elapsed.total_seconds()) / (3600)
        
        print(f"Name: {arg_Dic.model_name} | Epoch: {'[':>4}{epoch + 1:>4}/{arg_parameter.num_epochs}] | time left: {training_time_left:.2f} hours | loss: {epoch_ave_loss:.4f} | Val_mIoU: {val_mIoU:.2f} | Train_mIoU: {train_mIoU:.2f}")
        torch.save(model.state_dict(), osp.join(arg_Dic.model_dir, f"{epoch + 1:04d}_valmIoU_{int(val_mIoU):d}_trainmIoU_{int(train_mIoU):d}.pth"))

        if len(train_mIoU_list) >= 2:
            if train_mIoU_list[-1] > max(train_mIoU_list[:-1]):
                train_better_mIoU = train_mIoU_list[-1]
                train_better_mIoU_num = train_mIoU_list.index(train_better_mIoU)
                torch.save(model.state_dict(), osp.join(arg_Dic.train_higher_model_dir, f"{train_better_mIoU_num + 1:04d}_{int(train_better_mIoU):d}.pth"))
        else:
            torch.save(model.state_dict(), osp.join(arg_Dic.train_higher_model_dir, f"{epoch + 1:04d}_{int(train_mIoU):d}.pth"))

        if len(val_mIoU_list) >= 2:
            if val_mIoU_list[-1] > max(val_mIoU_list[:-1]):
                val_better_mIoU = val_mIoU_list[-1]
                val_better_mIoU_num = val_mIoU_list.index(val_better_mIoU)
                torch.save(model.state_dict(), osp.join(arg_Dic.val_higher_model_dir, f"{val_better_mIoU_num + 1:04d}_{int(val_better_mIoU):d}.pth"))
        else:
            torch.save(model.state_dict(), osp.join(arg_Dic.val_higher_model_dir, f"{epoch + 1:04d}_{int(val_mIoU):d}.pth"))

        #일 단위
        # training_time_left = ((total_epochs - (epoch + 1)) * t_elapsed.total_seconds()) / (3600*24)
        # print(f"Training time left: {training_time_left:.2f} days")
        
        # 텐서보드
        # print(sample_image.shape) > (B, C, H, W) > B에서 랜덤으로 하나 뽑아서 idx_random에 넣음
        idx_random_train = torch.randint(0, sample_image.size(0), (1,)).item()
        idx_random_val = torch.randint(0, sample_val_image.size(0), (1,)).item()
        # print(idx_random)
        # print(sample_image.shape) > (C, H, W)
        
        writer.add_image('Input/train_dataset_Image', denorm(sample_image[idx_random_train]), global_step=epoch)

        # sample_vis_gt = sample_vis_gt.permute(0, 3, 1, 2) #(B, H, W, C) > (B, C, H, W)
        # writer.add_image('Input/train_dataset_Gt', sample_vis_gt[idx_random_train], global_step=epoch)
        sample_gt = sample_gt[idx_random_train]
        sample_gt = sample_gt.detach().cpu().numpy()
        sample_gt = colorize_mask(sample_gt)  # shape: (H, W, 3), dtype: uint8
        sample_gt = torch.tensor(sample_gt, dtype=torch.uint8).permute(2, 0, 1)  # (3, H, W)
        writer.add_image('Input/train_dataset_Gt', sample_gt, global_step=epoch)

        writer.add_image('Input/val_dataset_Image', denorm(sample_val_image[idx_random_val]), global_step=epoch)

        sample_val_gt = sample_val_gt[idx_random_val]
        sample_val_gt = sample_val_gt.detach().cpu().numpy()
        sample_val_gt = colorize_mask(sample_val_gt)  # shape: (H, W, 3), dtype: uint8
        sample_val_gt = torch.tensor(sample_val_gt, dtype=torch.uint8).permute(2, 0, 1)  # (3, H, W)
        writer.add_image('Input/val_dataset_Gt', sample_val_gt, global_step=epoch)
        # writer.add_image('Input/val_dataset_Gt', sample_val_vis_gt[idx_random_val], global_step=epoch)

        # 예측 클래스에서 하나 선택
        predicted_class = predicted_class[idx_random_train] # shape: (H, W)
        predicted_class = predicted_class.detach().cpu().numpy()
        predicted_class = colorize_mask(predicted_class)  # shape: (H, W, 3), dtype: uint8
        predicted_class = torch.tensor(predicted_class, dtype=torch.uint8).permute(2, 0, 1)  # (3, H, W)
        writer.add_image('Results/train_predic_result', predicted_class, global_step=epoch)

        val_predicted_class = val_predicted_class[idx_random_val]
        val_predicted_class = val_predicted_class.detach().cpu().numpy()
        val_predicted_class = colorize_mask(val_predicted_class)  # shape: (H, W, 3), dtype: uint8
        val_predicted_class = torch.tensor(val_predicted_class, dtype=torch.uint8).permute(2, 0, 1)  # (3, H, W)
        writer.add_image('Results/val_predic_result', val_predicted_class, global_step=epoch)

        writer.add_scalar('Evaluation/Loss', epoch_ave_loss, global_step=epoch)
        writer.add_scalar('Evaluation/train_Accuracy', train_mIoU, global_step=epoch)
        writer.add_scalar('Evaluation/val_Accuracy', val_mIoU, global_step=epoch)
        writer.add_scalar('Evaluation/learning rate', optimizer.param_groups[0]['lr'], global_step=epoch)
    
    return model, train_mIoU_list, val_mIoU_list

def save_best_model(model, train_mIoU_list, val_mIoU_list):
    if len(train_mIoU_list) >= 2:
        train_max_mIoU = max(train_mIoU_list)
        train_max_mIoU_num = train_mIoU_list.index(train_max_mIoU)
        torch.save(model.state_dict(), osp.join(arg_Dic.train_bestModel_dir, f"{train_max_mIoU_num + 1:04d}_{int(train_max_mIoU):d}.pth"))

    if len(val_mIoU_list) >= 2:
        val_max_mIoU = max(val_mIoU_list)
        val_max_mIoU_num = val_mIoU_list.index(val_max_mIoU)
        torch.save(model.state_dict(), osp.join(arg_Dic.val_bestModel_dir, f"{val_max_mIoU_num + 1:04d}_{int(val_max_mIoU):d}.pth"))

if __name__ == "__main__":   
    model, train_mIoU_list, val_mIoU_list = main()
    save_best_model(model, train_mIoU_list, val_mIoU_list)