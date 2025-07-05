# -*- coding: utf-8 -*-
"""baseline_code_advanced_tta2.py

Advanced version with proper TTA implementation.

## Improvements Applied:
- Better model architecture (EfficientNet)
- Larger image size (224x224)
- Enhanced data augmentation
- Learning rate scheduler
- Validation data split
- Label smoothing loss
- Mixed precision training
- Proper Test time augmentation (TTA)
- More training epochs

## Contents
- Prepare Environments
- Import Library & Define Functions
- Hyper-parameters
- Load Data
- Train Model
- Inference & Save File
"""

import os
import time
import random

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# 로그 유틸리티 import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 현재 파일의 상위 디렉토리를 Python path에 추가
import utils.log_util as log

# 시드를 고정합니다.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)

# CUDA 10.2+ 환경에서 결정적 연산을 위한 환경 변수 설정
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 완전한 재현성을 위한 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CUDA 환경에서 결정적 연산 설정 (선택적)
try:
    torch.use_deterministic_algorithms(True)
    log.info("완전한 재현성 설정이 활성화되었습니다.")
except Exception as e:
    log.info(f"torch.use_deterministic_algorithms(True) 설정 실패: {e}")
    log.info("기본 재현성 설정만 사용됩니다.")

# 현재 스크립트 위치를 작업 디렉토리로 설정
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Label Smoothing Loss 클래스 정의
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# 데이터셋 클래스를 정의합니다.
class ImageDataset(Dataset):
    def __init__(self, csv_data, path, transform=None):
        if isinstance(csv_data, str):
            self.df = pd.read_csv(csv_data).values
        else:
            self.df = csv_data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target

# one epoch 학습을 위한 함수입니다.
def train_one_epoch(loader, model, optimizer, loss_fn, device, scaler):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader, desc="Training")
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():
            preds = model(image)
            loss = loss_fn(preds, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Training - Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

    return ret

# 검증을 위한 함수입니다. (TTA 적용)
def validate_one_epoch(loader, model, loss_fn, device, use_tta=True, tta_count=3):
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for image, targets in pbar:
            image = image.to(device)
            targets = targets.to(device)

            if use_tta:
                # TTA를 위한 여러 번 예측
                batch_preds = []
                for _ in range(tta_count):
                    with autocast():
                        preds = model(image)
                        batch_preds.append(preds.softmax(dim=1))
                
                # 평균내기
                final_preds = torch.stack(batch_preds).mean(0)
                loss = loss_fn(final_preds.log(), targets)  # log_softmax로 변환
            else:
                with autocast():
                    preds = model(image)
                    loss = loss_fn(preds, targets)
                    final_preds = preds.softmax(dim=1)

            val_loss += loss.item()
            preds_list.extend(final_preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())

            pbar.set_description(f"Validation - Loss: {loss.item():.4f}")

    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
    }

    return ret

# Validation TTA를 위한 고정된 transform 세트
def get_val_tta_transforms(img_size):
    """TTA를 위한 고정된 transform들 (간단하고 안정적인 변형들)"""
    return [
        # 원본
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 좌우 반전
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 상하 반전 (문서 이미지에 유용할 수 있음)
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 회전 (약간의 회전)
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 밝기 조정
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]

# 고정된 TTA를 사용하는 검증 함수
def validate_one_epoch_tta(val_data, model, loss_fn, device, img_size, data_path):
    """고정된 TTA를 사용한 validation"""
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []
    
    val_tta_transforms = get_val_tta_transforms(img_size)
    
    with torch.no_grad():
        for idx, (img_name, target) in enumerate(tqdm(val_data.values, desc="Validation TTA")):
            # 이미지 로드
            img_path = os.path.join(data_path, "train", img_name)  
            img = np.array(Image.open(img_path))
            
            # 각 TTA transform 적용하여 예측
            all_preds = []
            all_losses = []
            
            for transform in val_tta_transforms:
                transformed_img = transform(image=img)['image'].unsqueeze(0).to(device)
                target_tensor = torch.tensor([target]).to(device)
                
                with autocast():
                    preds = model(transformed_img)
                    loss = loss_fn(preds, target_tensor)
                
                all_preds.append(preds.softmax(dim=1))
                all_losses.append(loss.item())
            
            # 예측 평균
            final_pred = torch.stack(all_preds).mean(0)
            avg_loss = np.mean(all_losses)
            
            val_loss += avg_loss
            preds_list.extend(final_pred.argmax(dim=1).detach().cpu().numpy())
            targets_list.append(target)
    
    val_loss /= len(val_data)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')
    
    ret = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
    }
    
    return ret

# Test Time Augmentation을 위한 예측 함수 (tst_dataset 사용)
def predict_with_tta(model, dataset, device, img_size):
    """진짜 TTA를 적용한 예측 함수 - dataset을 사용"""
    model.eval()
    predictions = []
    
    # TTA transforms 가져오기 (validation과 동일)
    tta_transforms = get_val_tta_transforms(img_size)
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Real TTA Prediction"):
            # 원본 이미지를 파일에서 직접 로드 (transform 없이)
            img_name, _ = dataset.df[idx]
            img_path = os.path.join(dataset.path, img_name)
            original_img = np.array(Image.open(img_path))
            
            # 각 TTA transform 적용하여 예측
            all_preds = []
            
            for transform in tta_transforms:
                # 매번 원본 이미지에서 다른 변형 적용
                transformed_img = transform(image=original_img)['image'].unsqueeze(0).to(device)
                
                with autocast():
                    preds = model(transformed_img)
                
                all_preds.append(preds.softmax(dim=1))
            
            # 예측 평균
            final_pred = torch.stack(all_preds).mean(0)
            predictions.extend(final_pred.argmax(dim=1).detach().cpu().numpy())
    
    return predictions

"""## Hyper-parameters
* 학습 및 추론에 필요한 하이퍼파라미터들을 정의합니다.
"""

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {device}")

# data config
data_path = '../../input/data'

# model config
# model_name = 'tf_efficientnetv2_l'  # EfficientNet v2 Large 모델 사용
model_name = 'tf_efficientnetv2_xl'  # EfficientNet v2 Large 모델 사용
# training config
train_img_size = 384  # 훈련용 이미지 크기
test_img_size = 384   # 테스트용 이미지 크기
LR = 1e-4  # 더 큰 모델에 맞춰 학습률 조정
EPOCHS = 20  # 충분한 epoch
BATCH_SIZE = 10  # 이미지 크기에 맞춰 배치 크기 조정
num_workers = 0
weight_decay = 1e-4
label_smoothing = 0.1

"""## Load Data
* 학습, 검증, 테스트 데이터셋과 로더를 정의합니다.
"""

# 강화된 augmentation을 위한 transform 코드
trn_transform = A.Compose([
    A.Resize(height=train_img_size, width=train_img_size),
    # 다양한 데이터 증강 기법들
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.Blur(blur_limit=3, p=0.1),
    A.CLAHE(clip_limit=2.0, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# validation/test image 변환을 위한 transform 코드
# tst_transform_tta = val_transform_tta = trn_transform


# 검증 데이터 분할
train_df = pd.read_csv(f"{data_path}/train.csv")
train_data, val_data = train_test_split(
    train_df, 
    test_size=0.2, 
    stratify=train_df['target'], 
    random_state=SEED
)

log.info(f"Train samples: {len(train_data)}")
log.info(f"Validation samples: {len(val_data)}")

# Dataset 정의
trn_dataset = ImageDataset(
    train_data,
    f"{data_path}/train/",
    transform=trn_transform
)
val_dataset = ImageDataset(
    val_data,
    f"{data_path}/train/",
    transform=None  # TTA에서 원본 이미지를 직접 변형하므로 여기서는 None
)
tst_dataset = ImageDataset(
    f"{data_path}/sample_submission.csv",
    f"{data_path}/test/",
    transform=None  # TTA에서 원본 이미지를 직접 변형하므로 여기서는 None
)

log.info(f"Train dataset: {len(trn_dataset)}")
log.info(f"Validation dataset: {len(val_dataset)}")
log.info(f"Test dataset: {len(tst_dataset)}")

# DataLoader 정의
trn_loader = DataLoader(
    trn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
# test loader (TTA에서는 사용하지 않음)
tst_loader = DataLoader(
    tst_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

"""## Train Model
* 모델을 로드하고, 학습을 진행합니다.
"""

# load model
model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes=17
).to(device)

log.info(f"Model: {model_name}")
log.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss function, optimizer, scheduler 정의
loss_fn = LabelSmoothingLoss(classes=17, smoothing=label_smoothing)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = GradScaler()

# 최고 성능 모델 저장을 위한 변수
best_val_f1 = 0.0
best_model_path = "best_model.pth"

log.info("Starting training...")
for epoch in range(EPOCHS):
    log.info(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    
    # Training
    train_ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device, scaler)
    
    # Validation (진짜 TTA 적용으로 test 데이터와 더 유사한 조건에서 평가)
    val_ret = validate_one_epoch_tta(val_data, model, loss_fn, device, test_img_size, data_path)
    
    # Learning rate scheduler step
    scheduler.step()
    
    # 결과 출력
    current_lr = optimizer.param_groups[0]['lr']
    
    train_ret['epoch'] = epoch
    val_ret['epoch'] = epoch
    val_ret['lr'] = current_lr
    
    # 최고 성능 모델 저장
    if val_ret['val_f1'] > best_val_f1:
        best_val_f1 = val_ret['val_f1']
        torch.save(model.state_dict(), best_model_path)
        log.info(f"💾 Best model saved! F1: {best_val_f1:.4f}")
    
    log_message = ""
    for k, v in train_ret.items():
        log_message += f"train_{k}: {v:.4f} | "
    for k, v in val_ret.items():
        log_message += f"{k}: {v:.4f} | "
    
    log.info(log_message.rstrip(" | "))

# 최고 성능 모델 로드
log.info(f"\n🏆 Loading best model (F1: {best_val_f1:.4f})")
model.load_state_dict(torch.load(best_model_path))

"""## Inference & Save File
* 테스트 이미지에 대한 추론을 진행하고, 결과 파일을 저장합니다.
"""

log.info("\nStarting inference with Test Time Augmentation...")

# TTA로 예측 (tst_dataset 사용)
preds_list = predict_with_tta(model, tst_dataset, device, test_img_size)

# 결과 저장
sample_submission_df = pd.read_csv(f"{data_path}/sample_submission.csv")
pred_df = sample_submission_df.copy()
pred_df['target'] = preds_list

# ID 순서 확인 (이미 같은 순서로 처리했으므로 문제없음)
assert len(pred_df) == len(sample_submission_df)

output_path = "./output"
os.makedirs(output_path, exist_ok=True)
pred_df.to_csv(f"{output_path}/pred_advanced_tta2_efnv2xl_resize4.csv", index=False)

log.info(f"\n✅ Prediction completed and saved to {output_path}/pred_advanced_tta2_efnv2xl_resize4.csv")
log.info(f"Best validation F1 score: {best_val_f1:.4f}")

# 메모리 정리
if os.path.exists(best_model_path):
    os.remove(best_model_path)

pred_df.head()