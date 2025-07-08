# -*- coding: utf-8 -*-
"""baseline_code_kfold.py

K-Fold Cross Validation version of the document type classification baseline code.

## Improvements Applied:
- Stratified 5-Fold Cross Validation
- Better model architecture (EfficientNet)
- Larger image size (224x224)
- Enhanced data augmentation
- Learning rate scheduler
- Label smoothing loss
- Mixed precision training
- Test time augmentation (TTA)
- Model ensemble from all folds

## Contents
- Prepare Environments
- Import Library & Define Functions
- Hyper-parameters
- K-Fold Cross Validation Training
- Ensemble Inference & Save File
"""

import os
import time
import random
import cv2

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
from sklearn.model_selection import StratifiedKFold
import gc

# 로그 유틸리티 import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 현재 파일의 상위 디렉토리를 Python path에 추가
import utils.log_util as log


# 시드를 고정합니다.
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # CUDA 10.2+ 환경에서 결정적 연산을 위한 환경 변수 설정
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
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

SEED = 42
set_seed(SEED)
log.info(f"🌱 Random seed set to {SEED}")

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

# 검증을 위한 함수입니다.
def validate_one_epoch(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for image, targets in pbar:
            image = image.to(device)
            targets = targets.to(device)

            with autocast():
                preds = model(image)
                loss = loss_fn(preds, targets)

            val_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
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
            
            # 예측 평균 (확률 분포 반환)
            final_pred = torch.stack(all_preds).mean(0)
            predictions.append(final_pred.squeeze(0).detach().cpu().numpy())
    
    return np.array(predictions)

# 앙상블 예측 함수
def predict_ensemble(models, dataset, device, img_size):
    all_predictions = []
    
    for i, model in enumerate(models):
        log.info(f"🔮 Predicting with Fold {i+1} model...")
        fold_predictions = predict_with_tta(model, dataset, device, img_size)
        all_predictions.append(fold_predictions)
    
    # 모든 폴드의 예측을 평균 (shape: [num_samples, num_classes])
    ensemble_predictions = np.mean(all_predictions, axis=0)
    # 각 샘플에 대해 argmax 적용
    final_predictions = np.argmax(ensemble_predictions, axis=1)
    
    return final_predictions

"""## Hyper-parameters
* 학습 및 추론에 필요한 하이퍼파라미터들을 정의합니다.
"""

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"💻 Using device: {device}")

# data config
data_path = '../../input/data'

# model config
model_name = 'efficientnetv2_rw_m'  # 더 좋은 모델 사용

# training config
img_size = 480  # 이미지 크기 대폭 확대
LR = 1e-3  # 더 낮은 학습률
EPOCHS = 100  # early stopping을 위해 더 많은 epoch 설정
BATCH_SIZE = 10  # 큰 모델에 맞춰 배치 크기 조정
num_workers = 0
weight_decay = 1e-4
label_smoothing = 0.1
patience = 10  # early stopping patience

# K-Fold config
K_FOLDS = 5

"""## Data Preparation
* 데이터 로드 및 transform 정의
"""

# 강화된 augmentation을 위한 transform 코드
trn_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    # 다양한 데이터 증강 기법들
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.Blur(blur_limit=3, p=0.1),
    A.CLAHE(clip_limit=2.0, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# validation/test image 변환을 위한 transform 코드
# tst_transform_tta = val_transform_tta = trn_transform

# 데이터 로드
train_df = pd.read_csv(f"{data_path}/train.csv")
log.info(f"📂 Total training samples: {len(train_df)}")

# 클래스 분포 확인
class_counts = train_df['target'].value_counts().sort_index()
log.info(f"📊 Class distribution: {class_counts.to_dict()}")

"""## K-Fold Cross Validation Training
* 5폴드 층화 교차검증으로 모델을 학습합니다.
"""

# Early Stopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy()
        elif val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

# K-Fold 정의
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

# 폴드별 결과 저장
fold_scores = []
best_models = []

log.info(f"\n🔄 Starting {K_FOLDS}-Fold Cross Validation Training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
    log.info(f"\n{'='*60}")
    log.info(f"🔥 FOLD {fold+1}/{K_FOLDS}")
    log.info(f"{'='*60}")
    
    # 폴드별 데이터 분할
    train_fold = train_df.iloc[train_idx].reset_index(drop=True)
    val_fold = train_df.iloc[val_idx].reset_index(drop=True)
    
    log.info(f"📚 Train samples: {len(train_fold)}")
    log.info(f"📝 Validation samples: {len(val_fold)}")
    
    # 데이터셋 생성
    trn_dataset = ImageDataset(train_fold, f"{data_path}/train/", transform=trn_transform)
    val_dataset = ImageDataset(val_fold, f"{data_path}/train/", transform=None)  # TTA에서 원본 이미지를 직접 변형하므로 여기서는 None
    
    # DataLoader 생성
    trn_loader = DataLoader(
        trn_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    # val_loader는 TTA에서 사용하지 않으므로 생성하지 않음
    
    # 모델 초기화 (매 폴드마다 새로 시작)
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17
    ).to(device)
    
    log.info(f"🏗️ Model: {model_name}")
    log.info(f"🔢 Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function, optimizer, scheduler 정의
    loss_fn = LabelSmoothingLoss(classes=17, smoothing=label_smoothing)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler()
    
    # Early stopping 초기화
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001, restore_best_weights=True)
    
    # 최고 성능 모델 저장을 위한 변수
    best_val_f1 = 0.0
    best_model_state = None
    
    log.info(f"🚀 Starting Fold {fold+1} training...")
    
    # 폴드별 학습
    for epoch in range(EPOCHS):
        log.info(f"\n--- Fold {fold+1} | Epoch {epoch+1}/{EPOCHS} ---")
        
        # Training
        train_ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device, scaler)
        
        # Validation
        val_ret = validate_one_epoch_tta(val_fold, model, loss_fn, device, img_size, data_path)
        
        # Learning rate scheduler step
        scheduler.step()
        
        # 결과 출력
        current_lr = optimizer.param_groups[0]['lr']
        
        # 최고 성능 모델 저장
        if val_ret['val_f1'] > best_val_f1:
            best_val_f1 = val_ret['val_f1']
            best_model_state = model.state_dict().copy()
            log.info(f"💾 Fold {fold+1} Best model updated! F1: {best_val_f1:.4f}")
        
        # Early stopping 체크
        if early_stopping(val_ret['val_f1'], model):
            log.info(f"🛑 Early stopping triggered at epoch {epoch+1} for Fold {fold+1}")
            log.info(f"🎯 Best F1 score: {early_stopping.best_score:.4f}")
            break
        
        # 로그 출력
        log_msg = f"train_loss: {train_ret['train_loss']:.4f} | "
        log_msg += f"train_acc: {train_ret['train_acc']:.4f} | "
        log_msg += f"train_f1: {train_ret['train_f1']:.4f} | "
        log_msg += f"val_loss: {val_ret['val_loss']:.4f} | "
        log_msg += f"val_acc: {val_ret['val_acc']:.4f} | "
        log_msg += f"val_f1: {val_ret['val_f1']:.4f} | "
        log_msg += f"lr: {current_lr:.6f}"
        
        log.info(log_msg)
    
    # 폴드 완료
    # Early stopping이 활성화된 경우 best_weights를 사용, 그렇지 않으면 기존 방식 사용
    if early_stopping.best_weights is not None:
        final_f1 = early_stopping.best_score
        log.info(f"\n🎯 Fold {fold+1} completed with early stopping! Best F1: {final_f1:.4f}")
    else:
        final_f1 = best_val_f1
        log.info(f"\n🎯 Fold {fold+1} completed! Best F1: {final_f1:.4f}")
    
    fold_scores.append(final_f1)
    
    # 최고 성능 모델 저장 (early stopping이 활성화된 경우 best_weights 사용)
    if early_stopping.best_weights is not None:
        best_models.append(early_stopping.best_weights.copy())
    else:
        best_models.append(best_model_state.copy())
    
    # 메모리 정리
    del model, optimizer, scheduler, scaler, trn_loader
    del trn_dataset, val_dataset, train_fold, val_fold
    torch.cuda.empty_cache()
    gc.collect()

# K-Fold 결과 요약
log.info(f"\n{'='*60}")
log.info(f"📊 K-FOLD CROSS VALIDATION RESULTS")
log.info(f"{'='*60}")

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

log.info(f"📈 Individual fold scores: {[f'{score:.4f}' for score in fold_scores]}")
log.info(f"🎯 Mean F1 Score: {mean_score:.4f}")
log.info(f"📏 Standard Deviation: {std_score:.4f}")
log.info(f"📊 Score Range: {mean_score:.4f} ± {std_score:.4f}")
log.info(f"⬇️ Min Score: {min(fold_scores):.4f}")
log.info(f"⬆️ Max Score: {max(fold_scores):.4f}")

"""## Ensemble Inference & Save File
* 모든 폴드의 모델을 앙상블하여 테스트 이미지에 대한 추론을 진행합니다.
"""

log.info(f"\n🚀 Starting Ensemble Prediction...")

# 테스트 데이터셋 생성
tst_dataset = ImageDataset(
    f"{data_path}/sample_submission.csv",
    f"{data_path}/test/",
    transform=None  # TTA에서 원본 이미지를 직접 변형하므로 여기서는 None
)

# 앙상블을 위해 모든 폴드의 모델 로드
ensemble_models = []
for fold in range(K_FOLDS):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17
    ).to(device)
    model.load_state_dict(best_models[fold])
    ensemble_models.append(model)
    log.info(f"✅ Loaded Fold {fold+1} model")

# 앙상블 예측 실행
log.info(f"\n🔮 Running ensemble prediction with {K_FOLDS} models and TTA...")
preds_list = predict_ensemble(ensemble_models, tst_dataset, device, img_size)

# 결과 저장
pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = preds_list

sample_submission_df = pd.read_csv(f"{data_path}/sample_submission.csv")
assert (sample_submission_df['ID'] == pred_df['ID']).all()

output_path = "./output"
os.makedirs(output_path, exist_ok=True)
pred_df.to_csv(f"{output_path}/pred_advanced_kfold_tta2_efnv2rwm.csv", index=False)

log.info(f"\n✅ Ensemble prediction completed and saved to {output_path}/pred_advanced_kfold_tta2_efnv2rwm.csv")
log.info(f"📈 Final K-Fold CV Score: {mean_score:.4f} ± {std_score:.4f}")

# 메모리 정리
for model in ensemble_models:
    del model
torch.cuda.empty_cache()
gc.collect()

pred_df.head() 