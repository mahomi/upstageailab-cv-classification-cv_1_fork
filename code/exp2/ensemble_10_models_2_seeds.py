# -*- coding: utf-8 -*-
"""ensemble_10_models_2_seeds.py

5폴드 교차검증을 2번의 다른 랜덤시드로 실행하여 총 10개의 모델을 앙상블하는 스크립트.

## Features:
- 2개의 다른 랜덤시드로 5폴드 교차검증 실행
- 총 10개의 모델 생성 (5폴드 × 2시드)
- 증강 이미지 캐시 초기화로 새로운 증강 적용
- 앙상블 추론으로 최종 결과 생성

## Contents
- Import Library & Define Functions
- Hyper-parameters
- Multi-Seed K-Fold Cross Validation Training
- Ensemble Inference & Save File
"""

import os
import time
import random
import cv2
import shutil

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.log_util as log

# 시드를 고정합니다.
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    def __init__(self, csv_data, path, transform=None, cache_images=True, cache_augmented=True):
        if isinstance(csv_data, str):
            self.df = pd.read_csv(csv_data).values
        else:
            self.df = csv_data.values
        self.path = path
        self.transform = transform
        self.cache_images = cache_images
        self.cache_augmented = cache_augmented
        self.image_cache = {} if cache_images else None
        self.augmented_cache = {} if cache_augmented else None
        
        # 캐싱 통계
        self.stats = {
            'original_cache_hits': 0,
            'original_cache_misses': 0,
            'augmented_cache_hits': 0,
            'augmented_cache_misses': 0,
            'disk_loads': 0,
            'augmentations': 0
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        
        # 증강된 이미지가 캐시되어 있는지 확인
        if self.cache_augmented and self.augmented_cache is not None and name in self.augmented_cache:
            img = self.augmented_cache[name]
            self.stats['augmented_cache_hits'] += 1
        else:
            self.stats['augmented_cache_misses'] += 1
            
            # 원본 이미지 로드
            if self.cache_images and self.image_cache is not None and name in self.image_cache:
                img = self.image_cache[name]
                self.stats['original_cache_hits'] += 1
            else:
                self.stats['original_cache_misses'] += 1
                img = np.array(Image.open(os.path.join(self.path, name)))
                self.stats['disk_loads'] += 1
                if self.cache_images and self.image_cache is not None:
                    self.image_cache[name] = img
            
            # 증강 적용
            if self.transform:
                self.stats['augmentations'] += 1
                img = self.transform(image=img)['image']
                # 증강된 이미지 캐싱
                if self.cache_augmented and self.augmented_cache is not None:
                    self.augmented_cache[name] = img
        
        return img, target
    
    def clear_augmented_cache(self):
        """증강된 이미지 캐시를 초기화합니다."""
        if self.augmented_cache is not None:
            self.augmented_cache.clear()
            log.info("🧹 증강된 이미지 캐시가 초기화되었습니다.")
    
    def print_stats(self):
        """캐싱 통계를 출력합니다."""
        current_epoch_requests = self.stats['augmented_cache_hits'] + self.stats['augmented_cache_misses']
        
        if current_epoch_requests > 0:
            log.info(f"📊 Dataset Cache Statistics:")
            log.info(f"   Current epoch requests: {current_epoch_requests}")
            log.info(f"   Original cache hits: {self.stats['original_cache_hits']} ({self.stats['original_cache_hits']/current_epoch_requests*100:.1f}%)")
            log.info(f"   Original cache misses: {self.stats['original_cache_misses']} ({self.stats['original_cache_misses']/current_epoch_requests*100:.1f}%)")
            log.info(f"   Augmented cache hits: {self.stats['augmented_cache_hits']} ({self.stats['augmented_cache_hits']/current_epoch_requests*100:.1f}%)")
            log.info(f"   Augmented cache misses: {self.stats['augmented_cache_misses']} ({self.stats['augmented_cache_misses']/current_epoch_requests*100:.1f}%)")
            log.info(f"   Disk loads: {self.stats['disk_loads']}")
            log.info(f"   Augmentations applied: {self.stats['augmentations']}")
            log.info(f"   Original cache size: {len(self.image_cache) if self.image_cache else 0}")
            log.info(f"   Augmented cache size: {len(self.augmented_cache) if self.augmented_cache else 0}")
            
            if self.stats['augmented_cache_hits'] > 0:
                cache_efficiency = self.stats['augmented_cache_hits'] / current_epoch_requests * 100
                log.info(f"   Cache efficiency: {cache_efficiency:.1f}%")
    
    def reset_stats(self):
        """통계를 초기화합니다."""
        self.stats = {
            'original_cache_hits': 0,
            'original_cache_misses': 0,
            'augmented_cache_hits': 0,
            'augmented_cache_misses': 0,
            'disk_loads': 0,
            'augmentations': 0
        }

# one epoch 학습을 위한 함수입니다.
def train_one_epoch(loader, model, optimizer, loss_fn, device, scaler):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []
    
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Training")):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            preds = model(images)
            loss = loss_fn(preds, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        preds_list.append(preds.softmax(dim=1).detach().cpu().numpy())
        targets_list.append(targets.detach().cpu().numpy())
    
    train_loss /= len(loader)
    preds_list = np.concatenate(preds_list, axis=0)
    targets_list = np.concatenate(targets_list, axis=0)
    
    train_acc = accuracy_score(targets_list, np.argmax(preds_list, axis=1))
    train_f1 = f1_score(targets_list, np.argmax(preds_list, axis=1), average='macro')
    
    return {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1
    }

# one epoch 검증을 위한 함수입니다.
def validate_one_epoch(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Validation")):
            images = images.to(device)
            targets = targets.to(device)
            
            with autocast():
                preds = model(images)
                loss = loss_fn(preds, targets)
            
            val_loss += loss.item()
            preds_list.append(preds.softmax(dim=1).detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())
    
    val_loss /= len(loader)
    preds_list = np.concatenate(preds_list, axis=0)
    targets_list = np.concatenate(targets_list, axis=0)
    
    val_acc = accuracy_score(targets_list, np.argmax(preds_list, axis=1))
    val_f1 = f1_score(targets_list, np.argmax(preds_list, axis=1), average='macro')
    
    return {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1
    }

# TTA 변환 함수들
def get_val_tta_transforms(img_size):
    return [
        A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Resize(256, 256),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Resize(256, 256),
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Resize(256, 256),
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]

# TTA를 사용한 검증 함수
def validate_one_epoch_tta(val_data, model, loss_fn, device, img_size, data_path):
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []
    
    tta_transforms = get_val_tta_transforms(img_size)
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_data)), desc="TTA Validation"):
            name, target = val_data.iloc[idx]
            img = np.array(Image.open(os.path.join(data_path, name)))
            
            all_preds = []
            for transform in tta_transforms:
                transformed_img = transform(image=img)['image'].unsqueeze(0).to(device)
                with autocast():
                    preds = model(transformed_img)
                all_preds.append(preds.softmax(dim=1))
            
            # TTA 예측 평균
            final_pred = torch.stack(all_preds).mean(0)
            preds_list.append(final_pred.squeeze(0).detach().cpu().numpy())
            targets_list.append(target)
    
    preds_list = np.array(preds_list)
    targets_list = np.array(targets_list)
    
    # Loss 계산 (첫 번째 TTA 변환만 사용)
    val_loss = loss_fn(torch.tensor(preds_list).to(device), torch.tensor(targets_list).to(device)).item()
    
    val_acc = accuracy_score(targets_list, np.argmax(preds_list, axis=1))
    val_f1 = f1_score(targets_list, np.argmax(preds_list, axis=1), average='macro')
    
    return {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1
    }

# TTA를 사용한 예측 함수
def predict_with_tta(model, dataset, device, img_size):
    model.eval()
    predictions = []
    
    tta_transforms = get_val_tta_transforms(img_size)
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="TTA Prediction"):
            name, _ = dataset.df[idx]
            img = np.array(Image.open(os.path.join(dataset.path, name)))
            
            all_preds = []
            for transform in tta_transforms:
                transformed_img = transform(image=img)['image'].unsqueeze(0).to(device)
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
        log.info(f"🔮 Predicting with Model {i+1}/{len(models)}...")
        fold_predictions = predict_with_tta(model, dataset, device, img_size)
        all_predictions.append(fold_predictions)
    
    # 모든 모델의 예측을 평균 (shape: [num_samples, num_classes])
    ensemble_predictions = np.mean(all_predictions, axis=0)
    # 각 샘플에 대해 argmax 적용
    final_predictions = np.argmax(ensemble_predictions, axis=1)
    
    return final_predictions

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

"""## Hyper-parameters
* 학습 및 추론에 필요한 하이퍼파라미터들을 정의합니다.
"""

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"💻 Using device: {device}")

# data config
data_path = '../../input/data'

# model config
model_name = 'tf_efficientnetv2_xl'

# training config
img_size = 480
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 10
num_workers = 0
weight_decay = 1e-4
label_smoothing = 0.1
patience = 10

# K-Fold config
K_FOLDS = 5

# Multi-Seed config
SEEDS = [42, 123]  # 2개의 다른 랜덤시드

# 강화된 augmentation을 위한 transform 코드
def get_train_transform(img_size):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
        A.Resize(256, 256),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
        A.OneOf([
            A.Blur(blur_limit=2),
            A.MotionBlur(blur_limit=5),
            A.Defocus(radius=(1, 3)),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(0.0000002, 0.000001), mean=0, p=1.0),
            A.ImageCompression(quality_lower=40, quality_upper=60, p=1.0),
            A.CoarseDropout(max_holes=8, max_height=img_size//10, max_width=img_size//10),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
            A.GridDropout(ratio=0.5, p=0.5),
        ], p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# validation transform (train과 동일한 증강 적용)
def get_val_transform(img_size):
    return get_train_transform(img_size)

# 데이터 로드
train_df = pd.read_csv(f"{data_path}/train.csv")
log.info(f"📂 Total training samples: {len(train_df)}")

# 클래스 분포 확인
class_counts = train_df['target'].value_counts().sort_index()
log.info(f"📊 Class distribution: {class_counts.to_dict()}")

"""## Multi-Seed K-Fold Cross Validation Training
* 2개의 다른 랜덤시드로 5폴드 교차검증을 실행하여 총 10개의 모델을 생성합니다.
"""

# 모든 시드와 폴드의 결과 저장
all_fold_scores = []
all_best_models = []

log.info(f"\n🔄 Starting Multi-Seed K-Fold Cross Validation Training...")
log.info(f"🌱 Seeds: {SEEDS}")
log.info(f"📊 Total models to train: {len(SEEDS)} × {K_FOLDS} = {len(SEEDS) * K_FOLDS}")

for seed_idx, seed in enumerate(SEEDS):
    log.info(f"\n{'='*80}")
    log.info(f"🌱 SEED {seed_idx+1}/{len(SEEDS)}: {seed}")
    log.info(f"{'='*80}")
    
    # 시드 설정
    set_seed(seed)
    
    # 현재 시드의 transform 생성 (새로운 증강을 위해)
    trn_transform = get_train_transform(img_size)
    val_transform = get_val_transform(img_size)
    
    # K-Fold 정의
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    
    # 현재 시드의 폴드별 결과 저장
    fold_scores = []
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
        log.info(f"\n{'='*60}")
        log.info(f"🔥 SEED {seed} | FOLD {fold+1}/{K_FOLDS}")
        log.info(f"{'='*60}")
        
        # 폴드별 데이터 분할
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_df.iloc[val_idx].reset_index(drop=True)
        
        log.info(f"📚 Train samples: {len(train_fold)}")
        log.info(f"📝 Validation samples: {len(val_fold)}")
        
        # 데이터셋 생성 (증강 캐시 초기화)
        trn_dataset = ImageDataset(train_fold, f"{data_path}/train/", transform=trn_transform)
        val_dataset = ImageDataset(val_fold, f"{data_path}/train/", transform=val_transform)
        
        # 증강 캐시 초기화 (새로운 시드에서 새로운 증강을 위해)
        trn_dataset.clear_augmented_cache()
        val_dataset.clear_augmented_cache()
        
        # DataLoader 생성
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
            pin_memory=True,
            drop_last=False
        )
        
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
        
        log.info(f"🚀 Starting Seed {seed} | Fold {fold+1} training...")
        
        # 폴드별 학습
        for epoch in range(EPOCHS):
            log.info(f"\n--- Seed {seed} | Fold {fold+1} | Epoch {epoch+1}/{EPOCHS} ---")
            
            # 에포크 시작 시 통계 초기화
            trn_dataset.reset_stats()
            val_dataset.reset_stats()
            
            # Training
            train_ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device, scaler)
            
            # Validation
            val_ret = validate_one_epoch(val_loader, model, loss_fn, device)
            
            # Learning rate scheduler step
            scheduler.step()
            
            # 결과 출력
            current_lr = optimizer.param_groups[0]['lr']
            
            # 최고 성능 모델 저장
            if val_ret['val_f1'] > best_val_f1:
                best_val_f1 = val_ret['val_f1']
                best_model_state = model.state_dict().copy()
                log.info(f"💾 Seed {seed} | Fold {fold+1} Best model updated! F1: {best_val_f1:.4f}")
            
            # Early stopping 체크
            if early_stopping(val_ret['val_f1'], model):
                log.info(f"🛑 Early stopping triggered at epoch {epoch+1} for Seed {seed} | Fold {fold+1}")
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
            
            # 매 에포크마다 캐싱 통계 출력
            log.info(f"\n📊 Epoch {epoch+1} Cache Statistics:")
            trn_dataset.print_stats()
            val_dataset.print_stats()
        
        # 폴드 완료
        if early_stopping.best_weights is not None:
            final_f1 = early_stopping.best_score
            log.info(f"\n🎯 Seed {seed} | Fold {fold+1} completed with early stopping! Best F1: {final_f1:.4f}")
        else:
            final_f1 = best_val_f1
            log.info(f"\n🎯 Seed {seed} | Fold {fold+1} completed! Best F1: {final_f1:.4f}")
        
        fold_scores.append(final_f1)
        
        # 최고 성능 모델 저장
        if early_stopping.best_weights is not None:
            best_models.append(early_stopping.best_weights.copy())
        elif best_model_state is not None:
            best_models.append(best_model_state.copy())
        else:
            best_models.append(model.state_dict().copy())
        
        # 메모리 정리
        del model, optimizer, scheduler, scaler, trn_loader, val_loader
        del trn_dataset, val_dataset, train_fold, val_fold
        torch.cuda.empty_cache()
        gc.collect()
    
    # 현재 시드의 결과 저장
    all_fold_scores.extend(fold_scores)
    all_best_models.extend(best_models)
    
    # 현재 시드의 결과 요약
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 SEED {seed} K-FOLD RESULTS")
    log.info(f"{'='*60}")
    log.info(f"📈 Individual fold scores: {[f'{score:.4f}' for score in fold_scores]}")
    log.info(f"🎯 Mean F1 Score: {mean_score:.4f}")
    log.info(f"📏 Standard Deviation: {std_score:.4f}")
    log.info(f"📊 Score Range: {mean_score:.4f} ± {std_score:.4f}")
    log.info(f"⬇️ Min Score: {min(fold_scores):.4f}")
    log.info(f"⬆️ Max Score: {max(fold_scores):.4f}")

# 전체 결과 요약
log.info(f"\n{'='*80}")
log.info(f"📊 OVERALL MULTI-SEED K-FOLD CROSS VALIDATION RESULTS")
log.info(f"{'='*80}")

total_mean_score = np.mean(all_fold_scores)
total_std_score = np.std(all_fold_scores)

log.info(f"📈 All fold scores: {[f'{score:.4f}' for score in all_fold_scores]}")
log.info(f"🎯 Overall Mean F1 Score: {total_mean_score:.4f}")
log.info(f"📏 Overall Standard Deviation: {total_std_score:.4f}")
log.info(f"📊 Overall Score Range: {total_mean_score:.4f} ± {total_std_score:.4f}")
log.info(f"⬇️ Overall Min Score: {min(all_fold_scores):.4f}")
log.info(f"⬆️ Overall Max Score: {max(all_fold_scores):.4f}")
log.info(f"🔢 Total models trained: {len(all_best_models)}")

"""## Ensemble Inference & Save File
* 모든 시드와 폴드의 모델을 앙상블하여 테스트 이미지에 대한 추론을 진행합니다.
"""

log.info(f"\n🚀 Starting Ensemble Prediction with {len(all_best_models)} models...")

# 테스트 데이터셋 생성
tst_dataset = ImageDataset(
    f"{data_path}/sample_submission.csv",
    f"{data_path}/test/",
    transform=None  # TTA에서 원본 이미지를 직접 변형하므로 여기서는 None
)

# 앙상블을 위해 모든 모델 로드
ensemble_models = []
for i, model_state in enumerate(all_best_models):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17
    ).to(device)
    model.load_state_dict(model_state)
    ensemble_models.append(model)
    log.info(f"✅ Loaded Model {i+1}/{len(all_best_models)} (Seed {SEEDS[i//K_FOLDS]}, Fold {i%K_FOLDS+1})")

# 앙상블 예측 실행
log.info(f"\n🔮 Running ensemble prediction with {len(ensemble_models)} models and TTA...")
preds_list = predict_ensemble(ensemble_models, tst_dataset, device, img_size)

# 결과 저장
pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = preds_list

sample_submission_df = pd.read_csv(f"{data_path}/sample_submission.csv")
assert (sample_submission_df['ID'] == pred_df['ID']).all()

output_path = "./output"
os.makedirs(output_path, exist_ok=True)
pred_df.to_csv(f"{output_path}/pred_ensemble_10_models_2_seeds.csv", index=False)

log.info(f"\n✅ Ensemble prediction completed and saved to {output_path}/pred_ensemble_10_models_2_seeds.csv")
log.info(f"📈 Overall Multi-Seed K-Fold CV Score: {total_mean_score:.4f} ± {total_std_score:.4f}")
log.info(f"🎯 Total models used for ensemble: {len(ensemble_models)}")

# 메모리 정리
for model in ensemble_models:
    del model
torch.cuda.empty_cache()
gc.collect()

log.info(f"\n📊 Final Results Summary:")
log.info(f"   Seeds used: {SEEDS}")
log.info(f"   Folds per seed: {K_FOLDS}")
log.info(f"   Total models: {len(ensemble_models)}")
log.info(f"   Overall CV Score: {total_mean_score:.4f} ± {total_std_score:.4f}")
log.info(f"   Output file: {output_path}/pred_ensemble_10_models_2_seeds.csv")

pred_df.head() 