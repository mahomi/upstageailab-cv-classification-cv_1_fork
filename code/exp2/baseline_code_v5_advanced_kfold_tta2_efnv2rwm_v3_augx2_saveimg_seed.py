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

# Augraphy imports
try:
    from augraphy import (
        AugraphyPipeline, InkBleed, BleedThrough, ColorPaper, OneOf, NoiseTexturize,
        SubtleNoise, LightingGradient, ShadowCast
    )
    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False
    print("Warning: Augraphy not available. Skipping augraphy augmentations.")

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

# 시드 리스트 정의 (2개 시드 사용)
SEEDS = [42, 123]
# 초기 시드 설정
set_seed(SEEDS[0])
log.info(f"🌱 Initial random seed set to {SEEDS[0]}")

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
    def __init__(self, csv_data, path, transform=None, cache_images=True, cache_augmented=True, augmentation_multiplier=1, save_augmented_to_disk=True, current_seed=None, img_size=None):
        if isinstance(csv_data, str):
            self.df = pd.read_csv(csv_data).values
        else:
            self.df = csv_data.values
        self.path = path
        self.transform = transform
        self.cache_images = cache_images
        self.cache_augmented = cache_augmented
        self.augmentation_multiplier = augmentation_multiplier
        self.save_augmented_to_disk = save_augmented_to_disk
        self.current_seed = current_seed
        self.img_size = img_size
        self.image_cache = {} if cache_images else None
        self.augmented_cache = {} if cache_augmented else None
        
        # Augraphy 파이프라인 초기화
        self.augraphy_pipeline = get_augraphy_pipeline()
        if self.augraphy_pipeline:
            log.info("📸 Augraphy pipeline initialized")
        
        # 시드별 증강 이미지 저장 폴더 설정
        if self.save_augmented_to_disk and self.current_seed is not None:
            # 캐시 디렉토리를 data 폴더 안으로 변경 (input/data/train_cache)
            data_dir = os.path.dirname(os.path.dirname(self.path))  # input/data/train/ -> input/data/
            self.train_cache_dir = os.path.join(data_dir, 'train_cache')
            # img_size를 포함한 캐시 디렉토리명 생성
            cache_dir_name = f'img{self.img_size}_seed{self.current_seed}' if self.img_size is not None else f'seed_{self.current_seed}'
            self.seed_cache_dir = os.path.join(self.train_cache_dir, cache_dir_name)
            os.makedirs(self.seed_cache_dir, exist_ok=True)
            log.info(f"📁 Seed {self.current_seed} cache directory: {self.seed_cache_dir}")
        else:
            self.seed_cache_dir = None
        
        # 증강 배수가 1보다 큰 경우 데이터를 복제 (원본 제외, 증강된 데이터만 사용)
        if self.augmentation_multiplier > 1:
            original_df = self.df.copy()
            augmented_data = []
            
            # 증강된 데이터만 추가 (원본 제외)
            for i, (name, target) in enumerate(original_df):
                for aug_idx in range(1, self.augmentation_multiplier + 1):  # 1부터 augmentation_multiplier까지
                    augmented_data.append((name, target, aug_idx))  # aug_idx는 증강 인덱스
            
            self.df = np.array(augmented_data)
            log.info(f"📈 Data augmented (원본 제외): {len(original_df)} → {len(self.df)} samples (증강 x{self.augmentation_multiplier})")
        elif self.augmentation_multiplier == 1:
            # augmentation_multiplier가 1이면 원본 데이터만 사용
            original_df = self.df.copy()
            augmented_data = []
            
            # 원본 데이터 추가
            for i, (name, target) in enumerate(original_df):
                augmented_data.append((name, target, 0))  # 0은 원본을 의미
            
            self.df = np.array(augmented_data)
            log.info(f"📈 원본 데이터만 사용: {len(original_df)} samples")
        
        # 캐싱 통계
        self.stats = {
            'original_cache_hits': 0,
            'original_cache_misses': 0,
            'augmented_cache_hits': 0,
            'augmented_cache_misses': 0,
            'disk_cache_hits': 0,
            'disk_cache_misses': 0,
            'disk_loads': 0,
            'augmentations': 0,
            'augmented_saves': 0
        }

    def _get_augmented_cache_path(self, img_name, aug_idx):
        """증강된 이미지 캐시 파일 경로를 반환합니다."""
        if not self.save_augmented_to_disk or self.seed_cache_dir is None:
            return None
        # 파일명에서 확장자 제거하고 증강 인덱스와 함께 .pt 확장자 추가 (텐서 저장용)
        base_name = os.path.splitext(img_name)[0]
        return os.path.join(self.seed_cache_dir, f"{base_name}_aug_{aug_idx}.pt")
    
    def _get_augmented_visual_path(self, img_name, aug_idx):
        """증강된 이미지 시각적 확인용 파일 경로를 반환합니다."""
        if not self.save_augmented_to_disk or self.seed_cache_dir is None:
            return None
        # 파일명에서 확장자 제거하고 증강 인덱스와 함께 .jpg 확장자 추가 (시각적 확인용)
        base_name = os.path.splitext(img_name)[0]
        return os.path.join(self.seed_cache_dir, f"{base_name}_aug_{aug_idx}.jpg")
    
    def _get_tta_cache_path(self, img_name, tta_idx):
        """TTA 이미지 캐시 파일 경로를 반환합니다."""
        if not self.save_augmented_to_disk or self.seed_cache_dir is None:
            return None
        # 파일명에서 확장자 제거하고 TTA 인덱스와 함께 .pt 확장자 추가 (텐서 저장용)
        base_name = os.path.splitext(img_name)[0]
        return os.path.join(self.seed_cache_dir, f"{base_name}_tta_{tta_idx}.pt")
    
    def _get_tta_visual_path(self, img_name, tta_idx):
        """TTA 이미지 시각적 확인용 파일 경로를 반환합니다."""
        if not self.save_augmented_to_disk or self.seed_cache_dir is None:
            return None
        # 파일명에서 확장자 제거하고 TTA 인덱스와 함께 .jpg 확장자 추가 (시각적 확인용)
        base_name = os.path.splitext(img_name)[0]
        return os.path.join(self.seed_cache_dir, f"{base_name}_tta_{tta_idx}.jpg")
    
    def _load_image_from_cache(self, cache_path):
        """캐시 파일에서 이미지를 로드합니다."""
        if cache_path is None or not os.path.exists(cache_path):
            return None
        
        try:
            # 텐서 파일에서 직접 로드 (더 빠름)
            img_tensor = torch.load(cache_path, map_location='cpu')
            
            self.stats['disk_cache_hits'] += 1
            return img_tensor
            
        except Exception as e:
            log.warning(f"Failed to load cached tensor from {cache_path}: {e}")
            self.stats['disk_cache_misses'] += 1
            return None
    
    def _save_augmented_to_disk(self, img_name, aug_idx, img_tensor):
        """증강된 이미지를 디스크에 저장합니다."""
        if not self.save_augmented_to_disk or self.seed_cache_dir is None:
            return
        
        try:
            # 텐서를 직접 저장 (더 빠름)
            cache_path = self._get_augmented_cache_path(img_name, aug_idx)
            if cache_path is not None:
                torch.save(img_tensor.cpu(), cache_path)
                self.stats['augmented_saves'] += 1
            
            # 시각적 확인용 jpg 파일도 저장
            visual_path = self._get_augmented_visual_path(img_name, aug_idx)
            if visual_path is not None:
                # 텐서를 이미지로 변환
                img_array = img_tensor.cpu().numpy()
                
                # [C, H, W] -> [H, W, C] 변환
                img_array_hwc = img_array.transpose(1, 2, 0)
                
                # 정규화 역변환 (ImageNet 평균과 표준편차 사용)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                # 정규화 해제: normalized = (original - mean) / std -> original = normalized * std + mean
                img_array_denorm = img_array_hwc * std + mean
                
                # [0, 1] 범위로 클리핑하고 255를 곱해서 uint8로 변환
                img_array_uint8 = (np.clip(img_array_denorm, 0, 1) * 255).astype(np.uint8)
                
                pil_image = Image.fromarray(img_array_uint8)
                pil_image.save(visual_path, 'JPEG', quality=95)
                
        except Exception as e:
            log.warning(f"Failed to save augmented cache for {img_name}_aug{aug_idx}: {e}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.augmentation_multiplier > 1:
            name, target, aug_idx = self.df[idx]
            # aug_idx를 정수로 변환 (numpy array에서 문자열로 읽힐 수 있음)
            aug_idx = int(aug_idx)
        else:
            name, target = self.df[idx]
            aug_idx = 0
        
        # 증강된 이미지 캐시 확인 (메모리 캐시)
        cache_key = f"{name}_aug{aug_idx}" if aug_idx > 0 else name
        if self.cache_augmented and self.augmented_cache is not None and cache_key in self.augmented_cache:
            img = self.augmented_cache[cache_key]
            self.stats['augmented_cache_hits'] += 1
            target = int(target)
            return img, target
        
        self.stats['augmented_cache_misses'] += 1
        
        # 디스크 캐시 확인 (증강 이미지)
        if aug_idx > 0:
            cache_path = self._get_augmented_cache_path(name, aug_idx)
            cached_img = self._load_image_from_cache(cache_path)
            if cached_img is not None:
                # 메모리 캐시에도 저장
                if self.cache_augmented and self.augmented_cache is not None:
                    self.augmented_cache[cache_key] = cached_img
                target = int(target)
                return cached_img, target
        
        # 캐시에 없으면 이미지 로드 및 증강 처리
        # 원본 이미지 로드
        if self.cache_images and self.image_cache is not None and name in self.image_cache:
            img = self.image_cache[name]
            self.stats['original_cache_hits'] += 1
        else:
            self.stats['original_cache_misses'] += 1
            img = np.array(Image.open(os.path.join(self.path, name)))
            self.stats['disk_loads'] += 1
            
            # 메모리 캐시에 저장
            if self.cache_images and self.image_cache is not None:
                self.image_cache[name] = img
        
        # 증강 적용
        if self.transform:
            if aug_idx > 0:
                # 증강된 데이터: Augraphy + 랜덤 증강 + transform 적용
                self.stats['augmentations'] += 1
                
                # Augraphy 적용 (50% 확률)
                if self.augraphy_pipeline and random.random() < 0.5:
                    try:
                        # img가 numpy array인지 확인
                        if not isinstance(img, np.ndarray):
                            img = np.array(img)
                        # 흑백이면 3채널로 변환
                        if img.ndim == 2:
                            img = np.stack([img]*3, axis=-1)
                        if img.shape[2] == 1:
                            img = np.repeat(img, 3, axis=2)
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)
                        # Augraphy 적용
                        img = self.augraphy_pipeline(img)
                    except Exception as e:
                        log.warning(f"Augraphy failed for {name}: {e}")
                
                # Albumentations transform 적용
                img = self.transform(image=img)['image']
                
                # 증강된 이미지 캐싱
                if self.cache_augmented and self.augmented_cache is not None:
                    self.augmented_cache[cache_key] = img
                
                # 증강된 이미지를 디스크에 저장
                self._save_augmented_to_disk(name, aug_idx, img)
                
            else:
                # 원본 데이터: transform만 적용 (resize, normalize 등, 증강 없음)
                # 원본용 transform 생성 (증강 제외)
                original_transform = A.Compose([
                    A.LongestMaxSize(max_size=320, interpolation=cv2.INTER_AREA),  # img_size 하드코딩
                    A.PadIfNeeded(min_height=320, min_width=320,
                                border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
                img = original_transform(image=img)['image']
        
        # target을 정수로 변환 (numpy array에서 문자열로 읽힐 수 있음)
        target = int(target)
        return img, target
    
    def print_stats(self):
        """캐싱 통계를 출력합니다."""
        # 현재 에포크의 총 요청 수 (증강 캐시 히트 + 미스의 합)
        current_epoch_requests = self.stats['augmented_cache_hits'] + self.stats['augmented_cache_misses']
        
        if current_epoch_requests > 0:
            log.info(f"📊 Dataset Disk Cache Statistics (Memory Cache Disabled):")
            log.info(f"   Current epoch requests: {current_epoch_requests}")
            log.info(f"   Disk cache hits: {self.stats['disk_cache_hits']} ({self.stats['disk_cache_hits']/current_epoch_requests*100:.1f}%)")
            log.info(f"   Disk cache misses: {self.stats['disk_cache_misses']} ({self.stats['disk_cache_misses']/current_epoch_requests*100:.1f}%)")
            log.info(f"   Disk loads: {self.stats['disk_loads']}")
            log.info(f"   Augmented saves: {self.stats['augmented_saves']}")
            log.info(f"   Augmentations applied: {self.stats['augmentations']}")
            
            # 디스크 캐시 효율성 계산
            if self.stats['disk_cache_hits'] > 0:
                disk_cache_efficiency = self.stats['disk_cache_hits'] / current_epoch_requests * 100
                log.info(f"   Disk cache efficiency: {disk_cache_efficiency:.1f}%")
    
    def reset_stats(self):
        """통계를 초기화합니다."""
        self.stats = {
            'original_cache_hits': 0,
            'original_cache_misses': 0,
            'augmented_cache_hits': 0,
            'augmented_cache_misses': 0,
            'disk_cache_hits': 0,
            'disk_cache_misses': 0,
            'disk_loads': 0,
            'augmentations': 0,
            'augmented_saves': 0
        }

# one epoch 학습을 위한 함수입니다.
def train_one_epoch(loader, model, optimizer, loss_fn, device, scaler):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader, desc="Training")
    for image, targets in pbar:
        image = image.to(device, dtype=torch.float32)  # 명시적으로 float32로 변환
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
            image = image.to(device, dtype=torch.float32)  # 명시적으로 float32로 변환
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
            # A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            # A.PadIfNeeded(min_height=img_size, min_width=img_size,
            #               border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 좌우 반전
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(height=img_size, width=img_size),
            # A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            # A.PadIfNeeded(min_height=img_size, min_width=img_size,
            #               border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 상하 반전 (문서 이미지에 유용할 수 있음)
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(height=img_size, width=img_size),
            # A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            # A.PadIfNeeded(min_height=img_size, min_width=img_size,
            #               border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 회전 (약간의 회전)
        A.Compose([
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=1.0),
            A.Resize(height=img_size, width=img_size),
            # A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            # A.PadIfNeeded(min_height=img_size, min_width=img_size,
            #               border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 밝기 조정
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0),
            A.Resize(height=img_size, width=img_size),
            # A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            # A.PadIfNeeded(min_height=img_size, min_width=img_size,
            #               border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]

# Test Time Augmentation을 위한 예측 함수 (tst_dataset 사용)
def predict_with_tta(model, dataset, device, img_size, current_seed=None):
    """진짜 TTA를 적용한 예측 함수 - dataset을 사용"""
    model.eval()
    predictions = []
    
    # TTA transforms 가져오기 (validation과 동일)
    tta_transforms = get_val_tta_transforms(img_size)
    
    # 시드별 TTA 캐시 디렉토리 설정
    if current_seed is not None:
        data_dir = os.path.dirname(os.path.dirname(dataset.path))  # input/data/train/ -> input/data/
        train_cache_dir = os.path.join(data_dir, 'train_cache')
        # img_size를 포함한 캐시 디렉토리명 생성
        cache_dir_name = f'img{img_size}_seed{current_seed}' if img_size is not None else f'seed_{current_seed}'
        seed_cache_dir = os.path.join(train_cache_dir, cache_dir_name)
        os.makedirs(seed_cache_dir, exist_ok=True)
    else:
        seed_cache_dir = None
    
    def get_tta_cache_path(img_name, tta_idx):
        """TTA 이미지 캐시 파일 경로를 반환합니다."""
        if seed_cache_dir is None:
            return None
        base_name = os.path.splitext(img_name)[0]
        return os.path.join(seed_cache_dir, f"{base_name}_tta_{tta_idx}.pt")
    
    def get_tta_visual_path(img_name, tta_idx):
        """TTA 이미지 시각적 확인용 파일 경로를 반환합니다."""
        if seed_cache_dir is None:
            return None
        base_name = os.path.splitext(img_name)[0]
        return os.path.join(seed_cache_dir, f"{base_name}_tta_{tta_idx}.jpg")
    
    def load_tta_from_cache(cache_path):
        """캐시된 TTA 이미지를 로드합니다."""
        if cache_path is None or not os.path.exists(cache_path):
            return None
        
        try:
            img = Image.open(cache_path)
            img_array = np.array(img)
            
            # 정규화 적용 (ImageNet 평균과 표준편차 사용)
            img_array = img_array.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = (img_array - mean) / std
            
            # [H, W, C] -> [C, H, W] 변환 후 텐서로 변환
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
            return img_tensor
            
        except Exception as e:
            log.warning(f"Failed to load cached TTA image from {cache_path}: {e}")
            return None
    
    def save_tta_to_cache(img_name, tta_idx, img_tensor):
        """TTA 이미지를 캐시에 저장합니다."""
        if seed_cache_dir is None:
            return
        
        try:
            cache_path = get_tta_cache_path(img_name, tta_idx)
            if cache_path is not None:
                # 텐서를 이미지로 변환
                img_array = img_tensor.cpu().numpy()
                
                # [C, H, W] -> [H, W, C] 변환
                img_array_hwc = img_array.transpose(1, 2, 0)
                
                # 정규화 역변환 (ImageNet 평균과 표준편차 사용)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                # 정규화 해제: normalized = (original - mean) / std -> original = normalized * std + mean
                img_array_denorm = img_array_hwc * std + mean
                
                # [0, 1] 범위로 클리핑하고 255를 곱해서 uint8로 변환
                img_array_uint8 = (np.clip(img_array_denorm, 0, 1) * 255).astype(np.uint8)
                
                pil_image = Image.fromarray(img_array_uint8)
                pil_image.save(cache_path, 'JPEG', quality=95)
                
        except Exception as e:
            log.warning(f"Failed to save TTA cache for {img_name}_tta{tta_idx}: {e}")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Real TTA Prediction"):
            # 원본 이미지를 파일에서 직접 로드 (transform 없이)
            if len(dataset.df[idx]) == 3:
                img_name, _, _ = dataset.df[idx]  # augmentation_multiplier > 1인 경우
            else:
                img_name, _ = dataset.df[idx]  # augmentation_multiplier == 1인 경우
            img_path = os.path.join(dataset.path, img_name)
            original_img = np.array(Image.open(img_path))
            
            # 각 TTA transform 적용하여 예측
            all_preds = []
            
            for tta_idx, transform in enumerate(tta_transforms):
                # 캐시된 TTA 이미지 확인
                cache_path = get_tta_cache_path(img_name, tta_idx)
                cached_img = load_tta_from_cache(cache_path)
                
                if cached_img is not None:
                    # 캐시된 이미지 사용
                    transformed_img = cached_img.unsqueeze(0).to(device)
                else:
                    # 매번 원본 이미지에서 다른 변형 적용
                    transformed_img_tensor = transform(image=original_img)['image']
                    transformed_img = transformed_img_tensor.unsqueeze(0).to(device)
                    
                    # 캐시에 저장
                    save_tta_to_cache(img_name, tta_idx, transformed_img_tensor)
                
                with autocast():
                    # 입력 텐서를 float32로 명시적으로 변환
                    transformed_img = transformed_img.float()
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
# 현재 스크립트 위치를 기준으로 절대 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'input', 'data')
log.info(f"📂 Data path: {data_path}")

# model config
model_name = 'efficientnetv2_rw_m'  # 더 좋은 모델 사용

# training config
img_size = 320  # 이미지 크기 대폭 확대
LR = 1e-3  # 더 낮은 학습률
EPOCHS = 100  # early stopping을 위해 더 많은 epoch 설정
BATCH_SIZE = 16  # 큰 모델에 맞춰 배치 크기 조정
num_workers = 0
weight_decay = 1e-4
label_smoothing = 0.1
patience = 10  # early stopping patience

# Data augmentation config
AUGMENTATION_MULTIPLIER = 10 # 데이터 증강 배수 (기본값: 2배)
# 예: AUGMENTATION_MULTIPLIER = 3이면 원본 데이터 1개당 증강된 데이터 2개가 추가되어 총 3배가 됩니다.
# 원본 데이터는 그대로 유지되고, 추가로 증강된 데이터만 생성됩니다.

# K-Fold config
K_FOLDS = 5

"""## Data Preparation
* 데이터 로드 및 transform 정의
"""

# 강화된 augmentation을 위한 transform 코드
trn_transform = A.Compose([
    # 다양한 데이터 증강 기법들
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    # A.RandomRotate90(p=0.5),
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    # A.Blur(blur_limit=3, p=0.1),
    # A.CLAHE(clip_limit=2.0, p=0.2),

    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
    # A.OneOf([
    #     A.Blur(blur_limit=2),
    #     A.MotionBlur(blur_limit=5),
    #     A.Defocus(radius=(1, 3)),
    # ], p=0.5),
    A.GaussNoise(var_limit=(0.0000002, 0.000001), mean=0, p=0.3),  # 은은한 미세 노이즈
    # A.OneOf([
    #     A.GaussNoise(var_limit=(0.0000002, 0.000001), mean=0, p=1.0),  # 은은한 미세 노이즈
    #     # A.ImageCompression(quality_lower=40, quality_upper=60, p=1.0),  # 압축으로 전체적 얼룩
    #     # A.CoarseDropout(max_holes=8, max_height=img_size//10, max_width=img_size//10),
    # ], p=0.3),
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # A.OneOf([
    #     A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
    #     A.GridDropout(ratio=0.5, p=0.5),
    # ], p=0.4),
    
    A.Resize(height=img_size, width=img_size),
    # A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
    # A.PadIfNeeded(min_height=img_size, min_width=img_size,
    #                 border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),    
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def get_augraphy_pipeline():
    """Augraphy 파이프라인을 반환합니다."""
    if not AUGRAPHY_AVAILABLE:
        return None
    
    return AugraphyPipeline(
        ink_phase=[
            InkBleed(p=0.3), # 잉크 번짐
            BleedThrough(p=0.3), # 뒷면 잉크 비침
        ],
        paper_phase=[
            ColorPaper(p=0.3), # 종이 색상 변경
            OneOf([
                NoiseTexturize( # 테스트 데이터랑 비슷한 노이즈
                    sigma_range=(5, 15),
                    turbulence_range=(3, 9),
                    texture_width_range=(50, 500),
                    texture_height_range=(50, 500),
                    p=0.6
                ),
                SubtleNoise(
                    subtle_range=50,
                    p=0.4
                )
            ], p=0.3),
        ],
        post_phase=[
            LightingGradient( # 조명 그라데이션
                light_position=None,
                direction=90,
                max_brightness=255,
                min_brightness=0,
                mode="gaussian",
                transparency=0.5,
                p=0.3
            ),
            ShadowCast( # 그림자
                shadow_side=random.choice(["top", "bottom", "left", "right"]), # 그림자 위치
                shadow_vertices_range=(2, 3),
                shadow_width_range=(0.5, 0.8),
                shadow_height_range=(0.5, 0.8),
                shadow_color=(0, 0, 0),
                shadow_opacity_range=(0.5, 0.6),
                shadow_iterations_range=(1, 2),
                shadow_blur_kernel_range=(101, 301),
                p=0.3
            ),
        ],
    )

# validation transform (train과 동일한 증강 적용)
val_transform = trn_transform

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
# 초기 시드 설정을 위해 여기서는 skf 정의하지 않고 나중에 시드별로 정의

# 시드별 결과 저장
all_seed_scores = []
model_paths = []  # 모델 파일 경로를 저장

log.info(f"\n🔄 Starting {K_FOLDS}-Fold Cross Validation with {len(SEEDS)} seeds...")

# 모델 저장 경로 설정
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)
log.info(f"📂 Models will be saved to: {models_dir}")

# 시드별 결과 저장
all_seed_scores = []
model_paths = []  # 모델 파일 경로를 저장

# 시드별 학습 루프
for seed_idx, seed in enumerate(SEEDS):
    log.info(f"\n{'='*80}")
    log.info(f"🌱 SEED {seed_idx+1}/{len(SEEDS)}: {seed}")
    log.info(f"{'='*80}")
    
    # 시드 재설정
    set_seed(seed)
    
    # 이 시드에 대한 K-Fold 정의
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    
    # 폴드별 결과 저장
    fold_scores = []
    seed_model_paths = []  # 이 시드의 모델 파일 경로들
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
        log.info(f"\n{'='*60}")
        log.info(f"🔥 SEED {seed} | FOLD {fold+1}/{K_FOLDS}")
        log.info(f"{'='*60}")
        
        # 폴드별 데이터 분할
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_df.iloc[val_idx].reset_index(drop=True)
        
        log.info(f"📚 Train samples: {len(train_fold)}")
        log.info(f"📝 Validation samples: {len(val_fold)}")
        
        # 데이터셋 생성 (메모리 캐시 비활성화, 디스크 캐시만 사용)
        trn_dataset = ImageDataset(train_fold, f"{data_path}/train/", transform=trn_transform, cache_images=False, cache_augmented=False, augmentation_multiplier=AUGMENTATION_MULTIPLIER, save_augmented_to_disk=True, current_seed=seed, img_size=img_size)
        val_dataset = ImageDataset(val_fold, f"{data_path}/train/", transform=val_transform, cache_images=False, cache_augmented=False, augmentation_multiplier=AUGMENTATION_MULTIPLIER, save_augmented_to_disk=True, current_seed=seed, img_size=img_size)
        
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
        
        # 모델을 float32로 명시적으로 변환
        model = model.float()
        
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
            
            # 매 에포크마다 캐싱 통계 출력 (디스크 캐시만)
            log.info(f"\n📊 Epoch {epoch+1} Disk Cache Statistics:")
            trn_dataset.print_stats()
            val_dataset.print_stats()
        
        # 폴드 완료 후 모델 저장
        # Early stopping이 활성화된 경우 best_weights를 사용, 그렇지 않으면 기존 방식 사용
        if early_stopping.best_weights is not None:
            final_f1 = early_stopping.best_score
            final_model_state = early_stopping.best_weights
            log.info(f"\n🎯 Seed {seed} | Fold {fold+1} completed with early stopping! Best F1: {final_f1:.4f}")
        else:
            final_f1 = best_val_f1
            final_model_state = best_model_state if best_model_state is not None else model.state_dict()
            log.info(f"\n🎯 Seed {seed} | Fold {fold+1} completed! Best F1: {final_f1:.4f}")
        
        fold_scores.append(final_f1)
        
        # 모델을 디스크에 저장
        model_filename = f"model_seed{seed}_fold{fold+1}.pth"
        model_path = os.path.join(models_dir, model_filename)
        torch.save(final_model_state, model_path)
        seed_model_paths.append(model_path)
        log.info(f"💾 Model saved to: {model_path}")
        
        # 메모리 정리 (모델 상태 삭제)
        del model, optimizer, scheduler, scaler, trn_loader, val_loader
        del trn_dataset, val_dataset, train_fold, val_fold
        del best_model_state, final_model_state
        if early_stopping.best_weights is not None:
            del early_stopping.best_weights
        torch.cuda.empty_cache()
        gc.collect()
        
        log.info(f"🧹 Memory cleaned for Seed {seed} | Fold {fold+1}")
    
    # 시드별 결과 저장
    all_seed_scores.append(fold_scores)
    model_paths.append(seed_model_paths)
    
    # 시드별 결과 요약
    log.info(f"\n{'='*60}")
    log.info(f"📊 SEED {seed} K-FOLD RESULTS")
    log.info(f"{'='*60}")
    
    seed_mean_score = np.mean(fold_scores)
    seed_std_score = np.std(fold_scores)
    
    log.info(f"📈 Seed {seed} fold scores: {[f'{score:.4f}' for score in fold_scores]}")
    log.info(f"🎯 Seed {seed} Mean F1 Score: {seed_mean_score:.4f}")
    log.info(f"📏 Seed {seed} Standard Deviation: {seed_std_score:.4f}")
    log.info(f"📊 Seed {seed} Score Range: {seed_mean_score:.4f} ± {seed_std_score:.4f}")
    log.info(f"⬇️ Seed {seed} Min Score: {min(fold_scores):.4f}")
    log.info(f"⬆️ Seed {seed} Max Score: {max(fold_scores):.4f}")
    log.info(f"💾 Seed {seed} model paths: {seed_model_paths}")

# 전체 시드별 결과 요약
log.info(f"\n{'='*80}")
log.info(f"📊 ALL SEEDS K-FOLD CROSS VALIDATION RESULTS")
log.info(f"{'='*80}")

# 모든 시드의 모든 폴드 점수를 하나의 리스트로 만들기
all_scores = []
for seed_idx, seed_scores in enumerate(all_seed_scores):
    all_scores.extend(seed_scores)

overall_mean_score = np.mean(all_scores)
overall_std_score = np.std(all_scores)

log.info(f"📈 Total models trained: {len(all_scores)} (Seeds: {len(SEEDS)}, Folds per seed: {K_FOLDS})")
log.info(f"🎯 Overall Mean F1 Score: {overall_mean_score:.4f}")
log.info(f"📏 Overall Standard Deviation: {overall_std_score:.4f}")
log.info(f"📊 Overall Score Range: {overall_mean_score:.4f} ± {overall_std_score:.4f}")
log.info(f"⬇️ Overall Min Score: {min(all_scores):.4f}")
log.info(f"⬆️ Overall Max Score: {max(all_scores):.4f}")

# 시드별 평균 점수 출력
for seed_idx, (seed, seed_scores) in enumerate(zip(SEEDS, all_seed_scores)):
    seed_mean = np.mean(seed_scores)
    log.info(f"🌱 Seed {seed} average: {seed_mean:.4f}")

# 모든 모델 파일 경로 출력
all_model_paths = []
for seed_paths in model_paths:
    all_model_paths.extend(seed_paths)
log.info(f"💾 Total saved models: {len(all_model_paths)}")

"""## Ensemble Inference & Save File
* 모든 시드의 모든 폴드 모델을 앙상블하여 테스트 이미지에 대한 추론을 진행합니다.
"""

log.info(f"\n🚀 Starting Ensemble Prediction with {len(all_scores)} models...")

# 테스트 데이터셋 생성 (메모리 캐시 비활성화)
tst_dataset = ImageDataset(
    f"{data_path}/sample_submission.csv",
    f"{data_path}/test/",
    transform=None,  # TTA에서 원본 이미지를 직접 변형하므로 여기서는 None
    cache_images=False,  # 메모리 캐시 비활성화
    cache_augmented=False,  # 메모리 캐시 비활성화
    augmentation_multiplier=1,  # 테스트는 증강 적용하지 않음
    save_augmented_to_disk=False,  # 테스트는 증강하지 않으므로 저장하지 않음
    current_seed=None,  # 테스트 데이터는 시드 무관
    img_size=img_size
)

# 앙상블을 위해 저장된 모든 모델을 순차적으로 로드하여 예측
log.info(f"🔮 Running ensemble prediction with {len(all_model_paths)} models and TTA...")

# 모든 모델의 예측 결과를 저장할 리스트
all_predictions = []

# 각 모델을 순차적으로 로드하여 예측
for i, model_path in enumerate(all_model_paths):
    log.info(f"🔮 Loading and predicting with model {i+1}/{len(all_model_paths)}: {os.path.basename(model_path)}")
    
    # 모델 파일명에서 시드 추출 (예: model_seed42_fold1.pth -> 42)
    model_filename = os.path.basename(model_path)
    if 'seed' in model_filename:
        try:
            seed_part = model_filename.split('_')[1]  # 'seed42' 부분
            current_seed = int(seed_part.replace('seed', ''))  # 42
        except:
            current_seed = None
    else:
        current_seed = None
    
    # 모델 로드
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17
    ).to(device)
    
    # 모델을 float32로 명시적으로 변환
    model = model.float()
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 예측 수행 (해당 모델의 시드를 사용하여 TTA 캐시 활용)
    fold_predictions = predict_with_tta(model, tst_dataset, device, img_size, current_seed=current_seed)
    all_predictions.append(fold_predictions)
    
    # 메모리 정리
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    log.info(f"✅ Completed prediction with {os.path.basename(model_path)} (seed: {current_seed})")

# 모든 모델의 예측을 평균하여 최종 예측 생성
log.info(f"🎯 Averaging predictions from {len(all_predictions)} models...")
ensemble_predictions = np.mean(all_predictions, axis=0)
final_predictions = np.argmax(ensemble_predictions, axis=1)

# 결과 저장
# tst_dataset.df의 구조에 따라 적절한 컬럼 선택
if len(tst_dataset.df[0]) == 3:
    # augmentation_multiplier > 1인 경우: (name, target, aug_idx)
    pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target', 'aug_idx'])
    pred_df = pred_df[['ID', 'target']]  # aug_idx 컬럼 제거
else:
    # augmentation_multiplier == 1인 경우: (name, target)
    pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])

pred_df['target'] = final_predictions

sample_submission_df = pd.read_csv(f"{data_path}/sample_submission.csv")
assert (sample_submission_df['ID'] == pred_df['ID']).all()

output_path = "./output"
os.makedirs(output_path, exist_ok=True)
pred_df.to_csv(f"{output_path}/pred_advanced_kfold_tta2_efnv2rwm_v3_augx2_saveimg_seed_ensemble.csv", index=False)

log.info(f"\n✅ Ensemble prediction completed and saved to {output_path}/pred_advanced_kfold_tta2_efnv2rwm_v3_augx2_saveimg_seed_ensemble.csv")
log.info(f"📈 Final Overall CV Score: {overall_mean_score:.4f} ± {overall_std_score:.4f}")
log.info(f"🎯 Used {len(all_model_paths)} models for ensemble prediction")
log.info(f"💾 All model files saved in: {models_dir}")
log.info(f"🎨 Augmented images cached in: {os.path.join(data_path, 'train_cache')}")

# 캐시 통계 출력
log.info(f"\n📊 Final Cache Statistics:")
train_cache_path = os.path.join(data_path, 'train_cache')
if os.path.exists(train_cache_path):
    # 새로운 디렉토리명 패턴: img{img_size}_seed{seed}
    cache_dir_pattern = f'img{img_size}_seed'
    total_cache_dirs = len([d for d in os.listdir(train_cache_path) if d.startswith(cache_dir_pattern)])
    log.info(f"   Total seed cache directories: {total_cache_dirs}")
    for seed in SEEDS:
        seed_cache_dir = os.path.join(train_cache_path, f'img{img_size}_seed{seed}')
        if os.path.exists(seed_cache_dir):
            cache_files = len([f for f in os.listdir(seed_cache_dir) if f.endswith('.jpg')])
            log.info(f"   Seed {seed} cached images: {cache_files}")
        else:
            log.info(f"   Seed {seed} cache directory not found: {seed_cache_dir}")
else:
    log.info(f"   Train cache directory not found: {train_cache_path}")

# 메모리 정리
torch.cuda.empty_cache()
gc.collect()

pred_df.head() 