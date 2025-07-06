# -*- coding: utf-8 -*-
"""
데이터 증강 관련 기능들을 담은 모듈
- Albumentations 변환
- Augraphy 변환
- 혼합 증강
- TTA (Test Time Augmentation)
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
import random
from typing import Optional, Union, List, Tuple, Dict, Any
import warnings

# Augraphy 라이브러리 import (선택적)
try:
    import augraphy  # type: ignore
    from augraphy import *  # type: ignore
    AUGRAPHY_AVAILABLE = True
except ImportError:
    AUGRAPHY_AVAILABLE = False
    warnings.warn("Augraphy 라이브러리를 사용할 수 없습니다. albumentations만 사용됩니다.")

import log_util as log


class AugmentationConfig:
    """증강 설정을 관리하는 클래스"""
    def __init__(self, cfg):
        self.cfg = cfg.augmentation
        self.enabled = cfg.augmentation.enabled
        self.library = cfg.augmentation.library  # "albumentations", "augraphy", "mixed", "none"
        self.img_size = cfg.data.img_size
        
        # 각 단계별 증강 설정
        self.train_enabled = cfg.augmentation.train.enabled
        self.train_num_augmented_images = cfg.augmentation.train.num_augmented_images
        
        self.valid_enabled = cfg.augmentation.valid.enabled
        self.valid_num_augmented_images = cfg.augmentation.valid.num_augmented_images
        
        self.valid_tta_enabled = cfg.augmentation.valid_tta.enabled
        self.test_tta_enabled = cfg.augmentation.test_tta.enabled


def get_albumentations_augmentation(cfg: AugmentationConfig, is_train: bool = True) -> A.Compose:
    """Albumentations 기반 증강 변환 반환"""
    img_size = cfg.img_size
    
    if is_train:
        # 훈련용 증강
        augmentation = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.Blur(blur_limit=3, p=1.0),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=1.0),
                A.GridDistortion(p=1.0),
                A.ElasticTransform(p=1.0),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # 검증/테스트용 기본 변환
        augmentation = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return augmentation


def get_augraphy_augmentation(cfg: AugmentationConfig, is_train: bool = True) -> Optional[Any]:
    """Augraphy 기반 증강 변환 반환"""
    if not AUGRAPHY_AVAILABLE:
        log.warning("Augraphy가 사용 불가능합니다. albumentations로 대체됩니다.")
        return None
    
    if is_train:
        # 훈련용 증강 (문서 이미지에 특화)
        augmentation = augraphy.AugraphyPipeline(
            [
                # 잉크 관련 증강
                augraphy.InkBleed(p=0.3),
                augraphy.BleedThrough(p=0.3),
                
                # 노이즈 관련 증강
                augraphy.NoiseTexturize(p=0.3),
                augraphy.GaussianNoise(p=0.3),
                
                # 기하학적 변형
                augraphy.Geometric(p=0.3),
                
                # 페이지 관련 증강
                augraphy.PageBorder(p=0.3),
                augraphy.DirtyRollers(p=0.3),
                
                # 밝기/대비 조정
                augraphy.Brightness(p=0.3),
                augraphy.BrightnessTexturize(p=0.3),
            ]
        )
    else:
        # 검증/테스트용 - 기본 변환만
        augmentation = None
    
    return augmentation


def get_tta_transforms(cfg: AugmentationConfig) -> List[A.Compose]:
    """TTA용 변환들을 반환"""
    img_size = cfg.img_size
    
    tta_transforms = [
        # 원본
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 수평 뒤집기
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 수직 뒤집기
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 회전 (15도)
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Rotate(limit=15, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 밝기 조정
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    
    return tta_transforms


def apply_augraphy_transform(image: np.ndarray, augmentation: Any) -> np.ndarray:
    """Augraphy 변환을 이미지에 적용"""
    if augmentation is None:
        return image
    
    try:
        # Augraphy는 PIL 이미지나 numpy 배열을 사용
        augmented = augmentation(image)
        return augmented
    except Exception as e:
        log.warning(f"Augraphy 변환 실패: {e}")
        return image


class AugmentedDataset(Dataset):
    """증강된 데이터셋 클래스"""
    def __init__(self, base_dataset: Dataset, cfg: AugmentationConfig, is_train: bool = True):
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.is_train = is_train
        
        # 전역 증강 설정 확인
        if not cfg.enabled:
            # 증강 비활성화 시 원본 데이터셋과 동일하게 처리
            self.num_augmented_images = 1
        else:
            # train/valid에 따라 적절한 증강 이미지 개수 선택
            if is_train:
                self.num_augmented_images = cfg.train_num_augmented_images
            else:
                self.num_augmented_images = cfg.valid_num_augmented_images
        
        # 증강 변환 준비
        self.setup_augmentations()
        
        # 데이터셋 크기 계산
        self.base_size = len(base_dataset)  # type: ignore
        self.total_size = self.base_size * self.num_augmented_images
        
        stage_name = "훈련" if is_train else "검증"
        if not cfg.enabled:
            log.info(f"{stage_name} 데이터셋 생성: 원본 {self.base_size}개 -> 총 {self.total_size}개 (증강 비활성화)")
        else:
            log.info(f"{stage_name} 증강된 데이터셋 생성: 원본 {self.base_size}개 -> 총 {self.total_size}개 (증강 x{self.num_augmented_images})")
    
    def setup_augmentations(self):
        """증강 변환 설정"""
        if not self.cfg.enabled:
            self.augmentations = None
            return
        
        library = self.cfg.library
        
        if library == "albumentations":
            self.augmentations = [get_albumentations_augmentation(self.cfg, self.is_train)]
        elif library == "augraphy":
            aug = get_augraphy_augmentation(self.cfg, self.is_train)
            if aug is not None:
                self.augmentations = [aug]
            else:
                # Augraphy 실패 시 albumentations로 대체
                self.augmentations = [get_albumentations_augmentation(self.cfg, self.is_train)]
        elif library == "mixed":
            # 두 라이브러리 혼합
            albu_aug = get_albumentations_augmentation(self.cfg, self.is_train)
            augra_aug = get_augraphy_augmentation(self.cfg, self.is_train)
            
            if augra_aug is not None:
                self.augmentations = [albu_aug, augra_aug]
            else:
                self.augmentations = [albu_aug]
        else:
            self.augmentations = None
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # 원본 인덱스와 증강 인덱스 계산
        base_idx = idx % self.base_size
        aug_idx = idx // self.base_size
        
        # 원본 데이터 가져오기
        image, target = self.base_dataset[base_idx]
        
        # 증강 적용
        if self.augmentations is not None and len(self.augmentations) > 0:
            # 증강 방법 선택
            if self.cfg.library == "mixed":
                # 혼합 모드에서는 랜덤하게 선택
                aug_method = random.choice(self.augmentations)
            else:
                aug_method = self.augmentations[0]
            
            # 증강 적용
            if isinstance(aug_method, A.Compose):
                # Albumentations
                if isinstance(image, torch.Tensor):
                    # 이미 tensor인 경우 numpy로 변환
                    image_np = image.permute(1, 2, 0).numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    augmented = aug_method(image=image_np)
                    image = augmented['image']
                else:
                    # numpy array인 경우
                    augmented = aug_method(image=image)
                    image = augmented['image']
            else:
                # Augraphy
                if isinstance(image, torch.Tensor):
                    image_np = image.permute(1, 2, 0).numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    image_np = apply_augraphy_transform(image_np, aug_method)
                    
                    # 다시 tensor로 변환
                    basic_transform = A.Compose([
                        A.Resize(height=self.cfg.img_size, width=self.cfg.img_size),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ])
                    image = basic_transform(image=image_np)['image']
                else:
                    image = apply_augraphy_transform(image, aug_method)
        
        return image, target


class TTADataset(Dataset):
    """TTA용 데이터셋 클래스"""
    def __init__(self, base_dataset: Dataset, cfg: AugmentationConfig):
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.tta_transforms = get_tta_transforms(cfg)
        self.num_tta = len(self.tta_transforms)
        
        # 데이터셋 크기 계산
        self.base_size = len(base_dataset)  # type: ignore
        self.total_size = self.base_size * self.num_tta
        
        log.info(f"TTA 데이터셋 생성: 원본 {self.base_size}개 -> 총 {self.total_size}개 (TTA x{self.num_tta})")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # 원본 인덱스와 TTA 인덱스 계산
        # 한 이미지의 모든 TTA 변환을 연속으로 배치
        base_idx = idx // self.num_tta
        tta_idx = idx % self.num_tta
        
        # 원본 데이터 가져오기
        image, target = self.base_dataset[base_idx]
        
        # TTA 변환 적용
        transform = self.tta_transforms[tta_idx]
        
        if isinstance(image, torch.Tensor):
            # 이미 tensor인 경우 numpy로 변환
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            augmented = transform(image=image_np)
            image = augmented['image']
        else:
            # numpy array인 경우
            augmented = transform(image=image)
            image = augmented['image']
        
        return image, target, base_idx  # base_idx 추가로 원본 인덱스 추적


def create_augmented_dataset(base_dataset: Dataset, cfg: AugmentationConfig, is_train: bool = True) -> Dataset:
    """증강된 데이터셋 생성"""
    if not cfg.enabled:
        return base_dataset
    
    # 단계별 증강 확인
    if is_train and not cfg.train_enabled:
        return base_dataset
    elif not is_train and not cfg.valid_enabled:
        return base_dataset
    
    return AugmentedDataset(base_dataset, cfg, is_train)


def create_tta_dataset(base_dataset: Dataset, cfg: AugmentationConfig) -> Dataset:
    """TTA 데이터셋 생성"""
    if not cfg.enabled:
        return base_dataset
    
    return TTADataset(base_dataset, cfg) 