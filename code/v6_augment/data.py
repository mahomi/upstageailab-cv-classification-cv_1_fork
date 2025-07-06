# -*- coding: utf-8 -*-
"""
데이터 관련 기능들을 담은 모듈
- 데이터셋 클래스들
- 데이터 로딩 및 전처리
- Transform 정의
- 데이터 분할 로직
"""

import os
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from augmentations import AugmentationConfig, create_augmented_dataset, create_tta_dataset


class ImageDataset(Dataset):
    """기본 이미지 데이터셋 클래스"""
    def __init__(self, csv, path, transform=None):
        if isinstance(csv, str):
            self.df = pd.read_csv(csv).values
        else:
            self.df = csv.values
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


class IndexedImageDataset(Dataset):
    """인덱스 기반 이미지 데이터셋 클래스"""
    def __init__(self, df, path, transform=None):
        self.df = df
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df.iloc[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target


def get_transforms(img_size):
    """이미지 변환을 위한 transform들을 반환"""
    # 훈련용 transform
    train_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 테스트용 transform
    test_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, test_transform


def prepare_data_loaders(cfg, seed):
    """설정에 따라 데이터 로더들을 준비"""
    data_path = cfg.data.data_path
    img_size = cfg.data.img_size
    batch_size = cfg.training.batch_size
    num_workers = cfg.data.num_workers
    
    # Transform 준비
    train_transform, test_transform = get_transforms(img_size)
    
    # 증강 설정 준비
    aug_config = AugmentationConfig(cfg)
    
    # 전체 훈련 데이터 로드
    full_train_df = pd.read_csv(f"{data_path}/train.csv")
    
    # 테스트 데이터 로더 준비
    test_dataset = ImageDataset(
        f"{data_path}/sample_submission.csv",
        f"{data_path}/test/",
        transform=test_transform
    )
    
    # 테스트 TTA 적용 (필요한 경우)
    if aug_config.test_tta_enabled:
        test_dataset = create_tta_dataset(test_dataset, aug_config)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 검증 전략에 따른 데이터 분할
    validation_strategy = cfg.validation.strategy
    
    if validation_strategy == "holdout":
        train_loader, val_loader = _prepare_holdout_loaders(
            cfg, full_train_df, data_path, train_transform, test_transform, seed, aug_config
        )
        return train_loader, val_loader, test_loader, None
        
    elif validation_strategy == "kfold":
        folds = _prepare_kfold_splits(cfg, full_train_df, seed)
        return None, None, test_loader, (folds, full_train_df, data_path, train_transform, test_transform, aug_config)
        
    elif validation_strategy == "none":
        train_dataset = IndexedImageDataset(
            full_train_df, 
            f"{data_path}/train/", 
            transform=train_transform
        )
        
        # 훈련 데이터 증강 적용
        if aug_config.train_enabled:
            train_dataset = create_augmented_dataset(train_dataset, aug_config, is_train=True)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=False
        )
        return train_loader, None, test_loader, None
        
    else:
        raise ValueError(f"Unknown validation strategy: {validation_strategy}")


def _prepare_holdout_loaders(cfg, full_train_df, data_path, train_transform, test_transform, seed, aug_config):
    """Holdout 검증을 위한 데이터 로더 준비"""
    train_ratio = cfg.validation.holdout.train_ratio
    stratify = cfg.validation.holdout.stratify
    batch_size = cfg.training.batch_size
    num_workers = cfg.data.num_workers
    
    if stratify:
        train_df, val_df = train_test_split(
            full_train_df, 
            test_size=1-train_ratio, 
            stratify=full_train_df['target'], 
            random_state=seed
        )
    else:
        train_df, val_df = train_test_split(
            full_train_df, 
            test_size=1-train_ratio, 
            random_state=seed
        )
    
    # Dataset 정의
    train_dataset = IndexedImageDataset(
        train_df, 
        f"{data_path}/train/", 
        transform=train_transform
    )
    val_dataset = IndexedImageDataset(
        val_df, 
        f"{data_path}/train/", 
        transform=test_transform
    )
    
    # 증강 적용
    if aug_config.train_enabled:
        train_dataset = create_augmented_dataset(train_dataset, aug_config, is_train=True)
    
    if aug_config.valid_enabled:
        val_dataset = create_augmented_dataset(val_dataset, aug_config, is_train=False)
    
    # Valid TTA 적용
    if aug_config.valid_tta_enabled:
        val_dataset = create_tta_dataset(val_dataset, aug_config)
    
    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader


def _prepare_kfold_splits(cfg, full_train_df, seed):
    """K-Fold 교차 검증을 위한 분할 준비"""
    n_splits = cfg.validation.kfold.n_splits
    stratify = cfg.validation.kfold.stratify
    
    if stratify:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(skf.split(full_train_df, full_train_df['target']))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(kf.split(full_train_df))
    
    return folds


def get_kfold_loaders(fold_idx, folds, full_train_df, data_path, train_transform, test_transform, cfg, aug_config=None):
    """특정 fold에 대한 데이터 로더 반환"""
    train_idx, val_idx = folds[fold_idx]
    
    # 현재 fold의 데이터 분할
    train_df = full_train_df.iloc[train_idx]
    val_df = full_train_df.iloc[val_idx]
    
    # Dataset 정의
    train_dataset = IndexedImageDataset(
        train_df, 
        f"{data_path}/train/", 
        transform=train_transform
    )
    val_dataset = IndexedImageDataset(
        val_df, 
        f"{data_path}/train/", 
        transform=test_transform
    )
    
    # 증강 적용 (aug_config가 제공된 경우)
    if aug_config is not None:
        if aug_config.train_enabled:
            train_dataset = create_augmented_dataset(train_dataset, aug_config, is_train=True)
        
        if aug_config.valid_enabled:
            val_dataset = create_augmented_dataset(val_dataset, aug_config, is_train=False)
        
        # Valid TTA 적용
        if aug_config.valid_tta_enabled:
            val_dataset = create_tta_dataset(val_dataset, aug_config)
    
    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True, 
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, train_df, val_df 