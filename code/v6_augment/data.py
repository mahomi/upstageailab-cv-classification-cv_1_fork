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
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import warnings

# numpy matrix 경고 필터링
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*matrix subclass.*")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, message=".*matrix subclass.*")


def _should_use_pin_memory():
    """MPS 환경에서는 pin_memory를 사용하지 않도록 설정"""
    try:
        # MPS가 사용 가능하고 현재 사용 중인 경우 pin_memory=False
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return False
        # CUDA가 사용 가능한 경우 pin_memory=True
        elif torch.cuda.is_available():
            return True
        # CPU만 사용하는 경우 pin_memory=False
        else:
            return False
    except Exception:
        # 안전하게 False 반환
        return False


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


class AugmentedDataset(Dataset):
    """원본 데이터셋을 여러 번 반복하여 증강을 적용하는 래퍼"""

    def __init__(self, base_dataset, num_aug: int = 1, add_org: bool = False,
                 aug_transform=None, org_transform=None):
        # transform 매개변수들이 None인 경우 명시적으로 오류 발생
        if aug_transform is None:
            raise ValueError("aug_transform cannot be None. Please provide a valid transformation.")
        if org_transform is None:
            raise ValueError("org_transform cannot be None. Please provide a valid transformation.")
            
        self.base_dataset = base_dataset
        self.num_aug = max(0, int(num_aug))
        self.add_org = add_org
        self.aug_transform = aug_transform  # 증강 transform
        self.org_transform = org_transform  # 원본 transform (기본 transform)

    def __len__(self):
        if self.add_org:
            return len(self.base_dataset) * (self.num_aug + 1)
        else:
            return len(self.base_dataset) * self.num_aug

    def __getitem__(self, idx):
        base_idx = idx % len(self.base_dataset)
        aug_idx = idx // len(self.base_dataset)

        # base_dataset에서 원본 이미지 로드 (transform 없이)
        original_image, target = self.base_dataset[base_idx]

        # 원본 이미지 처리 (마지막 패스)
        if self.add_org and aug_idx == self.num_aug:
            image = self.org_transform(image=original_image)['image']
            return image, target, base_idx

        # 증강 이미지 처리
        image = self.aug_transform(image=original_image)['image']
        return image, target, base_idx


def _build_augmented_dataset(
    df: pd.DataFrame,
    path: str,
    transform,
    org_transform,
    aug_count: int,
    add_org: bool,
) -> Dataset:
    """Create dataset with optional augmentation."""

    base_ds = IndexedImageDataset(df, path, transform=None)
    if aug_count > 0:
        return AugmentedDataset(
            base_ds,
            aug_count,
            add_org,
            aug_transform=transform,
            org_transform=org_transform,
        )
    return IndexedImageDataset(df, path, transform=transform)


def _create_augraphy_lambda(intensity: float, ops: list[str] | None = None):
    """Create an Albumentations Lambda applying selected Augraphy transforms."""
    try:
        from augraphy import AugraphyPipeline
        from augraphy.augmentations import (
            BleedThrough,
            Brightness,
            BrightnessTexturize,
            ColorPaper,
            ColorShift,
            DirtyDrum,
            DirtyRollers,
            Geometric,
            InkBleed,
            InkMottling,
            InkShifter,
            Jpeg,
            LightingGradient,
            LinesDegradation,
            LowInkPeriodicLines,
            LowInkRandomLines,
            Markup,
            NoiseTexturize,
            PageBorder,
            Scribbles,
            ShadowCast,
            Stains,
            SubtleNoise,
        )

        all_ops: dict[str, tuple[str, object]] = {
            "ink_bleed": ("ink", InkBleed(p=0.2 * intensity)),
            "ink_mottling": ("ink", InkMottling(p=0.2 * intensity)),
            "ink_shifter": ("ink", InkShifter(p=0.2 * intensity)),
            "low_ink_random_lines": ("ink", LowInkRandomLines(p=0.2 * intensity)),
            "low_ink_periodic_lines": ("ink", LowInkPeriodicLines(p=0.2 * intensity)),
            "color_paper": ("paper", ColorPaper(p=0.3 * intensity)),
            "brightness_texturize": ("paper", BrightnessTexturize(p=0.3 * intensity)),
            "dirty_rollers": ("paper", DirtyRollers(p=0.2 * intensity)),
            "dirty_drum": ("paper", DirtyDrum(p=0.2 * intensity)),
            "stains": ("paper", Stains(p=0.3 * intensity)),
            "page_border": ("paper", PageBorder(p=0.2 * intensity)),
            "geometric": ("post", Geometric(rotate_range=(-15 * intensity, 15 * intensity), p=0.5 * intensity)),
            "lighting_gradient": ("post", LightingGradient(p=0.3 * intensity)),
            "brightness": ("post", Brightness(brightness_range=(1 - 0.3 * intensity, 1 + 0.3 * intensity), p=0.5 * intensity)),
            "color_shift": ("post", ColorShift(p=0.3 * intensity)),
            "noise_texturize": ("post", NoiseTexturize(p=0.3 * intensity)),
            "subtle_noise": ("post", SubtleNoise(p=0.3 * intensity)),
            "shadow_cast": ("post", ShadowCast(p=0.2 * intensity)),
            "lines_degradation": ("post", LinesDegradation(p=0.2 * intensity)),
            "bleed_through": ("post", BleedThrough(p=0.1 * intensity)),
            "markup": ("post", Markup(p=0.1 * intensity)),
            "scribbles": ("post", Scribbles(p=0.1 * intensity)),
            "jpeg": ("post", Jpeg(p=0.2 * intensity)),
        }

        if not ops or ops == ["all"]:
            ops = list(all_ops.keys())

        ink_phase = [aug for name, (phase, aug) in all_ops.items()
                     if phase == "ink" and name in ops]
        paper_phase = [aug for name, (phase, aug) in all_ops.items()
                       if phase == "paper" and name in ops]
        post_phase = [aug for name, (phase, aug) in all_ops.items()
                      if phase == "post" and name in ops]

        pipeline = AugraphyPipeline(
            ink_phase=ink_phase,
            paper_phase=paper_phase,
            post_phase=post_phase,
        )

        def _aug(image, **_):
            try:
                return {"image": pipeline(image)}
            except Exception:
                return {"image": image}

        return A.Lambda(image=_aug)
    except Exception:
        # Augraphy 설치되어 있지 않은 경우
        return None


def _get_albumentations_ops(intensity: float, img_size: int):
    """Return a dict of Albumentations transforms keyed by name"""
    limit15 = int(15 * intensity)
    return {
        "horizontal_flip": A.HorizontalFlip(p=0.5 * intensity),
        "vertical_flip": A.VerticalFlip(p=0.5 * intensity),
        "random_rotate90": A.RandomRotate90(p=0.5 * intensity),
        "rotate": A.Rotate(limit=limit15, p=0.5 * intensity),
        "transpose": A.Transpose(p=0.2 * intensity),
        "shift_scale_rotate": A.ShiftScaleRotate(
            shift_limit=0.1 * intensity,
            scale_limit=0.2 * intensity,
            rotate_limit=limit15,
            p=0.5 * intensity,
        ),
        "optical_distortion": A.OpticalDistortion(distort_limit=0.05 * intensity, shift_limit=0.05 * intensity, p=0.3 * intensity),
        "grid_distortion": A.GridDistortion(num_steps=5, distort_limit=0.3 * intensity, p=0.3 * intensity),
        "elastic_transform": A.ElasticTransform(alpha=1.0 * intensity, sigma=50 * intensity, alpha_affine=50 * intensity, p=0.3 * intensity),
        "perspective": A.Perspective(scale=(0.05 * intensity, 0.1 * intensity), p=0.3 * intensity),
        "random_brightness_contrast": A.RandomBrightnessContrast(brightness_limit=0.2 * intensity, contrast_limit=0.2 * intensity, p=0.5 * intensity),
        "color_jitter": A.ColorJitter(brightness=0.2 * intensity, contrast=0.2 * intensity, saturation=0.2 * intensity, hue=0.05 * intensity, p=0.3 * intensity),
        "hsv": A.HueSaturationValue(hue_shift_limit=20 * intensity, sat_shift_limit=30 * intensity, val_shift_limit=20 * intensity, p=0.3 * intensity),
        "random_gamma": A.RandomGamma(gamma_limit=(80, 120), p=0.3 * intensity),
        "clahe": A.CLAHE(p=0.3 * intensity),
        "to_gray": A.ToGray(p=0.1 * intensity),
        "channel_shuffle": A.ChannelShuffle(p=0.05 * intensity),
        "invert": A.InvertImg(p=0.05 * intensity),
        "gauss_noise": A.GaussNoise(var_limit=(10 * intensity, 50 * intensity), p=0.3 * intensity),
        "blur": A.Blur(blur_limit=3, p=0.2 * intensity),
        "motion_blur": A.MotionBlur(blur_limit=7, p=0.2 * intensity),
        "median_blur": A.MedianBlur(blur_limit=3, p=0.1 * intensity),
        "downscale": A.Downscale(scale_min=0.7, scale_max=0.95, interpolation=cv2.INTER_LINEAR, p=0.2 * intensity),
        "jpeg": A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2 * intensity),
        "sharpen": A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.2 * intensity),
        "emboss": A.Emboss(alpha=(0.1, 0.3), strength=(0.2, 0.5), p=0.2 * intensity),
        "piecewise_affine": A.PiecewiseAffine(scale=(0.01 * intensity, 0.05 * intensity), p=0.2 * intensity),

        # additional transformations
        "random_resized_crop": A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(1 - 0.3 * intensity, 1.0),
            ratio=(0.75, 1.33),
            p=0.5 * intensity,
        ),
        "center_crop": A.CenterCrop(
            height=int(img_size * (1 - 0.1 * intensity)),
            width=int(img_size * (1 - 0.1 * intensity)),
            p=0.3 * intensity,
        ),
        "random_crop": A.RandomCrop(
            height=int(img_size * (1 - 0.1 * intensity)),
            width=int(img_size * (1 - 0.1 * intensity)),
            p=0.3 * intensity,
        ),
        "pad_if_needed": A.PadIfNeeded(min_height=img_size, min_width=img_size, p=0.2 * intensity),
        "coarse_dropout": A.CoarseDropout(max_holes=2, max_height=8, max_width=8, p=0.2 * intensity),
        "channel_dropout": A.ChannelDropout(channel_drop_range=(1, 1), p=0.1 * intensity),
        "fancy_pca": A.FancyPCA(alpha=0.1 * intensity, p=0.2 * intensity),
        "equalize": A.Equalize(p=0.2 * intensity),
        "posterize": A.Posterize(num_bits=4, p=0.2 * intensity),
        "solarize": A.Solarize(p=0.2 * intensity),
        "iso_noise": A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1 * intensity, 0.5 * intensity), p=0.3 * intensity),
        "multiplicative_noise": A.MultiplicativeNoise(multiplier=(1 - 0.1 * intensity, 1 + 0.1 * intensity), p=0.2 * intensity),
        "random_grid_shuffle": A.RandomGridShuffle(grid=(2, 2), p=0.2 * intensity),
        "random_shadow": A.RandomShadow(p=0.2 * intensity),
        "random_sunflare": A.RandomSunFlare(p=0.2 * intensity),
        "random_fog": A.RandomFog(fog_coef_lower=0.1 * intensity, fog_coef_upper=0.3 * intensity, p=0.2 * intensity),
        "random_rain": A.RandomRain(p=0.2 * intensity),
        "random_snow": A.RandomSnow(p=0.2 * intensity),
        "zoom_blur": A.ZoomBlur(p=0.2 * intensity),
        "grid_dropout": A.GridDropout(ratio=0.5, p=0.2 * intensity),
        "gaussian_blur": A.GaussianBlur(blur_limit=(3, 5), p=0.2 * intensity),
        "glass_blur": A.GlassBlur(p=0.2 * intensity),
        "pixel_dropout": A.PixelDropout(dropout_prob=0.01, p=0.1 * intensity),
    }


def get_transforms(cfg, ops_key: str | None):
    """Return transforms for a given ops_key
    
    Args:
        cfg: Configuration object
        ops_key: Key for augmentation ops. If None, 
                returns basic transforms only (resize, normalize, totensor)
    """
    aug_cfg = getattr(cfg, "augmentation", {})
    method = getattr(aug_cfg, "method", "none").lower()
    intensity = float(getattr(aug_cfg, "intensity", 0))

    # If ops_key is None or empty, use empty list (basic transforms only)
    ops_list = aug_cfg.get(ops_key, []) if ops_key else []

    selected = []
    if method in ("albumentations", "mix") and intensity > 0 and ops_key is not None:
        ops_dict = _get_albumentations_ops(intensity, cfg.data.img_size)
        if not ops_list or ops_list == ["all"]:
            selected.extend(ops_dict.values())
        else:
            for name in ops_list:
                if name in ops_dict:
                    selected.append(ops_dict[name])

    aug_ops_keys = ("train_aug_ops", "valid_aug_ops",
                    "valid_tta_ops", "test_tta_ops")
    if (method in ("augraphy", "mix") and intensity > 0
            and ops_key and ops_key in aug_ops_keys):
        augraphy_aug = _create_augraphy_lambda(intensity, ops_list)
        if augraphy_aug is not None:
            selected.append(augraphy_aug)

    selected.extend([
        A.Resize(height=cfg.data.img_size, width=cfg.data.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return A.Compose(selected)


def prepare_data_loaders(cfg, seed):
    """설정에 따라 데이터 로더들을 준비"""
    # 개별 경로 설정 (필수 항목)
    train_images_path = cfg.data.train_images_path
    test_images_path = cfg.data.test_images_path
    train_csv_path = cfg.data.train_csv_path
    test_csv_path = cfg.data.test_csv_path
    
    batch_size = cfg.training.batch_size
    num_workers = cfg.data.num_workers

    # Transform 준비
    train_transform = get_transforms(cfg, "train_aug_ops")
    val_transform = get_transforms(cfg, "valid_aug_ops")
    test_transform = get_transforms(cfg, None)  # Basic transforms only

    # 원본 이미지용 transform (증강 없음)
    org_transform = get_transforms(cfg, None)  # Basic transforms only
    
    # 전체 훈련 데이터 로드
    full_train_df = pd.read_csv(train_csv_path)
    
    # 테스트 데이터 로더 준비
    test_dataset = ImageDataset(
        test_csv_path,
        test_images_path,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=_should_use_pin_memory()
    )
    
    # augmentation 설정
    aug_cfg = getattr(cfg, "augmentation", {})

    # 검증 전략에 따른 데이터 분할
    validation_strategy = cfg.validation.strategy
    
    if validation_strategy == "holdout":
        train_loader, val_loader = _prepare_holdout_loaders(
            cfg,
            full_train_df,
            train_images_path,
            train_transform,
            val_transform,
            seed,
            org_transform,
        )
        return train_loader, val_loader, test_loader, None
        
    elif validation_strategy == "kfold":
        folds = _prepare_kfold_splits(cfg, full_train_df, seed)
        return None, None, test_loader, (
            folds,
            full_train_df,
            train_images_path,
            train_transform,
            val_transform,
            test_transform,
        )
        
    elif validation_strategy == "none":
        train_dataset = _build_augmented_dataset(
            full_train_df,
            train_images_path,
            train_transform,
            org_transform,
            getattr(aug_cfg, "train_aug_count", 0),
            getattr(aug_cfg, "train_aug_add_org", False),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=_should_use_pin_memory(),
            drop_last=False,
        )
        return train_loader, None, test_loader, None
        
    else:
        raise ValueError(f"Unknown validation strategy: {validation_strategy}")


def _prepare_holdout_loaders(cfg, full_train_df, train_images_path,
                             train_transform, val_transform, seed,
                             org_transform):
    """Holdout 검증을 위한 데이터 로더 준비"""
    train_ratio = cfg.validation.holdout.train_ratio
    stratify = cfg.validation.holdout.stratify
    batch_size = cfg.training.batch_size
    num_workers = cfg.data.num_workers
    aug_cfg = getattr(cfg, "augmentation", {})
    
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
    train_dataset = _build_augmented_dataset(
        train_df,
        train_images_path,
        train_transform,
        org_transform,
        getattr(aug_cfg, "train_aug_count", 0),
        getattr(aug_cfg, "train_aug_add_org", False),
    )

    val_dataset = _build_augmented_dataset(
        val_df,
        train_images_path,
        val_transform,
        org_transform,
        getattr(aug_cfg, "valid_aug_count", 0),
        getattr(aug_cfg, "valid_aug_add_org", False),
    )
    
    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=_should_use_pin_memory(), 
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=_should_use_pin_memory()
    )
    
    return train_loader, val_loader


def _prepare_kfold_splits(cfg, full_train_df, seed):
    """K-Fold 교차 검증을 위한 분할 준비"""
    n_splits = cfg.validation.kfold.n_splits
    stratify = cfg.validation.kfold.stratify
    
    if stratify:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
        folds = list(skf.split(full_train_df, full_train_df['target']))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(kf.split(full_train_df))
    
    return folds


def get_kfold_loaders(fold_idx, folds, full_train_df, train_images_path,
                      train_transform, val_transform, cfg):
    """특정 fold에 대한 데이터 로더 반환"""
    train_idx, val_idx = folds[fold_idx]

    aug_cfg = getattr(cfg, "augmentation", {})

    # 원본 이미지용 transform (증강 없음)
    org_transform = get_transforms(cfg, None)  # Basic transforms only
    
    # 현재 fold의 데이터 분할
    train_df = full_train_df.iloc[train_idx]
    val_df = full_train_df.iloc[val_idx]
    
    # Dataset 정의
    train_dataset = _build_augmented_dataset(
        train_df,
        train_images_path,
        train_transform,
        org_transform,
        getattr(aug_cfg, "train_aug_count", 0),
        getattr(aug_cfg, "train_aug_add_org", False),
    )

    val_dataset = _build_augmented_dataset(
        val_df,
        train_images_path,
        val_transform,
        org_transform,
        getattr(aug_cfg, "valid_aug_count", 0),
        getattr(aug_cfg, "valid_aug_add_org", False),
    )
    
    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers, 
        pin_memory=_should_use_pin_memory(), 
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers, 
        pin_memory=_should_use_pin_memory()
    )
    
    return train_loader, val_loader, train_df, val_df 
