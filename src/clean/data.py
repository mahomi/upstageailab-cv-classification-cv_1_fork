import os
import json
import hashlib
from dataclasses import dataclass
from typing import Tuple, List, Optional, Iterable

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

# Basic logger
import logging
log = logging.getLogger(__name__)


def _should_pin_memory() -> bool:
    try:
        if torch.cuda.is_available():
            return True
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return False
    except Exception:
        pass
    return False


class ImageDataset(Dataset):
    """Dataset loading images defined by a CSV or DataFrame."""

    def __init__(self, csv_or_df: Iterable, image_dir: str, transform=None):
        if isinstance(csv_or_df, str):
            df = pd.read_csv(csv_or_df)
        else:
            df = pd.DataFrame(csv_or_df)
        self.df = df[['ID', 'target']]
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['ID'])
        img = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, int(row['target'])


class IndexedImageDataset(Dataset):
    """Dataset that keeps DataFrame indices stable."""

    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df = df[['ID', 'target']].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = np.array(Image.open(os.path.join(self.image_dir, row['ID'])).convert('RGB'))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, int(row['target'])


class AugmentedDataset(Dataset):
    """Wrap a base dataset and generate multiple augmentations per sample."""

    def __init__(self, base: Dataset, num_aug: int, add_org: bool = False,
                 aug_transform=None, org_transform=None):
        if aug_transform is None or org_transform is None:
            raise ValueError('Transforms must be provided')
        self.base = base
        self.num_aug = max(0, int(num_aug))
        self.add_org = add_org
        self.aug_transform = aug_transform
        self.org_transform = org_transform

    def __len__(self) -> int:
        mult = self.num_aug + 1 if self.add_org else self.num_aug
        return len(self.base) * mult

    def __getitem__(self, idx: int):
        base_idx = idx % len(self.base)
        aug_idx = idx // len(self.base)
        img, target = self.base[base_idx]
        if isinstance(img, dict):
            img = img['image']
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        if self.add_org and aug_idx == self.num_aug:
            img = self.org_transform(image=img)['image']
        else:
            img = self.aug_transform(image=img)['image']
        return img, target, base_idx


def _albumentations_ops(intensity: float, img_size: int) -> dict:
    limit = int(15 * intensity)
    ops = {
        'horizontal_flip': A.HorizontalFlip(p=0.5 * intensity),
        'vertical_flip': A.VerticalFlip(p=0.5 * intensity),
        'random_rotate90': A.RandomRotate90(p=0.5 * intensity),
        'rotate': A.Rotate(limit=limit, p=0.5 * intensity),
        'shift_scale_rotate': A.ShiftScaleRotate(0.1, 0.2, limit, p=0.5 * intensity),
        'optical_distortion': A.OpticalDistortion(p=0.3 * intensity),
        'grid_distortion': A.GridDistortion(p=0.3 * intensity),
        'random_brightness_contrast': A.RandomBrightnessContrast(p=0.5 * intensity),
        'color_jitter': A.ColorJitter(p=0.3 * intensity),
        'hsv': A.HueSaturationValue(p=0.3 * intensity),
        'random_gamma': A.RandomGamma(p=0.3 * intensity),
        'gauss_noise': A.GaussNoise(p=0.3 * intensity),
        'blur': A.Blur(blur_limit=3, p=0.2 * intensity),
        'motion_blur': A.MotionBlur(blur_limit=7, p=0.2 * intensity),
        'downscale': A.Downscale(p=0.2 * intensity),
        'jpeg': A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2 * intensity),
        'coarse_dropout': A.CoarseDropout(max_holes=2, max_height=8, max_width=8, p=0.2 * intensity),
        'random_resized_crop': A.RandomResizedCrop(img_size, img_size, p=0.5 * intensity),
        'center_crop': A.CenterCrop(int(img_size * 0.9), int(img_size * 0.9), p=0.3 * intensity),
        'random_crop': A.RandomCrop(int(img_size * 0.9), int(img_size * 0.9), p=0.3 * intensity),
        'pad_if_needed': A.PadIfNeeded(img_size, img_size, p=0.2 * intensity),
        'random_shadow': A.RandomShadow(p=0.2 * intensity),
        'random_sunflare': A.RandomSunFlare(p=0.2 * intensity),
        'random_fog': A.RandomFog(p=0.2 * intensity),
        'random_rain': A.RandomRain(p=0.2 * intensity),
        'random_snow': A.RandomSnow(p=0.2 * intensity),
        'posterize': A.Posterize(p=0.2 * intensity),
        'solarize': A.Solarize(p=0.2 * intensity),
        'equalize': A.Equalize(p=0.2 * intensity),
        'sharpen': A.Sharpen(p=0.2 * intensity),
    }
    return ops


def _augraphy_lambda(intensity: float, ops: Optional[List[str]] = None):
    try:
        from augraphy import AugraphyPipeline
        from augraphy.augmentations import InkBleed
    except Exception:
        return None
    if ops and 'ink_bleed' not in ops:
        return None
    pipeline = AugraphyPipeline(ink_phase=[InkBleed(p=0.2 * intensity)])

    def _aug(image, **kwargs):
        try:
            return pipeline(image)
        except Exception:
            return image

    return A.Lambda(image=_aug)


def get_transforms(cfg, ops_key: Optional[str]):
    aug_cfg = getattr(cfg, 'augment', {})
    method = aug_cfg.get('method', 'none')
    intensity = float(aug_cfg.get('intensity', 0))
    ops_list = aug_cfg.get(ops_key, []) if ops_key else []

    transforms: List[A.BasicTransform] = []
    if method in ('albumentations', 'mix') and intensity > 0 and ops_key:
        ops = _albumentations_ops(intensity, cfg.data.img_size)
        if not ops_list or ops_list == ['all']:
            transforms.extend(ops.values())
        else:
            for name in ops_list:
                if name in ops:
                    transforms.append(ops[name])
    if method in ('augraphy', 'mix') and intensity > 0 and ops_key:
        aug = _augraphy_lambda(intensity, ops_list)
        if aug is not None:
            transforms.append(aug)
    transforms.extend([
        A.Resize(cfg.data.img_size, cfg.data.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return A.Compose(transforms)


@dataclass
class LoaderBundle:
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]
    test_loader: DataLoader
    kfold_data: Optional[Tuple]


def _split_holdout(cfg, df: pd.DataFrame, seed: int):
    train_ratio = cfg.validation.holdout.train_ratio
    stratify = cfg.validation.holdout.get('stratify', False)
    if stratify:
        train_df, val_df = train_test_split(df, test_size=1-train_ratio,
                                            stratify=df['target'], random_state=seed)
    else:
        train_df, val_df = train_test_split(df, test_size=1-train_ratio,
                                            random_state=seed)
    return train_df, val_df


def _prepare_loader(df: pd.DataFrame, img_dir: str, transform, cfg, shuffle: bool):
    dataset = IndexedImageDataset(df, img_dir, transform=None)
    aug_cfg = getattr(cfg, 'augment', {})
    count = int(aug_cfg.get('train_aug_count' if shuffle else 'valid_aug_count', 0))
    add_org = bool(aug_cfg.get('train_aug_add_org' if shuffle else 'valid_aug_add_org', False))
    if count > 0:
        dataset = AugmentedDataset(dataset, count, add_org, get_transforms(cfg, 'train_aug_ops' if shuffle else 'valid_aug_ops'),
                                   get_transforms(cfg, None))
    else:
        dataset = IndexedImageDataset(df, img_dir, transform=get_transforms(cfg, 'train_aug_ops' if shuffle else 'valid_aug_ops'))
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=shuffle,
                        num_workers=cfg.data.num_workers, pin_memory=_should_pin_memory(), drop_last=False)
    return loader


def prepare_data_loaders(cfg, seed: int) -> LoaderBundle:
    train_csv = cfg.data.train_csv_path
    test_csv = cfg.data.test_csv_path
    train_img_dir = cfg.data.train_images_path
    test_img_dir = cfg.data.test_images_path
    batch_size = cfg.train.batch_size

    train_df = pd.read_csv(train_csv)
    test_dataset = ImageDataset(test_csv, test_img_dir, transform=get_transforms(cfg, None))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=_should_pin_memory())

    strategy = cfg.validation.strategy
    if strategy == 'holdout':
        tr_df, val_df = _split_holdout(cfg, train_df, seed)
        train_loader = _prepare_loader(tr_df, train_img_dir, get_transforms(cfg, 'train_aug_ops'), cfg, True)
        val_loader = _prepare_loader(val_df, train_img_dir, get_transforms(cfg, 'valid_aug_ops'), cfg, False)
        return LoaderBundle(train_loader, val_loader, test_loader, None)
    if strategy == 'kfold':
        n_splits = cfg.validation.kfold.n_splits
        stratify = cfg.validation.kfold.get('stratify', False)
        if stratify:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            folds = list(splitter.split(train_df, train_df['target']))
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            folds = list(splitter.split(train_df))
        bundle = (folds, train_df, train_img_dir,
                  get_transforms(cfg, 'train_aug_ops'),
                  get_transforms(cfg, 'valid_aug_ops'),
                  get_transforms(cfg, None))
        return LoaderBundle(None, None, test_loader, bundle)
    if strategy == 'none':
        loader = _prepare_loader(train_df, train_img_dir, get_transforms(cfg, 'train_aug_ops'), cfg, True)
        return LoaderBundle(loader, None, test_loader, None)
    raise ValueError(f'Unknown strategy: {strategy}')


def get_kfold_loaders(fold_idx: int, folds, full_df: pd.DataFrame, img_dir: str,
                      train_t, val_t, cfg):
    tr_idx, val_idx = folds[fold_idx]
    train_df = full_df.iloc[tr_idx]
    val_df = full_df.iloc[val_idx]
    train_loader = _prepare_loader(train_df, img_dir, train_t, cfg, True)
    val_loader = _prepare_loader(val_df, img_dir, val_t, cfg, False)
    return train_loader, val_loader, train_df, val_df
