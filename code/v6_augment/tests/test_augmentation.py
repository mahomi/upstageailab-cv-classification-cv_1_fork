import os
import sys
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import ImageDataset, AugmentedDataset, get_transforms
from inference import predict_single_model
from unittest.mock import patch
import pytest


def test_augmented_dataset_len():
    tmp = tempfile.mkdtemp()
    img_dir = os.path.join(tmp, "imgs")
    os.makedirs(img_dir)
    df = pd.DataFrame({'ID': ["a.jpg", "b.jpg"], 'target': [0, 1]})
    for name in df['ID']:
        Image.new('RGB', (10, 10), color='white').save(os.path.join(img_dir, name))
    dataset = ImageDataset(df, img_dir)
    
    # transform 생성
    aug_transform = A.Compose([A.Resize(10, 10), A.Normalize(), ToTensorV2()])
    org_transform = A.Compose([A.Resize(10, 10), A.Normalize(), ToTensorV2()])
    
    aug = AugmentedDataset(dataset, num_aug=2, aug_transform=aug_transform, org_transform=org_transform)
    # 기본값이 add_org=False이므로 원본 이미지가 포함되지 않음
    assert len(aug) == len(dataset) * 2


def test_augmented_dataset_add_org():
    """AugmentedDataset의 add_org 옵션 테스트"""
    tmp = tempfile.mkdtemp()
    img_dir = os.path.join(tmp, "imgs")
    os.makedirs(img_dir)
    df = pd.DataFrame({'ID': ["a.jpg", "b.jpg"], 'target': [0, 1]})
    for name in df['ID']:
        Image.new('RGB', (10, 10), color='white').save(os.path.join(img_dir, name))
    dataset = ImageDataset(df, img_dir)
    
    # transform 생성
    aug_transform = A.Compose([A.Resize(10, 10), A.Normalize(), ToTensorV2()])
    org_transform = A.Compose([A.Resize(10, 10), A.Normalize(), ToTensorV2()])
    
    # add_org=True인 경우
    aug_with_org = AugmentedDataset(dataset, num_aug=2, add_org=True, aug_transform=aug_transform, org_transform=org_transform)
    assert len(aug_with_org) == len(dataset) * 3  # original + 2 augmented = 3x
    
    # add_org=False인 경우 
    aug_without_org = AugmentedDataset(dataset, num_aug=2, add_org=False, aug_transform=aug_transform, org_transform=org_transform)
    assert len(aug_without_org) == len(dataset) * 2  # 2 augmented only = 2x


def test_predict_single_model_tta():
    tmp = tempfile.mkdtemp()
    img_dir = os.path.join(tmp, "imgs")
    os.makedirs(img_dir)
    df = pd.DataFrame({'ID': [f'{i}.jpg' for i in range(4)], 'target': [0]*4})
    for name in df['ID']:
        Image.new('RGB', (32, 32), color='white').save(os.path.join(img_dir, name))
    cfg = OmegaConf.create({
        'data': {'img_size': 32}, 
        'augmentation': {
            'method': 'albumentations', 
            'intensity': 0.5,
            'test_tta_ops': ['rotate']
        }
    })
    tta_transform = get_transforms(cfg, 'test_tta_ops')
    test_transform = get_transforms(cfg, None)
    dataset = ImageDataset(df, img_dir, transform=test_transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(32*32*3, 2))
    preds = predict_single_model(model, loader, torch.device('cpu'), tta_transform=tta_transform, tta_count=1, tta_add_org=False)
    assert len(preds) == len(dataset)


def test_predict_single_model_tta_add_org():
    """predict_single_model의 tta_add_org 옵션 테스트"""
    tmp = tempfile.mkdtemp()
    img_dir = os.path.join(tmp, "imgs")
    os.makedirs(img_dir)
    df = pd.DataFrame({'ID': [f'{i}.jpg' for i in range(4)], 'target': [0]*4})
    for name in df['ID']:
        Image.new('RGB', (32, 32), color='white').save(os.path.join(img_dir, name))
    cfg = OmegaConf.create({
        'data': {'img_size': 32}, 
        'augmentation': {
            'method': 'albumentations', 
            'intensity': 0.5,
            'test_tta_ops': ['rotate']
        }
    })
    tta_transform = get_transforms(cfg, 'test_tta_ops')
    test_transform = get_transforms(cfg, None)
    dataset = ImageDataset(df, img_dir, transform=test_transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(32*32*3, 2))
    
    # add_org=True와 add_org=False 모두 테스트
    preds_with_org = predict_single_model(model, loader, torch.device('cpu'), tta_transform=tta_transform, tta_count=1, tta_add_org=True)
    preds_without_org = predict_single_model(model, loader, torch.device('cpu'), tta_transform=tta_transform, tta_count=1, tta_add_org=False)
    
    assert len(preds_with_org) == len(dataset)
    assert len(preds_without_org) == len(dataset)


@patch('inference._predict_probs')
def test_skip_original_when_tta_no_org(mock_pred):
    """tta_add_org=False일 때 base 추론이 생략되는지 테스트"""
    tmp = tempfile.mkdtemp()
    img_dir = os.path.join(tmp, 'imgs')
    os.makedirs(img_dir)
    df = pd.DataFrame({'ID': [f'{i}.jpg' for i in range(2)], 'target': [0, 1]})
    for name in df['ID']:
        Image.new('RGB', (32, 32), color='white').save(os.path.join(img_dir, name))
    cfg = OmegaConf.create({'data': {'img_size': 32}, 'augmentation': {'method': 'albumentations', 'intensity': 0.5, 'test_tta_ops': ['rotate']}})
    tta_transform = get_transforms(cfg, 'test_tta_ops')
    test_transform = get_transforms(cfg, None)
    dataset = ImageDataset(df, img_dir, transform=test_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(32*32*3, 2))

    mock_pred.return_value = np.zeros((len(dataset), 2))
    predict_single_model(model, loader, torch.device('cpu'), tta_transform=tta_transform, tta_count=2, tta_add_org=False)
    assert mock_pred.call_count == 2


def test_get_transforms_many_ops():
    cfg = OmegaConf.create({'data': {'img_size': 32}, 'augmentation': {'method': 'albumentations', 'intensity': 1.0}})
    train_t = get_transforms(cfg, 'train_aug_ops')
    assert len(train_t.transforms) > 30


def test_get_transforms_basic():
    cfg = OmegaConf.create({'data': {'img_size': 32}})
    basic_t = get_transforms(cfg, None)
    assert len(basic_t.transforms) == 3  # resize, normalize, totensor


def test_custom_ops_selection():
    cfg = OmegaConf.create({'data': {'img_size': 32},
                            'augmentation': {'method': 'albumentations',
                                             'intensity': 1.0,
                                             'train_aug_ops': ['rotate']}})
    train_t = get_transforms(cfg, 'train_aug_ops')
    # rotate + resize/normalize/totensor
    names = [type(t).__name__ for t in train_t.transforms]
    assert any('Rotate' in n for n in names)
    assert len(train_t.transforms) <= 4


def test_new_ops_present():
    cfg = OmegaConf.create({'data': {'img_size': 32}, 'augmentation': {'method': 'albumentations', 'intensity': 1.0}})
    train_t = get_transforms(cfg, 'train_aug_ops')
    names = [type(t).__name__ for t in train_t.transforms]
    assert any('RandomSunFlare' in n for n in names)
    assert any('RandomFog' in n for n in names)


def test_get_transforms_augraphy_lambda():
    import importlib
    if importlib.util.find_spec('augraphy') is None:
        pytest.skip('augraphy not installed')
    cfg = OmegaConf.create({'data': {'img_size': 32}, 'augmentation': {'method': 'augraphy', 'intensity': 1.0}})
    train_t = get_transforms(cfg, 'train_aug_ops')
    assert any(isinstance(t, A.Lambda) for t in train_t.transforms)


def test_custom_augraphy_ops():
    import importlib
    if importlib.util.find_spec('augraphy') is None:
        pytest.skip('augraphy not installed')
    cfg = OmegaConf.create({'data': {'img_size': 32},
                            'augmentation': {'method': 'augraphy',
                                             'intensity': 1.0,
                                             'train_aug_ops': ['ink_bleed']}})
    train_t = get_transforms(cfg, 'train_aug_ops')
    assert any(isinstance(t, A.Lambda) for t in train_t.transforms)
