# -*- coding: utf-8 -*-
"""
증강 기능 테스트 모듈
"""

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tempfile
import os
from omegaconf import OmegaConf

from augmentations import (
    AugmentationConfig,
    get_albumentations_augmentation,
    get_augraphy_augmentation,
    get_tta_transforms,
    AugmentedDataset,
    TTADataset,
    create_augmented_dataset,
    create_tta_dataset,
    AUGRAPHY_AVAILABLE
)


class MockImageDataset(Dataset):
    """테스트용 Mock 이미지 데이터셋"""
    def __init__(self, size=10, img_shape=(32, 32, 3)):
        self.size = size
        self.img_shape = img_shape
        self.data = []
        
        for i in range(size):
            # 랜덤 이미지 생성
            image = np.random.randint(0, 255, img_shape, dtype=np.uint8)
            target = i % 3  # 3개 클래스
            self.data.append((image, target))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def mock_config():
    """테스트용 설정 생성"""
    config = {
        "data": {
            "img_size": 32
        },
        "augmentation": {
            "enabled": True,
            "library": "albumentations",
            "train": {
                "enabled": True,
                "num_augmented_images": 2
            },
            "valid": {
                "enabled": True,
                "num_augmented_images": 1
            },
            "valid_tta": {
                "enabled": True
            },
            "test_tta": {
                "enabled": True
            }
        }
    }
    return OmegaConf.create(config)


@pytest.fixture
def mock_dataset():
    """테스트용 Mock 데이터셋 생성"""
    return MockImageDataset(size=5)


class TestAugmentationConfig:
    """AugmentationConfig 클래스 테스트"""
    
    def test_config_initialization(self, mock_config):
        """설정 초기화 테스트"""
        aug_config = AugmentationConfig(mock_config)
        
        assert aug_config.enabled == True
        assert aug_config.library == "albumentations"
        assert aug_config.train_num_augmented_images == 2
        assert aug_config.valid_num_augmented_images == 1
        assert aug_config.img_size == 32
        assert aug_config.train_enabled == True
        assert aug_config.valid_enabled == True
        assert aug_config.valid_tta_enabled == True
        assert aug_config.test_tta_enabled == True
    
    def test_config_disabled(self, mock_config):
        """증강 비활성화 설정 테스트"""
        mock_config.augmentation.enabled = False
        aug_config = AugmentationConfig(mock_config)
        
        assert aug_config.enabled == False


class TestAugmentationFunctions:
    """증강 함수들 테스트"""
    
    def test_get_albumentations_augmentation(self, mock_config):
        """Albumentations 증강 생성 테스트"""
        aug_config = AugmentationConfig(mock_config)
        
        # 훈련용 증강
        train_aug = get_albumentations_augmentation(aug_config, is_train=True)
        assert train_aug is not None
        
        # 테스트용 증강
        test_aug = get_albumentations_augmentation(aug_config, is_train=False)
        assert test_aug is not None
    
    def test_get_augraphy_augmentation(self, mock_config):
        """Augraphy 증강 생성 테스트"""
        aug_config = AugmentationConfig(mock_config)
        
        # 훈련용 증강
        train_aug = get_augraphy_augmentation(aug_config, is_train=True)
        if AUGRAPHY_AVAILABLE:
            assert train_aug is not None
        else:
            assert train_aug is None
        
        # 테스트용 증강
        test_aug = get_augraphy_augmentation(aug_config, is_train=False)
        assert test_aug is None  # 테스트용은 항상 None
    
    def test_get_tta_transforms(self, mock_config):
        """TTA 변환 생성 테스트"""
        aug_config = AugmentationConfig(mock_config)
        tta_transforms = get_tta_transforms(aug_config)
        
        assert len(tta_transforms) == 5  # 원본 + 4개 변환
        assert all(transform is not None for transform in tta_transforms)


class TestAugmentedDataset:
    """AugmentedDataset 클래스 테스트"""
    
    def test_augmented_dataset_creation(self, mock_config, mock_dataset):
        """증강된 데이터셋 생성 테스트"""
        aug_config = AugmentationConfig(mock_config)
        augmented_dataset = AugmentedDataset(mock_dataset, aug_config, is_train=True)
        
        # 데이터셋 크기 확인 (훈련용 설정 사용)
        expected_size = len(mock_dataset) * aug_config.train_num_augmented_images
        assert len(augmented_dataset) == expected_size
        
        # 데이터 접근 테스트
        image, target = augmented_dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, (int, np.integer))
    
    def test_augmented_dataset_disabled(self, mock_config, mock_dataset):
        """증강 비활성화 시 데이터셋 테스트"""
        mock_config.augmentation.enabled = False
        aug_config = AugmentationConfig(mock_config)
        augmented_dataset = AugmentedDataset(mock_dataset, aug_config, is_train=True)
        
        # 원본 데이터셋과 동일한 크기
        assert len(augmented_dataset) == len(mock_dataset)
    
    def test_augmented_dataset_different_libraries(self, mock_config, mock_dataset):
        """다른 증강 라이브러리 테스트"""
        aug_config = AugmentationConfig(mock_config)
        
        # Albumentations
        aug_config.library = "albumentations"
        aug_dataset_albu = AugmentedDataset(mock_dataset, aug_config, is_train=True)
        assert len(aug_dataset_albu) == len(mock_dataset) * 2
        
        # Augraphy
        aug_config.library = "augraphy"
        aug_dataset_augra = AugmentedDataset(mock_dataset, aug_config, is_train=True)
        assert len(aug_dataset_augra) == len(mock_dataset) * 2
        
        # Mixed
        aug_config.library = "mixed"
        aug_dataset_mixed = AugmentedDataset(mock_dataset, aug_config, is_train=True)
        assert len(aug_dataset_mixed) == len(mock_dataset) * 2


class TestTTADataset:
    """TTADataset 클래스 테스트"""
    
    def test_tta_dataset_creation(self, mock_config, mock_dataset):
        """TTA 데이터셋 생성 테스트"""
        aug_config = AugmentationConfig(mock_config)
        tta_dataset = TTADataset(mock_dataset, aug_config)
        
        # 데이터셋 크기 확인 (원본 * TTA 개수)
        tta_transforms = get_tta_transforms(aug_config)
        expected_size = len(mock_dataset) * len(tta_transforms)
        assert len(tta_dataset) == expected_size
        
        # 데이터 접근 테스트
        image, target, base_idx = tta_dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, (int, np.integer))
        assert isinstance(base_idx, (int, np.integer))
        assert base_idx == 0
    
    def test_tta_dataset_base_idx_tracking(self, mock_config, mock_dataset):
        """TTA 데이터셋 base_idx 추적 테스트"""
        aug_config = AugmentationConfig(mock_config)
        tta_dataset = TTADataset(mock_dataset, aug_config)
        
        tta_transforms = get_tta_transforms(aug_config)
        num_tta = len(tta_transforms)
        
        # 첫 번째 원본 이미지의 모든 TTA 변환 확인
        for i in range(num_tta):
            image, target, base_idx = tta_dataset[i]
            assert base_idx == 0  # 첫 번째 원본 이미지
        
        # 두 번째 원본 이미지의 첫 번째 TTA 변환 확인
        image, target, base_idx = tta_dataset[num_tta]
        assert base_idx == 1  # 두 번째 원본 이미지


class TestHelperFunctions:
    """헬퍼 함수들 테스트"""
    
    def test_create_augmented_dataset(self, mock_config, mock_dataset):
        """증강된 데이터셋 생성 헬퍼 함수 테스트"""
        aug_config = AugmentationConfig(mock_config)
        
        # 훈련용 증강 데이터셋
        train_dataset = create_augmented_dataset(mock_dataset, aug_config, is_train=True)
        assert isinstance(train_dataset, AugmentedDataset)
        
        # 검증용 증강 데이터셋
        valid_dataset = create_augmented_dataset(mock_dataset, aug_config, is_train=False)
        assert isinstance(valid_dataset, AugmentedDataset)
    
    def test_create_augmented_dataset_disabled(self, mock_config, mock_dataset):
        """증강 비활성화 시 헬퍼 함수 테스트"""
        mock_config.augmentation.enabled = False
        aug_config = AugmentationConfig(mock_config)
        
        # 비활성화된 경우 원본 데이터셋 반환
        dataset = create_augmented_dataset(mock_dataset, aug_config, is_train=True)
        assert dataset is mock_dataset
    
    def test_create_tta_dataset(self, mock_config, mock_dataset):
        """TTA 데이터셋 생성 헬퍼 함수 테스트"""
        aug_config = AugmentationConfig(mock_config)
        
        # TTA 데이터셋 생성
        tta_dataset = create_tta_dataset(mock_dataset, aug_config)
        assert isinstance(tta_dataset, TTADataset)
    
    def test_create_tta_dataset_disabled(self, mock_config, mock_dataset):
        """TTA 비활성화 시 헬퍼 함수 테스트"""
        mock_config.augmentation.enabled = False
        aug_config = AugmentationConfig(mock_config)
        
        # 비활성화된 경우 원본 데이터셋 반환
        dataset = create_tta_dataset(mock_dataset, aug_config)
        assert dataset is mock_dataset


class TestDataLoaderIntegration:
    """DataLoader와의 통합 테스트"""
    
    def test_dataloader_with_augmented_dataset(self, mock_config, mock_dataset):
        """증강된 데이터셋과 DataLoader 통합 테스트"""
        aug_config = AugmentationConfig(mock_config)
        augmented_dataset = AugmentedDataset(mock_dataset, aug_config, is_train=True)
        
        # DataLoader 생성
        dataloader = DataLoader(augmented_dataset, batch_size=2, shuffle=False)
        
        # 배치 테스트
        for batch_idx, (images, targets) in enumerate(dataloader):
            assert isinstance(images, torch.Tensor)
            assert isinstance(targets, torch.Tensor)
            assert images.shape[0] <= 2  # 배치 크기
            assert len(images.shape) == 4  # [B, C, H, W]
            
            if batch_idx >= 2:  # 몇 개 배치만 테스트
                break
    
    def test_dataloader_with_tta_dataset(self, mock_config, mock_dataset):
        """TTA 데이터셋과 DataLoader 통합 테스트"""
        aug_config = AugmentationConfig(mock_config)
        tta_dataset = TTADataset(mock_dataset, aug_config)
        
        # DataLoader 생성
        dataloader = DataLoader(tta_dataset, batch_size=2, shuffle=False)
        
        # 배치 테스트
        for batch_idx, batch_data in enumerate(dataloader):
            assert len(batch_data) == 3  # image, target, base_idx
            images, targets, base_indices = batch_data
            
            assert isinstance(images, torch.Tensor)
            assert isinstance(targets, torch.Tensor)
            assert isinstance(base_indices, torch.Tensor)
            assert images.shape[0] <= 2  # 배치 크기
            assert len(images.shape) == 4  # [B, C, H, W]
            
            if batch_idx >= 2:  # 몇 개 배치만 테스트
                break


if __name__ == "__main__":
    pytest.main([__file__]) 