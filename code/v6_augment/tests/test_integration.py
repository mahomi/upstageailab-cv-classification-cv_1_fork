# -*- coding: utf-8 -*-
"""
통합 테스트 모듈
"""

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tempfile
import os
import pandas as pd
from omegaconf import OmegaConf
from unittest.mock import Mock, patch

from data import prepare_data_loaders, get_kfold_loaders
from training import validate_one_epoch_with_tta, train_one_epoch
from inference import predict_single_model_with_tta, predict_kfold_ensemble_with_tta
from augmentations import AugmentationConfig, TTADataset


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


class MockModel(torch.nn.Module):
    """테스트용 Mock 모델"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


@pytest.fixture
def mock_config():
    """테스트용 설정 생성"""
    config = {
        "data": {
            "data_path": "/tmp/test_data",
            "img_size": 32,
            "num_workers": 0
        },
        "training": {
            "batch_size": 4,
            "seed": 42,
            "lr": 0.001,
            "epochs": 2
        },
        "validation": {
            "strategy": "holdout",
            "holdout": {
                "train_ratio": 0.8,
                "stratify": True
            },
            "kfold": {
                "n_splits": 3,
                "stratify": True
            }
        },
                 "augmentation": {
             "enabled": True,
             "library": "albumentations",
             "train": {
                 "enabled": True,
                 "num_augmented_images": 2
             },
             "valid": {
                 "enabled": False,
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
def mock_csv_files():
    """테스트용 CSV 파일 생성"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # train.csv 생성
        train_data = {
            'filename': [f'train_{i}.jpg' for i in range(10)],
            'target': [i % 3 for i in range(10)]
        }
        train_df = pd.DataFrame(train_data)
        train_csv = os.path.join(temp_dir, 'train.csv')
        train_df.to_csv(train_csv, index=False)
        
        # sample_submission.csv 생성
        test_data = {
            'ID': [f'test_{i}.jpg' for i in range(5)],
            'target': [0] * 5
        }
        test_df = pd.DataFrame(test_data)
        test_csv = os.path.join(temp_dir, 'sample_submission.csv')
        test_df.to_csv(test_csv, index=False)
        
        yield temp_dir, train_csv, test_csv


class TestDataPreparation:
    """데이터 준비 통합 테스트"""
    
    @patch('data.pd.read_csv')
    @patch('data.ImageDataset')
    def test_prepare_data_loaders_with_augmentation(self, mock_image_dataset, mock_read_csv, mock_config):
        """증강이 활성화된 상태에서 데이터 로더 준비 테스트"""
        # Mock 설정
        mock_df = pd.DataFrame({
            'filename': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            'target': [0, 1, 2, 0]
        })
        mock_read_csv.return_value = mock_df
        mock_image_dataset.return_value = MockImageDataset(size=5)
        
        # 데이터 로더 준비
        train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(mock_config, 42)
        
        # 검증
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        assert kfold_data is None  # holdout 전략 사용
    
    @patch('data.pd.read_csv')
    @patch('data.ImageDataset')
    def test_prepare_data_loaders_with_kfold(self, mock_image_dataset, mock_read_csv, mock_config):
        """K-Fold 검증 데이터 로더 준비 테스트"""
        mock_config.validation.strategy = "kfold"
        
        # Mock 설정
        mock_df = pd.DataFrame({
            'filename': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'],
            'target': [0, 1, 2, 0, 1, 2]
        })
        mock_read_csv.return_value = mock_df
        mock_image_dataset.return_value = MockImageDataset(size=5)
        
        # 데이터 로더 준비
        train_loader, val_loader, test_loader, kfold_data = prepare_data_loaders(mock_config, 42)
        
        # 검증
        assert train_loader is None
        assert val_loader is None
        assert test_loader is not None
        assert kfold_data is not None
        assert len(kfold_data) == 6  # folds, full_train_df, data_path, train_transform, test_transform, aug_config


class TestTrainingIntegration:
    """훈련 통합 테스트"""
    
    def test_validate_one_epoch_with_tta(self, mock_config):
        """TTA 검증 테스트"""
        # Mock 데이터셋과 모델 생성
        base_dataset = MockImageDataset(size=3)
        aug_config = AugmentationConfig(mock_config)
        tta_dataset = TTADataset(base_dataset, aug_config)
        
        # DataLoader 생성
        tta_loader = DataLoader(tta_dataset, batch_size=2, shuffle=False)
        
        # Mock 모델과 손실 함수
        model = MockModel(num_classes=3)
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        # TTA 검증 실행
        result = validate_one_epoch_with_tta(tta_loader, model, loss_fn, device)
        
        # 검증
        assert 'val_loss' in result
        assert 'val_acc' in result
        assert 'val_f1' in result
        assert isinstance(result['val_loss'], float)
        assert isinstance(result['val_acc'], float)
        assert isinstance(result['val_f1'], float)
    
    def test_train_one_epoch(self, mock_config):
        """훈련 한 에포크 테스트"""
        # Mock 데이터셋과 로더 생성
        dataset = MockImageDataset(size=4)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Mock 모델, 옵티마이저, 손실 함수
        model = MockModel(num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        # 훈련 실행
        result = train_one_epoch(loader, model, optimizer, loss_fn, device)
        
        # 검증
        assert 'train_loss' in result
        assert 'train_acc' in result
        assert 'train_f1' in result
        assert isinstance(result['train_loss'], float)
        assert isinstance(result['train_acc'], float)
        assert isinstance(result['train_f1'], float)


class TestInferenceIntegration:
    """추론 통합 테스트"""
    
    def test_predict_single_model_with_tta(self, mock_config):
        """단일 모델 TTA 추론 테스트"""
        # Mock 데이터셋과 모델 생성
        base_dataset = MockImageDataset(size=3)
        aug_config = AugmentationConfig(mock_config)
        tta_dataset = TTADataset(base_dataset, aug_config)
        
        # DataLoader 생성
        tta_loader = DataLoader(tta_dataset, batch_size=2, shuffle=False)
        
        # Mock 모델
        model = MockModel(num_classes=3)
        model.eval()
        device = torch.device('cpu')
        
        # TTA 추론 실행
        predictions = predict_single_model_with_tta(model, tta_loader, device)
        
        # 검증
        assert isinstance(predictions, list)
        assert len(predictions) == len(base_dataset)
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
        assert all(0 <= pred < 3 for pred in predictions)
    
    def test_predict_kfold_ensemble_with_tta(self, mock_config):
        """K-Fold 앙상블 TTA 추론 테스트"""
        # Mock 데이터셋과 모델 생성
        base_dataset = MockImageDataset(size=3)
        aug_config = AugmentationConfig(mock_config)
        tta_dataset = TTADataset(base_dataset, aug_config)
        
        # DataLoader 생성
        tta_loader = DataLoader(tta_dataset, batch_size=2, shuffle=False)
        
        # Mock 모델들 (3개 fold)
        models = [MockModel(num_classes=3) for _ in range(3)]
        for model in models:
            model.eval()
        device = torch.device('cpu')
        
        # K-Fold TTA 추론 실행
        predictions = predict_kfold_ensemble_with_tta(models, tta_loader, device)
        
        # 검증
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(base_dataset)
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
        assert all(0 <= pred < 3 for pred in predictions)


class TestEndToEndIntegration:
    """종단간 통합 테스트"""
    
    def test_augmentation_config_propagation(self, mock_config):
        """증강 설정이 전체 파이프라인에 올바르게 전파되는지 테스트"""
        # AugmentationConfig 생성
        aug_config = AugmentationConfig(mock_config)
        
        # 설정 검증
        assert aug_config.enabled == True
        assert aug_config.library == "albumentations"
        assert aug_config.train_num_augmented_images == 2
        assert aug_config.valid_num_augmented_images == 1
        assert aug_config.train_enabled == True
        assert aug_config.valid_enabled == False
        assert aug_config.valid_tta_enabled == True
        assert aug_config.test_tta_enabled == True
    
    def test_tta_dataset_integration(self, mock_config):
        """TTA 데이터셋 통합 테스트"""
        # Mock 데이터셋 생성
        base_dataset = MockImageDataset(size=2)
        aug_config = AugmentationConfig(mock_config)
        
        # TTA 데이터셋 생성
        tta_dataset = TTADataset(base_dataset, aug_config)
        
        # 데이터 로더 생성
        tta_loader = DataLoader(tta_dataset, batch_size=3, shuffle=False)
        
        # 배치 처리 테스트
        batch_count = 0
        total_samples = 0
        
        for batch_data in tta_loader:
            assert len(batch_data) == 3  # image, target, base_idx
            images, targets, base_indices = batch_data
            
            batch_count += 1
            total_samples += len(images)
            
            # 배치 형태 검증
            assert isinstance(images, torch.Tensor)
            assert isinstance(targets, torch.Tensor)
            assert isinstance(base_indices, torch.Tensor)
            assert len(images.shape) == 4  # [B, C, H, W]
        
        # 전체 샘플 수 검증 (원본 2개 * TTA 5개 = 10개)
        expected_total = len(base_dataset) * 5  # 5개 TTA 변환
        assert total_samples == expected_total
    
    def test_different_augmentation_libraries(self, mock_config):
        """다른 증강 라이브러리들 테스트"""
        base_dataset = MockImageDataset(size=2)
        
        # Albumentations 테스트
        mock_config.augmentation.library = "albumentations"
        aug_config_albu = AugmentationConfig(mock_config)
        assert aug_config_albu.library == "albumentations"
        
        # Augraphy 테스트
        mock_config.augmentation.library = "augraphy"
        aug_config_augra = AugmentationConfig(mock_config)
        assert aug_config_augra.library == "augraphy"
        
        # Mixed 테스트
        mock_config.augmentation.library = "mixed"
        aug_config_mixed = AugmentationConfig(mock_config)
        assert aug_config_mixed.library == "mixed"
    
    def test_augmentation_disabled_fallback(self, mock_config):
        """증강 비활성화 시 폴백 테스트"""
        mock_config.augmentation.enabled = False
        aug_config = AugmentationConfig(mock_config)
        
        # 설정 검증
        assert aug_config.enabled == False
        
        # 데이터셋 생성 - 원본 데이터셋 반환되어야 함
        base_dataset = MockImageDataset(size=3)
        
        from augmentations import create_augmented_dataset, create_tta_dataset
        
        # 증강 데이터셋 생성 시도
        aug_dataset = create_augmented_dataset(base_dataset, aug_config, is_train=True)
        assert aug_dataset is base_dataset  # 원본 데이터셋 그대로 반환
        
        # TTA 데이터셋 생성 시도
        tta_dataset = create_tta_dataset(base_dataset, aug_config)
        assert tta_dataset is base_dataset  # 원본 데이터셋 그대로 반환


if __name__ == "__main__":
    pytest.main([__file__]) 