import os
import logging
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .utils import upload_to_wandb

log = logging.getLogger(__name__)


def predict_single_model(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                         tta_transform=None, tta_count: int = 0, tta_add_org: bool = False):
    log.info("추론 시작")
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.softmax(1)
            if tta_transform is not None and tta_count > 0:
                aug_preds = []
                for _ in range(tta_count):
                    aug_imgs = torch.stack([tta_transform(image=img.permute(1,2,0).cpu().numpy())['image'] for img in images]).to(device)
                    op = model(aug_imgs).softmax(1)
                    aug_preds.append(op)
                if tta_add_org:
                    aug_preds.append(preds)
                preds = torch.stack(aug_preds).mean(0)
            predictions.extend(preds.argmax(1).cpu().tolist())
    return predictions


def predict_kfold_ensemble(models: Sequence[torch.nn.Module], loader: DataLoader, device: torch.device):
    log.info("K-Fold 앙상블 예측 계산 중...")
    for m in models:
        m.eval()
    preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = [m(images).softmax(1) for m in models]
            mean_out = torch.stack(outputs).mean(0)
            preds.extend(mean_out.argmax(1).cpu().tolist())
    return np.array(preds)


def save_predictions(preds: Sequence[int], dataset, cfg: DictConfig) -> pd.DataFrame:
    if len(preds) != len(dataset.df):
        raise ValueError('Prediction length mismatch')
    df = pd.read_csv(cfg.data.test_csv_path)
    df['target'] = preds
    os.makedirs(cfg.output.dir, exist_ok=True)
    out_path = os.path.join(cfg.output.dir, cfg.output.filename)
    df.to_csv(out_path, index=False)
    log.info(f"예측 결과 저장 완료: {out_path}")
    return df


def run_inference(model_or_models, loader: DataLoader, dataset, cfg: DictConfig,
                  device: torch.device, is_kfold: bool):
    if is_kfold:
        preds = predict_kfold_ensemble(model_or_models, loader, device)
    else:
        preds = predict_single_model(model_or_models, loader, device)
    result_df = save_predictions(preds, dataset, cfg)
    if cfg.wandb.enabled:
        upload_to_wandb(result_df, cfg)
    return result_df
