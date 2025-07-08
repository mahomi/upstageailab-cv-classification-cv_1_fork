import os
import json
from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
)
import timm
from omegaconf import DictConfig

import logging
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Model creation and utilities
# -----------------------------------------------------------------------------

def create_model(name: str, pretrained: bool, num_classes: int) -> nn.Module:
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    sched_cfg = cfg.train.scheduler
    if not sched_cfg.enabled:
        return None
    name = sched_cfg.name
    if name == 'cosine':
        epochs = cfg.train.get('epochs', 1)
        params = sched_cfg.cosine
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=params.eta_min,
                                 last_epoch=params.last_epoch)
    if name == 'step':
        p = sched_cfg.step
        return StepLR(optimizer, step_size=p.step_size, gamma=p.gamma,
                      last_epoch=p.last_epoch)
    if name == 'plateau':
        p = sched_cfg.plateau
        return ReduceLROnPlateau(optimizer, mode=p.mode, factor=p.factor,
                                 patience=p.patience, threshold=p.threshold,
                                 threshold_mode=p.threshold_mode,
                                 cooldown=p.cooldown, min_lr=p.min_lr, eps=p.eps)
    if name == 'cosine_warm':
        p = sched_cfg.cosine_warm
        return CosineAnnealingWarmRestarts(optimizer, T_0=p.T_0, T_mult=p.T_mult,
                                           eta_min=p.eta_min, last_epoch=p.last_epoch)
    if name == 'none':
        return None
    raise ValueError('지원하지 않는 스케쥴러')


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        logprobs = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=1))


def create_optimizer(model: nn.Module, lr: float):
    return Adam(model.parameters(), lr=lr)


def create_model_and_optimizer(cfg: DictConfig, device: torch.device):
    model = create_model(cfg.model.name, cfg.model.pretrained, cfg.model.num_classes).to(device)
    optimizer = create_optimizer(model, cfg.train.lr)
    loss_fn: nn.Module
    if cfg.train.get('label_smoothing', {}).get('enabled', False):
        smoothing = cfg.train.label_smoothing.smoothing
        loss_fn = LabelSmoothingLoss(cfg.model.num_classes, smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss()
    scheduler = create_scheduler(optimizer, cfg)
    return model, optimizer, loss_fn, scheduler


def setup_model_and_optimizer(cfg: DictConfig, device: torch.device):
    """Compatibility wrapper used in tests."""
    return create_model_and_optimizer(cfg, device)


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str) -> nn.Module:
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    return model


def save_model_with_metadata(model: nn.Module, path: str, metadata: dict) -> None:
    torch.save({'model': model.state_dict(), 'metadata': metadata}, path)


def load_model_with_metadata(model: nn.Module, path: str) -> Tuple[nn.Module, dict]:
    obj = torch.load(path, map_location='cpu')
    model.load_state_dict(obj['model'])
    return model, obj.get('metadata', {})


def get_model_save_path(cfg: DictConfig, kind: str) -> str | None:
    if not cfg.model_save.get('enabled', False):
        return None
    filename = f"{cfg.model.name}_{kind}.pth"
    os.makedirs(cfg.model_save.dir, exist_ok=True)
    return os.path.join(cfg.model_save.dir, filename)


def get_model_info(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total,
        'trainable_params': trainable,
        'model_name': model.__class__.__name__,
    }
