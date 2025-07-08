import os
import logging
import random
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

import wandb

log = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best = None
        self.counter = 0

    def __call__(self, metrics: dict) -> bool:
        value = metrics[self.monitor]
        if self.best is None:
            self.best = value
            return False
        improvement = (value < self.best - self.min_delta) if self.mode == 'min' else (value > self.best + self.min_delta)
        if improvement:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    log.info(f"시드 고정 완료: {seed}")


def setup_wandb(cfg: DictConfig) -> None:
    if not cfg.wandb.enabled:
        log.info("wandb 비활성화됨")
        return
    entity = os.getenv('WANDB_ENTITY', cfg.wandb.entity)
    project = os.getenv('WANDB_PROJECT', cfg.wandb.project)
    wandb.init(project=project, entity=entity, name=cfg.wandb.run_name,
               tags=cfg.wandb.tags, notes=cfg.wandb.notes, config=dict(cfg))
    log.info(f"wandb 초기화 완료 - 프로젝트: {project}")


def finish_wandb(cfg: DictConfig) -> None:
    if cfg.wandb.enabled:
        wandb.finish()
        log.info("wandb 세션 종료")


def get_device(cfg: DictConfig) -> torch.device:
    if cfg.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    log.info(f"사용 장치: {device.type}")
    return device


def log_hyperparameters(cfg: DictConfig) -> None:
    msg = (
        f"하이퍼파라미터 설정 - 모델: {cfg.model.name}, 이미지 크기: {cfg.data.img_size}, "
        f"학습률: {cfg.train.lr}, 에포크: {cfg.train.epochs}, 배치 크기: {cfg.train.batch_size}"
    )
    log.info(msg)


def upload_to_wandb(df: Any, cfg: DictConfig) -> None:
    if cfg.wandb.enabled:
        table = wandb.Table(dataframe=df)
        wandb.log({'predictions': table})
