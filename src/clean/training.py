import logging
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, f1_score
from omegaconf import DictConfig

from .models import create_model_and_optimizer
from .utils import EarlyStopping

log = logging.getLogger(__name__)

AMP_AVAILABLE = hasattr(torch.cuda, 'amp')


def train_one_epoch(loader: DataLoader, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, loss_fn,
                    device: torch.device, scaler: Optional[torch.cuda.amp.GradScaler] = None):
    model.train()
    losses = []
    all_preds = []
    all_targets = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        preds = outputs.argmax(1).detach().cpu()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.cpu().tolist())
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return {
        'train_loss': float(sum(losses)/len(losses)),
        'train_acc': float(acc),
        'train_f1': float(f1),
    }


def validate_one_epoch(loader: DataLoader, model: torch.nn.Module, loss_fn,
                        device: torch.device):
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for images, tgt in loader:
            images, tgt = images.to(device), tgt.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, tgt)
            losses.append(loss.item())
            p = outputs.argmax(1).cpu()
            preds.extend(p.tolist())
            targets.extend(tgt.cpu().tolist())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    return {
        'val_loss': float(sum(losses)/len(losses)),
        'val_acc': float(acc),
        'val_f1': float(f1),
    }


def train_single_model(cfg: DictConfig, train_loader: DataLoader,
                        val_loader: Optional[DataLoader], device: torch.device):
    model, optimizer, loss_fn, scheduler = create_model_and_optimizer(cfg, device)
    scaler = torch.cuda.amp.GradScaler() if cfg.train.get('mixed_precision', {}).get('enabled', False) and device.type == 'cuda' and AMP_AVAILABLE else None
    early_cfg = cfg.validation.get('early_stopping', {})
    early = EarlyStopping(
        patience=early_cfg.get('patience', 10),
        min_delta=early_cfg.get('min_delta', 0.0),
        monitor=early_cfg.get('monitor', 'val_loss'),
        mode=early_cfg.get('mode', 'min'),
    ) if early_cfg.get('enabled', False) and val_loader is not None else None

    epochs = cfg.train.epochs
    for _ in range(epochs):
        train_metrics = train_one_epoch(train_loader, model, optimizer, loss_fn, device, scaler)
        if val_loader is not None:
            val_metrics = validate_one_epoch(val_loader, model, loss_fn, device)
            stop = early(val_metrics) if early is not None else False
        else:
            val_metrics = {}
            stop = False
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics.get('val_loss'))
            else:
                scheduler.step()
        if stop:
            break
    return model


def train_kfold_models(cfg: DictConfig, kfold_data: Tuple, device: torch.device) -> List[torch.nn.Module]:
    folds, df, img_dir, train_t, val_t, _ = kfold_data
    models: List[torch.nn.Module] = []
    for fold_idx in range(len(folds)):
        train_loader, val_loader, *_ = get_kfold_loaders(fold_idx, folds, df, img_dir, train_t, val_t, cfg)
        model = train_single_model(cfg, train_loader, val_loader, device)
        models.append(model)
    return models

from .data import get_kfold_loaders


def update_scheduler(scheduler, val_metrics, cfg: Optional[DictConfig]):
    if scheduler is None:
        return None
    if isinstance(scheduler, ReduceLROnPlateau):
        metric_name = 'val_loss'
        if cfg is not None:
            metric_name = cfg.train.scheduler.plateau.get('monitor', 'val_loss')
        metric = val_metrics.get(metric_name) if val_metrics else None
        scheduler.step(metric)
    else:
        scheduler.step()
    return scheduler.optimizer.param_groups[0]['lr']
