import os
import random
from dataclasses import dataclass, field
from typing import List

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional
    wandb = None

import hydra
from hydra.core.config_store import ConfigStore


# =============================
# Configuration dataclass
# =============================
@dataclass
class TrainConfig:
    data_path: str = "../input/data"
    output_dir: str = "../output"

    model_name: str = "tf_efficientnetv2_s"
    img_size: int = 224
    lr: float = 2e-4
    epochs: int = 5
    batch_size: int = 16
    num_workers: int = 0
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1

    seed: int = 42
    n_runs: int = 1
    kfold: bool = True
    n_splits: int = 5
    holdout_ratio: float = 0.2

    use_wandb: bool = False
    project: str = "doc_classification"
    tta_count: int = 5

    def update_for_run(self, run_idx: int):
        self.seed = self.seed + run_idx


cs = ConfigStore.instance()
cs.store(name="config", node=TrainConfig)


# =============================
# Utility functions
# =============================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(img_size: int):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(0.1, 0.1, 15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    # validation transform은 train과 동일
    val_transform = train_transform
    return train_transform, val_transform


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, target = self.df.iloc[idx]
        image = np.array(Image.open(os.path.join(self.img_dir, img_name)))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, target


def build_model(model_name: str, num_classes: int = 17):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model


# Training for one epoch

def train_epoch(loader, model, optimizer, loss_fn, device, scaler):
    model.train()
    losses = []
    preds, targets = [], []
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            output = model(images)
            loss = loss_fn(output, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        preds.extend(output.argmax(1).detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    return np.mean(losses), acc, f1


# Validation

def validate_epoch(loader, model, loss_fn, device):
    model.eval()
    losses = []
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="valid", leave=False):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                output = model(images)
                loss = loss_fn(output, labels)
            losses.append(loss.item())
            preds.extend(output.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    return np.mean(losses), acc, f1


# TTA prediction

def predict_tta(model, loader, device, tta_count: int):
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="tta", leave=False):
            batch_preds = []
            for _ in range(tta_count):
                with autocast():
                    out = model(images.to(device))
                    batch_preds.append(out.softmax(1))
            batch_pred = torch.stack(batch_preds).mean(0)
            preds.append(batch_pred.cpu().numpy())
    return np.vstack(preds)


# Main training loop per split

def run_training(train_df, val_df, cfg: TrainConfig, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, val_tf = get_transforms(cfg.img_size)

    trn_dataset = ImageDataset(train_df, f"{cfg.data_path}/train/", transform=train_tf)
    val_dataset = ImageDataset(val_df, f"{cfg.data_path}/train/", transform=val_tf)

    trn_loader = DataLoader(trn_dataset, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    model = build_model(cfg.model_name).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_f1 = -1
    best_state = None

    for epoch in range(cfg.epochs):
        trn_loss, trn_acc, trn_f1 = train_epoch(trn_loader, model, optimizer, loss_fn, device, scaler)
        val_loss, val_acc, val_f1 = validate_epoch(val_loader, model, loss_fn, device)
        scheduler.step()

        if cfg.use_wandb and wandb is not None:
            wandb.log({
                f"{run_name}/train_loss": trn_loss,
                f"{run_name}/train_acc": trn_acc,
                f"{run_name}/train_f1": trn_f1,
                f"{run_name}/val_loss": val_loss,
                f"{run_name}/val_acc": val_acc,
                f"{run_name}/val_f1": val_f1,
            })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()

    return best_state, best_f1


# =============================
# Main
# =============================

@hydra.main(config_path=None, config_name="config")
def main(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    all_models: List[dict] = []
    for run in range(cfg.n_runs):
        cfg.update_for_run(run)
        set_seed(cfg.seed)
        run_name = f"run_{run}"

        wb_run = None
        if cfg.use_wandb and wandb is not None:
            wb_run = wandb.init(project=cfg.project, name=run_name, config=cfg)

        if cfg.kfold:
            skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
            fold_models = []
            data = pd.read_csv(f"{cfg.data_path}/train.csv")
            for fold, (trn_idx, val_idx) in enumerate(skf.split(data, data["target"])):
                trn_df = data.iloc[trn_idx]
                val_df = data.iloc[val_idx]
                state, f1 = run_training(trn_df, val_df, cfg, f"{run_name}_fold{fold}")
                fold_models.append(state)
                if wb_run:
                    wandb.log({f"{run_name}/fold{fold}_best_f1": f1})
            all_models.append(fold_models)
        else:
            data = pd.read_csv(f"{cfg.data_path}/train.csv")
            train_df, val_df = train_test_split(
                data,
                test_size=cfg.holdout_ratio,
                stratify=data["target"],
                random_state=cfg.seed,
            )
            state, f1 = run_training(train_df, val_df, cfg, run_name)
            all_models.append([state])
            if wb_run:
                wandb.log({f"{run_name}/best_f1": f1})

        if wb_run:
            wb_run.finish()

    # Save models
    for i, models in enumerate(all_models):
        for j, state in enumerate(models):
            torch.save(state, os.path.join(cfg.output_dir, f"model_run{i}_fold{j}.pth"))


if __name__ == "__main__":
    main()
