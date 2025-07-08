import logging
from omegaconf import DictConfig
import hydra

from .utils import (set_seed, setup_wandb, finish_wandb, get_device,
                    log_hyperparameters)
from .data import prepare_data_loaders
from .training import train_single_model, train_kfold_models
from .inference import run_inference

log = logging.getLogger(__name__)


@hydra.main(config_path=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.train.seed)
    setup_wandb(cfg)
    log_hyperparameters(cfg)
    device = get_device(cfg)
    loaders = prepare_data_loaders(cfg, cfg.train.seed)

    if cfg.validation.strategy == 'kfold':
        models = train_kfold_models(cfg, loaders.kfold_data, device)
        run_inference(models, loaders.test_loader, loaders.test_loader.dataset, cfg, device, is_kfold=True)
    else:
        model = train_single_model(cfg, loaders.train_loader, loaders.val_loader, device)
        run_inference(model, loaders.test_loader, loaders.test_loader.dataset, cfg, device, is_kfold=False)

    finish_wandb(cfg)


if __name__ == '__main__':
    cfg = DictConfig({})
    main(cfg)
