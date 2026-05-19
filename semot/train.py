"""Hydra entry point for Sinkhorn divergence training.

Preferred launch method:
    spt run experiments/main.yaml [overrides...]

Direct Python launch (also works):
    python semot/train.py [overrides...]
"""

import hydra
from omegaconf import DictConfig
from stable_pretraining.config import instantiate_from_config


@hydra.main(config_path="../experiments", config_name="main", version_base=None)
def main(cfg: DictConfig) -> None:
    manager = instantiate_from_config(cfg)
    if callable(manager):
        manager()


if __name__ == "__main__":
    main()
