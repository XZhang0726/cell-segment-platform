"""
Training configuration management.

Provides configuration of training hyperparameters.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import yaml

from ..core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model configuration.
    model_name: str = "unet"
    n_channels: int = 3
    n_classes: int = 1
    bilinear: bool = True

    # Training configuration.
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Data configuration.
    image_size: tuple = (256, 256)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Loss function configuration.
    loss_type: str = "bce_dice"  # "dice", "bce_dice", "focal"
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # Optimizer configuration.
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    momentum: float = 0.9

    # Learning-rate scheduler configuration.
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Additional configuration.
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    save_dir: str = "checkpoints"
    log_interval: int = 10

    def save(self, path: str):
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

        logger.info(f"Saved config to {path}")

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        logger.info(f"Loaded config from {path}")
        return cls(**config_dict)


def get_default_config() -> TrainingConfig:
    """Get the default configuration."""
    return TrainingConfig()

