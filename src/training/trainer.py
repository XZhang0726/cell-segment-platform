"""
Model training utilities.

Provides a trainer for cell segmentation models.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import time

from ..core.models.unet import UNet
from .losses import DiceLoss, BCEDiceLoss, FocalLoss
from .metrics import calculate_metrics
from .config import TrainingConfig
from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Model trainer.

    Handles training, validation, and checkpoint management.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Training configuration.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Select the device.
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize the loss function.
        self.criterion = self._get_loss_function()

        # Initialize the optimizer.
        self.optimizer = self._get_optimizer()

        # Initialize the learning-rate scheduler.
        self.scheduler = self._get_scheduler()

        # Training state.
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0

        # Create the output directory.
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _get_loss_function(self) -> nn.Module:
        """Get the loss function."""
        loss_type = self.config.loss_type.lower()

        if loss_type == "dice":
            return DiceLoss()
        elif loss_type == "bce_dice":
            return BCEDiceLoss(
                bce_weight=self.config.bce_weight,
                dice_weight=self.config.dice_weight
            )
        elif loss_type == "focal":
            return FocalLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer."""
        optimizer_name = self.config.optimizer.lower()

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get the learning-rate scheduler."""
        scheduler_name = self.config.scheduler.lower()

        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor
            )
        elif scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor
            )
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}, using no scheduler")
            return None

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'iou': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            # Move data to the selected device.
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass.
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass.
            loss.backward()
            self.optimizer.step()

            # Record the loss.
            total_loss += loss.item()

            # Compute metrics.
            with torch.no_grad():
                batch_metrics = calculate_metrics(outputs, masks)
                for key in total_metrics:
                    total_metrics[key] += batch_metrics[key]

            # Write log output.
            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Epoch [{self.current_epoch}/{self.config.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Dice: {batch_metrics['dice']:.4f}"
                )

        # Compute averages.
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        avg_metrics['loss'] = avg_loss

        return avg_metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'iou': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}

        with torch.no_grad():
            for images, masks in self.val_loader:
                # Move data to the selected device.
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass.
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Record the loss.
                total_loss += loss.item()

                # Compute metrics.
                batch_metrics = calculate_metrics(outputs, masks)
                for key in total_metrics:
                    total_metrics[key] += batch_metrics[key]

        # Compute averages.
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        avg_metrics['loss'] = avg_loss

        return avg_metrics

    def save_checkpoint(self, filename: str = "checkpoint.pth"):
        """
        Save a checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = self.save_dir / filename
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_dice = checkpoint['best_val_dice']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch})")

    def train(self):
        """
        Run the complete training loop.
        """
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()

            # Training.
            train_metrics = self.train_epoch()

            # Validation.
            val_metrics = self.validate()

            # Update the learning rate.
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Record the current learning rate.
            current_lr = self.optimizer.param_groups[0]['lr']

            # Compute the epoch duration.
            epoch_time = time.time() - start_time

            # Log the epoch summary.
            logger.info(
                f"\nEpoch [{self.current_epoch}/{self.config.epochs}] Summary:\n"
                f"  Train Loss: {train_metrics['loss']:.4f} | Train Dice: {train_metrics['dice']:.4f}\n"
                f"  Val Loss: {val_metrics['loss']:.4f} | Val Dice: {val_metrics['dice']:.4f}\n"
                f"  Val IoU: {val_metrics['iou']:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f}\n"
                f"  Learning Rate: {current_lr:.6f} | Time: {epoch_time:.2f}s"
            )

            # Save the best model.
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint("best_loss.pth")
                logger.info(f"  Saved best loss model (loss: {self.best_val_loss:.4f})")

            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.save_checkpoint("best_dice.pth")
                logger.info(f"  Saved best dice model (dice: {self.best_val_dice:.4f})")

            # Save checkpoints periodically.
            if (self.current_epoch) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch}.pth")

        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation dice: {self.best_val_dice:.4f}")

