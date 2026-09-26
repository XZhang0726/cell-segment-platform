"""
Loss functions.

Provides loss functions for cell segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class DiceLoss(nn.Module):
    """
    Dice loss.

    Suitable for segmentation tasks, including those with class imbalance.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            pred: Predicted values (N, C, H, W).
            target: Target values (N, C, H, W).

        Returns:
            Dice loss value.
        """
        pred = torch.sigmoid(pred)

        # Flatten the tensors.
        pred = pred.view(-1)
        target = target.view(-1)

        # Compute the Dice coefficient.
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined binary cross-entropy and Dice loss.

    Combine binary cross-entropy with Dice loss.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined loss.

        Args:
            pred: Predicted values (N, C, H, W).
            target: Target values (N, C, H, W).

        Returns:
            Combined loss value.
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss.

    Address class imbalance by emphasizing difficult examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            pred: Predicted values (N, C, H, W).
            target: Target values (N, C, H, W).

        Returns:
            Focal loss value.
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()

