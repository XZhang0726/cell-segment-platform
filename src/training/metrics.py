"""
Evaluation metrics.

Provides evaluation metrics for cell segmentation.
"""
import torch
import numpy as np
from typing import Dict

from ..core.utils.logger import get_logger

logger = get_logger(__name__)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    Compute the Dice coefficient.

    Args:
        pred: Predicted values (N, C, H, W).
        target: Target values (N, C, H, W).
        smooth: Smoothing factor.

    Returns:
        Dice coefficient.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # Flatten the tensors.
    pred = pred.view(-1)
    target = target.view(-1)

    # Compute the Dice coefficient.
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """
    Compute intersection over union (IoU).

    Args:
        pred: Predicted values (N, C, H, W).
        target: Target values (N, C, H, W).
        smooth: Smoothing factor.

    Returns:
        IoU score.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # Flatten the tensors.
    pred = pred.view(-1)
    target = target.view(-1)

    # Compute IoU.
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute pixel accuracy.

    Args:
        pred: Predicted values (N, C, H, W).
        target: Target values (N, C, H, W).

    Returns:
        Pixel accuracy.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    correct = (pred == target).sum()
    total = target.numel()
    accuracy = correct / total

    return accuracy.item()


def precision_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute precision.

    Args:
        pred: Predicted values (N, C, H, W).
        target: Target values (N, C, H, W).
        smooth: Smoothing factor.

    Returns:
        Precision.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # Flatten the tensors.
    pred = pred.view(-1)
    target = target.view(-1)

    # Compute precision.
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    precision = (true_positive + smooth) / (predicted_positive + smooth)

    return precision.item()


def recall_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute recall.

    Args:
        pred: Predicted values (N, C, H, W).
        target: Target values (N, C, H, W).
        smooth: Smoothing factor.

    Returns:
        Recall.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # Flatten the tensors.
    pred = pred.view(-1)
    target = target.view(-1)

    # Compute recall.
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    recall = (true_positive + smooth) / (actual_positive + smooth)

    return recall.item()


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        pred: Predicted values (N, C, H, W).
        target: Target values (N, C, H, W).

    Returns:
        Dictionary containing all metrics.
    """
    metrics = {
        'dice': dice_coefficient(pred, target),
        'iou': iou_score(pred, target),
        'accuracy': pixel_accuracy(pred, target),
        'precision': precision_score(pred, target),
        'recall': recall_score(pred, target)
    }

    return metrics


