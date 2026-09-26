"""
Unit tests for loss functions
"""
import pytest
import torch
import sys
from pathlib import Path

# Add the project root to the import path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.losses import DiceLoss, BCEDiceLoss, FocalLoss


class TestDiceLoss:
    """Test the Dice loss function"""

    def test_dice_loss_perfect_prediction(self):
        """Test perfect predictions"""
        loss_fn = DiceLoss()
        pred = torch.ones(2, 1, 64, 64) * 10.0  # The sigmoid output is close to 1
        target = torch.ones(2, 1, 64, 64)
        loss = loss_fn(pred, target)

        # Perfect predictions should give a value close to 0
        assert loss.item() < 0.01

    def test_dice_loss_worst_prediction(self):
        """Test completely incorrect predictions"""
        loss_fn = DiceLoss()
        pred = torch.ones(2, 1, 64, 64) * 10.0  # The sigmoid output is close to 1
        target = torch.zeros(2, 1, 64, 64)
        loss = loss_fn(pred, target)

        # Completely incorrect predictions should give a value close to 1
        assert loss.item() > 0.9

    def test_dice_loss_shape(self):
        """Test the loss output shape"""
        loss_fn = DiceLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)

        # The loss should be a scalar
        assert loss.dim() == 0

    def test_dice_loss_gradient(self):
        """Test gradient computation"""
        loss_fn = DiceLoss()
        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        loss.backward()

        # Check that gradients are present
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


class TestBCEDiceLoss:
    """Test the combined binary cross-entropy and Dice loss"""

    def test_bce_dice_loss_forward(self):
        """Test the forward pass"""
        loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)

        # The loss should be a positive scalar
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_bce_dice_loss_weights(self):
        """Test different loss weights"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # Use only BCE
        loss_fn_bce = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)
        loss_bce = loss_fn_bce(pred, target)

        # Use only Dice
        loss_fn_dice = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)
        loss_dice = loss_fn_dice(pred, target)

        # The two loss values should differ
        assert abs(loss_bce.item() - loss_dice.item()) > 0.01

    def test_bce_dice_loss_gradient(self):
        """Test gradient computation"""
        loss_fn = BCEDiceLoss()
        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None


class TestFocalLoss:
    """Test the focal loss function"""

    def test_focal_loss_forward(self):
        """Test the forward pass"""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)

        # The loss should be a positive scalar
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_focal_loss_parameters(self):
        """Test different parameter configurations"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # Compare different gamma values
        loss_fn_1 = FocalLoss(alpha=0.25, gamma=1.0)
        loss_1 = loss_fn_1(pred, target)

        loss_fn_2 = FocalLoss(alpha=0.25, gamma=3.0)
        loss_2 = loss_fn_2(pred, target)

        # The loss values should differ
        assert abs(loss_1.item() - loss_2.item()) > 0.001

    def test_focal_loss_gradient(self):
        """Test gradient computation"""
        loss_fn = FocalLoss()
        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
