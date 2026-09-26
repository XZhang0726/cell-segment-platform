"""
Unit tests for evaluation metrics
"""
import pytest
import torch
import sys
from pathlib import Path

# Add the project root to the import path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.metrics import (
    dice_coefficient,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
    calculate_metrics
)


class TestDiceCoefficient:
    """Test the Dice coefficient"""

    def test_dice_perfect_prediction(self):
        """Test perfect predictions"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        dice = dice_coefficient(pred, target)

        # Perfect predictions should give a value close to 1
        assert dice > 0.99

    def test_dice_worst_prediction(self):
        """Test completely incorrect predictions"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.zeros(2, 1, 64, 64)
        dice = dice_coefficient(pred, target)

        # Completely incorrect predictions should give a value close to 0
        assert dice < 0.01

    def test_dice_return_type(self):
        """Test the return type"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        dice = dice_coefficient(pred, target)

        # The result should be a float
        assert isinstance(dice, float)
        assert 0 <= dice <= 1


class TestIoUScore:
    """Test the IoU score"""

    def test_iou_perfect_prediction(self):
        """Test perfect predictions"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        iou = iou_score(pred, target)

        # Perfect predictions should give a value close to 1
        assert iou > 0.99

    def test_iou_worst_prediction(self):
        """Test completely incorrect predictions"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.zeros(2, 1, 64, 64)
        iou = iou_score(pred, target)

        # Completely incorrect predictions should give a value close to 0
        assert iou < 0.01

    def test_iou_return_type(self):
        """Test the return type"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        iou = iou_score(pred, target)

        assert isinstance(iou, float)
        assert 0 <= iou <= 1


class TestPixelAccuracy:
    """Test pixel accuracy"""

    def test_accuracy_perfect_prediction(self):
        """Test perfect predictions"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        acc = pixel_accuracy(pred, target)

        # Perfect predictions should give a value close to 1
        assert acc > 0.99

    def test_accuracy_return_type(self):
        """Test the return type"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        acc = pixel_accuracy(pred, target)

        assert isinstance(acc, float)
        assert 0 <= acc <= 1


class TestPrecisionScore:
    """Test precision"""

    def test_precision_perfect_prediction(self):
        """Test perfect predictions"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        precision = precision_score(pred, target)

        assert precision > 0.99

    def test_precision_return_type(self):
        """Test the return type"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        precision = precision_score(pred, target)

        assert isinstance(precision, float)
        assert 0 <= precision <= 1


class TestRecallScore:
    """Test recall"""

    def test_recall_perfect_prediction(self):
        """Test perfect predictions"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        recall = recall_score(pred, target)

        assert recall > 0.99

    def test_recall_return_type(self):
        """Test the return type"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        recall = recall_score(pred, target)

        assert isinstance(recall, float)
        assert 0 <= recall <= 1


class TestCalculateMetrics:
    """Test the combined metric calculation"""

    def test_calculate_metrics_return_type(self):
        """Test the return type"""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        metrics = calculate_metrics(pred, target)

        # The result should be a dictionary
        assert isinstance(metrics, dict)

        # Check that all metrics are present
        expected_keys = ['dice', 'iou', 'accuracy', 'precision', 'recall']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert 0 <= metrics[key] <= 1

    def test_calculate_metrics_perfect_prediction(self):
        """Test all metrics with perfect predictions"""
        pred = torch.ones(2, 1, 64, 64) * 10.0
        target = torch.ones(2, 1, 64, 64)
        metrics = calculate_metrics(pred, target)

        # All metrics should be close to 1
        for key, value in metrics.items():
            assert value > 0.99, f"{key} should be close to 1 for perfect prediction"
