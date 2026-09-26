"""
Data augmentation.

Provides data augmentation transforms for cell segmentation.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional

from ..core.utils.logger import get_logger

logger = get_logger(__name__)


def get_training_augmentation(image_size: tuple = (256, 256)):
    """
    Get training augmentation transforms.

    Args:
        image_size: Target image dimensions (height, width).

    Returns:
        Composed albumentations transforms.
    """
    train_transform = A.Compose([
        # Geometric transforms.
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        ),

        # Elastic deformation for cell-image augmentation.
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.3
        ),

        # Grid distortion.
        A.GridDistortion(p=0.3),
    ])

    logger.info(f"Created training augmentation pipeline for size {image_size}")
    return train_transform


def get_validation_augmentation(image_size: tuple = (256, 256)):
    """
    Get validation transforms, applying resizing only.

    Args:
        image_size: Target image dimensions (height, width).

    Returns:
        Composed albumentations transforms.
    """
    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
    ])

    logger.info(f"Created validation augmentation pipeline for size {image_size}")
    return val_transform


def get_test_augmentation(image_size: tuple = (256, 256)):
    """
    Get test transforms, applying resizing only.

    Args:
        image_size: Target image dimensions (height, width).

    Returns:
        Composed albumentations transforms.
    """
    test_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
    ])

    logger.info(f"Created test augmentation pipeline for size {image_size}")
    return test_transform

