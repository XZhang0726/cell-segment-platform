"""
Unit tests for dataset loaders
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path
import tempfile
from PIL import Image

# Add the project root to the import path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import CellSegmentationDataset


class TestCellSegmentationDataset:
    """Test the cell segmentation dataset"""

    def create_test_images(self, tmpdir, num_images=5):
        """Create paired synthetic images and masks."""
        image_dir = Path(tmpdir) / "images"
        mask_dir = Path(tmpdir) / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_images):
            # Create an RGB image
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(image).save(image_dir / f"image_{i}.png")

            # Create a grayscale mask
            mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
            Image.fromarray(mask, mode='L').save(mask_dir / f"image_{i}.png")

        return image_dir, mask_dir

    def test_dataset_initialization(self):
        """Test dataset initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir = self.create_test_images(tmpdir, num_images=5)

            dataset = CellSegmentationDataset(
                image_dir=str(image_dir),
                mask_dir=str(mask_dir)
            )

            # Check the dataset size
            assert len(dataset) == 5

    def test_dataset_len(self):
        """Test the dataset length"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir = self.create_test_images(tmpdir, num_images=10)

            dataset = CellSegmentationDataset(
                image_dir=str(image_dir),
                mask_dir=str(mask_dir)
            )

            assert len(dataset) == 10

    def test_dataset_getitem(self):
        """Test retrieving one sample"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir = self.create_test_images(tmpdir, num_images=5)

            dataset = CellSegmentationDataset(
                image_dir=str(image_dir),
                mask_dir=str(mask_dir)
            )

            # Get the first sample
            image, mask = dataset[0]

            # Check the return type
            assert isinstance(image, torch.Tensor)
            assert isinstance(mask, torch.Tensor)

            # Check tensor shapes (C, H, W)
            assert image.dim() == 3
            assert mask.dim() == 3
            assert image.shape[0] == 3  # RGB image
            assert mask.shape[0] == 1   # Single-channel mask

    def test_dataset_with_image_size(self):
        """Test an explicit image size"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir, mask_dir = self.create_test_images(tmpdir, num_images=3)

            dataset = CellSegmentationDataset(
                image_dir=str(image_dir),
                mask_dir=str(mask_dir),
                image_size=(128, 128)
            )

            image, mask = dataset[0]

            # Check the resized dimensions
            assert image.shape == (3, 128, 128)
            assert mask.shape == (1, 128, 128)
