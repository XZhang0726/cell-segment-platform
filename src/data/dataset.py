"""
Cell segmentation dataset loader.

Provides a PyTorch Dataset for loading paired images and masks.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Tuple

from .image_io import ImageIO
from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class CellSegmentationDataset(Dataset):
    """
    Cell segmentation dataset.

    Args:
        image_dir: Path to the image directory.
        mask_dir: Path to the mask directory.
        transform: Data augmentation transform.
        image_size: Target image dimensions (height, width).
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        image_size: Optional[Tuple[int, int]] = None
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size

        # List all image files.
        self.image_files = self._get_image_files()

        logger.info(f"Loaded dataset: {len(self.image_files)} images from {image_dir}")

    def _get_image_files(self) -> List[Path]:
        """List all image files."""
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        image_files = []

        for ext in image_extensions:
            image_files.extend(self.image_dir.glob(f'*{ext}'))
            image_files.extend(self.image_dir.glob(f'*{ext.upper()}'))

        return sorted(image_files)

    def __len__(self) -> int:
        """Return the dataset size."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            (image, mask) tuple.
        """
        # Load an image.
        image_path = self.image_files[idx]
        image = ImageIO.load_image(image_path)

        # Load the corresponding mask.
        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists():
            # Try alternative mask filenames.
            mask_path = self.mask_dir / (image_path.stem + '_mask' + image_path.suffix)

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image: {image_path}")

        mask = ImageIO.load_image(mask_path, grayscale=True)

        # Resize the image.
        if self.image_size is not None:
            import cv2
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))

        # Apply data augmentation.
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Convert to tensors.
        image = self._to_tensor(image)
        mask = self._to_tensor(mask)

        return image, mask

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert a NumPy array to a PyTorch tensor.

        Args:
            image: NumPy array (H, W) or (H, W, C).

        Returns:
            PyTorch tensor (C, H, W) or (1, H, W).
        """
        # Normalize to [0, 1].
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Reorder dimensions.
        if image.ndim == 2:
            # Grayscale: (H, W) -> (1, H, W).
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:
            # Color: (H, W, C) -> (C, H, W).
            image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image.copy()).float()

