"""
Model inference utilities.

Provides pretrained model loading and prediction.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import cv2

from ..core.models.unet import UNet
from ..core.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Segmentation predictor.

    Loads a pretrained model and predicts segmentation masks for new images.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        n_channels: int = 3,
        n_classes: int = 1,
        bilinear: bool = True
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to pretrained model weights (.pth file).
            device: Execution device ("cuda" or "cpu").
            n_channels: Number of input image channels.
            n_classes: Number of output classes.
            bilinear: Use bilinear interpolation for upsampling.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Create the model.
        self.model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=bilinear
        )
        self.model.to(self.device)
        self.model.eval()

        # Load pretrained weights.
        if model_path is not None:
            self.load_model(model_path)
            logger.info(f"Loaded pretrained model from {model_path}")
        else:
            logger.warning("No pretrained model loaded. Using randomly initialized weights.")

        logger.info(f"Predictor initialized on device: {self.device}")

    def load_model(self, model_path: str):
        """
        Load pretrained model weights.

        Args:
            model_path: Path to the model weights file.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load weights.
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle supported checkpoint formats.
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Load from a training checkpoint.
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # Load a state_dict directly.
            self.model.load_state_dict(checkpoint)
            logger.info("Loaded model state dict")

        self.model.eval()

    def preprocess(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Preprocess the image.

        Args:
            image: Input image (H, W, C) or (H, W).
            target_size: Target dimensions (height, width); None preserves the original size.

        Returns:
            Preprocessed tensor (1, C, H, W).
        """
        # Resize.
        if target_size is not None:
            image = cv2.resize(image, (target_size[1], target_size[0]))

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

        # Add a batch dimension and convert to a tensor.
        image = torch.from_numpy(image.copy()).float()
        image = image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        return image

    def postprocess(
        self,
        output: torch.Tensor,
        threshold: float = 0.5,
        original_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Postprocess the model output.

        Args:
            output: Model output tensor (1, C, H, W).
            threshold: Binarization threshold.
            original_size: Original dimensions (height, width); resize back when provided.

        Returns:
            Binary mask (H, W).
        """
        # Apply sigmoid and convert to NumPy.
        output = torch.sigmoid(output)
        mask = output.squeeze().cpu().numpy()  # (H, W)

        # Binarize.
        mask = (mask > threshold).astype(np.uint8) * 255

        # Resize back to the original dimensions.
        if original_size is not None:
            mask = cv2.resize(mask, (original_size[1], original_size[0]))

        return mask

    def predict(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = (256, 256),
        threshold: float = 0.5,
        return_original_size: bool = True
    ) -> np.ndarray:
        """
        Predict a segmentation mask for one image.

        Args:
            image: Input image (H, W, C) or (H, W).
            target_size: Model input dimensions (height, width).
            threshold: Binarization threshold.
            return_original_size: Resize the result to the original image dimensions.

        Returns:
            Predicted binary mask (H, W).
        """
        original_size = image.shape[:2] if return_original_size else None

        # Preprocessing.
        input_tensor = self.preprocess(image, target_size)
        input_tensor = input_tensor.to(self.device)

        # Inference.
        with torch.no_grad():
            output = self.model(input_tensor)

        # Postprocessing.
        mask = self.postprocess(output, threshold, original_size)

        return mask


