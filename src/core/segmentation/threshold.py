"""
Threshold-based segmentation algorithms.

Provides several thresholding methods.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ThresholdSegmentation:
    """Threshold-based segmentation methods."""

    @staticmethod
    def otsu_threshold(
        image: np.ndarray,
        return_threshold: bool = False
    ) -> np.ndarray:
        """
        Segment using an automatically determined Otsu threshold.

        Args:
            image: Input grayscale image.
            return_threshold: Return the threshold as well as the image.

        Returns:
            Binary image, or (binary image, threshold) when return_threshold=True.
        """
        if image.ndim != 2:
            raise ValueError("Otsu threshold requires grayscale image")

        # Determine the threshold automatically using Otsu's method.
        threshold_value, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        logger.debug(f"Otsu threshold value: {threshold_value:.2f}")

        if return_threshold:
            return binary, threshold_value
        return binary

    @staticmethod
    def fixed_threshold(
        image: np.ndarray,
        threshold: int = 127,
        max_value: int = 255,
        threshold_type: str = 'binary'
    ) -> np.ndarray:
        """
        Segment using a fixed threshold.

        Args:
            image: Input grayscale image.
            threshold: Threshold value.
            max_value: Maximum output value.
            threshold_type: Threshold type ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv').

        Returns:
            Thresholded image.
        """
        if image.ndim != 2:
            raise ValueError("Fixed threshold requires grayscale image")

        threshold_types = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'trunc': cv2.THRESH_TRUNC,
            'tozero': cv2.THRESH_TOZERO,
            'tozero_inv': cv2.THRESH_TOZERO_INV
        }

        if threshold_type not in threshold_types:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

        _, binary = cv2.threshold(
            image, threshold, max_value, threshold_types[threshold_type]
        )

        logger.debug(f"Fixed threshold: {threshold}, type: {threshold_type}")
        return binary

    @staticmethod
    def adaptive_threshold(
        image: np.ndarray,
        max_value: int = 255,
        method: str = 'gaussian',
        threshold_type: str = 'binary',
        block_size: int = 11,
        C: int = 2
    ) -> np.ndarray:
        """
        Segment using adaptive thresholding.

        Args:
            image: Input grayscale image.
            max_value: Maximum output value.
            method: Adaptive thresholding method ('mean', 'gaussian').
            threshold_type: Threshold type ('binary', 'binary_inv').
            block_size: Neighborhood size; must be odd.
            C: Constant subtracted from the local mean or weighted mean.

        Returns:
            Thresholded image.
        """
        if image.ndim != 2:
            raise ValueError("Adaptive threshold requires grayscale image")

        if block_size % 2 == 0:
            block_size += 1
            logger.warning(f"Block size must be odd, adjusted to {block_size}")

        adaptive_methods = {
            'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
            'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        }

        threshold_types = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV
        }

        if method not in adaptive_methods:
            raise ValueError(f"Unknown adaptive method: {method}")

        if threshold_type not in threshold_types:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

        binary = cv2.adaptiveThreshold(
            image,
            max_value,
            adaptive_methods[method],
            threshold_types[threshold_type],
            block_size,
            C
        )

        logger.debug(f"Adaptive threshold: method={method}, block_size={block_size}")
        return binary

    @staticmethod
    def multi_otsu(
        image: np.ndarray,
        n_classes: int = 3
    ) -> np.ndarray:
        """
        Segment using multiple Otsu thresholds.

        Args:
            image: Input grayscale image.
            n_classes: Number of classes.

        Returns:
            Segmented image.
        """
        if image.ndim != 2:
            raise ValueError("Multi-Otsu requires grayscale image")

        from skimage.filters import threshold_multiotsu

        # Compute multiple thresholds.
        thresholds = threshold_multiotsu(image, classes=n_classes)

        # Segment the image using the thresholds.
        segmented = np.digitize(image, bins=thresholds)

        logger.debug(f"Multi-Otsu thresholds: {thresholds}")
        return segmented.astype(np.uint8)


# Convenience functions.
def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Convenience wrapper for Otsu thresholding."""
    return ThresholdSegmentation.otsu_threshold(image)


def adaptive_threshold(image: np.ndarray, block_size: int = 11) -> np.ndarray:
    """Convenience wrapper for adaptive thresholding."""
    return ThresholdSegmentation.adaptive_threshold(image, block_size=block_size)


def fixed_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Convenience wrapper for fixed thresholding."""
    return ThresholdSegmentation.fixed_threshold(image, threshold=threshold)
