"""
Morphological image operations.

Provides common morphological operations.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MorphologicalOps:
    """Morphological operations."""

    @staticmethod
    def get_kernel(shape: str = 'rect', size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        Create a morphological kernel.

        Args:
            shape: Kernel shape ('rect', 'ellipse', 'cross').
            size: Kernel size.

        Returns:
            Morphological kernel.
        """
        shapes = {
            'rect': cv2.MORPH_RECT,
            'ellipse': cv2.MORPH_ELLIPSE,
            'cross': cv2.MORPH_CROSS
        }

        if shape not in shapes:
            raise ValueError(f"Unknown kernel shape: {shape}")

        kernel = cv2.getStructuringElement(shapes[shape], size)
        return kernel

    @staticmethod
    def erode(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect',
        iterations: int = 1
    ) -> np.ndarray:
        """
        Apply erosion.

        Args:
            image: Input image.
            kernel_size: Kernel size.
            kernel_shape: Kernel shape.
            iterations: Number of iterations.

        Returns:
            Eroded image.
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        eroded = cv2.erode(image, kernel, iterations=iterations)

        logger.debug(f"Erosion: kernel_size={kernel_size}, iterations={iterations}")
        return eroded

    @staticmethod
    def dilate(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect',
        iterations: int = 1
    ) -> np.ndarray:
        """
        Apply dilation.

        Args:
            image: Input image.
            kernel_size: Kernel size.
            kernel_shape: Kernel shape.
            iterations: Number of iterations.

        Returns:
            Dilated image.
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        dilated = cv2.dilate(image, kernel, iterations=iterations)

        logger.debug(f"Dilation: kernel_size={kernel_size}, iterations={iterations}")
        return dilated

    @staticmethod
    def opening(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect'
    ) -> np.ndarray:
        """
        Apply opening: erosion followed by dilation to remove small objects.

        Args:
            image: Input image.
            kernel_size: Kernel size.
            kernel_shape: Kernel shape.

        Returns:
            Opened image.
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        logger.debug(f"Opening: kernel_size={kernel_size}")
        return opened

    @staticmethod
    def closing(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect'
    ) -> np.ndarray:
        """
        Apply closing: dilation followed by erosion to fill small holes.

        Args:
            image: Input image.
            kernel_size: Kernel size.
            kernel_shape: Kernel shape.

        Returns:
            Closed image.
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        logger.debug(f"Closing: kernel_size={kernel_size}")
        return closed

    @staticmethod
    def gradient(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect'
    ) -> np.ndarray:
        """
        Compute the morphological gradient: dilation minus erosion to extract boundaries.

        Args:
            image: Input image.
            kernel_size: Kernel size.
            kernel_shape: Kernel shape.

        Returns:
            Gradient image.
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

        logger.debug(f"Morphological gradient: kernel_size={kernel_size}")
        return gradient

    @staticmethod
    def tophat(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect'
    ) -> np.ndarray:
        """
        Apply the top-hat transform: original minus opening to extract bright details.

        Args:
            image: Input image.
            kernel_size: Kernel size.
            kernel_shape: Kernel shape.

        Returns:
            Top-hat transformed image.
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

        logger.debug(f"Top-hat transform: kernel_size={kernel_size}")
        return tophat

    @staticmethod
    def blackhat(
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        kernel_shape: str = 'rect'
    ) -> np.ndarray:
        """
        Apply the black-hat transform: closing minus original to extract dark details.

        Args:
            image: Input image.
            kernel_size: Kernel size.
            kernel_shape: Kernel shape.

        Returns:
            Black-hat transformed image.
        """
        kernel = MorphologicalOps.get_kernel(kernel_shape, kernel_size)
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

        logger.debug(f"Black-hat transform: kernel_size={kernel_size}")
        return blackhat


# Convenience functions.
def erode(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Convenience wrapper for erosion."""
    return MorphologicalOps.erode(image, kernel_size)


def dilate(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Convenience wrapper for dilation."""
    return MorphologicalOps.dilate(image, kernel_size)


def opening(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Convenience wrapper for opening."""
    return MorphologicalOps.opening(image, kernel_size)


def closing(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Convenience wrapper for closing."""
    return MorphologicalOps.closing(image, kernel_size)
