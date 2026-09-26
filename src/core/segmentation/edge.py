"""
Edge detection algorithms.

Provides several edge detection methods.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EdgeDetection:
    """Edge detection methods."""

    @staticmethod
    def canny(
        image: np.ndarray,
        threshold1: int = 50,
        threshold2: int = 150,
        aperture_size: int = 3,
        L2gradient: bool = False
    ) -> np.ndarray:
        """
        Detect edges using the Canny algorithm.

        Args:
            image: Input grayscale image.
            threshold1: Lower hysteresis threshold.
            threshold2: Upper hysteresis threshold.
            aperture_size: Aperture size of the Sobel operator.
            L2gradient: Use the L2 norm to compute gradient magnitude.

        Returns:
            Edge image.
        """
        if image.ndim != 2:
            raise ValueError("Canny edge detection requires grayscale image")

        edges = cv2.Canny(
            image,
            threshold1,
            threshold2,
            apertureSize=aperture_size,
            L2gradient=L2gradient
        )

        logger.debug(f"Canny edge detection: threshold1={threshold1}, threshold2={threshold2}")
        return edges

    @staticmethod
    def sobel(
        image: np.ndarray,
        dx: int = 1,
        dy: int = 1,
        ksize: int = 3
    ) -> np.ndarray:
        """
        Detect edges using the Sobel operator.

        Args:
            image: Input grayscale image.
            dx: Derivative order in the x direction.
            dy: Derivative order in the y direction.
            ksize: Sobel kernel size.

        Returns:
            Edge image.
        """
        if image.ndim != 2:
            raise ValueError("Sobel edge detection requires grayscale image")

        # Compute gradients in the x and y directions.
        if dx > 0:
            grad_x = cv2.Sobel(image, cv2.CV_64F, dx, 0, ksize=ksize)
            grad_x = np.abs(grad_x)
        else:
            grad_x = 0

        if dy > 0:
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, dy, ksize=ksize)
            grad_y = np.abs(grad_y)
        else:
            grad_y = 0

        # Combine the gradients.
        if dx > 0 and dy > 0:
            edges = np.sqrt(grad_x**2 + grad_y**2)
        elif dx > 0:
            edges = grad_x
        else:
            edges = grad_y

        # Normalize to [0, 255].
        edges = np.uint8(np.clip(edges, 0, 255))

        logger.debug(f"Sobel edge detection: dx={dx}, dy={dy}, ksize={ksize}")
        return edges

    @staticmethod
    def laplacian(
        image: np.ndarray,
        ksize: int = 3
    ) -> np.ndarray:
        """
        Detect edges using the Laplacian operator.

        Args:
            image: Input grayscale image.
            ksize: Kernel size.

        Returns:
            Edge image.
        """
        if image.ndim != 2:
            raise ValueError("Laplacian edge detection requires grayscale image")

        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
        laplacian = np.abs(laplacian)
        laplacian = np.uint8(np.clip(laplacian, 0, 255))

        logger.debug(f"Laplacian edge detection: ksize={ksize}")
        return laplacian

    @staticmethod
    def scharr(
        image: np.ndarray,
        dx: int = 1,
        dy: int = 0
    ) -> np.ndarray:
        """
        Detect edges using the Scharr operator, a refinement of the Sobel operator.

        Args:
            image: Input grayscale image.
            dx: Derivative order in the x direction.
            dy: Derivative order in the y direction.

        Returns:
            Edge image.
        """
        if image.ndim != 2:
            raise ValueError("Scharr edge detection requires grayscale image")

        scharr = cv2.Scharr(image, cv2.CV_64F, dx, dy)
        scharr = np.abs(scharr)
        scharr = np.uint8(np.clip(scharr, 0, 255))

        logger.debug(f"Scharr edge detection: dx={dx}, dy={dy}")
        return scharr


# Convenience functions.
def canny_edge(image: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
    """Convenience wrapper for Canny edge detection."""
    return EdgeDetection.canny(image, threshold1, threshold2)


def sobel_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Convenience wrapper for Sobel edge detection."""
    return EdgeDetection.sobel(image, dx=1, dy=1, ksize=ksize)


def laplacian_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Convenience wrapper for Laplacian edge detection."""
    return EdgeDetection.laplacian(image, ksize=ksize)
