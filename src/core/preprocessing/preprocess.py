"""
Image preprocessing.

Provides common image preprocessing operations.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from skimage import exposure, filters

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """Image preprocessing methods."""

    @staticmethod
    def normalize(
        image: np.ndarray,
        method: str = 'minmax',
        clip_percentile: Tuple[float, float] = (1, 99)
    ) -> np.ndarray:
        """
        Normalize an image.

        Args:
            image: Input image.
            method: Normalization method ('minmax', 'zscore', 'percentile').
            clip_percentile: Percentile clipping range, used only by the percentile method.

        Returns:
            Normalized image.
        """
        image = image.astype(np.float32)

        if method == 'minmax':
            # Apply min-max normalization to [0, 1].
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = image

        elif method == 'zscore':
            # Apply z-score standardization.
            mean = image.mean()
            std = image.std()
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean

        elif method == 'percentile':
            # Clip by percentiles, then normalize.
            p_low, p_high = np.percentile(image, clip_percentile)
            image_clipped = np.clip(image, p_low, p_high)
            normalized = (image_clipped - p_low) / (p_high - p_low)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        logger.debug(f"Normalized image using {method} method")
        return normalized

    @staticmethod
    def resize(
        image: np.ndarray,
        size: Tuple[int, int],
        interpolation: str = 'bilinear'
    ) -> np.ndarray:
        """
        Resize the image.

        Args:
            image: Input image.
            size: Target size (width, height).
            interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic', 'lanczos').

        Returns:
            Resized image.
        """
        interp_methods = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }

        if interpolation not in interp_methods:
            raise ValueError(f"Unknown interpolation method: {interpolation}")

        resized = cv2.resize(image, size, interpolation=interp_methods[interpolation])
        logger.debug(f"Resized image to {size}")
        return resized

    @staticmethod
    def denoise(
        image: np.ndarray,
        method: str = 'gaussian',
        **kwargs
    ) -> np.ndarray:
        """
        Denoise the image.

        Args:
            image: Input image.
            method: Denoising method ('gaussian', 'median', 'bilateral', 'nlm').
            **kwargs: Method-specific parameters.

        Returns:
            Denoised image.
        """
        if method == 'gaussian':
            # Gaussian filtering.
            ksize = kwargs.get('ksize', 5)
            sigma = kwargs.get('sigma', 0)
            denoised = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        elif method == 'median':
            # Median filtering.
            ksize = kwargs.get('ksize', 5)
            denoised = cv2.medianBlur(image, ksize)

        elif method == 'bilateral':
            # Bilateral filtering.
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        elif method == 'nlm':
            # Non-local means denoising.
            h = kwargs.get('h', 10)
            template_window_size = kwargs.get('template_window_size', 7)
            search_window_size = kwargs.get('search_window_size', 21)

            if image.ndim == 2:
                denoised = cv2.fastNlMeansDenoising(
                    image, None, h, template_window_size, search_window_size
                )
            else:
                denoised = cv2.fastNlMeansDenoisingColored(
                    image, None, h, h, template_window_size, search_window_size
                )

        else:
            raise ValueError(f"Unknown denoising method: {method}")

        logger.debug(f"Denoised image using {method} method")
        return denoised

    @staticmethod
    def enhance_contrast(
        image: np.ndarray,
        method: str = 'clahe',
        **kwargs
    ) -> np.ndarray:
        """
        Enhance image contrast.

        Args:
            image: Input image.
            method: Enhancement method ('clahe', 'histogram_eq', 'adaptive_eq').
            **kwargs: Method-specific parameters.

        Returns:
            Contrast-enhanced image.
        """
        if method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clip_limit = kwargs.get('clip_limit', 2.0)
            tile_grid_size = kwargs.get('tile_grid_size', (8, 8))

            if image.ndim == 2:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                enhanced = clahe.apply(image)
            else:
                # For color images, apply CLAHE to the L channel in LAB color space.
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        elif method == 'histogram_eq':
            # Histogram equalization.
            if image.ndim == 2:
                enhanced = cv2.equalizeHist(image)
            else:
                # For color images, operate on the Y channel in YUV color space.
                yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

        elif method == 'adaptive_eq':
            # Adaptive histogram equalization.
            clip_limit = kwargs.get('clip_limit', 0.03)
            enhanced = exposure.equalize_adapthist(image, clip_limit=clip_limit)
            # Convert back to the original data type and value range.
            if image.dtype == np.uint8:
                enhanced = (enhanced * 255).astype(np.uint8)

        else:
            raise ValueError(f"Unknown contrast enhancement method: {method}")

        logger.debug(f"Enhanced contrast using {method} method")
        return enhanced


# Convenience functions.
def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Convenience wrapper for image normalization."""
    return ImagePreprocessor.normalize(image, method=method)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Convenience wrapper for image resizing."""
    return ImagePreprocessor.resize(image, size)


def denoise_image(image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
    """Convenience wrapper for image denoising."""
    return ImagePreprocessor.denoise(image, method=method)


def enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """Convenience wrapper for contrast enhancement."""
    return ImagePreprocessor.enhance_contrast(image, method=method)
