"""
Watershed segmentation algorithms.

Provides watershed methods for separating touching cells.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage as ndi

from ..utils.logger import get_logger

logger = get_logger(__name__)


class WatershedSegmentation:
    """Watershed segmentation methods."""

    @staticmethod
    def watershed_basic(
        image: np.ndarray,
        markers: np.ndarray
    ) -> np.ndarray:
        """
        Apply basic watershed segmentation.

        Args:
            image: Input grayscale or color image.
            markers: Marker image with a distinct positive integer for each region.

        Returns:
            Segmented label image.
        """
        # Convert grayscale to color; OpenCV watershed requires three channels.
        if image.ndim == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()

        # Ensure markers use the int32 data type.
        markers = markers.astype(np.int32)

        # Run the watershed algorithm.
        markers = cv2.watershed(image_color, markers)

        logger.debug(f"Watershed segmentation: {len(np.unique(markers))} regions")
        return markers

    @staticmethod
    def watershed_distance_transform(
        binary_image: np.ndarray,
        min_distance: int = 10,
        return_markers: bool = False
    ) -> np.ndarray:
        """
        Separate touching cells using distance-transform watershed segmentation.

        Args:
            binary_image: Binary image with white foreground.
            min_distance: Minimum distance between local maxima.
            return_markers: Return seed markers as well as the segmentation.

        Returns:
            Segmented label image, or (segmentation, markers) when return_markers=True.
        """
        if binary_image.ndim != 2:
            raise ValueError("Distance transform watershed requires binary image")

        # Compute the distance transform.
        distance = ndi.distance_transform_edt(binary_image)

        # Find local maxima to use as seeds.
        from skimage.feature import peak_local_max
        local_max = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=binary_image
        )

        # Create the marker image.
        markers = np.zeros_like(binary_image, dtype=np.int32)
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

        # Expand the markers.
        markers = ndi.label(markers)[0]

        # Run watershed segmentation.
        labels = cv2.watershed(
            cv2.cvtColor((binary_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
            markers
        )

        logger.debug(f"Distance transform watershed: {len(local_max)} seeds, {len(np.unique(labels))} regions")

        if return_markers:
            return labels, markers
        return labels

    @staticmethod
    def watershed_marker_controlled(
        image: np.ndarray,
        binary_mask: np.ndarray,
        sure_fg_erosion: int = 3,
        sure_bg_dilation: int = 3
    ) -> np.ndarray:
        """
        Apply marker-controlled watershed segmentation.

        Args:
            image: Input grayscale or color image.
            binary_mask: Binary mask with white foreground.
            sure_fg_erosion: Erosion kernel size for identifying confident foreground.
            sure_bg_dilation: Dilation kernel size for identifying confident background.

        Returns:
            Segmented label image.
        """
        # Identify the background region using dilation.
        kernel = np.ones((sure_bg_dilation, sure_bg_dilation), np.uint8)
        sure_bg = cv2.dilate(binary_mask, kernel, iterations=1)

        # Identify confident foreground using erosion.
        kernel = np.ones((sure_fg_erosion, sure_fg_erosion), np.uint8)
        sure_fg = cv2.erode(binary_mask, kernel, iterations=1)

        # Identify the unknown region.
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label foreground objects.
        _, markers = cv2.connectedComponents(sure_fg)

        # Assign label 1 to the background and labels starting at 2 to foreground objects.
        markers = markers + 1
        markers[unknown == 255] = 0

        # Run watershed segmentation.
        if image.ndim == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()

        markers = cv2.watershed(image_color, markers)

        logger.debug(f"Marker-controlled watershed: {len(np.unique(markers))} regions")
        return markers

    @staticmethod
    def visualize_watershed(
        image: np.ndarray,
        markers: np.ndarray,
        show_boundaries: bool = True
    ) -> np.ndarray:
        """
        Visualize watershed segmentation results.

        Args:
            image: Original image.
            markers: Watershed label image.
            show_boundaries: Highlight region boundaries.

        Returns:
            Visualization image.
        """
        # Create a color label image.
        if image.ndim == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()

        if show_boundaries:
            # Highlight boundaries in red.
            result[markers == -1] = [0, 0, 255]

        return result


# Convenience functions.
def watershed_distance(binary_image: np.ndarray, min_distance: int = 10) -> np.ndarray:
    """Convenience wrapper for distance-transform watershed segmentation."""
    return WatershedSegmentation.watershed_distance_transform(binary_image, min_distance)


def watershed_marker(
    image: np.ndarray,
    binary_mask: np.ndarray,
    erosion: int = 3
) -> np.ndarray:
    """Convenience wrapper for marker-controlled watershed segmentation."""
    return WatershedSegmentation.watershed_marker_controlled(
        image, binary_mask, erosion, erosion
    )

