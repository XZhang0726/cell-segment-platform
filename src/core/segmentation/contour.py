"""
Contour detection and analysis.

Provides contour detection, feature extraction, and analysis methods.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ContourAnalysis:
    """Contour analysis methods."""

    @staticmethod
    def find_contours(
        binary_image: np.ndarray,
        mode: str = 'external',
        method: str = 'simple'
    ) -> List[np.ndarray]:
        """
        Find contours.

        Args:
            binary_image: Binary image.
            mode: Contour retrieval mode ('external', 'list', 'tree', 'ccomp').
            method: Contour approximation method ('none', 'simple', 'tc89_l1', 'tc89_kcos').

        Returns:
            List of contours.
        """
        if binary_image.ndim != 2:
            raise ValueError("Contour detection requires binary image")

        # Contour retrieval modes.
        modes = {
            'external': cv2.RETR_EXTERNAL,
            'list': cv2.RETR_LIST,
            'tree': cv2.RETR_TREE,
            'ccomp': cv2.RETR_CCOMP
        }

        # Contour approximation methods.
        methods = {
            'none': cv2.CHAIN_APPROX_NONE,
            'simple': cv2.CHAIN_APPROX_SIMPLE,
            'tc89_l1': cv2.CHAIN_APPROX_TC89_L1,
            'tc89_kcos': cv2.CHAIN_APPROX_TC89_KCOS
        }

        if mode not in modes:
            raise ValueError(f"Unknown contour mode: {mode}")
        if method not in methods:
            raise ValueError(f"Unknown contour method: {method}")

        # Find contours.
        contours, _ = cv2.findContours(
            binary_image,
            modes[mode],
            methods[method]
        )

        logger.debug(f"Found {len(contours)} contours")
        return contours

    @staticmethod
    def filter_contours(
        contours: List[np.ndarray],
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        min_perimeter: Optional[float] = None,
        max_perimeter: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Filter contours by area and perimeter.

        Args:
            contours: List of contours.
            min_area: Minimum area.
            max_area: Maximum area.
            min_perimeter: Minimum perimeter.
            max_perimeter: Maximum perimeter.

        Returns:
            Filtered list of contours.
        """
        filtered = []

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Check area limits.
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            # Check perimeter limits.
            if min_perimeter is not None and perimeter < min_perimeter:
                continue
            if max_perimeter is not None and perimeter > max_perimeter:
                continue

            filtered.append(contour)

        logger.debug(f"Filtered {len(contours)} -> {len(filtered)} contours")
        return filtered

    @staticmethod
    def get_contour_properties(contour: np.ndarray) -> Dict[str, float]:
        """
        Compute contour properties.

        Args:
            contour: A single contour.

        Returns:
            Dictionary of contour properties.
        """
        properties = {}

        # Basic properties.
        properties['area'] = cv2.contourArea(contour)
        properties['perimeter'] = cv2.arcLength(contour, True)

        # Circularity (4*pi*area/perimeter^2).
        if properties['perimeter'] > 0:
            properties['circularity'] = (4 * np.pi * properties['area']) / (properties['perimeter'] ** 2)
        else:
            properties['circularity'] = 0

        # Bounding box.
        x, y, w, h = cv2.boundingRect(contour)
        properties['bbox_x'] = x
        properties['bbox_y'] = y
        properties['bbox_width'] = w
        properties['bbox_height'] = h
        properties['bbox_area'] = w * h

        # Aspect ratio.
        if h > 0:
            properties['aspect_ratio'] = w / h
        else:
            properties['aspect_ratio'] = 0

        # Extent (contour area / bounding-box area).
        if properties['bbox_area'] > 0:
            properties['extent'] = properties['area'] / properties['bbox_area']
        else:
            properties['extent'] = 0

        # Convex hull.
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        properties['convex_hull_area'] = hull_area

        # Solidity (contour area / convex-hull area).
        if hull_area > 0:
            properties['solidity'] = properties['area'] / hull_area
        else:
            properties['solidity'] = 0

        # Equivalent diameter.
        properties['equivalent_diameter'] = np.sqrt(4 * properties['area'] / np.pi)

        return properties

    @staticmethod
    def draw_contours(
        image: np.ndarray,
        contours: List[np.ndarray],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        fill: bool = False
    ) -> np.ndarray:
        """
        Draw contours on an image.

        Args:
            image: Input image.
            contours: List of contours.
            color: Contour color (B, G, R).
            thickness: Line thickness; -1 fills the contour.
            fill: Fill contours.

        Returns:
            Image with contours drawn.
        """
        # Copy the image to avoid modifying the original.
        if image.ndim == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()

        # Draw the contours.
        thickness_value = -1 if fill else thickness
        cv2.drawContours(result, contours, -1, color, thickness_value)

        logger.debug(f"Drew {len(contours)} contours")
        return result

    @staticmethod
    def get_bounding_boxes(
        contours: List[np.ndarray],
        rotated: bool = False
    ) -> List[Tuple]:
        """
        Get contour bounding boxes.

        Args:
            contours: List of contours.
            rotated: Use rotated bounding boxes.

        Returns:
            List of bounding boxes.
        """
        boxes = []

        for contour in contours:
            if rotated:
                # Rotated bounding box (center, size, angle).
                if len(contour) >= 5:
                    box = cv2.minAreaRect(contour)
                    boxes.append(box)
            else:
                # Axis-aligned bounding box (x, y, w, h).
                box = cv2.boundingRect(contour)
                boxes.append(box)

        logger.debug(f"Got {len(boxes)} bounding boxes")
        return boxes

    @staticmethod
    def fit_ellipse(
        contours: List[np.ndarray],
        min_points: int = 5
    ) -> List[Optional[Tuple]]:
        """
        Fit ellipses to contours.

        Args:
            contours: List of contours.
            min_points: Minimum number of points required for ellipse fitting.

        Returns:
            List of ellipse parameters (center, axis lengths, angle).
        """
        ellipses = []

        for contour in contours:
            if len(contour) >= min_points:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    ellipses.append(ellipse)
                except:
                    ellipses.append(None)
            else:
                ellipses.append(None)

        logger.debug(f"Fitted {sum(e is not None for e in ellipses)} ellipses")
        return ellipses


# Convenience functions.
def find_contours(binary_image: np.ndarray, mode: str = 'external') -> List[np.ndarray]:
    """Convenience wrapper for contour detection."""
    return ContourAnalysis.find_contours(binary_image, mode)


def filter_by_area(
    contours: List[np.ndarray],
    min_area: float = 100,
    max_area: Optional[float] = None
) -> List[np.ndarray]:
    """Convenience wrapper for filtering contours by area."""
    return ContourAnalysis.filter_contours(contours, min_area=min_area, max_area=max_area)


def get_properties(contour: np.ndarray) -> Dict[str, float]:
    """Convenience wrapper for contour property extraction."""
    return ContourAnalysis.get_contour_properties(contour)


def draw_contours(
    image: np.ndarray,
    contours: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Convenience wrapper for drawing contours."""
    return ContourAnalysis.draw_contours(image, contours, color)
