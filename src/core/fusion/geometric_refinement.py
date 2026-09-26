"""
Geometric boundary refinement.

Use watershed segmentation to refine boundaries in regions with high conflict.
Combine geometric refinement with DST conflict information for hybrid fusion.
"""
import numpy as np
from typing import Optional, Tuple
from scipy.ndimage import gaussian_gradient_magnitude, distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import label
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_gradient_map(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute the image gradient map.

    Args:
        image: Original image (H, W) or (H, W, C).
        sigma: Gaussian smoothing parameter.

    Returns:
        Gradient map (H, W).
    """
    # Convert color images to grayscale.
    if len(image.shape) == 3:
        # Convert to grayscale using a weighted channel average.
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image.copy()

    # Scale standard 8-bit intensities to [0, 1] when the maximum exceeds 1.
    if gray.max() > 1:
        gray = gray.astype(np.float32) / 255.0

    # Compute the gradient magnitude.
    gradient = gaussian_gradient_magnitude(gray, sigma=sigma)

    return gradient


def watershed_refinement(image: np.ndarray,
                         fused_mask: np.ndarray,
                         conflict_map: Optional[np.ndarray] = None,
                         conflict_threshold: float = 0.5,
                         gradient_sigma: float = 1.0) -> Tuple[np.ndarray, dict]:
    """
    Refine boundaries in high-conflict regions using watershed segmentation.

    This hybrid fusion method applies the following rules:
    - Low-conflict regions: preserve the DST fusion result.
    - High-conflict regions: refine boundaries using watershed segmentation.

    Args:
        image: Original image (H, W) or (H, W, C).
        fused_mask: Labeled mask produced by DST fusion (H, W).
        conflict_map: Conflict map (H, W); None applies watershed to all foreground regions.
        conflict_threshold: Threshold above which regions undergo watershed refinement.
        gradient_sigma: Gaussian smoothing parameter for gradient computation.

    Returns:
        (refined_mask, stats): Refined mask and refinement statistics.
    """
    logger.info(f"Starting watershed boundary refinement: conflict_threshold={conflict_threshold}")

    # 1. Compute the gradient map as the watershed elevation surface.
    gradient = compute_gradient_map(image, sigma=gradient_sigma)

    # 2. Identify high-conflict regions.
    if conflict_map is not None:
        high_conflict_mask = conflict_map > conflict_threshold
        high_conflict_pixels = np.sum(high_conflict_mask)
        logger.info(f"High-conflict pixels: {high_conflict_pixels} ({high_conflict_pixels / conflict_map.size * 100:.2f}%)")
    else:
        # Apply watershed to all foreground regions when no conflict map is available.
        high_conflict_mask = fused_mask > 0
        high_conflict_pixels = np.sum(high_conflict_mask)
        logger.info(f"Applying watershed to all foreground regions: {high_conflict_pixels} pixels")

    # 3. Extract confident regions as seed markers.
    # Strategy 1: use low-conflict regions as confident seeds.
    markers = None
    num_markers = 0

    if conflict_map is not None:
        # Select confident seeds below 60% of the conflict threshold.
        certain_threshold = conflict_threshold * 0.6
        certain_mask = (conflict_map < certain_threshold) & (fused_mask > 0)
        markers = label(certain_mask)
        num_markers = np.max(markers)
        logger.info(f"Strategy 1 (low-conflict regions): seeds={num_markers}, threshold={certain_threshold:.3f}")

    # Strategy 2: if strategy 1 fails, use distance-transform cell centers as seeds.
    if num_markers == 0:
        logger.info("Strategy 1 failed; trying strategy 2 (distance transform)")
        distance = distance_transform_edt(fused_mask > 0)
        # Use local maxima of the distance transform as seeds.
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(distance, size=10)
        certain_mask = (distance == local_max) & (distance > 3)  # Keep peaks more than 3 pixels from the boundary.
        markers = label(certain_mask)
        num_markers = np.max(markers)
        logger.info(f"Strategy 2 (distance transform): seeds={num_markers}")

    # Strategy 3: if strategy 2 fails, use fused-mask labels directly as seeds.
    if num_markers == 0:
        logger.info("Strategy 2 failed; trying strategy 3 (fused-mask labels)")
        markers = fused_mask.copy()
        num_markers = np.max(markers)
        logger.info(f"Strategy 3 (fused-mask labels): seeds={num_markers}")

    if num_markers == 0:
        logger.warning("No confident seeds found with any strategy; returning the original fusion result")
        return fused_mask, {
            'refined': False,
            'reason': 'no_markers',
            'high_conflict_pixels': high_conflict_pixels if conflict_map is not None else 0
        }

    # 4. Run watershed on the foreground before applying selected updates.
    # Restrict watershed segmentation to foreground regions.
    watershed_mask = fused_mask > 0

    try:
        # Run watershed segmentation.
        watershed_result = watershed(gradient, markers, mask=watershed_mask)

        # 5. Preserve low-conflict regions and use watershed results in high-conflict regions.
        refined_mask = fused_mask.copy()
        if conflict_map is not None:
            refined_mask[high_conflict_mask] = watershed_result[high_conflict_mask]
        else:
            refined_mask = watershed_result

        # 6. Compute statistics.
        refined_pixels_mask = refined_mask != fused_mask
        refined_pixels = np.sum(refined_pixels_mask)
        stats = {
            'refined': True,
            'num_markers': num_markers,
            'high_conflict_pixels': high_conflict_pixels if conflict_map is not None else 0,
            'refined_pixels': refined_pixels,
            'refined_percentage': refined_pixels / fused_mask.size * 100,
            'refined_mask': refined_pixels_mask  # Include a binary mask of refined pixels.
        }

        logger.info(f"Watershed refinement complete: modified {refined_pixels} pixels ({stats['refined_percentage']:.2f}%)")

        return refined_mask, stats

    except Exception as e:
        logger.error(f"Watershed refinement failed: {e}")
        return fused_mask, {
            'refined': False,
            'reason': 'watershed_failed',
            'error': str(e)
        }
