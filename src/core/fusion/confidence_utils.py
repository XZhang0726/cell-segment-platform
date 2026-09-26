"""
Confidence utilities.

Generate synthetic confidence maps for models that do not provide confidence scores.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import List
from ..utils.logger import get_logger

logger = get_logger(__name__)


def generate_confidence_from_mask(mask: np.ndarray,
                                  base_confidence: float = 0.8,
                                  boundary_penalty: float = 0.3) -> np.ndarray:
    """
    Generate a synthetic confidence map from a segmentation mask.

    Use the distance transform to assign lower confidence near boundaries and higher confidence at centers.
    This heuristic represents the greater uncertainty often found at object boundaries.

    Args:
        mask: Labeled mask (H, W) containing instance IDs.
        base_confidence: Base confidence in the range [0, 1].
        boundary_penalty: Boundary penalty coefficient in the range [0, 1].

    Returns:
        Confidence map (H, W) in the range [0, 1].
    """
    H, W = mask.shape
    confidence_map = np.zeros((H, W), dtype=np.float32)

    # Get all instance IDs.
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]  # Exclude the background.

    for inst_id in instance_ids:
        # Extract this instance's binary mask.
        binary_mask = (mask == inst_id).astype(np.uint8)

        # Compute the distance transform, measuring distance to the boundary.
        distance = distance_transform_edt(binary_mask)

        # Normalize distances to [0, 1].
        if distance.max() > 0:
            normalized_distance = distance / distance.max()
        else:
            normalized_distance = np.zeros_like(distance)

        # Assign higher confidence at the center and lower confidence near the boundary.
        # confidence = base - penalty * (1 - normalized_distance)
        instance_confidence = base_confidence - boundary_penalty * (1 - normalized_distance)
        instance_confidence = np.clip(instance_confidence, 0.0, 1.0)

        # Write to the confidence map.
        confidence_map[binary_mask > 0] = instance_confidence[binary_mask > 0]

    return confidence_map


def generate_confidence_maps(masks_list: List[np.ndarray],
                            model_names: List[str],
                            model_reliabilities: dict) -> List[np.ndarray]:
    """
    Generate confidence maps for multiple models.

    Adjust base confidence according to the configured model reliability:
    - Models assigned higher reliability receive higher base confidence.
    - Models assigned lower reliability receive lower base confidence.

    Args:
        masks_list: List of masks.
        model_names: List of model names.
        model_reliabilities: Dictionary of model reliability values.

    Returns:
        List of confidence maps.
    """
    logger.info(f"Generating synthetic confidence maps for {len(masks_list)} models...")

    confidences_list = []

    for mask, model_name in zip(masks_list, model_names):
        # Adjust base confidence according to model reliability.
        reliability = model_reliabilities.get(model_name, 0.8)
        base_confidence = reliability * 0.9  # Set confidence slightly below reliability.

        # Generate the confidence map.
        confidence_map = generate_confidence_from_mask(
            mask,
            base_confidence=base_confidence,
            boundary_penalty=0.3
        )

        confidences_list.append(confidence_map)

        logger.debug(f"  {model_name}: base_confidence={base_confidence:.2f}, "
                    f"mean={np.mean(confidence_map[mask > 0]):.3f}")

    logger.info("Confidence maps generated")

    return confidences_list
