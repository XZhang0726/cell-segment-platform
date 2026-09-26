"""
Uncertainty estimation.

Compute disagreement heatmaps and consistency metrics across models.
"""
import numpy as np
from typing import List, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_disagreement_map(masks_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Compute the disagreement map.

    Args:
        masks_list: List of labeled masks from multiple models.

    Returns:
        disagreement_map: Uncertainty heatmap in the range [0, 1].
        consistency_score: Global consistency score.
    """
    logger.info(f"Computing disagreement map for {len(masks_list)} models")

    # 1. Convert labeled masks to binary foreground/background masks.
    binary_masks = []
    for mask in masks_list:
        binary = (mask > 0).astype(np.float32)
        binary_masks.append(binary)

    # 2. Stack all binary masks.
    stack = np.stack(binary_masks, axis=0)

    # 3. Compute the fraction of foreground votes at each pixel.
    vote_ratio = np.mean(stack, axis=0)  # Range: [0, 1].

    # 4. Compute disagreement.
    # Disagreement is maximal at vote_ratio=0.5, when models are evenly split.
    # Disagreement is minimal at vote_ratio=0 or 1, when all models agree.
    disagreement = 1.0 - np.abs(2 * vote_ratio - 1)

    # 5. Compute the global consistency score.
    consistency_score = 1.0 - np.mean(disagreement)

    logger.info(f"Global consistency score: {consistency_score:.4f}")

    return disagreement, consistency_score


def compute_model_consistency(masks_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Compute pairwise consistency between models.

    Args:
        masks_list: List of labeled masks from multiple models.

    Returns:
        consistency_matrix: Pairwise model IoU matrix.
        avg_consistency: Mean consistency.
    """
    logger.info(f"Computing consistency across {len(masks_list)} models")

    n_models = len(masks_list)
    consistency_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(i, n_models):
            if i == j:
                consistency_matrix[i, j] = 1.0
            else:
                # Compute the overall IoU between two models.
                mask_i = (masks_list[i] > 0)
                mask_j = (masks_list[j] > 0)
                intersection = np.logical_and(mask_i, mask_j)
                union = np.logical_or(mask_i, mask_j)
                iou = np.sum(intersection) / (np.sum(union) + 1e-8)
                consistency_matrix[i, j] = iou
                consistency_matrix[j, i] = iou

    # Compute mean consistency, excluding the diagonal.
    mask = ~np.eye(n_models, dtype=bool)
    avg_consistency = np.mean(consistency_matrix[mask])

    logger.info(f"Mean model consistency: {avg_consistency:.4f}")

    return consistency_matrix, avg_consistency
