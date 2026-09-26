"""
Instance matching.

Use an IoU threshold and greedy matching to identify predictions of the same cell across models.
"""
import numpy as np
from typing import List, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute the intersection over union (IoU) of two binary masks.

    Args:
        mask1: First binary mask.
        mask2: Second binary mask.

    Returns:
        IoU in the range [0, 1].
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    union_sum = np.sum(union)
    if union_sum == 0:
        return 0.0

    return np.sum(intersection) / union_sum


def extract_instances(mask: np.ndarray, model_idx: int) -> List[dict]:
    """
    Extract all instances from a labeled mask.

    Args:
        mask: Labeled mask with a unique integer ID for each cell.
        model_idx: Model index.

    Returns:
        List of instances containing {model_idx, instance_id, binary_mask, area}.
    """
    instances = []
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]  # Skip the background.

    for inst_id in unique_ids:
        binary_mask = (mask == inst_id)
        area = np.sum(binary_mask)

        instances.append({
            'model_idx': model_idx,
            'instance_id': int(inst_id),
            'binary_mask': binary_mask,
            'area': int(area)
        })

    return instances


def match_instances(masks_list: List[np.ndarray],
                   iou_threshold: float = 0.5) -> List[List[Tuple[int, int]]]:
    """
    Match instances across multiple models.

    Greedy matching procedure:
    1. Extract instances from all models to build a candidate pool.
    2. Sort by area, largest first.
    3. Find instances whose IoU exceeds the threshold for each candidate.
    4. Group the matched instances.

    Args:
        masks_list: List of labeled masks from multiple models.
        iou_threshold: IoU threshold for matching.

    Returns:
        List of matched groups containing [(model_idx, instance_id), ...].
    """
    logger.info(f"Starting instance matching: models={len(masks_list)}, IoU threshold={iou_threshold}")

    # 1. Extract instances from all models to build the candidate pool.
    proposals = []
    for model_idx, mask in enumerate(masks_list):
        instances = extract_instances(mask, model_idx)
        proposals.extend(instances)
        logger.info(f"Model {model_idx}: extracted {len(instances)} instances")

    logger.info(f"Candidate pool: {len(proposals)} instances")

    # 2. Sort by area, prioritizing larger cells for more robust matching.
    sorted_indices = np.argsort([-p['area'] for p in proposals])

    # 3. Perform greedy matching.
    matched_groups = []
    processed_indices = set()

    for i in sorted_indices:
        if i in processed_indices:
            continue

        # Start a group with the current instance.
        current_proposal = proposals[i]
        current_group = [(current_proposal['model_idx'], current_proposal['instance_id'])]
        processed_indices.add(i)

        # Find remaining instances whose IoU with the current instance meets the threshold.
        for j in sorted_indices:
            if j in processed_indices:
                continue

            other_proposal = proposals[j]

            # Skip instances from the same model; a model cannot match itself.
            if other_proposal['model_idx'] == current_proposal['model_idx']:
                continue

            # Compute IoU.
            iou = compute_iou(current_proposal['binary_mask'],
                            other_proposal['binary_mask'])

            if iou > iou_threshold:
                current_group.append((other_proposal['model_idx'],
                                    other_proposal['instance_id']))
                processed_indices.add(j)

        # Append the matched group to the results.
        matched_groups.append(current_group)

    logger.info(f"Matching complete: {len(matched_groups)} instance groups")

    # Summarize matching statistics.
    multi_model_groups = [g for g in matched_groups if len(g) > 1]
    logger.info(f"{len(multi_model_groups)} groups contain predictions from multiple models")

    return matched_groups
