"""
Optimized instance matching.

Use bounding-box prefiltering and cropped mask regions to accelerate matching.
Performance depends on the number, size, and overlap of instances.
"""
import numpy as np
from typing import List, Tuple, Dict
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_bbox(binary_mask: np.ndarray) -> Dict[str, int]:
    """
    Compute the bounding box of a binary mask.

    Args:
        binary_mask: Binary mask.

    Returns:
        Bounding-box dictionary {x_min, x_max, y_min, y_max}.
    """
    y_coords, x_coords = np.where(binary_mask)

    if len(y_coords) == 0:
        return {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}

    return {
        'x_min': int(np.min(x_coords)),
        'x_max': int(np.max(x_coords)),
        'y_min': int(np.min(y_coords)),
        'y_max': int(np.max(y_coords))
    }


def boxes_overlap(box1: Dict[str, int], box2: Dict[str, int]) -> bool:
    """
    Quickly check whether two bounding boxes overlap.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        Whether the bounding boxes overlap.
    """
    return not (box1['x_max'] < box2['x_min'] or
                box1['x_min'] > box2['x_max'] or
                box1['y_max'] < box2['y_min'] or
                box1['y_min'] > box2['y_max'])


def compute_iou_cropped(mask1: np.ndarray, bbox1: Dict[str, int],
                       mask2: np.ndarray, bbox2: Dict[str, int]) -> float:
    """
    Compute the IoU of two instances using cropped regions.

    Compute the intersection only in the overlapping region to avoid a full-image scan.

    Args:
        mask1: First binary mask (full image).
        bbox1: First bounding box.
        mask2: Second binary mask (full image).
        bbox2: Second bounding box.

    Returns:
        IoU in the range [0, 1].
    """
    # Determine the overlapping region.
    overlap_x_min = max(bbox1['x_min'], bbox2['x_min'])
    overlap_x_max = min(bbox1['x_max'], bbox2['x_max'])
    overlap_y_min = max(bbox1['y_min'], bbox2['y_min'])
    overlap_y_max = min(bbox1['y_max'], bbox2['y_max'])

    # Crop to the overlapping region.
    crop1 = mask1[overlap_y_min:overlap_y_max+1, overlap_x_min:overlap_x_max+1]
    crop2 = mask2[overlap_y_min:overlap_y_max+1, overlap_x_min:overlap_x_max+1]

    # Compute the intersection within the overlapping region.
    intersection = np.sum(np.logical_and(crop1, crop2))

    # Compute the union, accounting for the full instance areas.
    # Union = area1 + area2 - intersection.
    area1 = (bbox1['x_max'] - bbox1['x_min'] + 1) * (bbox1['y_max'] - bbox1['y_min'] + 1)
    area2 = (bbox2['x_max'] - bbox2['x_min'] + 1) * (bbox2['y_max'] - bbox2['y_min'] + 1)

    # Use actual pixel counts for accurate areas.
    area1_actual = np.sum(mask1[bbox1['y_min']:bbox1['y_max']+1,
                                bbox1['x_min']:bbox1['x_max']+1])
    area2_actual = np.sum(mask2[bbox2['y_min']:bbox2['y_max']+1,
                                bbox2['x_min']:bbox2['x_max']+1])

    union = area1_actual + area2_actual - intersection

    if union == 0:
        return 0.0

    return intersection / union


def extract_instances_optimized(mask: np.ndarray, model_idx: int) -> List[dict]:
    """
    Extract all instances from a labeled mask using the optimized representation.

    Optimizations:
    1. Compute and store bounding boxes.
    2. Store references to the full mask to avoid copies.
    3. Precompute instance areas.

    Args:
        mask: Labeled mask with a unique integer ID for each cell.
        model_idx: Model index.

    Returns:
        List of instances containing {model_idx, instance_id, mask_ref, bbox, area}.
    """
    instances = []
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]  # Skip the background.

    for inst_id in unique_ids:
        binary_mask = (mask == inst_id)
        bbox = compute_bbox(binary_mask)
        area = int(np.sum(binary_mask))

        instances.append({
            'model_idx': model_idx,
            'instance_id': int(inst_id),
            'mask_ref': mask,  # Store a reference to the original mask.
            'inst_id_value': int(inst_id),  # Use the instance ID to extract the binary mask on demand.
            'bbox': bbox,
            'area': area
        })

    return instances


def match_instances_optimized(masks_list: List[np.ndarray],
                              iou_threshold: float = 0.5) -> List[List[Tuple[int, int]]]:
    """
    Match instances across models using the optimized implementation.

    Optimization strategies:
    1. Bounding-box prefiltering: compute IoU only for overlapping boxes.
    2. Area prefiltering: skip pairs whose areas differ too much.
    3. Cropped-region IoU: compute intersections only in overlapping regions.

    Performance depends on the number, size, and overlap of instances.

    Args:
        masks_list: List of labeled masks from multiple models.
        iou_threshold: IoU threshold for matching.

    Returns:
        List of matched groups containing [(model_idx, instance_id), ...].
    """
    logger.info(f"Starting optimized instance matching: models={len(masks_list)}, IoU threshold={iou_threshold}")

    # 1. Extract instances from all models to build the candidate pool.
    proposals = []
    for model_idx, mask in enumerate(masks_list):
        instances = extract_instances_optimized(mask, model_idx)
        proposals.extend(instances)
        logger.info(f"Model {model_idx}: extracted {len(instances)} instances")

    logger.info(f"Candidate pool: {len(proposals)} instances")

    # 2. Sort by area, prioritizing larger cells for more robust matching.
    sorted_indices = np.argsort([-p['area'] for p in proposals])

    # 3. Perform optimized greedy matching.
    matched_groups = []
    processed_indices = set()

    # Track optimization statistics.
    total_pairs_checked = 0
    bbox_filtered = 0
    area_filtered = 0
    iou_computed = 0

    for i in sorted_indices:
        if i in processed_indices:
            continue

        # Start a group with the current instance.
        current_proposal = proposals[i]
        current_group = [(current_proposal['model_idx'], current_proposal['instance_id'])]
        processed_indices.add(i)

        # Extract the current binary mask on demand.
        current_binary_mask = (current_proposal['mask_ref'] == current_proposal['inst_id_value'])

        # Find remaining instances whose IoU with the current instance meets the threshold.
        for j in sorted_indices:
            if j in processed_indices:
                continue

            other_proposal = proposals[j]
            total_pairs_checked += 1

            # Skip instances from the same model.
            if other_proposal['model_idx'] == current_proposal['model_idx']:
                continue

            # Optimization 1: bounding-box prefiltering.
            if not boxes_overlap(current_proposal['bbox'], other_proposal['bbox']):
                bbox_filtered += 1
                continue

            # Optimization 2: area prefiltering.
            # If areas differ too much, IoU cannot exceed the threshold.
            area_ratio = min(current_proposal['area'], other_proposal['area']) / \
                        max(current_proposal['area'], other_proposal['area'])
            if area_ratio < iou_threshold:
                area_filtered += 1
                continue

            # Extract the other instance's binary mask.
            other_binary_mask = (other_proposal['mask_ref'] == other_proposal['inst_id_value'])

            # Compute IoU using the optimized implementation.
            iou = compute_iou_cropped(current_binary_mask, current_proposal['bbox'],
                                     other_binary_mask, other_proposal['bbox'])
            iou_computed += 1

            if iou > iou_threshold:
                current_group.append((other_proposal['model_idx'],
                                    other_proposal['instance_id']))
                processed_indices.add(j)

        # Append the matched group to the results.
        matched_groups.append(current_group)

    logger.info(f"Matching complete: {len(matched_groups)} instance groups")
    logger.info(f"Optimization statistics: checked={total_pairs_checked} pairs, bbox_filtered={bbox_filtered}, "
               f"area_filtered={area_filtered}, IoU_computations={iou_computed}")
    logger.info(f"IoU computations reduced by {100*(1-iou_computed/max(total_pairs_checked,1)):.1f}%")

    # Summarize matching statistics.
    multi_model_groups = [g for g in matched_groups if len(g) > 1]
    logger.info(f"{len(multi_model_groups)} groups contain predictions from multiple models")

    return matched_groups
