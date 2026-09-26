"""
Segmentation fusion engine.

Fuse matched instances at the pixel level using several fusion strategies.
Includes voting, union, intersection, and Dempster-Shafer evidence fusion.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from ..utils.logger import get_logger
from .dempster_shafer import DempsterShaferFusion, FusionResult, handle_conflict

logger = get_logger(__name__)


def _majority_voting(binary_masks: List[np.ndarray]) -> np.ndarray:
    """
    Retain pixels supported by at least half the masks, including ties.

    Args:
        binary_masks: List of binary masks.

    Returns:
        Fused binary mask.
    """
    stack = np.stack(binary_masks, axis=0)
    vote_count = np.sum(stack, axis=0)
    return vote_count >= (len(binary_masks) / 2)


def _weighted_voting(binary_masks: List[np.ndarray],
                    weights: Optional[List[float]],
                    group: List[Tuple[int, int]]) -> np.ndarray:
    """
    Apply weighted voting using model-specific weights.

    Args:
        binary_masks: List of binary masks.
        weights: List of model weights.
        group: Matched group of (model_idx, instance_id) tuples.

    Returns:
        Fused binary mask.
    """
    if weights is None:
        weights = [1.0] * len(binary_masks)

    weighted_sum = np.zeros_like(binary_masks[0], dtype=np.float32)
    total_weight = 0.0

    for i, (model_idx, _) in enumerate(group):
        weighted_sum += binary_masks[i] * weights[model_idx]
        total_weight += weights[model_idx]

    return (weighted_sum / total_weight) >= 0.5


def _union_fusion(binary_masks: List[np.ndarray]) -> np.ndarray:
    """
    Take the union, favoring recall.

    Args:
        binary_masks: List of binary masks.

    Returns:
        Fused binary mask.
    """
    stack = np.stack(binary_masks, axis=0)
    return np.any(stack, axis=0)


def _intersection_fusion(binary_masks: List[np.ndarray]) -> np.ndarray:
    """
    Take the intersection, favoring precision.

    Args:
        binary_masks: List of binary masks.

    Returns:
        Fused binary mask.
    """
    stack = np.stack(binary_masks, axis=0)
    return np.all(stack, axis=0)


def fuse_instances(matched_groups: List[List[Tuple[int, int]]],
                  masks_list: List[np.ndarray],
                  strategy: str = 'majority',
                  weights: Optional[List[float]] = None,
                  min_vote_count: int = 2) -> np.ndarray:
    """
    Fuse groups of matched instances.

    Args:
        matched_groups: List of matched groups containing (model_idx, instance_id) tuples.
        masks_list: List of original labeled masks.
        strategy: Fusion strategy ('majority', 'weighted', 'union', 'intersection').
        weights: Model weights for the weighted strategy.
        min_vote_count: Minimum number of distinct contributing models per group.

    Returns:
        Fused labeled mask.
    """
    logger.info(f"Starting fusion: strategy={strategy}, min_votes={min_vote_count}")

    # 1. Initialize the output mask.
    fused_mask = np.zeros_like(masks_list[0], dtype=np.int32)
    instance_id = 1

    # Statistics.
    total_groups = len(matched_groups)
    fused_count = 0
    skipped_count = 0

    # 2. Iterate over matched groups.
    for group in matched_groups:
        # Check the minimum vote requirement.
        contributing_models = set([model_idx for model_idx, _ in group])
        if len(contributing_models) < min_vote_count:
            skipped_count += 1
            continue

        # 3. Extract the binary mask of each instance in the group.
        binary_masks = []
        for model_idx, inst_id in group:
            mask = (masks_list[model_idx] == inst_id)
            binary_masks.append(mask)

        # 4. Apply the selected fusion strategy.
        if strategy == 'majority':
            consensus_mask = _majority_voting(binary_masks)
        elif strategy == 'weighted':
            consensus_mask = _weighted_voting(binary_masks, weights, group)
        elif strategy == 'union':
            consensus_mask = _union_fusion(binary_masks)
        elif strategy == 'intersection':
            consensus_mask = _intersection_fusion(binary_masks)
        else:
            logger.warning(f"Unknown fusion strategy: {strategy}; using majority voting")
            consensus_mask = _majority_voting(binary_masks)

        # 5. Write the fused mask.
        if np.sum(consensus_mask) > 0:
            fused_mask[consensus_mask] = instance_id
            instance_id += 1
            fused_count += 1

    logger.info(f"Fusion complete: total_groups={total_groups}, fused={fused_count}, skipped={skipped_count}")

    return fused_mask


def fuse_instances_dst(matched_groups: List[List[Tuple[int, int]]],
                       masks_list: List[np.ndarray],
                       confidences_list: List[Optional[np.ndarray]],
                       model_names: List[str],
                       model_reliabilities: Dict[str, float],
                       image: Optional[np.ndarray] = None,
                       min_vote_count: int = 2,
                       conflict_threshold: float = 0.6,
                       enable_watershed: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Fuse instances using Dempster-Shafer theory.

    Compared with simple voting, Dempster-Shafer theory (DST) provides:
    1. Explicit uncertainty modeling.
    2. Quantification of conflicts between models.
    3. A formal mathematical framework for combining evidence.
    4. Belief intervals and conflict maps.

    Args:
        matched_groups: List of matched groups containing (model_idx, instance_id) tuples.
        masks_list: List of original labeled masks.
        confidences_list: List of per-model confidence maps; individual entries may be None.
        model_names: List of model names.
        model_reliabilities: Model reliability mapping, e.g. {'cellpose': 0.9, 'cellvit': 0.85}.
        image: Optional original image (H, W) or (H, W, C) for watershed boundary refinement.
        min_vote_count: Minimum number of distinct contributing models per group.
        conflict_threshold: Raw-conflict review threshold and adjusted-conflict flagging threshold.
        enable_watershed: Enable watershed boundary refinement (default: True).

    Returns:
        (Fused labeled mask, dictionary of DST statistics).

        Statistics include:
        - 'conflict_distribution': Summary statistics for adjusted instance conflict.
        - 'average_uncertainty': Mean uncertainty of accepted instances.
        - 'high_conflict_instances': List of instances with high conflict.
        - 'fusion_results': Detailed fusion results for each instance.
        - 'watershed_refinement': Watershed refinement statistics, when enabled.
    """
    logger.info(f"Starting DST fusion: min_votes={min_vote_count}, conflict_threshold={conflict_threshold}")

    # 1. Initialize the DST fusion engine.
    dst_engine = DempsterShaferFusion(model_reliabilities)

    # 2. Initialize the output.
    fused_mask = np.zeros_like(masks_list[0], dtype=np.int32)
    instance_id = 1

    # 3. Initialize statistics.
    total_groups = len(matched_groups)
    fused_count = 0
    skipped_count = 0
    high_conflict_count = 0

    # 4. Track strategy usage.
    strategy_counts = {
        'ULTRA_AGGRESSIVE': 0,      # Ultra-aggressive: any one model (conf>0.75, conflict<0.2).
        'AGGRESSIVE': 0,             # Aggressive: max(1, floor(0.3*n)) votes.
        'RELAXED': 0,                # Relaxed: max(1, floor(0.4*n)) votes.
        'STANDARD_HIGH_CONF': 0,     # Standard, high confidence: 50% threshold.
        'STANDARD_MID_CONF': 0,      # Standard, medium confidence: 50% threshold.
        'STANDARD_LOW_CONF': 0,      # Standard, low confidence: 50% threshold.
        'STRICT': 0,                 # Strict: max(1, floor(0.6*n)) votes.
        'STRICT_LOW_CONF': 0,        # Low confidence: max(1, floor(0.6*n)) votes.
        'ULTRA_STRICT': 0,           # Ultra-strict: max(2, floor(0.7*n)) votes.
        'INTERSECTION': 0            # Intersection: all models must agree.
    }

    # 5. Store fusion results for each instance.
    fusion_results = []
    high_conflict_instances = []

    # Iterate over matched groups.
    for group_idx, group in enumerate(matched_groups):
        # Check the vote count.
        contributing_models = set([model_idx for model_idx, _ in group])
        if len(contributing_models) < min_vote_count:
            skipped_count += 1
            continue

        # 6. Prepare DST input as a list of (model_name, confidence) tuples.
        dst_input = []
        binary_masks = []

        for model_idx, inst_id in group:
            model_name = model_names[model_idx]

            # Extract the binary mask of this instance.
            mask = (masks_list[model_idx] == inst_id)
            binary_masks.append(mask)

            # Obtain the confidence score.
            if confidences_list[model_idx] is not None:
                # Use mean confidence over the instance region.
                confidence = np.mean(confidences_list[model_idx][mask])
            else:
                # Use a default confidence of 0.8 when no confidence map is available.
                confidence = 0.8

            dst_input.append((model_name, float(confidence)))

        # 7. Apply DST fusion.
        try:
            result: FusionResult = dst_engine.fuse_instances(dst_input)

            # 7.5. Compute shape disagreement to improve conflict detection.
            # Increase conflict when model masks differ substantially in shape.
            shape_disagreement = 0.0
            if len(binary_masks) >= 2:
                # Compute IoU for every pair of masks.
                ious = []
                for i in range(len(binary_masks)):
                    for j in range(i + 1, len(binary_masks)):
                        intersection = np.sum(binary_masks[i] & binary_masks[j])
                        union = np.sum(binary_masks[i] | binary_masks[j])
                        if union > 0:
                            iou = intersection / union
                            ious.append(iou)

                if ious:
                    avg_iou = np.mean(ious)
                    # Shape disagreement = 1 - mean IoU.
                    # A low IoU indicates greater shape disagreement.
                    shape_disagreement = 1.0 - avg_iou

            # Adjust conflict by combining DST conflict with shape disagreement.
            # Use a weighted average: 0.6 for DST conflict and 0.4 for shape disagreement.
            adjusted_conflict = result.conflict * 0.6 + shape_disagreement * 0.4

            # 8. Apply the acceptance policy to the original DST conflict score.
            conflict_info = handle_conflict(result, {'low': 0.3, 'medium': conflict_threshold, 'high': 0.9})

            # 9. Decide whether to accept the instance based on conflict handling.
            if conflict_info['action'] in ['accept', 'accept_with_caution']:
                # Adapt the DST fusion strategy to confidence and conflict.
                stack = np.stack(binary_masks, axis=0)
                vote_count = np.sum(stack, axis=0)

                # Select a strategy using a two-dimensional confidence-by-conflict decision matrix.
                conf = result.confidence
                conflict = adjusted_conflict  # Use adjusted conflict, including shape disagreement.
                n_models = len(binary_masks)
                strategy_used = ""

                # Strategy 1: high confidence (conf > 0.75).
                if conf > 0.75:
                    if conflict < 0.2:
                        # Very high confidence and very low conflict: ultra-aggressive (any one model).
                        consensus_mask = vote_count >= 1
                        strategy_used = "ULTRA_AGGRESSIVE"
                    elif conflict < 0.4:
                        # High confidence and low conflict: max(1, floor(0.3*n)) votes.
                        consensus_mask = vote_count >= max(1, int(n_models * 0.3))
                        strategy_used = "AGGRESSIVE"
                    else:
                        # High confidence and moderate-to-high conflict: standard (50% threshold).
                        consensus_mask = vote_count >= (n_models / 2)
                        strategy_used = "STANDARD_HIGH_CONF"

                # Strategy 2: medium confidence (0.55 < conf <= 0.75).
                elif conf > 0.55:
                    if conflict < 0.3:
                        # Moderate-to-high confidence and low conflict: max(1, floor(0.4*n)) votes.
                        consensus_mask = vote_count >= max(1, int(n_models * 0.4))
                        strategy_used = "RELAXED"
                    elif conflict < 0.5:
                        # Medium confidence and medium conflict: standard (50% threshold).
                        consensus_mask = vote_count >= (n_models / 2)
                        strategy_used = "STANDARD_MID_CONF"
                    else:
                        # Medium confidence and high conflict: max(1, floor(0.6*n)) votes.
                        consensus_mask = vote_count >= max(1, int(n_models * 0.6))
                        strategy_used = "STRICT"

                # Strategy 3: low confidence (conf <= 0.55).
                else:
                    if conflict < 0.4:
                        # Low confidence and low conflict: standard (50% threshold).
                        consensus_mask = vote_count >= (n_models / 2)
                        strategy_used = "STANDARD_LOW_CONF"
                    elif conflict < 0.6:
                        # Low confidence and medium conflict: max(1, floor(0.6*n)) votes.
                        consensus_mask = vote_count >= max(1, int(n_models * 0.6))
                        strategy_used = "STRICT_LOW_CONF"
                    else:
                        # Low confidence and high conflict: max(2, floor(0.7*n)) votes or intersection.
                        if n_models >= 3:
                            consensus_mask = vote_count >= max(2, int(n_models * 0.7))
                            strategy_used = "ULTRA_STRICT"
                        else:
                            consensus_mask = vote_count == n_models
                            strategy_used = "INTERSECTION"

                logger.debug(f"Group {group_idx}: conf={result.confidence:.3f}, conflict={result.conflict:.3f}, "
                           f"strategy={strategy_used}, models={len(binary_masks)}")

                # Track strategy usage.
                strategy_counts[strategy_used] += 1

                if np.sum(consensus_mask) > 0:
                    fused_mask[consensus_mask] = instance_id

                    # Record the fusion result.
                    fusion_results.append({
                        'instance_id': instance_id,
                        'group_idx': group_idx,
                        'decision': result.decision,
                        'confidence': result.confidence,
                        'conflict': adjusted_conflict,  # Use adjusted conflict.
                        'dst_conflict': result.conflict,  # Preserve the original DST conflict score.
                        'shape_disagreement': shape_disagreement,  # Include shape disagreement.
                        'uncertainty': result.uncertainty,
                        'belief_cell': result.belief_cell,
                        'plausibility_cell': result.plausibility_cell,
                        'status': conflict_info['status'],
                        'strategy': strategy_used  # Include strategy information.
                    })

                    instance_id += 1
                    fused_count += 1

                    # Flag instances with high conflict using the adjusted score.
                    if adjusted_conflict >= conflict_threshold:
                        high_conflict_count += 1
                        high_conflict_instances.append({
                            'instance_id': instance_id - 1,
                            'conflict': adjusted_conflict,
                            'dst_conflict': result.conflict,
                            'shape_disagreement': shape_disagreement,
                            'uncertainty': result.uncertainty
                        })
            else:
                # Reject the instance or request manual review.
                logger.warning(f"Group {group_idx} rejected: {conflict_info.get('warning', conflict_info.get('error'))}")
                skipped_count += 1

        except Exception as e:
            logger.error(f"DST fusion failed for group {group_idx}: {e}")
            skipped_count += 1

    # 10. Generate summary statistics.
    # Compute confidence and conflict distributions.
    confidences = [r['confidence'] for r in fusion_results]
    conflicts = [r['conflict'] for r in fusion_results]

    dst_stats = {
        'total_groups': total_groups,
        'fused_count': fused_count,
        'skipped_count': skipped_count,
        'high_conflict_count': high_conflict_count,
        'fusion_results': fusion_results,
        'high_conflict_instances': high_conflict_instances,
        'average_conflict': np.mean(conflicts) if conflicts else 0.0,
        'average_uncertainty': np.mean([r['uncertainty'] for r in fusion_results]) if fusion_results else 0.0,
        'strategy_counts': strategy_counts,  # Include strategy counts.
        # Include confidence and conflict distribution statistics.
        'confidence_distribution': {
            'min': np.min(confidences) if confidences else 0.0,
            'max': np.max(confidences) if confidences else 0.0,
            'mean': np.mean(confidences) if confidences else 0.0,
            'std': np.std(confidences) if confidences else 0.0
        },
        'conflict_distribution': {
            'min': np.min(conflicts) if conflicts else 0.0,
            'max': np.max(conflicts) if conflicts else 0.0,
            'mean': np.mean(conflicts) if conflicts else 0.0,
            'std': np.std(conflicts) if conflicts else 0.0
        }
    }

    logger.info(f"DST fusion complete: total_groups={total_groups}, fused={fused_count}, skipped={skipped_count}, "
               f"high_conflict={high_conflict_count}, mean_conflict={dst_stats['average_conflict']:.3f}")

    # Log confidence and conflict distributions.
    logger.info(f"Confidence distribution: min={dst_stats['confidence_distribution']['min']:.3f}, "
               f"max={dst_stats['confidence_distribution']['max']:.3f}, "
               f"mean={dst_stats['confidence_distribution']['mean']:.3f}, "
               f"std={dst_stats['confidence_distribution']['std']:.3f}")
    logger.info(f"Conflict distribution: min={dst_stats['conflict_distribution']['min']:.3f}, "
               f"max={dst_stats['conflict_distribution']['max']:.3f}, "
               f"mean={dst_stats['conflict_distribution']['mean']:.3f}, "
               f"std={dst_stats['conflict_distribution']['std']:.3f}")

    # Log strategy usage statistics.
    logger.info(f"Strategy usage:")
    for strategy, count in strategy_counts.items():
        if count > 0:
            percentage = (count / fused_count * 100) if fused_count > 0 else 0
            logger.info(f"  {strategy}: {count} ({percentage:.1f}%)")

    # 11. Apply watershed boundary refinement when enabled and fused instances exist.
    if enable_watershed and fused_count > 0 and image is not None:
        logger.info(f"Starting watershed boundary refinement: fused_instances={fused_count}, high_conflict_instances={high_conflict_count}")

        # Build an instance-level conflict map.
        conflict_map_instance = np.zeros_like(fused_mask, dtype=np.float32)
        for result in fusion_results:
            instance_mask = fused_mask == result['instance_id']
            conflict_map_instance[instance_mask] = result['conflict']

        # Apply watershed refinement.
        from .geometric_refinement import watershed_refinement
        refined_mask, refinement_stats = watershed_refinement(
            image=image,
            fused_mask=fused_mask,
            conflict_map=conflict_map_instance,
            conflict_threshold=conflict_threshold,
            gradient_sigma=1.0
        )

        # Update statistics.
        dst_stats['watershed_refinement'] = refinement_stats

        if refinement_stats['refined']:
            logger.info(f"Watershed refinement complete: modified {refinement_stats['refined_pixels']} pixels "
                       f"({refinement_stats['refined_percentage']:.2f}%)")
            fused_mask = refined_mask
        else:
            logger.warning(f"Watershed refinement failed: {refinement_stats.get('reason', 'unknown')}")
    elif enable_watershed and high_conflict_count > 0 and image is None:
        logger.warning("Watershed refinement is enabled, but no original image was provided; skipping refinement")
        dst_stats['watershed_refinement'] = {'refined': False, 'reason': 'no_image'}
    else:
        dst_stats['watershed_refinement'] = {'refined': False, 'reason': 'disabled_or_no_conflict'}

    return fused_mask, dst_stats
