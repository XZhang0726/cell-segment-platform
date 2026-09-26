"""
Dempster-Shafer Theory (DST) Fusion for Cell Instance Segmentation

This module implements evidence-based fusion using Dempster-Shafer Theory,
which provides a mathematical framework for combining evidence from multiple
segmentation models while explicitly modeling uncertainty and conflict.

Key Concepts:
- Mass Function: Represents evidence distribution over hypotheses
- Dempster's Combination Rule: Combines evidence from multiple sources
- Conflict Coefficient: Quantifies disagreement between models
- Belief/Plausibility: Lower and upper bounds of probability

Author: Claude Sonnet 4.5
Date: 2026-01-27
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class FusionResult:
    """Data class for fusion results.

    Attributes:
        mass_function: Final mass-function dictionary.
        decision: Decision ('Cell', 'Background', 'Cell|Background').
        confidence: Confidence in the range [0, 1].
        conflict: Sum of pairwise conflict coefficients; may exceed 1 for multiple models.
        uncertainty: Uncertainty in the range [0, 1].
        belief_cell: Belief assigned to Cell in the range [0, 1].
        plausibility_cell: Plausibility assigned to Cell in the range [0, 1].
    """
    mass_function: Dict[str, float]
    decision: str
    confidence: float
    conflict: float
    uncertainty: float
    belief_cell: float
    plausibility_cell: float


class DempsterShaferFusion:
    """
    Dempster-Shafer evidence fusion engine.

    Combine predictions from multiple cell segmentation models within a formal
    framework for handling uncertainty and conflicting evidence.

    Example:
        >>> fusion_engine = DempsterShaferFusion({
        ...     'cellpose': 0.9,
        ...     'cellvit': 0.85,
        ...     'cellsam': 0.8
        ... })
        >>> matched_group = [
        ...     ('cellpose', 0.8),
        ...     ('cellvit', 0.6),
        ...     ('cellsam', 0.9)
        ... ]
        >>> result = fusion_engine.fuse_instances(matched_group)
        >>> print(f"Decision: {result.decision}, Confidence: {result.confidence:.3f}")
    """

    def __init__(self, model_reliabilities: Dict[str, float]):
        """
        Initialize the fusion engine.

        Args:
            model_reliabilities: Per-model reliability parameters in the range [0, 1].
                Example: {'cellpose': 0.9, 'cellvit': 0.85, 'cellsam': 0.8}.
                Higher reliability gives the model's evidence greater weight.
        """
        self.reliabilities = model_reliabilities
        self.hypotheses = ['Cell', 'Background', 'Cell|Background']
        logger.info(f"Initialized DST Fusion Engine with reliabilities: {model_reliabilities}")

    def compute_mass_from_confidence(self,
                                    model_name: str,
                                    confidence: float) -> Dict[str, float]:
        """
        Compute a mass function from a confidence score.

        Convert a model's confidence score to a DST mass function.

        Formula:
            m(Cell) = confidence × reliability
            m(Background) = (1 - confidence) × reliability
            m(Cell|Background) = 1 - reliability

        Args:
            model_name: Model name.
            confidence: Confidence in the range [0, 1].

        Returns:
            Dictionary mapping hypotheses to mass values.
        """
        reliability = self.reliabilities.get(model_name, 0.8)

        mass_func = {
            'Cell': confidence * reliability,
            'Background': (1 - confidence) * reliability,
            'Cell|Background': 1 - reliability
        }

        # Verify that mass values sum to 1.
        total = sum(mass_func.values())
        assert abs(total - 1.0) < 1e-6, f"Mass function sum is {total}, should be 1.0"

        return mass_func

    def compute_intersection(self, A: str, B: str) -> str:
        """
        Compute the intersection of two hypotheses.

        Intersection rules:
        - Cell ∩ Cell = Cell
        - Cell ∩ Background = ∅ (conflict)
        - Cell ∩ Uncertain = Cell
        - Background ∩ Background = Background
        - Uncertain ∩ Uncertain = Uncertain

        Args:
            A, B: Hypothesis strings, e.g. 'Cell', 'Background', 'Cell|Background'.

        Returns:
            Intersection string; the empty-set symbol denotes conflict.
        """
        # Parse hypotheses as sets.
        set_A = set(A.split('|'))
        set_B = set(B.split('|'))

        # Compute the intersection.
        intersection = set_A & set_B

        if len(intersection) == 0:
            return '∅'  # An empty set denotes conflict.
        else:
            return '|'.join(sorted(intersection))

    def dempster_combine(self,
                        m1: Dict[str, float],
                        m2: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """
        Dempster's rule of combination.

        Combine two mass functions to compute joint evidence and the conflict coefficient.

        Mathematical formula:
            m₁₂(C) = [Σ m₁(A) × m₂(B)] / (1 - K)
                     A∩B=C

            K = Σ m₁(A) × m₂(B)  (conflict coefficient)
                A∩B=∅

        Args:
            m1, m2: Two mass functions.

        Returns:
            (Combined mass function, conflict coefficient).

        Raises:
            ValueError: If the conflict coefficient is >= 1.0 (total conflict).
        """
        combined = {}
        conflict = 0.0

        # Compute all possible intersections.
        for (A, m1_A) in m1.items():
            for (B, m2_B) in m2.items():
                intersection = self.compute_intersection(A, B)

                if intersection == '∅':
                    # Conflict: the two hypotheses are incompatible.
                    conflict += m1_A * m2_B
                else:
                    # Accumulate evidence for the corresponding intersection.
                    if intersection not in combined:
                        combined[intersection] = 0.0
                    combined[intersection] += m1_A * m2_B

        # Check for total conflict.
        if conflict >= 1.0:
            logger.error(f"Complete conflict detected! K={conflict}")
            raise ValueError(f"Total conflict: K={conflict}; fusion is not possible")

        # Normalize.
        normalization = 1.0 - conflict
        for key in combined:
            combined[key] /= normalization

        logger.debug(f"Dempster combination: conflict={conflict:.3f}, combined={combined}")

        return combined, conflict

    def combine_multiple(self,
                        mass_functions: List[Dict[str, float]]) -> Tuple[Dict[str, float], float]:
        """
        Combine multiple mass functions.

        Iteratively apply Dempster's rule to combine model evidence into a single mass function.

        Args:
            mass_functions: List of mass functions.

        Returns:
            (Final combined mass function, sum of conflict coefficients from each combination).
            The accumulated conflict is not normalized and may exceed 1.

        Raises:
            ValueError: If the list of mass functions is empty.
        """
        if len(mass_functions) == 0:
            raise ValueError("At least one mass function is required")

        if len(mass_functions) == 1:
            return mass_functions[0], 0.0

        # Combine iteratively.
        combined = mass_functions[0]
        total_conflict = 0.0

        for i in range(1, len(mass_functions)):
            combined, conflict = self.dempster_combine(combined, mass_functions[i])
            total_conflict += conflict

        logger.info(f"Combined {len(mass_functions)} mass functions, total conflict: {total_conflict:.3f}")

        return combined, total_conflict

    def compute_belief(self,
                      mass_function: Dict[str, float],
                      hypothesis: str) -> float:
        """
        Compute the belief function Bel(A).

        Belief is the lower bound on support for hypothesis A.

        Formula:
            Bel(A) = Σ m(B), for all B ⊆ A

        Args:
            mass_function: Mass function.
            hypothesis: Target hypothesis.

        Returns:
            Belief in the range [0, 1].
        """
        belief = 0.0
        target_set = set(hypothesis.split('|'))

        for B, mass in mass_function.items():
            B_set = set(B.split('|'))
            if B_set.issubset(target_set):
                belief += mass

        return belief

    def compute_plausibility(self,
                            mass_function: Dict[str, float],
                            hypothesis: str) -> float:
        """
        Compute the plausibility function Pl(A).

        Plausibility is the upper bound on support for hypothesis A.

        Formula:
            Pl(A) = Σ m(B), for all B ∩ A ≠ ∅

        Args:
            mass_function: Mass function.
            hypothesis: Target hypothesis.

        Returns:
            Plausibility in the range [0, 1].
        """
        plausibility = 0.0
        target_set = set(hypothesis.split('|'))

        for B, mass in mass_function.items():
            B_set = set(B.split('|'))
            if len(B_set & target_set) > 0:  # The hypotheses overlap.
                plausibility += mass

        return plausibility

    def decide(self, mass_function: Dict[str, float]) -> Tuple[str, float]:
        """
        Select the singleton hypothesis with the largest mass.

        Choose the singleton hypothesis with the highest mass as the final decision.

        Args:
            mass_function: Mass function.

        Returns:
            (Decision, confidence).
        """
        max_mass = 0.0
        decision = 'Cell|Background'  # Default to uncertainty.

        for hypothesis, mass in mass_function.items():
            if '|' not in hypothesis:  # Singleton hypothesis.
                if mass > max_mass:
                    max_mass = mass
                    decision = hypothesis

        return decision, max_mass

    def fuse_instances(self,
                      matched_group: List[Tuple[str, float]]) -> FusionResult:
        """
        Fuse a group of matched instances.

        This primary fusion interface accepts matched instances from different models
        and returns a fusion result with uncertainty measures.

        Args:
            matched_group: [(model_name, confidence), ...]
                Example: [('cellpose', 0.8), ('cellvit', 0.6), ('cellsam', 0.9)].

        Returns:
            FusionResult containing the decision, confidence, conflict, and related measures.
        """
        logger.debug(f"Fusing {len(matched_group)} instances: {matched_group}")

        # Step 1: compute a mass function for each model.
        mass_functions = []
        for model_name, confidence in matched_group:
            mass = self.compute_mass_from_confidence(model_name, confidence)
            mass_functions.append(mass)
            logger.debug(f"  {model_name}: confidence={confidence:.3f}, mass={mass}")

        # Step 2: combine all mass functions.
        combined_mass, total_conflict = self.combine_multiple(mass_functions)

        # Step 3: make a decision.
        decision, confidence = self.decide(combined_mass)

        # Step 4: compute belief and plausibility.
        belief_cell = self.compute_belief(combined_mass, 'Cell')
        plausibility_cell = self.compute_plausibility(combined_mass, 'Cell')

        # Step 5: compute uncertainty.
        uncertainty = combined_mass.get('Cell|Background', 0.0)

        result = FusionResult(
            mass_function=combined_mass,
            decision=decision,
            confidence=confidence,
            conflict=total_conflict,
            uncertainty=uncertainty,
            belief_cell=belief_cell,
            plausibility_cell=plausibility_cell
        )

        logger.info(f"DST Fusion Result: decision={decision}, confidence={confidence:.3f}, "
                   f"conflict={total_conflict:.3f}, uncertainty={uncertainty:.3f}")

        return result


def handle_conflict(result: FusionResult,
                   conflict_thresholds: Optional[Dict[str, float]] = None) -> Dict:
    """
    Choose a handling strategy based on the conflict level.

    Conflict levels:
    - Low conflict (K < 0.3): accept fusion normally.
    - Moderate conflict (0.3 <= K < 0.6): reduce confidence.
    - High conflict (0.6 <= K < 0.9): flag for manual review.
    - Conflict at or above the rejection threshold (K >= 0.9): reject fusion.

    Args:
        result: Fusion result.
        conflict_thresholds: Dictionary of conflict thresholds.

    Returns:
        Result dictionary with status, decision, confidence, and recommended action.
    """
    if conflict_thresholds is None:
        conflict_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9
        }

    K = result.conflict

    if K < conflict_thresholds['low']:
        # Low conflict: accept normally.
        return {
            'status': 'NORMAL',
            'decision': result.decision,
            'confidence': result.confidence,
            'action': 'accept',
            'message': f'Low conflict (K={K:.3f}); fusion accepted'
        }

    elif K < conflict_thresholds['medium']:
        # Moderate conflict: reduce confidence.
        adjusted_confidence = result.confidence * (1 - K * 0.5)
        return {
            'status': 'MEDIUM_CONFLICT',
            'decision': result.decision,
            'confidence': adjusted_confidence,
            'action': 'accept_with_caution',
            'warning': f'Moderate conflict (K={K:.3f}); confidence adjusted'
        }

    elif K < conflict_thresholds['high']:
        # High conflict: flag for review.
        return {
            'status': 'HIGH_CONFLICT',
            'decision': 'Uncertain',
            'confidence': 0.0,
            'action': 'manual_review',
            'warning': f'High conflict (K={K:.3f}); manual review required'
        }

    else:
        # Conflict exceeds the configured rejection threshold.
        return {
            'status': 'COMPLETE_CONFLICT',
            'decision': 'Error',
            'confidence': 0.0,
            'action': 'reject',
            'error': f'Conflict too high (K={K:.3f}); fusion rejected'
        }


def generate_conflict_map(masks_list: List[np.ndarray],
                         confidences_list: List[np.ndarray],
                         model_names: List[str],
                         fusion_engine: DempsterShaferFusion) -> np.ndarray:
    """
    Generate a pixel-level conflict map.

    Accumulate conflict between foreground predictions at each pixel
    to visualize regions of model disagreement as a heatmap.

    Args:
        masks_list: List of model segmentation masks [(H, W), ...].
        confidences_list: List of confidence maps [(H, W), ...].
        model_names: List of model names.
        fusion_engine: DST fusion engine.

    Returns:
        Conflict map (H, W); larger values indicate stronger conflict. Accumulated
        scores may exceed 1, and failed combinations receive a fallback score of 1.
    """
    H, W = masks_list[0].shape
    conflict_map = np.zeros((H, W), dtype=np.float32)

    logger.info(f"Generating conflict map for {H}x{W} image with {len(masks_list)} models")

    for i in range(H):
        for j in range(W):
            # Collect all model predictions for this pixel.
            pixel_predictions = []
            for k, (mask, conf, name) in enumerate(zip(masks_list,
                                                       confidences_list,
                                                       model_names)):
                if mask[i, j] > 0:  # This model predicts foreground.
                    confidence = conf[i, j] if conf is not None else 0.5
                    pixel_predictions.append((name, confidence))

            # Fuse when at least two models predict foreground.
            if len(pixel_predictions) >= 2:
                try:
                    result = fusion_engine.fuse_instances(pixel_predictions)
                    conflict_map[i, j] = result.conflict
                except Exception as e:
                    logger.warning(f"Fusion failed at pixel ({i},{j}): {e}")
                    conflict_map[i, j] = 1.0  # Assign a fallback conflict score if fusion fails.

    logger.info(f"Conflict map generated: mean={np.mean(conflict_map):.3f}, "
               f"max={np.max(conflict_map):.3f}")

    return conflict_map
