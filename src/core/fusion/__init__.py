"""
Multi-model segmentation fusion.

Provides instance matching, mask fusion, and uncertainty estimation.
Includes simple fusion strategies and Dempster-Shafer evidence fusion.
"""

# Use optimized instance matching.
from .instance_matcher_optimized import match_instances_optimized as match_instances
from .fusion_engine import fuse_instances, fuse_instances_dst
from .uncertainty import compute_disagreement_map, compute_model_consistency

# Keep the original compute_iou implementation for compatibility.
from .instance_matcher import compute_iou

# Dempster-Shafer evidence fusion.
from .dempster_shafer import (
    DempsterShaferFusion,
    FusionResult,
    handle_conflict,
    generate_conflict_map
)

# Confidence utilities.
from .confidence_utils import generate_confidence_maps

# Geometric boundary refinement.
from .geometric_refinement import watershed_refinement, compute_gradient_map

__all__ = [
    # Core functionality.
    'compute_iou',
    'match_instances',

    # Simple fusion strategies.
    'fuse_instances',

    # Dempster-Shafer fusion.
    'fuse_instances_dst',
    'DempsterShaferFusion',
    'FusionResult',
    'handle_conflict',
    'generate_conflict_map',

    # Confidence utilities.
    'generate_confidence_maps',

    # Geometric boundary refinement.
    'watershed_refinement',
    'compute_gradient_map',

    # Uncertainty estimation.
    'compute_disagreement_map',
    'compute_model_consistency'
]
