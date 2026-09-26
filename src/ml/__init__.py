"""
Machine learning utilities for cell feature analysis.

Includes clustering, dimensionality reduction, and feature analysis.
"""
from .clustering import (
    perform_kmeans,
    perform_dbscan,
    perform_hierarchical,
    perform_gmm,
    evaluate_clustering,
    find_optimal_clusters,
    preprocess_features
)

from .dimensionality_reduction import (
    apply_pca,
    apply_tsne,
    apply_umap
)

from .feature_analysis import (
    analyze_feature_importance,
    compute_feature_correlation,
    select_top_features
)

__all__ = [
    # Clustering analysis
    'perform_kmeans',
    'perform_dbscan',
    'perform_hierarchical',
    'perform_gmm',
    'evaluate_clustering',
    'find_optimal_clusters',
    'preprocess_features',

    # Dimensionality reduction
    'apply_pca',
    'apply_tsne',
    'apply_umap',

    # Feature analysis
    'analyze_feature_importance',
    'compute_feature_correlation',
    'select_top_features'
]
