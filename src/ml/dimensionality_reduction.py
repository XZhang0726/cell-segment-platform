"""
Dimensionality reduction for feature visualization.

Project high-dimensional features into two or three dimensions.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from loguru import logger

# Import the optional UMAP dependency.
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


def apply_pca(features_df: pd.DataFrame, n_components: int = 2, exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, PCA, List[str]]:
    """
    Reduce feature dimensionality with PCA.

    Args:
        features_df: DataFrame of cell features.
        n_components: Number of output dimensions, typically 2 or 3.
        exclude_cols: Column names to exclude.

    Returns:
        components: Projected feature data.
        pca: Fitted PCA model.
        feature_cols: Names of the feature columns used.
    """
    if exclude_cols is None:
        exclude_cols = [
            'sequential_id', 'cell_id',
            'centroid_x', 'centroid_y',
            'bbox_min_row', 'bbox_min_col',
            'bbox_max_row', 'bbox_max_col'
        ]

    # Select feature columns.
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    features = features_df[feature_cols].values

    # Handle NaN and infinite values.
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Features contain NaN or Inf, replacing with column means")
        features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values

    # Apply PCA.
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(features)

    # Compute explained variance ratios.
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()

    logger.info(f"PCA completed: {n_components} components explain {total_variance*100:.2f}% of variance")

    return components, pca, feature_cols


def apply_tsne(features_df: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0,
               n_iter: int = 1000, exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Embed features with t-SNE.

    Args:
        features_df: DataFrame of cell features.
        n_components: Number of output dimensions, typically 2 or 3.
        perplexity: Perplexity, typically 5-50; default is 30.
        n_iter: Number of optimization iterations.
        exclude_cols: Column names to exclude.

    Returns:
        components: Projected feature data.
        feature_cols: Names of the feature columns used.
    """
    if exclude_cols is None:
        exclude_cols = [
            'sequential_id', 'cell_id',
            'centroid_x', 'centroid_y',
            'bbox_min_row', 'bbox_min_col',
            'bbox_max_row', 'bbox_max_col'
        ]

    # Select feature columns.
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    features = features_df[feature_cols].values

    # Handle NaN and infinite values.
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Features contain NaN or Inf, replacing with column means")
        features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values

    # Adjust perplexity to the sample count.
    n_samples = len(features)
    if perplexity >= n_samples:
        perplexity = max(5, n_samples // 3)
        logger.warning(f"Perplexity too large for {n_samples} samples, adjusted to {perplexity}")

    # Apply t-SNE.
    logger.info(f"Running t-SNE (this may take a while for large datasets)...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=n_iter,
                random_state=42, verbose=0)
    components = tsne.fit_transform(features)

    logger.info(f"t-SNE completed: {n_components} components, perplexity={perplexity}")

    return components, feature_cols


def apply_umap(features_df: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15,
               min_dist: float = 0.1, exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Embed features with UMAP.

    Args:
        features_df: DataFrame of cell features.
        n_components: Number of output dimensions, typically 2 or 3.
        n_neighbors: Number of neighbors, typically 5-50; default is 15.
        min_dist: Minimum embedding distance, from 0.0 to 0.99; default is 0.1.
        exclude_cols: Column names to exclude.

    Returns:
        components: Projected feature data.
        feature_cols: Names of the feature columns used.
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not installed. Install with: pip install umap-learn")

    if exclude_cols is None:
        exclude_cols = [
            'sequential_id', 'cell_id',
            'centroid_x', 'centroid_y',
            'bbox_min_row', 'bbox_min_col',
            'bbox_max_row', 'bbox_max_col'
        ]

    # Select feature columns.
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    features = features_df[feature_cols].values

    # Handle NaN and infinite values.
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Features contain NaN or Inf, replacing with column means")
        features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values

    # Adjust n_neighbors to the sample count.
    n_samples = len(features)
    if n_neighbors >= n_samples:
        n_neighbors = max(2, n_samples // 2)
        logger.warning(f"n_neighbors too large for {n_samples} samples, adjusted to {n_neighbors}")

    # Apply UMAP.
    logger.info(f"Running UMAP...")
    umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                      random_state=42, verbose=False)
    components = umap_model.fit_transform(features)

    logger.info(f"UMAP completed: {n_components} components, n_neighbors={n_neighbors}, min_dist={min_dist}")

    return components, feature_cols
