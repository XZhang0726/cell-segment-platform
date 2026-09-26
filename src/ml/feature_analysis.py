"""
Feature importance, correlation, and selection utilities.

Analyze feature importance, compute correlations, and select representative features.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from loguru import logger


def analyze_feature_importance(features_df: pd.DataFrame, cluster_labels: np.ndarray,
                               exclude_cols: Optional[List[str]] = None, top_n: int = 20) -> pd.DataFrame:
    """
    Estimate feature importance for distinguishing cluster labels.

    Fit a random forest classifier to assess feature contributions to cluster separation.

    Args:
        features_df: DataFrame of cell features.
        cluster_labels: Array of cluster labels.
        exclude_cols: Column names to exclude.
        top_n: Number of highest-ranked features to return.

    Returns:
        DataFrame of feature importance scores.
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

    # Check the number of clusters.
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    if n_clusters < 2:
        logger.warning("Less than 2 clusters, cannot analyze feature importance")
        return pd.DataFrame()

    # Exclude DBSCAN noise points labeled -1.
    mask = cluster_labels != -1
    features_clean = features[mask]
    labels_clean = cluster_labels[mask]

    if len(features_clean) == 0:
        logger.warning("No valid samples after removing noise points")
        return pd.DataFrame()

    # Train a random forest classifier.
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(features_clean, labels_clean)

    # Retrieve feature importances.
    importances = rf.feature_importances_

    # Create the results DataFrame.
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    })

    # Sort by importance.
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # Return only the top N features.
    importance_df = importance_df.head(top_n)

    logger.info(f"Feature importance analysis completed: top {len(importance_df)} features identified")

    return importance_df


def compute_feature_correlation(features_df: pd.DataFrame, exclude_cols: Optional[List[str]] = None,
                                method: str = 'spearman') -> pd.DataFrame:
    """
    Compute pairwise feature correlations.

    Args:
        features_df: DataFrame of cell features.
        exclude_cols: Column names to exclude.
        method: Correlation method: 'pearson' or 'spearman'.

    Returns:
        Correlation matrix as a DataFrame.
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
    features = features_df[feature_cols]

    # Compute the correlation matrix.
    if method == 'pearson':
        corr_matrix = features.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = features.corr(method='spearman')
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    logger.info(f"Feature correlation matrix computed using {method} method")

    return corr_matrix


def select_top_features(features_df: pd.DataFrame, cluster_labels: np.ndarray,
                       n_features: int = 20, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Select representative high-importance features.

    Use feature importance and correlation to select a representative subset.

    Args:
        features_df: DataFrame of cell features.
        cluster_labels: Array of cluster labels.
        n_features: Number of features to select.
        exclude_cols: Column names to exclude.

    Returns:
        Names of the selected feature columns.
    """
    # Analyze feature importance.
    importance_df = analyze_feature_importance(features_df, cluster_labels, exclude_cols, top_n=n_features*2)

    if importance_df.empty:
        logger.warning("Cannot select features: importance analysis failed")
        return []

    # Retrieve high-importance features.
    top_features = importance_df['feature'].tolist()

    # Compute correlations among candidate features.
    if exclude_cols is None:
        exclude_cols = []

    # Restrict the correlation matrix to candidate features.
    features_subset = features_df[[f for f in top_features if f in features_df.columns]]
    corr_matrix = features_subset.corr(method='spearman').abs()

    # Remove highly correlated features, retaining the more important feature.
    selected_features = []
    for feature in top_features:
        if feature not in features_subset.columns:
            continue

        # Check correlation with already selected features.
        is_redundant = False
        for selected in selected_features:
            if selected in corr_matrix.columns and feature in corr_matrix.index:
                if corr_matrix.loc[feature, selected] > 0.9:  # Correlation threshold
                    is_redundant = True
                    break

        if not is_redundant:
            selected_features.append(feature)

        # Stop when the target feature count is reached.
        if len(selected_features) >= n_features:
            break

    logger.info(f"Selected {len(selected_features)} features from {len(top_features)} candidates")

    return selected_features
