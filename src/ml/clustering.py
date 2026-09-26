"""
Clustering analysis for cell morphology features.

Group cells by morphology with several unsupervised clustering algorithms.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from loguru import logger


def preprocess_features(features_df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """
    Exclude metadata columns and standardize numeric features.

    Args:
        features_df: DataFrame of cell features.
        exclude_cols: Column names to exclude, or None to use the default exclusions.

    Returns:
        features_scaled: Standardized feature array.
        feature_cols: Names of the feature columns used.
        scaler: Fitted scaler, also usable for inverse transformation.
    """
    if exclude_cols is None:
        # Exclude identifiers, positions, and bounding boxes by default.
        exclude_cols = [
            'sequential_id', 'cell_id',
            'centroid_x', 'centroid_y',
            'bbox_min_row', 'bbox_min_col',
            'bbox_max_row', 'bbox_max_col'
        ]

    # Select feature columns.
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]

    # Extract feature data.
    features = features_df[feature_cols].values

    # Check for NaN and infinite values.
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.warning("Features contain NaN or Inf values, replacing with column means")
        features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values

    # Standardize features.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    logger.info(f"Preprocessed {len(feature_cols)} features for {len(features_df)} cells")

    return features_scaled, feature_cols, scaler


def perform_kmeans(features_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    Cluster features with k-means.

    Args:
        features_df: DataFrame of cell features.
        n_clusters: Number of clusters.
        random_state: Random seed.

    Returns:
        labels: Array of cluster labels.
        info: Dictionary containing the fitted model, inertia, and other metadata.
    """
    # Preprocess the features.
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # Run k-means clustering.
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    # Compute clustering quality metrics.
    silhouette = silhouette_score(features_scaled, labels) if n_clusters > 1 else 0
    davies_bouldin = davies_bouldin_score(features_scaled, labels) if n_clusters > 1 else 0
    calinski_harabasz = calinski_harabasz_score(features_scaled, labels) if n_clusters > 1 else 0

    info = {
        'model': kmeans,
        'inertia': kmeans.inertia_,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'n_clusters': n_clusters,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"K-means clustering completed: {n_clusters} clusters, silhouette={silhouette:.3f}")

    return labels, info


def perform_dbscan(features_df: pd.DataFrame, eps: Optional[float] = None, min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
    """
    Cluster features with density-based DBSCAN.

    Args:
        features_df: DataFrame of cell features.
        eps: Neighborhood radius, estimated automatically when None.
        min_samples: Minimum neighborhood sample count for a core point.

    Returns:
        labels: Cluster labels, with -1 indicating noise.
        info: Dictionary of clustering metadata.
    """
    # Preprocess the features.
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # Estimate eps from nearest-neighbor distances when it is not specified.
    if eps is None:
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(features_scaled)
        distances, _ = neighbors.kneighbors(features_scaled)
        distances = np.sort(distances[:, -1])
        # Use the 90th distance percentile as eps.
        eps = np.percentile(distances, 90)
        logger.info(f"Auto-estimated eps={eps:.3f}")

    # Run DBSCAN clustering.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_scaled)

    # Count clusters, excluding noise.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Compute clustering metrics after excluding noise.
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 0:
            silhouette = silhouette_score(features_scaled[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(features_scaled[mask], labels[mask])
            calinski_harabasz = calinski_harabasz_score(features_scaled[mask], labels[mask])
        else:
            silhouette = davies_bouldin = calinski_harabasz = 0
    else:
        silhouette = davies_bouldin = calinski_harabasz = 0

    info = {
        'model': dbscan,
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"DBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points, silhouette={silhouette:.3f}")

    return labels, info


def perform_hierarchical(features_df: pd.DataFrame, n_clusters: int = 3, linkage: str = 'ward') -> Tuple[np.ndarray, Dict]:
    """
    Perform agglomerative hierarchical clustering.

    Args:
        features_df: DataFrame of cell features.
        n_clusters: Number of clusters.
        linkage: Linkage method: 'ward', 'complete', 'average', or 'single'.

    Returns:
        labels: Array of cluster labels.
        info: Dictionary of clustering metadata.
    """
    # Preprocess the features.
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # Run hierarchical clustering.
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hierarchical.fit_predict(features_scaled)

    # Compute clustering quality metrics.
    silhouette = silhouette_score(features_scaled, labels) if n_clusters > 1 else 0
    davies_bouldin = davies_bouldin_score(features_scaled, labels) if n_clusters > 1 else 0
    calinski_harabasz = calinski_harabasz_score(features_scaled, labels) if n_clusters > 1 else 0

    info = {
        'model': hierarchical,
        'n_clusters': n_clusters,
        'linkage': linkage,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"Hierarchical clustering completed: {n_clusters} clusters, linkage={linkage}, silhouette={silhouette:.3f}")

    return labels, info


def perform_gmm(features_df: pd.DataFrame, n_components: int = 3, random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    Cluster features with a Gaussian mixture model.

    Args:
        features_df: DataFrame of cell features.
        n_components: Number of Gaussian components.
        random_state: Random seed.

    Returns:
        labels: Array of cluster labels.
        info: Dictionary of clustering metadata.
    """
    # Preprocess the features.
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # Fit a Gaussian mixture model.
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, covariance_type='full')
    labels = gmm.fit_predict(features_scaled)

    # Compute clustering quality metrics.
    silhouette = silhouette_score(features_scaled, labels) if n_components > 1 else 0
    davies_bouldin = davies_bouldin_score(features_scaled, labels) if n_components > 1 else 0
    calinski_harabasz = calinski_harabasz_score(features_scaled, labels) if n_components > 1 else 0

    info = {
        'model': gmm,
        'n_components': n_components,
        'bic': gmm.bic(features_scaled),
        'aic': gmm.aic(features_scaled),
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"GMM clustering completed: {n_components} components, BIC={gmm.bic(features_scaled):.2f}, silhouette={silhouette:.3f}")

    return labels, info


def evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate clustering quality.

    Args:
        features: Feature array.
        labels: Array of cluster labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        return {
            'silhouette_score': 0.0,
            'davies_bouldin_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'n_clusters': n_clusters
        }

    # Exclude DBSCAN noise points labeled -1.
    mask = labels != -1
    if mask.sum() == 0:
        return {
            'silhouette_score': 0.0,
            'davies_bouldin_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'n_clusters': 0
        }

    features_clean = features[mask]
    labels_clean = labels[mask]

    # Compute evaluation metrics.
    silhouette = silhouette_score(features_clean, labels_clean)
    davies_bouldin = davies_bouldin_score(features_clean, labels_clean)
    calinski_harabasz = calinski_harabasz_score(features_clean, labels_clean)

    return {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'n_clusters': n_clusters
    }


def find_optimal_clusters(features_df: pd.DataFrame, max_k: int = 10, method: str = 'kmeans') -> Dict:
    """
    Evaluate candidate cluster counts and select the best silhouette score.

    Args:
        features_df: DataFrame of cell features.
        max_k: Maximum number of clusters to evaluate.
        method: Clustering method: 'kmeans' or 'gmm'.

    Returns:
        Dictionary of evaluation results for the tested cluster counts.
    """
    # Preprocess the features.
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    results = {
        'k_values': [],
        'inertias': [],
        'silhouette_scores': [],
        'davies_bouldin_scores': [],
        'calinski_harabasz_scores': []
    }

    if method == 'gmm':
        results['bic_scores'] = []
        results['aic_scores'] = []

    logger.info(f"Finding optimal number of clusters (k=2 to {max_k})...")

    for k in range(2, max_k + 1):
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            inertia = kmeans.inertia_
            results['inertias'].append(inertia)
        elif method == 'gmm':
            gmm = GaussianMixture(n_components=k, random_state=42)
            labels = gmm.fit_predict(features_scaled)
            results['bic_scores'].append(gmm.bic(features_scaled))
            results['aic_scores'].append(gmm.aic(features_scaled))

        # Compute evaluation metrics.
        silhouette = silhouette_score(features_scaled, labels)
        davies_bouldin = davies_bouldin_score(features_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, labels)

        results['k_values'].append(k)
        results['silhouette_scores'].append(silhouette)
        results['davies_bouldin_scores'].append(davies_bouldin)
        results['calinski_harabasz_scores'].append(calinski_harabasz)

    # Select the best k using the silhouette score.
    best_k_idx = np.argmax(results['silhouette_scores'])
    best_k = results['k_values'][best_k_idx]

    results['best_k'] = best_k
    results['best_silhouette'] = results['silhouette_scores'][best_k_idx]

    logger.info(f"Optimal k={best_k} with silhouette score={results['best_silhouette']:.3f}")

    return results
