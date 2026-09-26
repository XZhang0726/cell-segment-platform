"""
Anomaly detection for cell feature data.

Identify cell samples with atypical morphology using several anomaly detection methods.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a noninteractive plotting backend.
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from loguru import logger


def preprocess_features(features_df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """
    Preprocess and standardize feature data.

    Args:
        features_df: DataFrame of cell features.
        exclude_cols: Column names to exclude.

    Returns:
        features_scaled: Standardized feature array.
        feature_cols: Names of the selected feature columns.
        scaler: Fitted feature scaler.
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
        features_df_clean = pd.DataFrame(features, columns=feature_cols)
        features = features_df_clean.fillna(features_df_clean.mean()).values

    # Standardize features.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    logger.info(f"Preprocessed {len(feature_cols)} features for {len(features_df)} cells")

    return features_scaled, feature_cols, scaler


def detect_isolation_forest(features_df: pd.DataFrame, contamination: float = 0.1,
                            random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    Detect anomalies with an isolation forest.

    Isolation forests isolate samples through random feature splits.
    Anomalies tend to require fewer splits and have shorter isolation paths.

    Args:
        features_df: DataFrame of cell features.
        contamination: Expected anomaly fraction in the interval (0, 0.5].
        random_state: Random seed.

    Returns:
        labels: Label array: 1 for normal samples and -1 for anomalies.
        info: Dictionary containing detection metadata and scores.
    """
    # Preprocess the features.
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # Fit an isolation forest.
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    labels = iso_forest.fit_predict(features_scaled)

    # Retrieve anomaly scores; lower values indicate stronger anomalies.
    scores = iso_forest.score_samples(features_scaled)

    # Count anomalous samples.
    n_anomalies = (labels == -1).sum()
    n_normal = (labels == 1).sum()
    anomaly_ratio = n_anomalies / len(labels)

    info = {
        'model': iso_forest,
        'method': 'Isolation Forest',
        'contamination': contamination,
        'n_anomalies': int(n_anomalies),
        'n_normal': int(n_normal),
        'anomaly_ratio': float(anomaly_ratio),
        'scores': scores,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"Isolation Forest completed: {n_anomalies} anomalies ({anomaly_ratio:.1%}), {n_normal} normal")

    return labels, info


def detect_lof(features_df: pd.DataFrame, contamination: float = 0.1,
               n_neighbors: int = 20) -> Tuple[np.ndarray, Dict]:
    """
    Detect anomalies with Local Outlier Factor.

    LOF compares the local density of each sample with that of its neighbors.
    Samples with substantially lower local density are treated as anomalies.

    Args:
        features_df: DataFrame of cell features.
        contamination: Expected anomaly fraction in the interval (0, 0.5].
        n_neighbors: Number of neighbors.

    Returns:
        labels: Label array: 1 for normal samples and -1 for anomalies.
        info: Dictionary containing detection metadata and scores.
    """
    # Preprocess the features.
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # Adjust n_neighbors to the sample count.
    n_samples = len(features_scaled)
    if n_neighbors >= n_samples:
        n_neighbors = max(2, n_samples // 2)
        logger.warning(f"n_neighbors too large for {n_samples} samples, adjusted to {n_neighbors}")

    # Fit Local Outlier Factor.
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=n_neighbors,
        novelty=False
    )
    labels = lof.fit_predict(features_scaled)

    # Retrieve negative outlier factors; more negative values indicate stronger anomalies.
    scores = lof.negative_outlier_factor_

    # Count anomalous samples.
    n_anomalies = (labels == -1).sum()
    n_normal = (labels == 1).sum()
    anomaly_ratio = n_anomalies / len(labels)

    info = {
        'model': lof,
        'method': 'Local Outlier Factor',
        'contamination': contamination,
        'n_neighbors': n_neighbors,
        'n_anomalies': int(n_anomalies),
        'n_normal': int(n_normal),
        'anomaly_ratio': float(anomaly_ratio),
        'scores': scores,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"LOF completed: {n_anomalies} anomalies ({anomaly_ratio:.1%}), {n_normal} normal")

    return labels, info


def detect_one_class_svm(features_df: pd.DataFrame, nu: float = 0.1,
                         kernel: str = 'rbf', gamma: str = 'scale') -> Tuple[np.ndarray, Dict]:
    """
    Detect anomalies with a one-class support vector machine.

    A one-class SVM learns a boundary around the data and flags samples outside it.

    Args:
        features_df: DataFrame of cell features.
        nu: Upper bound on the fraction of training errors, in (0, 1].
        kernel: Kernel type: 'rbf', 'linear', 'poly', or 'sigmoid'.
        gamma: Kernel coefficient: 'scale', 'auto', or a float.

    Returns:
        labels: Label array: 1 for normal samples and -1 for anomalies.
        info: Dictionary containing detection metadata and scores.
    """
    # Preprocess the features.
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # Fit a one-class SVM.
    oc_svm = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma
    )
    labels = oc_svm.fit_predict(features_scaled)

    # Retrieve decision scores; negative values indicate anomalies.
    scores = oc_svm.decision_function(features_scaled)

    # Count anomalous samples.
    n_anomalies = (labels == -1).sum()
    n_normal = (labels == 1).sum()
    anomaly_ratio = n_anomalies / len(labels)

    info = {
        'model': oc_svm,
        'method': 'One-Class SVM',
        'nu': nu,
        'kernel': kernel,
        'gamma': gamma,
        'n_anomalies': int(n_anomalies),
        'n_normal': int(n_normal),
        'anomaly_ratio': float(anomaly_ratio),
        'scores': scores,
        'feature_cols': feature_cols,
        'scaler': scaler
    }

    logger.info(f"One-Class SVM completed: {n_anomalies} anomalies ({anomaly_ratio:.1%}), {n_normal} normal")

    return labels, info


def detect_elliptic_envelope(features_df: pd.DataFrame, contamination: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    Detect anomalies with an elliptic envelope.

    Fit a robust Gaussian covariance estimate and detect samples outside its envelope.
    This method is suited to approximately multivariate-normal data.

    Args:
        features_df: DataFrame of cell features.
        contamination: Expected anomaly fraction in the interval (0, 0.5].

    Returns:
        labels: Label array: 1 for normal samples and -1 for anomalies.
        info: Dictionary containing detection metadata and scores.
    """
    # Preprocess the features.
    features_scaled, feature_cols, scaler = preprocess_features(features_df)

    # Fit an elliptic envelope.
    try:
        elliptic = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        labels = elliptic.fit_predict(features_scaled)

        # Retrieve squared Mahalanobis distances; larger values indicate stronger anomalies.
        scores = elliptic.decision_function(features_scaled)

        # Count anomalous samples.
        n_anomalies = (labels == -1).sum()
        n_normal = (labels == 1).sum()
        anomaly_ratio = n_anomalies / len(labels)

        info = {
            'model': elliptic,
            'method': 'Elliptic Envelope',
            'contamination': contamination,
            'n_anomalies': int(n_anomalies),
            'n_normal': int(n_normal),
            'anomaly_ratio': float(anomaly_ratio),
            'scores': scores,
            'feature_cols': feature_cols,
            'scaler': scaler
        }

        logger.info(f"Elliptic Envelope completed: {n_anomalies} anomalies ({anomaly_ratio:.1%}), {n_normal} normal")

    except Exception as e:
        logger.error(f"Elliptic Envelope failed: {str(e)}")
        # On failure, return labels marking every sample as normal.
        labels = np.ones(len(features_scaled), dtype=int)
        info = {
            'model': None,
            'method': 'Elliptic Envelope',
            'contamination': contamination,
            'n_anomalies': 0,
            'n_normal': len(labels),
            'anomaly_ratio': 0.0,
            'scores': np.zeros(len(labels)),
            'feature_cols': feature_cols,
            'scaler': scaler,
            'error': str(e)
        }

    return labels, info


def get_anomaly_statistics(features_df: pd.DataFrame, labels: np.ndarray,
                           exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare feature statistics for normal and anomalous samples.

    Args:
        features_df: DataFrame of cell features.
        labels: Anomaly labels: 1 for normal samples and -1 for anomalies.
        exclude_cols: Column names to exclude.

    Returns:
        DataFrame of summary statistics.
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

    # Separate normal and anomalous samples.
    normal_mask = labels == 1
    anomaly_mask = labels == -1

    stats_list = []

    for col in feature_cols:
        normal_values = features_df.loc[normal_mask, col]
        anomaly_values = features_df.loc[anomaly_mask, col]

        stats = {
            'feature': col,
            'normal_mean': normal_values.mean() if len(normal_values) > 0 else 0,
            'normal_std': normal_values.std() if len(normal_values) > 0 else 0,
            'anomaly_mean': anomaly_values.mean() if len(anomaly_values) > 0 else 0,
            'anomaly_std': anomaly_values.std() if len(anomaly_values) > 0 else 0,
            'difference': abs(normal_values.mean() - anomaly_values.mean()) if len(normal_values) > 0 and len(anomaly_values) > 0 else 0
        }

        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.sort_values('difference', ascending=False).reset_index(drop=True)

    logger.info(f"Computed anomaly statistics for {len(feature_cols)} features")

    return stats_df


def _reduce_to_2d(features_scaled: np.ndarray) -> np.ndarray:
    """
    Project features to two dimensions with PCA for visualization.

    Args:
        features_scaled: Standardized feature array.

    Returns:
        Two-dimensional feature array.
    """
    if features_scaled.shape[1] <= 2:
        return features_scaled

    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features_scaled)

    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    return features_2d


def visualize_isolation_forest(features_scaled: np.ndarray, labels: np.ndarray,
                                scores: np.ndarray, info: Dict) -> plt.Figure:
    """
    Visualize isolation forest results and illustrate spatial partitioning.

    Isolation forests recursively partition the feature space with random splits.
    The figure illustrates this process and the distribution of anomaly scores.

    Args:
        features_scaled: Standardized feature array.
        labels: Anomaly labels: 1 for normal samples and -1 for anomalies.
        scores: Anomaly scores; lower values indicate stronger anomalies.
        info: Detection metadata dictionary.

    Returns:
        Matplotlib Figure instance.
    """
    # Reduce the features to two dimensions for visualization.
    features_2d = _reduce_to_2d(features_scaled)

    # Create a high-resolution figure at 300 DPI.
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), dpi=300)
    fig.suptitle('Isolation Forest: Space Partitioning Visualization', fontsize=20, fontweight='bold')

    normal_mask = labels == 1
    anomaly_mask = labels == -1

    # 1. Spatial partitioning: illustrate the isolation forest decision surface.
    ax1 = axes[0, 0]

    # Create a grid for the decision boundary.
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict anomaly scores at grid locations.
    model = info['model']
    # Grid points must be mapped to the original feature space.
    if features_scaled.shape[1] > 2:
        # A PCA inverse transform provides an approximation.
        pca = PCA(n_components=2, random_state=42)
        pca.fit(features_scaled)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        # For this illustration, fit a separate isolation forest in two dimensions.
        if features_scaled.shape[1] > 2:
            # Refit the model on the first two principal components for visualization.
            iso_forest_2d = IsolationForest(
                contamination=info['contamination'],
                random_state=42,
                n_estimators=100
            )
            iso_forest_2d.fit(features_2d)
            Z = iso_forest_2d.score_samples(grid_points)
        else:
            Z = model.score_samples(grid_points)
    else:
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.score_samples(grid_points)

    Z = Z.reshape(xx.shape)

    # Plot anomaly score contours to illustrate spatial partitioning.
    contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.7)
    ax1.contour(xx, yy, Z, levels=[scores[anomaly_mask].max()],
                colors='red', linewidths=4, linestyles='dashed', label='Decision Boundary')

    # Draw sample points with increased visibility.
    ax1.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                c='blue', alpha=0.7, s=60, label='Normal', edgecolors='k', linewidth=1.0)
    ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                c='red', alpha=0.9, s=120, label='Anomaly', edgecolors='k', linewidth=1.5, marker='^')

    ax1.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax1.set_title('Space Partitioning by Isolation Forest', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    plt.colorbar(contour, ax=ax1, label='Anomaly Score')
    ax1.grid(True, alpha=0.2)

    # 2. Score distribution: isolated anomalies tend to have shorter tree paths.
    ax2 = axes[0, 1]

    # Lower anomaly scores are associated with shorter isolation paths.
    # Use scores as a visual proxy for path length.
    path_lengths = -scores  # Lower scores indicate shorter isolation paths.

    ax2.hist(path_lengths[normal_mask], bins=30, alpha=0.6, color='blue',
             label=f'Normal (mean={path_lengths[normal_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.hist(path_lengths[anomaly_mask], bins=30, alpha=0.6, color='red',
             label=f'Anomaly (mean={path_lengths[anomaly_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.axvline(x=path_lengths[normal_mask].mean(), color='blue', linestyle='--', linewidth=3)
    ax2.axvline(x=path_lengths[anomaly_mask].mean(), color='red', linestyle='--', linewidth=3)

    ax2.set_xlabel('Path Length (shorter = more anomalous)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax2.set_title('Isolation Path Length Distribution', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add explanatory text with a readable font size.
    ax2.text(0.02, 0.98, 'Anomalies have shorter\nisolation paths',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 3. Tree partition illustration: show example random splits.
    ax3 = axes[1, 0]

    # Select representative partition lines for illustration.
    # Simulate the random partitioning process.
    np.random.seed(42)
    n_splits = 8  # Display eight partition lines.

    # Draw the sample distribution in the background.
    ax3.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                c='lightblue', alpha=0.4, s=50, edgecolors='none')
    ax3.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                c='lightcoral', alpha=0.6, s=90, marker='^', edgecolors='none')

    # Draw random partition lines with increased width.
    colors = plt.cm.Set3(np.linspace(0, 1, n_splits))
    for i in range(n_splits):
        # Randomly choose a horizontal or vertical split.
        if np.random.rand() > 0.5:
            # Vertical split
            split_pos = np.random.uniform(x_min, x_max)
            ax3.axvline(x=split_pos, color=colors[i], linestyle='-', linewidth=3, alpha=0.8)
        else:
            # Horizontal split
            split_pos = np.random.uniform(y_min, y_max)
            ax3.axhline(y=split_pos, color=colors[i], linestyle='-', linewidth=3, alpha=0.8)

    ax3.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax3.set_title('Random Space Partitioning (Sample Splits)', fontsize=15, fontweight='bold')
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.grid(True, alpha=0.2)

    # Add explanatory text.
    ax3.text(0.02, 0.98, 'Colored lines show\nrandom splits',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    Algorithm: Isolation Forest

    How it works:
    • Randomly selects features and split values
    • Recursively partitions the space
    • Anomalies are isolated faster (shorter paths)
    • Creates an ensemble of isolation trees

    Detection Results:
    • Total Samples: {len(labels)}
    • Normal: {info['n_normal']} ({info['n_normal']/len(labels)*100:.1f}%)
    • Anomaly: {info['n_anomalies']} ({info['anomaly_ratio']*100:.1f}%)

    Parameters:
    • Contamination: {info['contamination']}
    • Number of Trees: 100

    Anomaly Score Statistics:
    • Normal mean: {scores[normal_mask].mean():.4f}
    • Anomaly mean: {scores[anomaly_mask].mean():.4f}
    • Threshold: {scores[anomaly_mask].max():.4f}

    Interpretation:
    • Lower scores indicate anomalies
    • Anomalies require fewer splits to isolate
    • Red dashed line shows decision boundary
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    logger.info("Isolation Forest space partitioning visualization created")

    return fig


def visualize_lof(features_scaled: np.ndarray, labels: np.ndarray,
                  scores: np.ndarray, info: Dict) -> plt.Figure:
    """
    Visualize LOF results and illustrate local density relationships.

    LOF compares the local density of each sample with that of its neighbors.
    The figure shows relative density scores and nearest-neighbor relationships.

    Args:
        features_scaled: Standardized feature array.
        labels: Anomaly labels: 1 for normal samples and -1 for anomalies.
        scores: Negative outlier factors; more negative values indicate stronger anomalies.
        info: Detection metadata dictionary.

    Returns:
        Matplotlib Figure instance.
    """
    # Reduce the features to two dimensions for visualization.
    features_2d = _reduce_to_2d(features_scaled)

    # Create a high-resolution figure at 300 DPI.
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), dpi=300)
    fig.suptitle('Local Outlier Factor (LOF): Local Density Visualization', fontsize=20, fontweight='bold')

    normal_mask = labels == 1
    anomaly_mask = labels == -1

    # 1. Local density visualization: illustrate the LOF principle.
    ax1 = axes[0, 0]

    # Use negative outlier factors as a visual proxy for relative density.
    # Transform LOF scores into a density-like visualization metric.
    density_scores = -scores  # Invert the scores so normal samples have larger values.

    # Draw a density-colored scatter plot.
    scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=density_scores, cmap='RdYlGn_r', alpha=0.7, s=100,
                          edgecolors='k', linewidth=1.0)

    # Highlight anomalous points.
    ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                facecolors='none', edgecolors='red', linewidths=4, s=200,
                marker='o', label='Anomaly (Low Density)')

    ax1.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax1.set_title('Local Density Distribution', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Local Density (higher = denser)', fontsize=12)
    ax1.grid(True, alpha=0.2)

    # 2. LOF score distribution
    ax2 = axes[0, 1]

    ax2.hist(scores[normal_mask], bins=30, alpha=0.6, color='blue',
             label=f'Normal (mean={scores[normal_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.hist(scores[anomaly_mask], bins=30, alpha=0.6, color='red',
             label=f'Anomaly (mean={scores[anomaly_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.axvline(x=scores[normal_mask].mean(), color='blue', linestyle='--', linewidth=3)
    ax2.axvline(x=scores[anomaly_mask].mean(), color='red', linestyle='--', linewidth=3)

    ax2.set_xlabel('LOF Score (more negative = more anomalous)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax2.set_title('LOF Score Distribution', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add explanatory text.
    ax2.text(0.02, 0.98, 'Lower density regions\nhave more negative scores',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 3. Nearest-neighbor visualization: show local neighborhood relationships.
    ax3 = axes[1, 0]

    # Plot all samples.
    ax3.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                c='lightblue', alpha=0.5, s=60, edgecolors='k', linewidth=0.7)
    ax3.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                c='lightcoral', alpha=0.7, s=100, marker='^', edgecolors='k', linewidth=1.0)

    # Select representative anomalies and display their nearest neighbors.
    from sklearn.neighbors import NearestNeighbors
    n_neighbors = min(info['n_neighbors'], len(features_2d) - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(features_2d)

    # Show neighbor relationships for up to three anomalous samples.
    anomaly_indices = np.where(anomaly_mask)[0]
    if len(anomaly_indices) > 0:
        # Select the anomalies with the lowest LOF scores.
        top_anomalies = anomaly_indices[np.argsort(scores[anomaly_indices])[:min(3, len(anomaly_indices))]]

        colors_neighbors = ['red', 'orange', 'purple']
        for idx, anomaly_idx in enumerate(top_anomalies):
            # Find the nearest neighbors.
            distances, indices = nbrs.kneighbors([features_2d[anomaly_idx]])

            # Draw connecting lines.
            for neighbor_idx in indices[0][1:]:  # Skip the query sample itself.
                ax3.plot([features_2d[anomaly_idx, 0], features_2d[neighbor_idx, 0]],
                        [features_2d[anomaly_idx, 1], features_2d[neighbor_idx, 1]],
                        color=colors_neighbors[idx], alpha=0.5, linewidth=2.0)

            # Highlight anomalous samples.
            ax3.scatter(features_2d[anomaly_idx, 0], features_2d[anomaly_idx, 1],
                       c=colors_neighbors[idx], s=300, marker='*', edgecolors='black',
                       linewidth=2.5, zorder=10, label=f'Anomaly {idx+1}')

    ax3.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax3.set_title(f'k-Nearest Neighbors (k={n_neighbors})', fontsize=15, fontweight='bold')
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.2)

    # Add explanatory text.
    ax3.text(0.02, 0.98, 'Lines show k-nearest\nneighbors of anomalies',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    Algorithm: Local Outlier Factor (LOF)

    How it works:
    • Compares local density of each sample
    • Density based on k-nearest neighbors
    • Low density samples are anomalies
    • Considers local neighborhood structure

    Detection Results:
    • Total Samples: {len(labels)}
    • Normal: {info['n_normal']} ({info['n_normal']/len(labels)*100:.1f}%)
    • Anomaly: {info['n_anomalies']} ({info['anomaly_ratio']*100:.1f}%)

    Parameters:
    • Contamination: {info['contamination']}
    • Number of Neighbors: {info['n_neighbors']}

    LOF Score Statistics:
    • Normal mean: {scores[normal_mask].mean():.4f}
    • Anomaly mean: {scores[anomaly_mask].mean():.4f}
    • Threshold: {scores[anomaly_mask].min():.4f}

    Interpretation:
    • Scores close to -1.0 = normal density
    • More negative scores = lower local density
    • Anomalies have fewer nearby neighbors
    • Lines show k-nearest neighbor connections
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    logger.info("LOF local density visualization created")

    return fig


def visualize_one_class_svm(features_scaled: np.ndarray, labels: np.ndarray,
                             scores: np.ndarray, info: Dict) -> plt.Figure:
    """
    Visualize a one-class SVM decision boundary and kernel-based decision surface.

    A one-class SVM uses a kernel-defined feature space to estimate a decision boundary.
    The figure shows the boundary, support vectors, and decision scores.

    Args:
        features_scaled: Standardized feature array.
        labels: Anomaly labels: 1 for normal samples and -1 for anomalies.
        scores: Decision scores; negative values indicate anomalies.
        info: Detection metadata dictionary.

    Returns:
        Matplotlib Figure instance.
    """
    # Reduce the features to two dimensions for visualization.
    features_2d = _reduce_to_2d(features_scaled)

    # Create a high-resolution figure at 300 DPI.
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), dpi=300)
    fig.suptitle('One-Class SVM: Decision Boundary Visualization', fontsize=20, fontweight='bold')

    normal_mask = labels == 1
    anomaly_mask = labels == -1

    # 1. Decision boundary: illustrate the boundary learned by the SVM.
    ax1 = axes[0, 0]

    # Create a grid for the decision boundary.
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Fit a separate one-class SVM in two dimensions for visualization.
    from sklearn.svm import OneClassSVM
    svm_2d = OneClassSVM(
        nu=info['nu'],
        kernel=info['kernel'],
        gamma=info['gamma']
    )
    svm_2d.fit(features_2d)

    # Evaluate the decision function on the grid.
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_2d.decision_function(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot filled decision-function contours.
    contour = ax1.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 25),
                           cmap='RdYlBu', alpha=0.7)
    # Draw the zero-level decision boundary.
    ax1.contour(xx, yy, Z, levels=[0], colors='red', linewidths=4,
                linestyles='solid', label='Decision Boundary')
    # Draw margin contours.
    ax1.contour(xx, yy, Z, levels=[-0.5, 0.5], colors='orange',
                linewidths=3, linestyles='dashed', alpha=0.8)

    # Plot sample points.
    ax1.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                c='blue', alpha=0.7, s=60, label='Normal', edgecolors='k', linewidth=1.0)
    ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                c='red', alpha=0.9, s=120, label='Anomaly', edgecolors='k', linewidth=1.5, marker='^')

    # Mark support vectors.
    support_vectors_2d = features_2d[svm_2d.support_]
    ax1.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1],
                s=250, facecolors='none', edgecolors='green', linewidths=3.5,
                label='Support Vectors')

    ax1.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Decision Boundary ({info["kernel"]} kernel)', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    plt.colorbar(contour, ax=ax1, label='Decision Function Value')
    ax1.grid(True, alpha=0.2)

    # 2. Decision score distribution
    ax2 = axes[0, 1]

    ax2.hist(scores[normal_mask], bins=30, alpha=0.6, color='blue',
             label=f'Normal (mean={scores[normal_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.hist(scores[anomaly_mask], bins=30, alpha=0.6, color='red',
             label=f'Anomaly (mean={scores[anomaly_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=4, label='Decision Boundary (0)')
    ax2.axvline(x=scores[normal_mask].mean(), color='blue', linestyle=':', linewidth=3)
    ax2.axvline(x=scores[anomaly_mask].mean(), color='red', linestyle=':', linewidth=3)

    ax2.set_xlabel('Decision Function Value', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax2.set_title('Decision Function Distribution', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add explanatory text.
    ax2.text(0.02, 0.98, 'Positive = Normal\nNegative = Anomaly',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 3. Kernel-based decision surface visualization
    ax3 = axes[1, 0]

    # Encode decision scores with color in the scatter plot.
    scatter = ax3.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=scores, cmap='RdYlBu', alpha=0.8, s=100,
                          edgecolors='k', linewidth=1.0)

    # Draw the decision boundary.
    ax3.contour(xx, yy, Z, levels=[0], colors='red', linewidths=4, linestyles='solid')

    # Mark support vectors.
    ax3.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1],
                s=250, facecolors='none', edgecolors='green', linewidths=3.5)

    ax3.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax3.set_title('Kernel Function Effect', fontsize=15, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Distance to Boundary', fontsize=12)
    ax3.grid(True, alpha=0.2)

    # Add explanatory text.
    ax3.text(0.02, 0.98, f'Kernel: {info["kernel"]}\nSupport Vectors: {len(svm_2d.support_)}',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    Algorithm: One-Class SVM

    How it works:
    • Learns a boundary around normal data
    • Uses kernel trick to map to high-dimensional space
    • Maximizes margin from origin in feature space
    • Support vectors define the boundary

    Detection Results:
    • Total Samples: {len(labels)}
    • Normal: {info['n_normal']} ({info['n_normal']/len(labels)*100:.1f}%)
    • Anomaly: {info['n_anomalies']} ({info['anomaly_ratio']*100:.1f}%)
    • Support Vectors: {len(svm_2d.support_)} ({len(svm_2d.support_)/len(labels)*100:.1f}%)

    Parameters:
    • Nu: {info['nu']} (upper bound on anomaly fraction)
    • Kernel: {info['kernel']}
    • Gamma: {info['gamma']}

    Decision Function Statistics:
    • Normal mean: {scores[normal_mask].mean():.4f}
    • Anomaly mean: {scores[anomaly_mask].mean():.4f}
    • Boundary: 0.0

    Interpretation:
    • Positive values = inside boundary (normal)
    • Negative values = outside boundary (anomaly)
    • Green circles = support vectors
    • Red line = decision boundary
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    logger.info("One-Class SVM decision boundary visualization created")

    return fig


def visualize_elliptic_envelope(features_scaled: np.ndarray, labels: np.ndarray,
                                  scores: np.ndarray, info: Dict) -> plt.Figure:
    """
    Visualize elliptic envelope results and covariance boundaries.

    The elliptic envelope estimates a Gaussian boundary for anomaly detection.
    The figure shows ellipse boundaries and distance-based scores.

    Args:
        features_scaled: Standardized feature array.
        labels: Anomaly labels: 1 for normal samples and -1 for anomalies.
        scores: Squared Mahalanobis distances; larger values indicate stronger anomalies.
        info: Detection metadata dictionary.

    Returns:
        Matplotlib Figure instance.
    """
    # Reduce the features to two dimensions for visualization.
    features_2d = _reduce_to_2d(features_scaled)

    # Create a high-resolution figure at 300 DPI.
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), dpi=300)
    fig.suptitle('Elliptic Envelope: Gaussian Distribution Boundary', fontsize=20, fontweight='bold')

    normal_mask = labels == 1
    anomaly_mask = labels == -1

    # 1. Elliptic envelope boundary: illustrate the Gaussian model.
    ax1 = axes[0, 0]

    # Create a grid for the decision boundary.
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Fit a separate elliptic envelope in two dimensions for visualization.
    from sklearn.covariance import EllipticEnvelope
    elliptic_2d = EllipticEnvelope(
        contamination=info['contamination'],
        random_state=42
    )
    elliptic_2d.fit(features_2d)

    # Compute squared Mahalanobis distances at grid locations.
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = elliptic_2d.decision_function(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot distance contours to illustrate the elliptic envelope.
    contour = ax1.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 25),
                           cmap='RdYlBu', alpha=0.7)
    # Draw the zero-level decision boundary.
    ax1.contour(xx, yy, Z, levels=[0], colors='red', linewidths=4,
                linestyles='solid', label='Elliptic Boundary')
    # Draw additional contour levels.
    ax1.contour(xx, yy, Z, levels=[-2, -1, 1, 2], colors='orange',
                linewidths=2.5, linestyles='dashed', alpha=0.8)

    # Plot sample points.
    ax1.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1],
                c='blue', alpha=0.7, s=60, label='Normal', edgecolors='k', linewidth=1.0)
    ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1],
                c='red', alpha=0.9, s=120, label='Anomaly', edgecolors='k', linewidth=1.5, marker='^')

    # Mark the distribution center.
    center = elliptic_2d.location_
    ax1.scatter(center[0], center[1], c='green', s=400, marker='X',
                edgecolors='black', linewidths=2.5, label='Distribution Center', zorder=10)

    ax1.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax1.set_title('Elliptic Envelope Boundary', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    plt.colorbar(contour, ax=ax1, label='Mahalanobis Distance')
    ax1.grid(True, alpha=0.2)

    # 2. Mahalanobis distance distribution
    ax2 = axes[0, 1]

    ax2.hist(scores[normal_mask], bins=30, alpha=0.6, color='blue',
             label=f'Normal (mean={scores[normal_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.hist(scores[anomaly_mask], bins=30, alpha=0.6, color='red',
             label=f'Anomaly (mean={scores[anomaly_mask].mean():.3f})', edgecolor='black', linewidth=1.2)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=4, label='Boundary (0)')
    ax2.axvline(x=scores[normal_mask].mean(), color='blue', linestyle=':', linewidth=3)
    ax2.axvline(x=scores[anomaly_mask].mean(), color='red', linestyle=':', linewidth=3)

    ax2.set_xlabel('Mahalanobis Distance', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax2.set_title('Mahalanobis Distance Distribution', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add explanatory text.
    ax2.text(0.02, 0.98, 'Positive = Inside ellipse\nNegative = Outside ellipse',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 3. Covariance ellipse: illustrate the fitted Gaussian distribution.
    ax3 = axes[1, 0]

    # Plot a distance-colored scatter plot.
    scatter = ax3.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=scores, cmap='RdYlBu', alpha=0.8, s=100,
                          edgecolors='k', linewidth=1.0)

    # Draw the elliptic boundary.
    ax3.contour(xx, yy, Z, levels=[0], colors='red', linewidths=4, linestyles='solid')
    # Draw multiple ellipse contours.
    ax3.contour(xx, yy, Z, levels=[-2, -1, 1, 2], colors='orange',
                linewidths=2.5, linestyles='dashed', alpha=0.8)

    # Mark the distribution center.
    ax3.scatter(center[0], center[1], c='green', s=400, marker='X',
                edgecolors='black', linewidths=2.5, zorder=10)

    ax3.set_xlabel('Feature 1 (PC1)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Feature 2 (PC2)', fontsize=14, fontweight='bold')
    ax3.set_title('Covariance Ellipse (Gaussian Assumption)', fontsize=15, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Mahalanobis Distance', fontsize=12)
    ax3.grid(True, alpha=0.2)

    # Add explanatory text.
    ax3.text(0.02, 0.98, 'Dashed lines:\nConfidence intervals',
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = f"""
    Algorithm: Elliptic Envelope

    How it works:
    • Assumes data follows Gaussian distribution
    • Fits an ellipse around normal data
    • Uses Mahalanobis distance to measure outliers
    • Robust covariance estimation

    Detection Results:
    • Total Samples: {len(labels)}
    • Normal: {info['n_normal']} ({info['n_normal']/len(labels)*100:.1f}%)
    • Anomaly: {info['n_anomalies']} ({info['anomaly_ratio']*100:.1f}%)

    Parameters:
    • Contamination: {info['contamination']}

    Mahalanobis Distance Statistics:
    • Normal mean: {scores[normal_mask].mean():.4f}
    • Anomaly mean: {scores[anomaly_mask].mean():.4f}
    • Boundary: 0.0

    Interpretation:
    • Positive values = inside ellipse (normal)
    • Negative values = outside ellipse (anomaly)
    • Green X = distribution center
    • Red line = elliptic boundary
    • Orange dashed = confidence intervals
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    logger.info("Elliptic Envelope boundary visualization created")

    return fig
