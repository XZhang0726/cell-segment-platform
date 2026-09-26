"""
Virtual screening with trained supervised models.

Apply trained models to new feature tables for batch prediction and screening.
Includes confidence scores, result ranking, and candidate selection.

Features:
1. Load trained models for prediction.
2. Compute prediction confidence scores.
3. Rank and filter results.
4. Visualize screening results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import joblib
from loguru import logger

# Sklearn imports
from sklearn.metrics import r2_score

# ============================================================================
# CORE SCREENING FUNCTIONS
# ============================================================================

def screen_dataset(
    model_path: str,
    data_df: pd.DataFrame,
    confidence_method: str = 'probability',
    min_confidence: Optional[float] = None,
    return_probabilities: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Screen a feature dataset with a saved model.

    Args:
        model_path: Path to the saved model.
        data_df: Feature DataFrame to screen.
        confidence_method: Confidence method ('probability' or 'distance'); retained for API compatibility.
        min_confidence: Minimum confidence score, or None to disable filtering.
        return_probabilities: Include per-class probabilities for classification.

    Returns:
        results_df: DataFrame containing predictions and confidence scores.
        info: Dictionary of screening summary statistics.
    """
    logger.info(f"Loading model from {model_path}")

    # Load the model.
    try:
        model_package = joblib.load(model_path)
        model = model_package['model']
        task_type = model_package.get('task_type', 'classification')
        feature_names = model_package.get('feature_names', [])
        scaler = model_package.get('scaler')
        encoder = model_package.get('encoder')
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    logger.info(f"Model loaded: task_type={task_type}, n_features={len(feature_names)}")

    # Validate feature columns.
    missing_features = [f for f in feature_names if f not in data_df.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        raise ValueError(f"Missing features in data: {missing_features}")

    # Extract features.
    X = data_df[feature_names]

    # Apply the scaler.
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    # Generate predictions.
    logger.info(f"Screening {len(data_df)} samples...")
    predictions = model.predict(X_scaled)

    # Create the results DataFrame.
    results_df = data_df.copy()
    results_df['prediction'] = predictions

    # Compute confidence scores.
    if task_type == 'classification':
        if hasattr(model, 'predict_proba') and return_probabilities:
            probabilities = model.predict_proba(X_scaled)

            # Add probabilities for each class.
            for i in range(probabilities.shape[1]):
                results_df[f'probability_class_{i}'] = probabilities[:, i]

            # Use the maximum class probability as confidence.
            if confidence_method == 'probability':
                confidence = np.max(probabilities, axis=1)
            else:
                confidence = np.max(probabilities, axis=1)
        else:
            confidence = np.ones(len(predictions))
    else:
        # Regression uses a constant confidence of 1; this is not calibrated uncertainty.
        confidence = np.ones(len(predictions))

    results_df['confidence'] = confidence

    # Filter samples below the confidence threshold.
    if min_confidence is not None:
        n_before = len(results_df)
        results_df = results_df[results_df['confidence'] >= min_confidence]
        n_after = len(results_df)
        logger.info(f"Filtered by confidence: {n_before} -> {n_after} samples")

    # Summary statistics
    info = {
        'n_samples': len(data_df),
        'n_screened': len(results_df),
        'task_type': task_type,
        'model_path': model_path,
        'confidence_method': confidence_method,
        'min_confidence': min_confidence,
        'prediction_stats': {
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'min': float(predictions.min()),
            'max': float(predictions.max())
        }
    }

    if task_type == 'classification':
        unique, counts = np.unique(predictions, return_counts=True)
        info['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))

    logger.info(f"Screening completed: {len(results_df)} samples")

    return results_df, info


def batch_screen_files(
    model_path: str,
    data_files: List[str],
    output_dir: str,
    confidence_threshold: float = 0.7,
    merge_results: bool = True
) -> Optional[pd.DataFrame]:
    """
    Screen multiple CSV files with a saved model.

    Args:
        model_path: Path to the saved model.
        data_files: List of input data file paths.
        output_dir: Directory for output files.
        confidence_threshold: Minimum confidence score.
        merge_results: Whether to combine results from all files.

    Returns:
        merged_results_df: Combined results DataFrame when merge_results is True.
    """
    logger.info(f"Batch screening {len(data_files)} files")

    # Create the output directory.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, data_file in enumerate(data_files):
        try:
            logger.info(f"Processing file {i+1}/{len(data_files)}: {data_file}")

            # Read the data.
            data_df = pd.read_csv(data_file)

            # Run screening.
            results_df, info = screen_dataset(
                model_path, data_df,
                min_confidence=confidence_threshold
            )

            # Save the results.
            output_file = Path(output_dir) / f"screened_{Path(data_file).stem}.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")

            if merge_results:
                all_results.append(results_df)

        except Exception as e:
            logger.error(f"Failed to process {data_file}: {str(e)}")
            continue

    # Merge results.
    if merge_results and all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        merged_file = Path(output_dir) / "merged_results.csv"
        merged_df.to_csv(merged_file, index=False)
        logger.info(f"Merged results saved to {merged_file}")
        return merged_df

    return None


# ============================================================================
# CONFIDENCE SCORING FUNCTIONS
# ============================================================================

def compute_confidence_probability(
    model: Any,
    X: np.ndarray
) -> np.ndarray:
    """
    Compute confidence as the maximum predicted class probability.

    Args:
        model: Trained model.
        X: Feature array.

    Returns:
        confidence_scores: Array of confidence scores.
    """
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        # Use the maximum class probability as confidence.
        confidence = np.max(probabilities, axis=1)
    else:
        # Return a constant score of 1 when probability prediction is unavailable.
        confidence = np.ones(len(X))

    return confidence


def compute_prediction_intervals(
    model: Any,
    X: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate heuristic prediction intervals for regression.

    Uses a heuristic based on prediction variability; intervals are not calibrated.

    Args:
        model: Trained model.
        X: Feature array.
        confidence_level: Requested confidence level.

    Returns:
        lower_bounds: Array of lower bounds.
        upper_bounds: Array of upper bounds.
    """
    predictions = model.predict(X)

    # Approximate intervals under a normal-error assumption.
    # Estimate the standard deviation heuristically; training residuals are unavailable.
    # More reliable intervals require residual information from model training.
    std_estimate = predictions.std() * 0.1  # Heuristic estimate

    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    lower_bounds = predictions - z_score * std_estimate
    upper_bounds = predictions + z_score * std_estimate

    return lower_bounds, upper_bounds


# ============================================================================
# RESULT RANKING FUNCTIONS
# ============================================================================

def rank_by_prediction(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    confidence_col: str = 'confidence',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Sort results by prediction and then confidence.

    Args:
        results_df: Results DataFrame.
        prediction_col: Prediction column name.
        confidence_col: Confidence column name.
        ascending: Whether to sort in ascending order.

    Returns:
        ranked_df: Sorted DataFrame.
    """
    # Sort by prediction, then confidence.
    ranked_df = results_df.sort_values(
        by=[prediction_col, confidence_col],
        ascending=[ascending, False]
    ).reset_index(drop=True)

    return ranked_df


def filter_by_confidence(
    results_df: pd.DataFrame,
    confidence_col: str = 'confidence',
    min_confidence: float = 0.7
) -> pd.DataFrame:
    """
    Filter results by a minimum confidence score.

    Args:
        results_df: Results DataFrame.
        confidence_col: Confidence column name.
        min_confidence: Minimum confidence score.

    Returns:
        filtered_df: Filtered DataFrame.
    """
    filtered_df = results_df[results_df[confidence_col] >= min_confidence].copy()
    logger.info(f"Filtered by confidence >= {min_confidence}: {len(results_df)} -> {len(filtered_df)} samples")
    return filtered_df


def filter_by_prediction_range(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Filter results by a prediction interval.

    Args:
        results_df: Results DataFrame.
        prediction_col: Prediction column name.
        min_value: Minimum prediction value.
        max_value: Maximum prediction value.

    Returns:
        filtered_df: Filtered DataFrame.
    """
    filtered_df = results_df.copy()

    if min_value is not None:
        filtered_df = filtered_df[filtered_df[prediction_col] >= min_value]

    if max_value is not None:
        filtered_df = filtered_df[filtered_df[prediction_col] <= max_value]

    logger.info(f"Filtered by prediction range [{min_value}, {max_value}]: {len(results_df)} -> {len(filtered_df)} samples")

    return filtered_df


def select_top_candidates(
    results_df: pd.DataFrame,
    n_candidates: int = 100,
    criteria: str = 'prediction',
    confidence_threshold: float = 0.5,
    prediction_col: str = 'prediction',
    confidence_col: str = 'confidence'
) -> pd.DataFrame:
    """
    Select the highest-ranked candidates.

    Args:
        results_df: Results DataFrame.
        n_candidates: Number of candidates to return.
        criteria: 'prediction', 'confidence', 'combined'
        confidence_threshold: Minimum confidence score.
        prediction_col: Prediction column name.
        confidence_col: Confidence column name.

    Returns:
        top_candidates_df: DataFrame containing the selected candidates.
    """
    # Apply the confidence filter first.
    filtered_df = results_df[results_df[confidence_col] >= confidence_threshold].copy()

    if len(filtered_df) == 0:
        logger.warning(f"No samples meet confidence threshold {confidence_threshold}")
        return pd.DataFrame()

    # Sort by the selected criterion.
    if criteria == 'prediction':
        sorted_df = filtered_df.sort_values(prediction_col, ascending=False)
    elif criteria == 'confidence':
        sorted_df = filtered_df.sort_values(confidence_col, ascending=False)
    elif criteria == 'combined':
        # Combined score: prediction multiplied by confidence.
        filtered_df['combined_score'] = filtered_df[prediction_col] * filtered_df[confidence_col]
        sorted_df = filtered_df.sort_values('combined_score', ascending=False)
    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    # Select the top N candidates.
    top_candidates = sorted_df.head(n_candidates)

    logger.info(f"Selected top {len(top_candidates)} candidates (criteria={criteria})")

    return top_candidates


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_prediction_distribution(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    task_type: str = 'classification'
) -> plt.Figure:
    """
    Plot the prediction distribution.

    Args:
        results_df: Results DataFrame.
        prediction_col: Prediction column name.
        task_type: 'classification' or 'regression'

    Returns:
        fig: Matplotlib Figure instance.
    """
    predictions = results_df[prediction_col]

    # Adjust the figure width to the number of categories.
    unique_values = np.unique(predictions)
    n_categories = len(unique_values)

    if task_type == 'classification' or n_categories <= 10:
        # Scale categorical plots to the category count.
        fig_width = max(4, min(10, n_categories * 2))
    else:
        fig_width = 10

    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    if task_type == 'classification' or n_categories <= 10:
        # Classification: bar chart
        unique, counts = np.unique(predictions, return_counts=True)
        x_pos = np.arange(len(unique))
        bar_width = 0.6
        ax.bar(x_pos, counts, width=bar_width, color='steelblue', edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(unique)
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        # Regression: histogram
        ax.hist(predictions, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Predicted Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_confidence_distribution(
    results_df: pd.DataFrame,
    confidence_col: str = 'confidence'
) -> plt.Figure:
    """
    Plot the confidence score distribution.

    Args:
        results_df: Results DataFrame.
        confidence_col: Confidence column name.

    Returns:
        fig: Matplotlib Figure instance.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    confidence = results_df[confidence_col]

    ax.hist(confidence, bins=30, color='green', edgecolor='black', alpha=0.7)
    ax.axvline(confidence.mean(), color='r', linestyle='--', lw=2,
               label=f'Mean = {confidence.mean():.3f}')
    ax.axvline(confidence.median(), color='orange', linestyle='--', lw=2,
               label=f'Median = {confidence.median():.3f}')

    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_prediction_and_confidence(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    confidence_col: str = 'confidence',
    task_type: str = 'classification'
) -> plt.Figure:
    """
    Plot prediction and confidence distributions in a 1 x 2 layout.

    Args:
        results_df: Results DataFrame.
        prediction_col: Prediction column name.
        confidence_col: Confidence column name.
        task_type: 'classification' or 'regression'

    Returns:
        fig: Matplotlib Figure instance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    predictions = results_df[prediction_col]
    confidence = results_df[confidence_col]

    # Left panel: prediction distribution
    unique_values = np.unique(predictions)
    n_categories = len(unique_values)

    if task_type == 'classification' or n_categories <= 10:
        unique, counts = np.unique(predictions, return_counts=True)
        x_pos = np.arange(len(unique))
        bar_width = 0.6
        ax1.bar(x_pos, counts, width=bar_width, color='steelblue', edgecolor='black')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(unique)
        ax1.set_xlabel('Predicted Class', fontsize=11)
    else:
        ax1.hist(predictions, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Predicted Value', fontsize=11)

    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Right panel: confidence distribution
    ax2.hist(confidence, bins=30, color='green', edgecolor='black', alpha=0.7)
    ax2.axvline(confidence.mean(), color='r', linestyle='--', lw=2,
                label=f'Mean = {confidence.mean():.3f}')
    ax2.axvline(confidence.median(), color='orange', linestyle='--', lw=2,
                label=f'Median = {confidence.median():.3f}')
    ax2.set_xlabel('Confidence Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_top_candidates(
    results_df: pd.DataFrame,
    top_n: int = 20,
    prediction_col: str = 'prediction',
    confidence_col: str = 'confidence'
) -> plt.Figure:
    """
    Visualize the top-ranked candidates.

    Args:
        results_df: Results DataFrame.
        top_n: Number of top candidates to display.
        prediction_col: Prediction column name.
        confidence_col: Confidence column name.

    Returns:
        fig: Matplotlib Figure instance.
    """
    # Select the top N candidates.
    top_df = results_df.head(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Prediction bar chart
    x_pos = np.arange(len(top_df))
    ax1.bar(x_pos, top_df[prediction_col], color='steelblue', edgecolor='black')
    ax1.set_xlabel('Candidate Rank', fontsize=12)
    ax1.set_ylabel('Predicted Value', fontsize=12)
    ax1.set_title(f'Top {top_n} Candidates - Predictions', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Confidence bar chart
    colors = ['green' if c >= 0.7 else 'orange' if c >= 0.5 else 'red'
              for c in top_df[confidence_col]]
    ax2.bar(x_pos, top_df[confidence_col], color=colors, edgecolor='black')
    ax2.axhline(y=0.7, color='green', linestyle='--', lw=2, alpha=0.5, label='High Confidence')
    ax2.axhline(y=0.5, color='orange', linestyle='--', lw=2, alpha=0.5, label='Medium Confidence')
    ax2.set_xlabel('Candidate Rank', fontsize=12)
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_title(f'Top {top_n} Candidates - Confidence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_confidence_intervals(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    lower_col: str = 'lower_bound',
    upper_col: str = 'upper_bound',
    top_n: int = 50
) -> plt.Figure:
    """
    Plot regression predictions and prediction intervals.

    Args:
        results_df: Results DataFrame.
        prediction_col: Prediction column name.
        lower_col: Lower-bound column name.
        upper_col: Upper-bound column name.
        top_n: Number of samples to display.

    Returns:
        fig: Matplotlib Figure instance.
    """
    # Select the top N candidates.
    top_df = results_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    x_pos = np.arange(len(top_df))

    # Plot predictions.
    ax.plot(x_pos, top_df[prediction_col], 'o-', color='blue', linewidth=2,
            markersize=6, label='Prediction')

    # Plot prediction intervals.
    if lower_col in top_df.columns and upper_col in top_df.columns:
        ax.fill_between(x_pos, top_df[lower_col], top_df[upper_col],
                        alpha=0.3, color='blue', label='95% Confidence Interval')

    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    ax.set_title(f'Prediction with Confidence Intervals (Top {top_n})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_prediction_vs_confidence(
    results_df: pd.DataFrame,
    prediction_col: str = 'prediction',
    confidence_col: str = 'confidence'
) -> plt.Figure:
    """
    Plot predictions against confidence scores.

    Args:
        results_df: Results DataFrame.
        prediction_col: Prediction column name.
        confidence_col: Confidence column name.

    Returns:
        fig: Matplotlib Figure instance.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Color scatter points by confidence.
    scatter = ax.scatter(results_df[prediction_col], results_df[confidence_col],
                        c=results_df[confidence_col], cmap='RdYlGn',
                        s=50, alpha=0.6, edgecolors='k', linewidths=0.5)

    # Add a colorbar.
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidence Score', fontsize=12)

    # Add a confidence threshold line.
    ax.axhline(y=0.7, color='green', linestyle='--', lw=2, alpha=0.5, label='High Confidence')
    ax.axhline(y=0.5, color='orange', linestyle='--', lw=2, alpha=0.5, label='Medium Confidence')

    ax.set_xlabel('Predicted Value', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('Prediction vs Confidence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

