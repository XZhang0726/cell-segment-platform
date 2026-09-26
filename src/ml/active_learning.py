"""
Active learning utilities.

Provides active learning strategies and Bayesian optimization for sample selection.
Includes uncertainty sampling, query by committee, and Bayesian optimization workflows.

Main capabilities:
1. Uncertainty sampling: least confidence, margin, and entropy.
2. Query by committee.
3. Bayesian optimization loop.
4. Active learning workflow.
5. Uncertainty visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from loguru import logger

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Scikit-optimize imports for Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

# Scipy imports
from scipy.stats import entropy

# ============================================================================
# UNCERTAINTY SAMPLING STRATEGIES
# ============================================================================

def uncertainty_sampling(
    model: Any,
    X_pool: np.ndarray,
    n_samples: int = 10,
    method: str = 'least_confident'
) -> np.ndarray:
    """
    Select samples using predictive uncertainty.

    Choose samples for annotation where the model is most uncertain.

    Args:
        model: Trained classifier supporting predict_proba.
        X_pool: Pool of unlabeled samples.
        n_samples: Number of samples to select.
        method: Uncertainty measure.
            - 'least_confident': One minus the largest predicted probability.
            - 'margin': Difference between the two largest predicted probabilities.
            - 'entropy': Predictive information entropy.

    Returns:
        selected_indices: Array of selected sample indices.
    """
    if not hasattr(model, 'predict_proba'):
        logger.error("Model does not support predict_proba, cannot use uncertainty sampling")
        raise ValueError("Model must support predict_proba for uncertainty sampling")

    # Get predicted probabilities.
    probabilities = model.predict_proba(X_pool)

    # Compute uncertainty scores.
    if method == 'least_confident':
        # Least confidence: 1 - max(p).
        uncertainty_scores = 1 - np.max(probabilities, axis=1)

    elif method == 'margin':
        # Margin: largest minus second-largest probability; smaller margins indicate greater uncertainty.
        if probabilities.shape[1] < 2:
            logger.warning("Only one class, using least_confident method instead")
            uncertainty_scores = 1 - np.max(probabilities, axis=1)
        else:
            # Sort each row to extract the two largest probabilities.
            sorted_probs = np.sort(probabilities, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            uncertainty_scores = -margin  # Negate the margin so smaller margins receive higher scores.

    elif method == 'entropy':
        # Higher predictive entropy indicates greater uncertainty.
        uncertainty_scores = entropy(probabilities.T)

    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

    # Select samples with the highest uncertainty.
    selected_indices = np.argsort(uncertainty_scores)[-n_samples:][::-1]

    logger.info(f"Selected {len(selected_indices)} samples using {method} uncertainty sampling")
    logger.info(f"Uncertainty scores range: [{uncertainty_scores.min():.4f}, {uncertainty_scores.max():.4f}]")

    return selected_indices


def query_by_committee(
    models: List[Any],
    X_pool: np.ndarray,
    n_samples: int = 10,
    disagreement: str = 'vote_entropy'
) -> np.ndarray:
    """
    Select samples using query by committee.

    Use predictions from a committee of trained models to select samples with the greatest disagreement.

    Args:
        models: List of trained committee members.
        X_pool: Pool of unlabeled samples.
        n_samples: Number of samples to select.
        disagreement: Disagreement measure.
            - 'vote_entropy': Entropy of class votes for classification.
            - 'variance': Variance of predictions for regression.

    Returns:
        selected_indices: Array of selected sample indices.
    """
    if len(models) < 2:
        logger.error("Need at least 2 models for query by committee")
        raise ValueError("Need at least 2 models for committee")

    # Check whether this is a classification task.
    is_classification = hasattr(models[0], 'predict_proba')

    if disagreement == 'vote_entropy' and is_classification:
        # Collect predictions from all models.
        all_predictions = np.array([model.predict(X_pool) for model in models])

        # Compute vote entropy.
        disagreement_scores = []
        for i in range(X_pool.shape[0]):
            votes = all_predictions[:, i]
            # Compute the fraction of votes for each class.
            unique, counts = np.unique(votes, return_counts=True)
            vote_probs = counts / len(models)
            # Compute entropy.
            vote_entropy = entropy(vote_probs)
            disagreement_scores.append(vote_entropy)

        disagreement_scores = np.array(disagreement_scores)

    elif disagreement == 'variance':
        # Collect predictions from all models.
        all_predictions = np.array([model.predict(X_pool) for model in models])

        # Compute prediction variance.
        disagreement_scores = np.var(all_predictions, axis=0)

    else:
        raise ValueError(f"Unknown disagreement method: {disagreement}")

    # Select samples with the greatest disagreement.
    selected_indices = np.argsort(disagreement_scores)[-n_samples:][::-1]

    logger.info(f"Selected {len(selected_indices)} samples using query by committee ({disagreement})")
    logger.info(f"Disagreement scores range: [{disagreement_scores.min():.4f}, {disagreement_scores.max():.4f}]")

    return selected_indices


def expected_improvement_sampling(
    model: Any,
    X_pool: np.ndarray,
    y_pool_estimated: np.ndarray,
    n_samples: int = 10
) -> np.ndarray:
    """
    Select samples using a heuristic proxy for expected improvement.

    Rank samples by uncertainty or prediction discrepancy as a proxy for potential improvement.

    Args:
        model: Trained model.
        X_pool: Pool of unlabeled samples.
        y_pool_estimated: Estimated pool targets used by the regression discrepancy heuristic.
        n_samples: Number of samples to select.

    Returns:
        selected_indices: Array of selected sample indices.
    """
    # Get predictions.
    predictions = model.predict(X_pool)

    # Estimate improvement using uncertainty or discrepancies from estimated targets.
    if hasattr(model, 'predict_proba'):
        # For classification, use uncertainty as a proxy for expected improvement.
        probabilities = model.predict_proba(X_pool)
        uncertainty = 1 - np.max(probabilities, axis=1)
        expected_improvement = uncertainty
    else:
        # For regression, use the absolute difference between predictions and estimated targets.
        expected_improvement = np.abs(predictions - y_pool_estimated)

    # Select samples with the highest improvement proxy scores.
    selected_indices = np.argsort(expected_improvement)[-n_samples:][::-1]

    logger.info(f"Selected {len(selected_indices)} samples using expected improvement sampling")
    logger.info(f"Expected improvement range: [{expected_improvement.min():.4f}, {expected_improvement.max():.4f}]")

    return selected_indices


# ============================================================================
# BAYESIAN OPTIMIZATION FUNCTIONS
# ============================================================================

def fit_gaussian_process(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: Optional[Any] = None
) -> Any:
    """
    Fit a Gaussian process for uncertainty estimation.

    Args:
        X_train: Training features.
        y_train: Training targets.
        kernel: Covariance kernel; None uses a constant kernel multiplied by an RBF kernel.

    Returns:
        gp_model: Fitted Gaussian process model.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    if kernel is None:
        # Default kernel: constant kernel * RBF kernel.
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,
        normalize_y=True
    )

    gp_model.fit(X_train, y_train)

    logger.info(f"Gaussian Process fitted with kernel: {gp_model.kernel_}")

    return gp_model


def compute_acquisition_function(
    model: Any,
    X_pool: np.ndarray,
    acquisition: str = 'ei',
    xi: float = 0.01,
    kappa: float = 1.96
) -> np.ndarray:
    """
    Compute acquisition function values.

    Args:
        model: Gaussian process or another model supporting predictive uncertainty.
        X_pool: Candidate sample pool.
        acquisition: Acquisition function type.
            - 'ei': Expected improvement.
            - 'ucb': Upper confidence bound.
            - 'pi': Probability of improvement.
        xi: Exploration parameter for EI and PI.
        kappa: Exploration parameter for UCB.

    Returns:
        acquisition_values: Array of acquisition function values.
    """
    from scipy.stats import norm

    # Get predictive means and standard deviations.
    if hasattr(model, 'predict') and hasattr(model, 'predict'):
        # Gaussian process prediction.
        try:
            mu, sigma = model.predict(X_pool, return_std=True)
        except:
            # If return_std is unsupported, use point predictions with zero standard deviation.
            mu = model.predict(X_pool)
            sigma = np.zeros_like(mu)
    else:
        raise ValueError("Model must support predict with return_std=True")

    # Avoid division by zero.
    sigma = np.maximum(sigma, 1e-9)

    if acquisition == 'ei':
        # Expected Improvement
        # Find the current best value.
        if hasattr(model, 'y_train_'):
            f_best = np.max(model.y_train_)
        else:
            f_best = np.max(mu)

        # Compute improvement.
        improvement = mu - f_best - xi
        Z = improvement / sigma

        # Compute expected improvement.
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        acquisition_values = ei

    elif acquisition == 'ucb':
        # Upper Confidence Bound
        acquisition_values = mu + kappa * sigma

    elif acquisition == 'pi':
        # Probability of Improvement
        if hasattr(model, 'y_train_'):
            f_best = np.max(model.y_train_)
        else:
            f_best = np.max(mu)

        improvement = mu - f_best - xi
        Z = improvement / sigma
        acquisition_values = norm.cdf(Z)

    else:
        raise ValueError(f"Unknown acquisition function: {acquisition}")

    logger.info(f"Computed {acquisition} acquisition function: range [{acquisition_values.min():.4f}, {acquisition_values.max():.4f}]")

    return acquisition_values


def bayesian_optimization_loop(
    objective_function: Optional[Callable] = None,
    X_train_initial: np.ndarray = None,
    y_train_initial: np.ndarray = None,
    X_pool: np.ndarray = None,
    n_iterations: int = 10,
    acquisition: str = 'ei',
    model_type: str = 'gp',
    samples_per_iteration: int = 1,
    random_state: int = 42
) -> Dict:
    """
    Run the Bayesian optimization loop.

    Args:
        objective_function: Objective callable; None logs a warning and uses zero placeholder targets.
        X_train_initial: Initial training features.
        y_train_initial: Initial training targets.
        X_pool: Candidate sample pool.
        n_iterations: Number of optimization iterations.
        acquisition: Acquisition function ('ei', 'ucb', 'pi').
        model_type: Surrogate model type ('gp', 'rf', 'gbrt').
        samples_per_iteration: Number of samples selected per iteration.
        random_state: Random seed.

    Returns:
        results: Dictionary containing the optimization history.
            - 'selected_samples': Selected sample indices at each iteration.
            - 'acquisition_values': Acquisition values at each iteration.
            - 'best_values': Best value at each iteration.
            - 'model_history': Fitted model history.
            - 'X_train_history': Training feature history.
            - 'y_train_history': Training target history.
    """
    np.random.seed(random_state)

    # Initialize state.
    X_train = X_train_initial.copy()
    y_train = y_train_initial.copy()
    X_pool_remaining = X_pool.copy()

    # Record history.
    selected_samples_history = []
    acquisition_values_history = []
    best_values_history = []
    model_history = []
    X_train_history = [X_train.copy()]
    y_train_history = [y_train.copy()]

    logger.info(f"Starting Bayesian optimization: {n_iterations} iterations, acquisition={acquisition}")

    for iteration in range(n_iterations):
        logger.info(f"Iteration {iteration + 1}/{n_iterations}")

        # Train the surrogate model.
        if model_type == 'gp':
            model = fit_gaussian_process(X_train, y_train)
        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            model.fit(X_train, y_train)
        elif model_type == 'gbrt':
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Compute acquisition values.
        if model_type == 'gp':
            acq_values = compute_acquisition_function(model, X_pool_remaining, acquisition=acquisition)
        else:
            # For non-GP models, rank candidates directly by predicted values.
            predictions = model.predict(X_pool_remaining)
            acq_values = predictions  # Point-prediction fallback.

        # Select samples.
        selected_indices = np.argsort(acq_values)[-samples_per_iteration:][::-1]
        selected_samples_history.append(selected_indices)
        acquisition_values_history.append(acq_values)

        # Prepare selected samples for objective evaluation.
        X_selected = X_pool_remaining[selected_indices]

        if objective_function is not None:
            # Evaluate the objective function.
            y_selected = np.array([objective_function(x) for x in X_selected])
        else:
            # Without an objective callable, use zero placeholders and warn the caller.
            logger.warning("No objective function provided, cannot evaluate selected samples")
            y_selected = np.zeros(len(selected_indices))

        # Update the training set.
        X_train = np.vstack([X_train, X_selected])
        y_train = np.concatenate([y_train, y_selected])

        # Remove selected samples from the pool.
        mask = np.ones(len(X_pool_remaining), dtype=bool)
        mask[selected_indices] = False
        X_pool_remaining = X_pool_remaining[mask]

        # Record the best value.
        best_value = np.max(y_train)
        best_values_history.append(best_value)

        # Record history.
        model_history.append(model)
        X_train_history.append(X_train.copy())
        y_train_history.append(y_train.copy())

        logger.info(f"Selected {len(selected_indices)} samples, best value so far: {best_value:.4f}")

        # Stop early if the pool is exhausted.
        if len(X_pool_remaining) == 0:
            logger.info("Pool exhausted, stopping optimization")
            break

    results = {
        'selected_samples': selected_samples_history,
        'acquisition_values': acquisition_values_history,
        'best_values': best_values_history,
        'model_history': model_history,
        'X_train_history': X_train_history,
        'y_train_history': y_train_history,
        'final_model': model_history[-1] if model_history else None,
        'n_iterations': len(best_values_history)
    }

    logger.info(f"Bayesian optimization completed: {len(best_values_history)} iterations")

    return results


# ============================================================================
# ACTIVE LEARNING WORKFLOW
# ============================================================================

def active_learning_workflow(
    X_train_initial: np.ndarray,
    y_train_initial: np.ndarray,
    X_pool: np.ndarray,
    y_pool_true: np.ndarray,
    model_name: str = 'random_forest',
    task_type: str = 'classification',
    strategy: str = 'uncertainty',
    n_iterations: int = 10,
    samples_per_iteration: int = 10,
    random_state: int = 42
) -> Dict:
    """
    Run the complete active learning workflow.

    Args:
        X_train_initial: Initial training features.
        y_train_initial: Initial training targets.
        X_pool: Pool of unlabeled samples.
        y_pool_true: True pool labels used to simulate annotation.
        model_name: Model name ('random_forest', 'svm', 'logistic', etc.).
        task_type: Task type ('classification', 'regression').
        strategy: Sampling strategy ('uncertainty', 'qbc', 'random').
        n_iterations: Number of iterations.
        samples_per_iteration: Number of samples selected per iteration.
        random_state: Random seed.

    Returns:
        results: Dictionary containing the complete workflow history.
            - 'iteration_metrics': Performance metrics at each iteration.
            - 'selected_indices': Selected sample indices at each iteration.
            - 'final_model': Final trained model.
            - 'training_history': Complete training history.
            - 'X_train_history': Training feature history.
            - 'y_train_history': Training target history.
    """
    np.random.seed(random_state)

    # Initialize state.
    X_train = X_train_initial.copy()
    y_train = y_train_initial.copy()
    X_pool_remaining = X_pool.copy()
    y_pool_remaining = y_pool_true.copy()

    # Record history.
    iteration_metrics = []
    selected_indices_history = []
    X_train_history = [X_train.copy()]
    y_train_history = [y_train.copy()]

    # Select the model.
    if task_type == 'classification':
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_name == 'svm':
            from sklearn.svm import SVC
            base_model = SVC(probability=True, random_state=random_state)
        elif model_name == 'logistic':
            from sklearn.linear_model import LogisticRegression
            base_model = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            base_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif model_name == 'svm':
            from sklearn.svm import SVR
            base_model = SVR()
        else:
            from sklearn.ensemble import RandomForestRegressor
            base_model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    logger.info(f"Starting active learning: {n_iterations} iterations, strategy={strategy}")

    for iteration in range(n_iterations):
        logger.info(f"Iteration {iteration + 1}/{n_iterations}")

        # Train the model.
        model = base_model.__class__(**base_model.get_params())
        model.fit(X_train, y_train)

        # Evaluate the current model.
        if task_type == 'classification':
            train_score = accuracy_score(y_train, model.predict(X_train))
            # Evaluate on the full pool, including labeled and unlabeled samples.
            all_X = np.vstack([X_train, X_pool_remaining])
            all_y = np.concatenate([y_train, y_pool_remaining])
            test_score = accuracy_score(all_y, model.predict(all_X))
            metric_name = 'accuracy'
        else:
            train_score = r2_score(y_train, model.predict(X_train))
            all_X = np.vstack([X_train, X_pool_remaining])
            all_y = np.concatenate([y_train, y_pool_remaining])
            test_score = r2_score(all_y, model.predict(all_X))
            metric_name = 'r2_score'

        iteration_metrics.append({
            'iteration': iteration + 1,
            'n_train': len(X_train),
            'train_score': train_score,
            'test_score': test_score,
            'metric_name': metric_name
        })

        logger.info(f"Train {metric_name}: {train_score:.4f}, Test {metric_name}: {test_score:.4f}")

        # Stop early if the pool is exhausted.
        if len(X_pool_remaining) == 0:
            logger.info("Pool exhausted, stopping active learning")
            break

        # Select samples.
        if strategy == 'uncertainty':
            if task_type == 'classification':
                selected_indices = uncertainty_sampling(
                    model, X_pool_remaining,
                    n_samples=min(samples_per_iteration, len(X_pool_remaining)),
                    method='entropy'
                )
            else:
                # For regression, use random sampling as a simplified fallback.
                selected_indices = np.random.choice(
                    len(X_pool_remaining),
                    size=min(samples_per_iteration, len(X_pool_remaining)),
                    replace=False
                )

        elif strategy == 'qbc':
            # Train the committee.
            committee = []
            for i in range(3):
                committee_model = base_model.__class__(**base_model.get_params())
                # Use bootstrap sampling.
                bootstrap_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                committee_model.fit(X_train[bootstrap_indices], y_train[bootstrap_indices])
                committee.append(committee_model)

            selected_indices = query_by_committee(
                committee, X_pool_remaining,
                n_samples=min(samples_per_iteration, len(X_pool_remaining)),
                disagreement='vote_entropy' if task_type == 'classification' else 'variance'
            )

        elif strategy == 'random':
            # Random sampling baseline.
            selected_indices = np.random.choice(
                len(X_pool_remaining),
                size=min(samples_per_iteration, len(X_pool_remaining)),
                replace=False
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        selected_indices_history.append(selected_indices)

        # Prepare selected samples for objective evaluation.
        X_selected = X_pool_remaining[selected_indices]
        y_selected = y_pool_remaining[selected_indices]

        # Update the training set.
        X_train = np.vstack([X_train, X_selected])
        y_train = np.concatenate([y_train, y_selected])

        # Remove selected samples from the pool.
        mask = np.ones(len(X_pool_remaining), dtype=bool)
        mask[selected_indices] = False
        X_pool_remaining = X_pool_remaining[mask]
        y_pool_remaining = y_pool_remaining[mask]

        # Record history.
        X_train_history.append(X_train.copy())
        y_train_history.append(y_train.copy())

    # Train the final model.
    final_model = base_model.__class__(**base_model.get_params())
    final_model.fit(X_train, y_train)

    results = {
        'iteration_metrics': iteration_metrics,
        'selected_indices': selected_indices_history,
        'final_model': final_model,
        'training_history': {
            'X_train': X_train_history,
            'y_train': y_train_history
        },
        'X_train_history': X_train_history,
        'y_train_history': y_train_history,
        'n_iterations': len(iteration_metrics),
        'strategy': strategy,
        'task_type': task_type
    }

    logger.info(f"Active learning completed: {len(iteration_metrics)} iterations")

    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_uncertainty_intervals(
    X: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    X_new: Optional[np.ndarray] = None,
    y_new: Optional[np.ndarray] = None,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    title: str = 'Uncertainty Intervals',
    feature_idx: int = 0
) -> plt.Figure:
    """
    Plot predictive uncertainty intervals.

    Use fill_between to shade intervals around model predictions.

    Args:
        X: Features at prediction locations.
        y_pred: Predictive means.
        y_std: Predictive standard deviations.
        X_new: Optional features of newly selected samples.
        y_new: Optional labels of newly selected samples.
        X_train: Optional training sample features.
        y_train: Optional training sample labels.
        title: Plot title.
        feature_idx: Feature index to plot when X is multidimensional.

    Returns:
        fig: Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # For multidimensional X, use only the requested feature.
    if X.ndim > 1:
        X_plot = X[:, feature_idx]
    else:
        X_plot = X

    # Sort points for a continuous plot.
    sort_idx = np.argsort(X_plot)
    X_sorted = X_plot[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    y_std_sorted = y_std[sort_idx]

    # Plot the predictive mean.
    ax.plot(X_sorted, y_pred_sorted, 'b-', linewidth=2.5, label='Mean Prediction', zorder=3)

    # Plot a nominal 95% interval (mean +/- 1.96 * std).
    ax.fill_between(
        X_sorted,
        y_pred_sorted - 1.96 * y_std_sorted,
        y_pred_sorted + 1.96 * y_std_sorted,
        alpha=0.2, color='blue', label='95% Confidence', zorder=1
    )

    # Plot a nominal 68% interval (mean +/- 1 * std).
    ax.fill_between(
        X_sorted,
        y_pred_sorted - y_std_sorted,
        y_pred_sorted + y_std_sorted,
        alpha=0.3, color='blue', label='68% Confidence', zorder=2
    )

    # Plot training samples.
    if X_train is not None and y_train is not None:
        if X_train.ndim > 1:
            X_train_plot = X_train[:, feature_idx]
        else:
            X_train_plot = X_train
        ax.scatter(X_train_plot, y_train, c='green', s=80, marker='o',
                   edgecolors='black', linewidths=1, alpha=0.7,
                   label='Training Samples', zorder=4)

    # Plot newly selected samples.
    if X_new is not None and y_new is not None:
        if X_new.ndim > 1:
            X_new_plot = X_new[:, feature_idx]
        else:
            X_new_plot = X_new
        ax.scatter(X_new_plot, y_new, c='red', s=150, marker='*',
                   edgecolors='black', linewidths=1.5,
                   label='Selected Samples', zorder=5)

    ax.set_xlabel('Feature Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Target Value', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_acquisition_function(
    X_pool: np.ndarray,
    acquisition_values: np.ndarray,
    selected_idx: Optional[np.ndarray] = None,
    title: str = 'Acquisition Function',
    feature_idx: int = 0
) -> plt.Figure:
    """
    Plot the acquisition function.

    Args:
        X_pool: Candidate sample pool.
        acquisition_values: Acquisition function values.
        selected_idx: Optional selected sample indices.
        title: Plot title.
        feature_idx: Feature index to plot.

    Returns:
        fig: Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # For multidimensional X, use only the requested feature.
    if X_pool.ndim > 1:
        X_plot = X_pool[:, feature_idx]
    else:
        X_plot = X_pool

    # Sort points for a continuous plot.
    sort_idx = np.argsort(X_plot)
    X_sorted = X_plot[sort_idx]
    acq_sorted = acquisition_values[sort_idx]

    # Draw the acquisition function.
    ax.plot(X_sorted, acq_sorted, 'g-', linewidth=2, label='Acquisition Function')
    ax.fill_between(X_sorted, 0, acq_sorted, alpha=0.3, color='green')

    # Highlight selected samples.
    if selected_idx is not None:
        X_selected = X_plot[selected_idx]
        acq_selected = acquisition_values[selected_idx]
        ax.scatter(X_selected, acq_selected, c='red', s=150, marker='*',
                   edgecolors='black', linewidths=1.5,
                   label='Selected Samples', zorder=5)

    ax.set_xlabel('Feature Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Acquisition Value', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_optimization_trajectory(
    results: Dict,
    metric: str = 'test_score'
) -> plt.Figure:
    """
    Plot the optimization trajectory.

    Show model performance across iterations.

    Args:
        results: Result dictionary returned by active_learning_workflow.
        metric: Metric to plot ('test_score', 'train_score').

    Returns:
        fig: Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    iteration_metrics = results['iteration_metrics']
    iterations = [m['iteration'] for m in iteration_metrics]

    if metric == 'test_score':
        scores = [m['test_score'] for m in iteration_metrics]
        label = 'Test Score'
    elif metric == 'train_score':
        scores = [m['train_score'] for m in iteration_metrics]
        label = 'Train Score'
    else:
        scores = [m.get(metric, 0) for m in iteration_metrics]
        label = metric

    # Plot performance curves.
    ax.plot(iterations, scores, 'o-', linewidth=2.5, markersize=8,
            color='steelblue', label=label)

    # Add a reference line for the best performance.
    best_score = max(scores)
    best_iter = iterations[scores.index(best_score)]
    ax.axhline(y=best_score, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Best: {best_score:.4f} (Iter {best_iter})')

    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_title('Optimization Trajectory', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_convergence(
    results: Dict,
    show_confidence: bool = True
) -> plt.Figure:
    """
    Plot convergence.

    Show performance improvement and convergence trends.

    Args:
        results: Result dictionary from active_learning_workflow or bayesian_optimization_loop.
        show_confidence: Show an uncertainty band around the moving average.

    Returns:
        fig: Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Check the result type.
    if 'iteration_metrics' in results:
        # Active learning results.
        iteration_metrics = results['iteration_metrics']
        iterations = [m['iteration'] for m in iteration_metrics]
        scores = [m['test_score'] for m in iteration_metrics]
        ylabel = 'Test Score'
    elif 'best_values' in results:
        # Bayesian optimization results.
        iterations = list(range(1, len(results['best_values']) + 1))
        scores = results['best_values']
        ylabel = 'Best Value'
    else:
        raise ValueError("Unknown results format")

    # Plot the convergence curve.
    ax.plot(iterations, scores, 'o-', linewidth=2.5, markersize=8,
            color='darkgreen', label='Performance')

    # Compute the moving average and standard deviation for the uncertainty band.
    if show_confidence and len(scores) > 3:
        window = min(3, len(scores))
        moving_avg = pd.Series(scores).rolling(window=window, center=True, min_periods=1).mean()
        moving_std = pd.Series(scores).rolling(window=window, center=True, min_periods=1).std()

        ax.plot(iterations, moving_avg, '--', linewidth=2, color='orange',
                label=f'Moving Average (window={window})')

        if not moving_std.isna().all():
            ax.fill_between(iterations,
                           moving_avg - moving_std,
                           moving_avg + moving_std,
                           alpha=0.2, color='orange')

    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title('Convergence Plot', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_learning_progress(
    iteration_metrics: List[Dict],
    metrics_to_plot: List[str] = ['test_score', 'train_score']
) -> plt.Figure:
    """
    Plot learning progress.

    Show multiple metrics across iterations.

    Args:
        iteration_metrics: List of per-iteration metrics.
        metrics_to_plot: List of metrics to display.

    Returns:
        fig: Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    iterations = [m['iteration'] for m in iteration_metrics]

    colors = ['steelblue', 'darkgreen', 'coral', 'purple']

    for i, metric in enumerate(metrics_to_plot):
        if metric in iteration_metrics[0]:
            values = [m[metric] for m in iteration_metrics]
            color = colors[i % len(colors)]
            ax.plot(iterations, values, 'o-', linewidth=2, markersize=6,
                   color=color, label=metric.replace('_', ' ').title())

    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Learning Progress', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_exploration_space_2d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    selected_idx: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    feature_indices: Tuple[int, int] = (0, 1)
) -> plt.Figure:
    """
    Plot the exploration space in two dimensions.

    Args:
        X_train: Training sample features.
        y_train: Training sample labels.
        X_pool: Sample pool features.
        selected_idx: Optional selected sample indices.
        feature_names: Optional list of feature names.
        feature_indices: Two feature indices to plot.

    Returns:
        fig: Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    idx1, idx2 = feature_indices

    # Plot training samples.
    scatter_train = ax.scatter(
        X_train[:, idx1], X_train[:, idx2],
        c=y_train, cmap='viridis', s=100,
        edgecolors='black', linewidths=1.5,
        alpha=0.8, label='Training Samples'
    )

    # Plot the sample pool.
    ax.scatter(
        X_pool[:, idx1], X_pool[:, idx2],
        c='lightgray', s=50, alpha=0.5,
        edgecolors='gray', linewidths=0.5,
        label='Pool Samples'
    )

    # Plot selected samples.
    if selected_idx is not None:
        ax.scatter(
            X_pool[selected_idx, idx1], X_pool[selected_idx, idx2],
            c='red', s=200, marker='*',
            edgecolors='black', linewidths=2,
            label='Selected Samples', zorder=5
        )

    # Set axis labels.
    if feature_names is not None:
        xlabel = feature_names[idx1]
        ylabel = feature_names[idx2]
    else:
        xlabel = f'Feature {idx1}'
        ylabel = f'Feature {idx2}'

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title('2D Exploration Space', fontsize=16, fontweight='bold', pad=20)

    # Add a color bar.
    cbar = plt.colorbar(scatter_train, ax=ax)
    cbar.set_label('Target Value', fontsize=12)

    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_exploration_space_3d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pool: np.ndarray,
    selected_idx: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    feature_indices: Tuple[int, int, int] = (0, 1, 2)
) -> plt.Figure:
    """
    Plot the exploration space in three dimensions.

    Args:
        X_train: Training sample features.
        y_train: Training sample labels.
        X_pool: Sample pool features.
        selected_idx: Optional selected sample indices.
        feature_names: Optional list of feature names.
        feature_indices: Three feature indices to plot.

    Returns:
        fig: Matplotlib Figure object.
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    idx1, idx2, idx3 = feature_indices

    # Plot training samples.
    scatter_train = ax.scatter(
        X_train[:, idx1], X_train[:, idx2], X_train[:, idx3],
        c=y_train, cmap='viridis', s=100,
        edgecolors='black', linewidths=1.5,
        alpha=0.8, label='Training Samples'
    )

    # Plot the sample pool.
    ax.scatter(
        X_pool[:, idx1], X_pool[:, idx2], X_pool[:, idx3],
        c='lightgray', s=30, alpha=0.3,
        edgecolors='gray', linewidths=0.5,
        label='Pool Samples'
    )

    # Plot selected samples.
    if selected_idx is not None:
        ax.scatter(
            X_pool[selected_idx, idx1],
            X_pool[selected_idx, idx2],
            X_pool[selected_idx, idx3],
            c='red', s=200, marker='*',
            edgecolors='black', linewidths=2,
            label='Selected Samples'
        )

    # Set axis labels.
    if feature_names is not None:
        xlabel = feature_names[idx1]
        ylabel = feature_names[idx2]
        zlabel = feature_names[idx3]
    else:
        xlabel = f'Feature {idx1}'
        ylabel = f'Feature {idx2}'
        zlabel = f'Feature {idx3}'

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_zlabel(zlabel, fontsize=12, fontweight='bold')
    ax.set_title('3D Exploration Space', fontsize=16, fontweight='bold', pad=20)

    # Add a color bar.
    cbar = plt.colorbar(scatter_train, ax=ax, shrink=0.8)
    cbar.set_label('Target Value', fontsize=12)

    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    return fig


def plot_sample_selection_heatmap(
    selected_indices_history: List[np.ndarray],
    n_pool_samples: int
) -> plt.Figure:
    """
    Plot a sample selection heatmap.

    Show which samples were selected at each iteration.

    Args:
        selected_indices_history: List of selected sample indices for each iteration.
        n_pool_samples: Total number of samples in the pool.

    Returns:
        fig: Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Create the selection matrix.
    n_iterations = len(selected_indices_history)
    selection_matrix = np.zeros((n_iterations, n_pool_samples))

    for i, selected_idx in enumerate(selected_indices_history):
        selection_matrix[i, selected_idx] = 1

    # Draw the heatmap.
    im = ax.imshow(selection_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    # Set axis labels.
    ax.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_title('Sample Selection Heatmap', fontsize=16, fontweight='bold', pad=20)

    # Set axis ticks.
    ax.set_yticks(range(n_iterations))
    ax.set_yticklabels([f'Iter {i+1}' for i in range(n_iterations)])

    # Add a color bar.
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selected', fontsize=12)

    plt.tight_layout()
    return fig
