"""
Machine learning module tests

Exercise the supervised_learning, virtual_screening, and active_learning
modules with synthetic classification and regression data.

Test coverage:
1. Supervised learning tests for classification and regression
2. Virtual screening module tests
3. Active learning module tests
"""

import sys
from pathlib import Path

# Add the src directory to the import path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")

# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_classification_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic classification dataset

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        random_state: Random seed

    Returns:
        Features and target stored in a DataFrame
    """
    from sklearn.datasets import make_classification

    logger.info(f"Generating classification data: {n_samples} samples, {n_features} features, {n_classes} classes")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=n_classes,
        random_state=random_state,
        flip_y=0.05  # Randomly replace 5% of the labels
    )

    # Create a DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y

    logger.info(f"Classification data generated: shape={df.shape}")
    logger.info(f"Class distribution: {df['target'].value_counts().to_dict()}")

    return df


def generate_regression_data(
    n_samples: int = 1000,
    n_features: int = 20,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic regression dataset

    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed

    Returns:
        Features and target stored in a DataFrame
    """
    from sklearn.datasets import make_regression

    logger.info(f"Generating regression data: {n_samples} samples, {n_features} features")

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        noise=10.0,
        random_state=random_state
    )

    # Create a DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y

    logger.info(f"Regression data generated: shape={df.shape}")
    logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    return df


def split_data_for_active_learning(
    df: pd.DataFrame,
    n_initial: int = 50,
    n_pool: int = 500,
    random_state: int = 42
) -> tuple:
    """
    Split the data into an initial training set and an active learning pool

    Args:
        df: Complete dataset
        n_initial: Initial training set size
        n_pool: Sample pool size
        random_state: Random seed

    Returns:
        (X_train_initial, y_train_initial, X_pool, y_pool)
    """
    np.random.seed(random_state)

    # Separate the features and target
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Shuffle randomly
    indices = np.random.permutation(len(df))

    # Split indices into the initial training set and candidate pool
    train_indices = indices[:n_initial]
    pool_indices = indices[n_initial:n_initial + n_pool]

    X_train_initial = X[train_indices]
    y_train_initial = y[train_indices]
    X_pool = X[pool_indices]
    y_pool = y[pool_indices]

    logger.info(f"Data split: initial_train={len(X_train_initial)}, pool={len(X_pool)}")

    return X_train_initial, y_train_initial, X_pool, y_pool


# ============================================================================
# SUPERVISED LEARNING TESTS
# ============================================================================

def test_supervised_learning_classification():
    """Test supervised classification"""
    from ml.supervised_learning import train_supervised_model, save_model, load_model

    logger.info("\n" + "="*80)
    logger.info("Testing Supervised Learning - Classification")
    logger.info("="*80)

    # Generate data
    df = generate_classification_data(n_samples=500, n_features=15, n_classes=3)

    # Train the model
    logger.info("\n1. Training Random Forest classifier...")
    model, results = train_supervised_model(
        data_df=df,
        target_column='target',
        task_type='classification',
        model_name='random_forest',
        feature_selection='mutual_info',
        feature_scaling='standard',
        hyperparameter_tuning=False,
        test_size=0.2,
        cv_folds=3,
        random_state=42
    )

    # Check the results
    logger.info(f"[OK] Model trained successfully")
    logger.info(f"[OK] Accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"[OK] F1 score: {results['metrics']['f1_score']:.4f}")
    logger.info(f"[OK] Number of features used: {len(results['feature_names'])}")

    # Test model serialization and loading
    logger.info("\n2. Testing model save/load...")
    model_path = Path(__file__).parent / 'test_model_clf.pkl'
    save_model(model, str(model_path), metadata=results['train_info'],
               feature_names=results['feature_names'],
               scaler=results['scaler'],
               task_type=results['task_type'])
    logger.info(f"[OK] Model saved to {model_path}")

    loaded_model, loaded_package = load_model(str(model_path))
    logger.info(f"[OK] Model loaded successfully")
    logger.info(f"[OK] Task type: {loaded_package['task_type']}")

    # Clean up
    model_path.unlink()

    logger.info("\n[PASS] Classification test passed!")
    return True


def test_supervised_learning_regression():
    """Test supervised regression"""
    from ml.supervised_learning import train_supervised_model

    logger.info("\n" + "="*80)
    logger.info("Testing Supervised Learning - Regression")
    logger.info("="*80)

    # Generate data
    df = generate_regression_data(n_samples=500, n_features=15)

    # Train the model
    logger.info("\n1. Training Random Forest regressor...")
    model, results = train_supervised_model(
        data_df=df,
        target_column='target',
        task_type='regression',
        model_name='random_forest',
        feature_selection='correlation',
        feature_scaling='standard',
        hyperparameter_tuning=False,
        test_size=0.2,
        cv_folds=3,
        random_state=42
    )

    # Check the results
    logger.info(f"[OK] Model trained successfully")
    logger.info(f"[OK] R² score: {results['metrics']['r2_score']:.4f}")
    logger.info(f"[OK] RMSE: {results['metrics']['rmse']:.4f}")
    logger.info(f"[OK] MAE: {results['metrics']['mae']:.4f}")

    logger.info("\n[PASS] Regression test passed!")
    return True


def test_supervised_learning_automl():
    """Test automated model comparison"""
    from ml.supervised_learning import compare_models_automl

    logger.info("\n" + "="*80)
    logger.info("Testing Supervised Learning - AutoML")
    logger.info("="*80)

    # Generate data
    df = generate_classification_data(n_samples=300, n_features=10, n_classes=2)

    # AutoML comparison
    logger.info("\n1. Comparing multiple models...")
    best_model, comparison_df, all_results = compare_models_automl(
        data_df=df,
        target_column='target',
        task_type='classification',
        models_to_compare=['random_forest', 'logistic', 'svm'],
        test_size=0.2,
        cv_folds=3,
        tune_best=False,
        random_state=42
    )

    # Check the results
    logger.info(f"[OK] Compared {len(comparison_df)} models")
    logger.info(f"[OK] Best model: {comparison_df.iloc[0]['model']}")
    logger.info(f"[OK] Best accuracy: {comparison_df.iloc[0]['test_accuracy']:.4f}")
    logger.info("\nModel comparison:")
    logger.info(comparison_df.to_string())

    logger.info("\n[PASS] AutoML test passed!")
    return True


# ============================================================================
# VIRTUAL SCREENING TESTS
# ============================================================================

def test_virtual_screening():
    """Test the virtual screening module"""
    from ml.supervised_learning import train_supervised_model, save_model
    from ml.virtual_screening import screen_dataset, select_top_candidates

    logger.info("\n" + "="*80)
    logger.info("Testing Virtual Screening")
    logger.info("="*80)

    # Generate training and screening data
    logger.info("\n1. Generating data...")
    train_df = generate_classification_data(n_samples=300, n_features=10, n_classes=2)
    screen_df = generate_classification_data(n_samples=200, n_features=10, n_classes=2)
    screen_df = screen_df.drop('target', axis=1)  # Remove the target column

    # Train the model
    logger.info("\n2. Training model...")
    model, results = train_supervised_model(
        data_df=train_df,
        target_column='target',
        task_type='classification',
        model_name='random_forest',
        test_size=0.2,
        random_state=42
    )

    # Save the model
    model_path = Path(__file__).parent / 'test_model_screening.pkl'
    save_model(model, str(model_path),
               metadata=results['train_info'],
               feature_names=results['feature_names'],
               scaler=results['scaler'],
               task_type=results['task_type'])
    logger.info(f"[OK] Model saved to {model_path}")

    # Virtual screening
    logger.info("\n3. Screening dataset...")
    results_df, info = screen_dataset(
        model_path=str(model_path),
        data_df=screen_df,
        confidence_method='probability',
        min_confidence=0.6,
        return_probabilities=True
    )

    logger.info(f"[OK] Screened {info['n_samples']} samples")
    logger.info(f"[OK] {info['n_screened']} samples passed the confidence threshold")
    logger.info(f"[OK] Class distribution: {info.get('class_distribution', {})}")

    # Select the top candidates
    logger.info("\n4. Selecting top candidates...")
    top_candidates = select_top_candidates(
        results_df=results_df,
        n_candidates=20,
        criteria='combined',
        confidence_threshold=0.5
    )

    logger.info(f"[OK] Selected {len(top_candidates)} top candidates")
    logger.info(f"[OK] Average confidence: {top_candidates['confidence'].mean():.4f}")

    # Clean up
    model_path.unlink()

    logger.info("\n[PASS] Virtual screening test passed!")
    return True


# ============================================================================
# ACTIVE LEARNING TESTS
# ============================================================================

def test_active_learning():
    """Test the active learning module"""
    from ml.active_learning import active_learning_workflow, uncertainty_sampling

    logger.info("\n" + "="*80)
    logger.info("Testing Active Learning")
    logger.info("="*80)

    # Generate data
    logger.info("\n1. Generating data...")
    df = generate_classification_data(n_samples=600, n_features=10, n_classes=2)
    X_train_initial, y_train_initial, X_pool, y_pool = split_data_for_active_learning(
        df, n_initial=50, n_pool=300, random_state=42
    )

    # Active learning workflow
    logger.info("\n2. Running active learning workflow...")
    results = active_learning_workflow(
        X_train_initial=X_train_initial,
        y_train_initial=y_train_initial,
        X_pool=X_pool,
        y_pool_true=y_pool,
        model_name='random_forest',
        task_type='classification',
        strategy='uncertainty',
        n_iterations=5,
        samples_per_iteration=10,
        random_state=42
    )

    logger.info(f"[OK] Completed {results['n_iterations']} iterations")
    logger.info(f"[OK] Strategy: {results['strategy']}")

    # Display the change in performance
    logger.info("\n3. Performance across iterations:")
    for metric in results['iteration_metrics']:
        logger.info(f"  Iteration {metric['iteration']}: "
                   f"train={metric['train_score']:.4f}, "
                   f"test={metric['test_score']:.4f}")

    final_score = results['iteration_metrics'][-1]['test_score']
    initial_score = results['iteration_metrics'][0]['test_score']
    improvement = final_score - initial_score

    logger.info(f"\n[OK] Initial score: {initial_score:.4f}")
    logger.info(f"[OK] Final score: {final_score:.4f}")
    logger.info(f"[OK] Score change: {improvement:.4f}")

    logger.info("\n[PASS] Active learning test passed!")
    return True


def test_bayesian_optimization():
    """Test Bayesian optimization"""
    from ml.active_learning import bayesian_optimization_loop, fit_gaussian_process

    logger.info("\n" + "="*80)
    logger.info("Testing Bayesian Optimization")
    logger.info("="*80)

    # Generate data
    logger.info("\n1. Generating data...")
    df = generate_regression_data(n_samples=200, n_features=5)
    X_train_initial, y_train_initial, X_pool, y_pool = split_data_for_active_learning(
        df, n_initial=20, n_pool=100, random_state=42
    )

    # Define a simple objective function for testing
    def objective_function(x):
        return y_pool[0]  # Simplified implementation

    # Bayesian optimization
    logger.info("\n2. Running Bayesian optimization...")
    results = bayesian_optimization_loop(
        objective_function=None,  # Use supervised learning mode
        X_train_initial=X_train_initial,
        y_train_initial=y_train_initial,
        X_pool=X_pool[:50],  # Use a smaller candidate pool
        n_iterations=3,
        acquisition='ei',
        model_type='gp',
        samples_per_iteration=5,
        random_state=42
    )

    logger.info(f"[OK] Completed {results['n_iterations']} iterations")
    logger.info(f"[OK] Selected {len(results['selected_samples'])} batches of samples")

    logger.info("\n[PASS] Bayesian optimization test passed!")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("MACHINE LEARNING MODULES TEST SUITE")
    logger.info("="*80)
    logger.info(f"Testing supervised_learning, virtual_screening, and active_learning modules")
    logger.info(f"Using synthetic data for workflow checks\n")

    results = {}

    # Test supervised learning
    try:
        results['classification'] = test_supervised_learning_classification()
    except Exception as e:
        logger.error(f"[FAIL] Classification test failed: {str(e)}")
        results['classification'] = False

    try:
        results['regression'] = test_supervised_learning_regression()
    except Exception as e:
        logger.error(f"[FAIL] Regression test failed: {str(e)}")
        results['regression'] = False

    try:
        results['automl'] = test_supervised_learning_automl()
    except Exception as e:
        logger.error(f"[FAIL] AutoML test failed: {str(e)}")
        results['automl'] = False

    # Test virtual screening
    try:
        results['virtual_screening'] = test_virtual_screening()
    except Exception as e:
        logger.error(f"[FAIL] Virtual screening test failed: {str(e)}")
        results['virtual_screening'] = False

    # Test active learning
    try:
        results['active_learning'] = test_active_learning()
    except Exception as e:
        logger.error(f"[FAIL] Active learning test failed: {str(e)}")
        results['active_learning'] = False

    try:
        results['bayesian_opt'] = test_bayesian_optimization()
    except Exception as e:
        logger.error(f"[FAIL] Bayesian optimization test failed: {str(e)}")
        results['bayesian_opt'] = False

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "[PASS] PASSED" if passed_flag else "[FAIL] FAILED"
        logger.info(f"{test_name:25s}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n[SUCCESS] All tests passed successfully!")
        return 0
    else:
        logger.warning(f"\n[WARNING] {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())



