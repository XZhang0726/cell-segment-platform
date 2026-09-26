"""
Supervised learning utilities.

Provides feature engineering, model training, evaluation, and prediction workflows.
Supports classification, regression, and automated model comparison and selection.

Main capabilities:
1. Feature engineering: selection, scaling, encoding, and polynomial features.
2. Model training: multiple scikit-learn estimators and hyperparameter tuning.
3. Model evaluation: metrics and visualizations.
4. Model persistence: save and load models with metadata.
5. AutoML: automated model comparison and selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import joblib
from pathlib import Path
import warnings

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, RFE, f_classif, f_regression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)

# Classification models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Regression models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

from loguru import logger

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CONSTANTS AND CONFIGURATIONS
# ============================================================================

# Classification models registry
CLASSIFICATION_MODELS = {
    'random_forest': RandomForestClassifier,
    'svm': SVC,
    'logistic': LogisticRegression,
    'gradient_boosting': GradientBoostingClassifier,
    'knn': KNeighborsClassifier,
    'decision_tree': DecisionTreeClassifier,
    'adaboost': AdaBoostClassifier,
    'extra_trees': ExtraTreesClassifier,
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS['xgboost'] = XGBClassifier

# Regression models registry
REGRESSION_MODELS = {
    'random_forest': RandomForestRegressor,
    'svr': SVR,
    'linear': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'elastic_net': ElasticNet,
    'gradient_boosting': GradientBoostingRegressor,
    'decision_tree': DecisionTreeRegressor,
}

if XGBOOST_AVAILABLE:
    REGRESSION_MODELS['xgboost'] = XGBRegressor

# Default hyperparameter grids for classification
PARAM_GRIDS_CLASSIFICATION = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    },
    'logistic': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'decision_tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    'adaboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'extra_trees': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
}

if XGBOOST_AVAILABLE:
    PARAM_GRIDS_CLASSIFICATION['xgboost'] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

# Default hyperparameter grids for regression
PARAM_GRIDS_REGRESSION = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'svr': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'epsilon': [0.01, 0.1, 0.2]
    },
    'ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    'lasso': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    'elastic_net': {
        'alpha': [0.01, 0.1, 1, 10],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    },
    'decision_tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
}

if XGBOOST_AVAILABLE:
    PARAM_GRIDS_REGRESSION['xgboost'] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

# Default model parameters
DEFAULT_PARAMS_CLASSIFICATION = {
    'random_forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
    'svm': {'probability': True, 'random_state': 42},
    'logistic': {'random_state': 42, 'max_iter': 1000},
    'gradient_boosting': {'n_estimators': 100, 'random_state': 42},
    'knn': {'n_neighbors': 5},
    'decision_tree': {'random_state': 42},
    'adaboost': {'n_estimators': 100, 'random_state': 42},
    'extra_trees': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
}

if XGBOOST_AVAILABLE:
    DEFAULT_PARAMS_CLASSIFICATION['xgboost'] = {
        'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'
    }

DEFAULT_PARAMS_REGRESSION = {
    'random_forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
    'svr': {},
    'linear': {},
    'ridge': {'random_state': 42},
    'lasso': {'random_state': 42},
    'elastic_net': {'random_state': 42},
    'gradient_boosting': {'n_estimators': 100, 'random_state': 42},
    'decision_tree': {'random_state': 42},
}

if XGBOOST_AVAILABLE:
    DEFAULT_PARAMS_REGRESSION['xgboost'] = {
        'n_estimators': 100, 'random_state': 42, 'n_jobs': -1
    }


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def select_features_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.1,
    method: str = 'pearson'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Select features based on correlation.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        threshold: Absolute correlation threshold.
        method: Correlation method ('pearson', 'spearman', 'kendall').

    Returns:
        X_selected: DataFrame of selected features.
        info: Dictionary containing feature correlations.
    """
    logger.info(f"Selecting features by correlation (method={method}, threshold={threshold})")

    # Compute correlations between features and the target.
    correlations = {}
    for col in X.columns:
        if method == 'pearson':
            corr = X[col].corr(y, method='pearson')
        elif method == 'spearman':
            corr = X[col].corr(y, method='spearman')
        elif method == 'kendall':
            corr = X[col].corr(y, method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        correlations[col] = corr

    # Select features whose correlation exceeds the threshold.
    selected_features = [col for col, corr in correlations.items() if abs(corr) >= threshold]

    if len(selected_features) == 0:
        logger.warning(f"No features meet correlation threshold {threshold}, keeping all features")
        selected_features = list(X.columns)

    X_selected = X[selected_features]

    info = {
        'method': 'correlation',
        'correlation_method': method,
        'threshold': threshold,
        'n_features_original': len(X.columns),
        'n_features_selected': len(selected_features),
        'selected_features': selected_features,
        'correlations': correlations
    }

    logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")

    return X_selected, info


def select_features_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: Optional[int] = None,
    task_type: str = 'classification'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Select features based on mutual information.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        n_features: Number of features to select; None selects the top 50%.
        task_type: 'classification' or 'regression'

    Returns:
        X_selected: DataFrame of selected features.
        info: Dictionary containing feature importances.
    """
    logger.info(f"Selecting features by mutual information (task_type={task_type})")

    if n_features is None:
        n_features = max(1, len(X.columns) // 2)

    n_features = min(n_features, len(X.columns))

    # Compute mutual information.
    if task_type == 'classification':
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)

    # Create a feature importance DataFrame.
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    # Select the top n features.
    selected_features = mi_df.head(n_features)['feature'].tolist()
    X_selected = X[selected_features]

    info = {
        'method': 'mutual_info',
        'task_type': task_type,
        'n_features_original': len(X.columns),
        'n_features_selected': len(selected_features),
        'selected_features': selected_features,
        'mi_scores': dict(zip(mi_df['feature'], mi_df['mi_score']))
    }

    logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")

    return X_selected, info


def select_features_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: Optional[int] = None,
    estimator: Optional[Any] = None,
    task_type: str = 'classification'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Select features using recursive feature elimination (RFE).

    Args:
        X: Feature DataFrame.
        y: Target Series.
        n_features: Number of features to select; None selects the top 50%.
        estimator: Feature selection estimator; None uses the default.
        task_type: 'classification' or 'regression'

    Returns:
        X_selected: DataFrame of selected features.
        info: Dictionary containing feature rankings.
    """
    logger.info(f"Selecting features by RFE (task_type={task_type})")

    if n_features is None:
        n_features = max(1, len(X.columns) // 2)

    n_features = min(n_features, len(X.columns))

    # Use the default estimator.
    if estimator is None:
        if task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

    # Run RFE.
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X, y)

    # Get the selected features.
    selected_features = X.columns[rfe.support_].tolist()
    X_selected = X[selected_features]

    # Get feature rankings.
    feature_ranking = dict(zip(X.columns, rfe.ranking_))

    info = {
        'method': 'rfe',
        'task_type': task_type,
        'n_features_original': len(X.columns),
        'n_features_selected': len(selected_features),
        'selected_features': selected_features,
        'feature_ranking': feature_ranking
    }

    logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")

    return X_selected, info


def select_features_tree_based(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: Optional[int] = None,
    threshold: str = 'median',
    task_type: str = 'classification'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Select features using a tree-based model.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        n_features: Number of features to select; None uses threshold.
        threshold: Importance threshold ('mean', 'median', or a float).
        task_type: 'classification' or 'regression'

    Returns:
        X_selected: DataFrame of selected features.
        info: Dictionary containing feature importances.
    """
    logger.info(f"Selecting features by tree-based importance (task_type={task_type})")

    # Train a tree-based model.
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X, y)

    # Get feature importances.
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Select features.
    if n_features is not None:
        n_features = min(n_features, len(X.columns))
        selected_features = importance_df.head(n_features)['feature'].tolist()
    else:
        # Apply the importance threshold.
        if threshold == 'mean':
            thresh_value = importances.mean()
        elif threshold == 'median':
            thresh_value = np.median(importances)
        else:
            thresh_value = float(threshold)

        selected_features = importance_df[importance_df['importance'] >= thresh_value]['feature'].tolist()

        if len(selected_features) == 0:
            logger.warning(f"No features meet threshold {thresh_value}, keeping top 50%")
            n_features = max(1, len(X.columns) // 2)
            selected_features = importance_df.head(n_features)['feature'].tolist()

    X_selected = X[selected_features]

    info = {
        'method': 'tree_based',
        'task_type': task_type,
        'threshold': threshold,
        'n_features_original': len(X.columns),
        'n_features_selected': len(selected_features),
        'selected_features': selected_features,
        'feature_importance': dict(zip(importance_df['feature'], importance_df['importance']))
    }

    logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")

    return X_selected, info


def scale_features(
    X: Union[pd.DataFrame, np.ndarray],
    method: str = 'standard',
    scaler: Optional[Any] = None,
    fit: bool = True
) -> Tuple[np.ndarray, Any]:
    """
    Scale features.

    Args:
        X: Feature DataFrame or array.
        method: Scaling method ('standard', 'minmax', 'robust').
        scaler: Fitted scaler; None creates a new scaler.
        fit: Fit the scaler; True for training and False for prediction.

    Returns:
        X_scaled: Scaled feature array.
        scaler: Scaler object.
    """
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    if fit:
        X_scaled = scaler.fit_transform(X)
        logger.info(f"Features scaled using {method} method (fitted)")
    else:
        X_scaled = scaler.transform(X)
        logger.info(f"Features scaled using {method} method (transform only)")

    return X_scaled, scaler


def encode_categorical(
    X: pd.DataFrame,
    method: str = 'onehot',
    encoder: Optional[Any] = None,
    fit: bool = True,
    categorical_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Any]:
    """
    Encode categorical variables.

    Args:
        X: Feature DataFrame.
        method: Encoding method ('onehot', 'label').
        encoder: Fitted encoder; None creates a new encoder.
        fit: Fit the encoder; True for training and False for prediction.
        categorical_cols: List of categorical columns; None detects them automatically.

    Returns:
        X_encoded: Encoded DataFrame.
        encoder: Encoder object.
    """
    # Detect categorical columns automatically.
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) == 0:
        logger.info("No categorical columns found, returning original DataFrame")
        return X, None

    logger.info(f"Encoding {len(categorical_cols)} categorical columns using {method} method")

    if method == 'onehot':
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        if fit:
            encoded = encoder.fit_transform(X[categorical_cols])
        else:
            encoded = encoder.transform(X[categorical_cols])

        # Create new column names.
        feature_names = encoder.get_feature_names_out(categorical_cols)

        # Create the encoded DataFrame.
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)

        # Combine with noncategorical columns.
        non_categorical_cols = [col for col in X.columns if col not in categorical_cols]
        X_encoded = pd.concat([X[non_categorical_cols], encoded_df], axis=1)

    elif method == 'label':
        X_encoded = X.copy()
        if encoder is None:
            encoder = {}

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                encoder[col] = le
            else:
                le = encoder.get(col)
                if le is not None:
                    X_encoded[col] = le.transform(X[col].astype(str))

    else:
        raise ValueError(f"Unknown encoding method: {method}")

    logger.info(f"Categorical encoding completed: {X_encoded.shape[1]} features")

    return X_encoded, encoder


def generate_polynomial_features(
    X: pd.DataFrame,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate polynomial features.

    Args:
        X: Feature DataFrame.
        degree: Polynomial degree.
        interaction_only: Generate interaction terms only.
        include_bias: Include a bias term.

    Returns:
        X_poly: DataFrame containing polynomial features.
        feature_names: List of generated feature names.
    """
    from sklearn.preprocessing import PolynomialFeatures

    logger.info(f"Generating polynomial features (degree={degree}, interaction_only={interaction_only})")

    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    X_poly_array = poly.fit_transform(X)

    # Generate feature names.
    feature_names = poly.get_feature_names_out(X.columns)

    X_poly = pd.DataFrame(X_poly_array, columns=feature_names, index=X.index)

    logger.info(f"Polynomial features generated: {X_poly.shape[1]} features (from {X.shape[1]} original)")

    return X_poly, feature_names.tolist()


def detect_feature_interactions(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 10,
    task_type: str = 'classification'
) -> pd.DataFrame:
    """
    Identify feature interactions.

    Use a random forest to estimate the importance of feature interactions.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        top_n: Number of highest-ranked interactions to return.
        task_type: 'classification' or 'regression'

    Returns:
        interactions_df: DataFrame of feature pairs and their interaction importances.
    """
    logger.info(f"Detecting feature interactions (top_n={top_n})")

    # Generate all second-order feature interactions.
    from itertools import combinations

    interactions = []
    feature_pairs = list(combinations(X.columns, 2))

    # Limit the number of interactions to control feature expansion.
    if len(feature_pairs) > 100:
        logger.warning(f"Too many feature pairs ({len(feature_pairs)}), sampling 100 pairs")
        import random
        random.seed(42)
        feature_pairs = random.sample(feature_pairs, 100)

    # Create interaction features.
    X_interactions = X.copy()
    for feat1, feat2 in feature_pairs:
        interaction_name = f"{feat1}_x_{feat2}"
        X_interactions[interaction_name] = X[feat1] * X[feat2]
        interactions.append((feat1, feat2, interaction_name))

    # Train a model to estimate feature importances.
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_interactions, y)

    # Get interaction feature importances.
    importances = model.feature_importances_
    feature_names = X_interactions.columns

    # Retain only interaction feature importances.
    interaction_importances = []
    for feat1, feat2, interaction_name in interactions:
        idx = list(feature_names).index(interaction_name)
        importance = importances[idx]
        interaction_importances.append({
            'feature_1': feat1,
            'feature_2': feat2,
            'interaction': interaction_name,
            'importance': importance
        })

    # Create and sort the DataFrame.
    interactions_df = pd.DataFrame(interaction_importances)
    interactions_df = interactions_df.sort_values('importance', ascending=False).head(top_n)

    logger.info(f"Top {len(interactions_df)} feature interactions identified")

    return interactions_df


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    task_type: str = 'classification',
    params: Optional[Dict] = None,
    cv_folds: int = 5
) -> Tuple[Any, Dict]:
    """
    Train a single model.

    Args:
        X_train: Training features.
        y_train: Training targets.
        model_name: Model name.
        task_type: 'classification' or 'regression'
        params: Model parameter dictionary; None uses default parameters.
        cv_folds: Number of cross-validation folds.

    Returns:
        model: Trained model.
        info: Dictionary of training details and cross-validation scores.
    """
    logger.info(f"Training {model_name} model (task_type={task_type})")

    # Get the model class.
    if task_type == 'classification':
        if model_name not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unknown classification model: {model_name}")
        model_class = CLASSIFICATION_MODELS[model_name]
        default_params = DEFAULT_PARAMS_CLASSIFICATION.get(model_name, {})
    else:
        if model_name not in REGRESSION_MODELS:
            raise ValueError(f"Unknown regression model: {model_name}")
        model_class = REGRESSION_MODELS[model_name]
        default_params = DEFAULT_PARAMS_REGRESSION.get(model_name, {})

    # Merge parameters.
    if params is None:
        params = default_params
    else:
        params = {**default_params, **params}

    # Create the model.
    model = model_class(**params)

    # Train the model.
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()

    # Run cross-validation.
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, n_jobs=-1)

    info = {
        'model_name': model_name,
        'task_type': task_type,
        'params': params,
        'training_time': training_time,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_samples': len(X_train),
        'n_features': X_train.shape[1]
    }

    logger.info(f"Model trained: CV score = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, info


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    task_type: str = 'classification',
    search_method: str = 'grid',
    param_grid: Optional[Dict] = None,
    n_iter: int = 50,
    cv_folds: int = 5
) -> Tuple[Any, Dict]:
    """
    Tune model hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training targets.
        model_name: Model name.
        task_type: 'classification' or 'regression'
        search_method: 'grid' or 'random'
        param_grid: Parameter grid; None uses the default.
        n_iter: Number of randomized search iterations.
        cv_folds: Number of cross-validation folds.

    Returns:
        best_model: Best-performing model.
        info: Dictionary of best parameters, cross-validation scores, and search history.
    """
    logger.info(f"Hyperparameter tuning for {model_name} (method={search_method})")

    # Get the model class and parameter grid.
    if task_type == 'classification':
        if model_name not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unknown classification model: {model_name}")
        model_class = CLASSIFICATION_MODELS[model_name]
        default_param_grid = PARAM_GRIDS_CLASSIFICATION.get(model_name, {})
    else:
        if model_name not in REGRESSION_MODELS:
            raise ValueError(f"Unknown regression model: {model_name}")
        model_class = REGRESSION_MODELS[model_name]
        default_param_grid = PARAM_GRIDS_REGRESSION.get(model_name, {})

    # Use the default or supplied parameter grid.
    if param_grid is None:
        param_grid = default_param_grid

    if not param_grid:
        logger.warning(f"No parameter grid defined for {model_name}, using default parameters")
        return train_single_model(X_train, y_train, model_name, task_type, cv_folds=cv_folds)

    # Create the base model.
    base_model = model_class()

    # Run the search.
    start_time = datetime.now()
    if search_method == 'grid':
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            n_jobs=-1,
            verbose=0
        )
    elif search_method == 'random':
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown search method: {search_method}")

    search.fit(X_train, y_train)
    tuning_time = (datetime.now() - start_time).total_seconds()

    best_model = search.best_estimator_

    info = {
        'model_name': model_name,
        'task_type': task_type,
        'search_method': search_method,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'tuning_time': tuning_time,
        'n_samples': len(X_train),
        'n_features': X_train.shape[1],
        'cv_results': {
            'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': search.cv_results_['std_test_score'].tolist(),
            'params': search.cv_results_['params']
        }
    }

    logger.info(f"Hyperparameter tuning completed: best score = {search.best_score_:.4f}")
    logger.info(f"Best parameters: {search.best_params_}")

    return best_model, info


def train_multiple_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = 'classification',
    models_to_try: Optional[List[str]] = None,
    cv_folds: int = 5
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train multiple models and compare their performance.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        task_type: 'classification' or 'regression'
        models_to_try: List of models to evaluate; None includes all supported models.
        cv_folds: Number of cross-validation folds.

    Returns:
        models_dict: Dictionary mapping model names to trained models.
        comparison_df: DataFrame of model comparison results.
    """
    logger.info(f"Training multiple models (task_type={task_type})")

    # Determine which models to train.
    if models_to_try is None:
        if task_type == 'classification':
            models_to_try = list(CLASSIFICATION_MODELS.keys())
        else:
            models_to_try = list(REGRESSION_MODELS.keys())

    models_dict = {}
    cv_scores_dict = {}  # Store cross-validation scores for each model.
    results = []

    for model_name in models_to_try:
        try:
            logger.info(f"Training {model_name}...")

            # Train the model.
            model, train_info = train_single_model(
                X_train, y_train, model_name, task_type, cv_folds=cv_folds
            )

            # Evaluate on the test set.
            y_pred = model.predict(X_test)

            if task_type == 'classification':
                from sklearn.metrics import accuracy_score, f1_score
                test_score = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                result = {
                    'model': model_name,
                    'cv_mean': train_info['cv_mean'],
                    'cv_std': train_info['cv_std'],
                    'test_accuracy': test_score,
                    'test_f1': f1,
                    'training_time': train_info['training_time']
                }
            else:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                test_r2 = r2_score(y_test, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_mae = mean_absolute_error(y_test, y_pred)

                result = {
                    'model': model_name,
                    'cv_mean': train_info['cv_mean'],
                    'cv_std': train_info['cv_std'],
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'training_time': train_info['training_time']
                }

            results.append(result)
            models_dict[model_name] = model
            cv_scores_dict[model_name] = np.array(train_info['cv_scores'])  # Store cross-validation scores.

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {str(e)}")
            continue

    # Create the comparison DataFrame.
    comparison_df = pd.DataFrame(results)

    # Sort the results.
    if task_type == 'classification':
        comparison_df = comparison_df.sort_values('test_accuracy', ascending=False)
    else:
        comparison_df = comparison_df.sort_values('test_r2', ascending=False)

    logger.info(f"Trained {len(models_dict)} models successfully")

    return models_dict, comparison_df, cv_scores_dict


# ============================================================================
# MODEL EVALUATION FUNCTIONS
# ============================================================================

def evaluate_classification(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate a classification model.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test targets.
        class_names: List of class names.

    Returns:
        metrics: Dictionary containing accuracy, precision, recall, f1, roc_auc, and confusion_matrix.
    """
    logger.info("Evaluating classification model")

    y_pred = model.predict(X_test)

    # Basic metrics.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Compute ROC AUC when probability prediction is supported.
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = None
    except Exception as e:
        logger.warning(f"Could not compute ROC AUC: {str(e)}")
        roc_auc = None

    # Confusion matrix.
    cm = confusion_matrix(y_test, y_pred)

    # Classification report.
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'n_samples': len(y_test),
        'n_classes': len(np.unique(y_test))
    }

    logger.info(f"Classification metrics: accuracy={accuracy:.4f}, f1={f1:.4f}")

    return metrics


def evaluate_regression(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Evaluate a regression model.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test targets.

    Returns:
        metrics: Dictionary containing MSE, RMSE, MAE, R2, and adjusted_R2.
    """
    logger.info("Evaluating regression model")

    y_pred = model.predict(X_test)

    # Compute metrics.
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Compute adjusted R-squared.
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Mean absolute percentage error (MAPE).
    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
    except:
        # MAPE is undefined or unstable when y_test contains zeros.
        mape = None

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'adjusted_r2': adjusted_r2,
        'mape': mape,
        'n_samples': len(y_test)
    }

    logger.info(f"Regression metrics: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    return metrics


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> plt.Figure:
    """
    Plot feature importances.

    Args:
        model: Trained model.
        feature_names: List of feature names.
        top_n: Number of highest-ranked features to display.

    Returns:
        fig: Matplotlib Figure object.
    """
    # Get feature importances.
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        logger.warning("Model does not have feature_importances_ or coef_ attribute")
        return None

    # Create the DataFrame.
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    # Draw the plot.
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    colors = sns.color_palette("viridis", len(importance_df))
    ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.

    Returns:
        fig: Matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Infer class names from the data when they are not provided.
    if class_names is None:
        class_names = sorted(set(y_true) | set(y_pred))
        class_names = [str(c) for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    return fig


def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> plt.Figure:
    """
    Compare ROC curves for multiple models.

    Args:
        models_dict: {model_name: model_object}
        X_test: Test features.
        y_test: Test targets.

    Returns:
        fig: Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    colors = sns.color_palette("husl", len(models_dict))

    for (model_name, model), color in zip(models_dict.items(), colors):
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)

                # Binary classification.
                if y_proba.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, lw=2,
                           label=f'{model_name} (AUC = {roc_auc:.3f})')
        except Exception as e:
            logger.warning(f"Could not plot ROC curve for {model_name}: {str(e)}")
            continue

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> plt.Figure:
    """
    Plot predicted versus observed values for regression.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        fig: Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Scatter plot.
    ax.scatter(y_true, y_pred, alpha=0.5, s=50, edgecolors='k', linewidths=0.5)

    # Identity line (y = x).
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # Compute R-squared.
    r2 = r2_score(y_true, y_pred)

    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'Prediction vs Actual (R² = {r2:.4f})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> plt.Figure:
    """
    Plot regression residuals.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        fig: Matplotlib Figure object.
    """
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Residual scatter plot.
    ax1.scatter(y_pred, residuals, alpha=0.5, s=50, edgecolors='k', linewidths=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Residual histogram.
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig


def plot_learning_curves(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5
) -> plt.Figure:
    """
    Plot learning curves.

    Show training and validation performance as the number of samples increases.

    Args:
        model: Trained model.
        X: Feature array.
        y: Target array.
        cv_folds: Number of cross-validation folds.

    Returns:
        fig: Matplotlib Figure object.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv_folds, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='blue')

    ax.plot(train_sizes, val_mean, 'o-', color='green', label='Validation Score', linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='green')

    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_cv_scores(
    cv_scores: np.ndarray,
    metric_name: str = 'Score'
) -> plt.Figure:
    """
    Plot the distribution of cross-validation scores.

    Args:
        cv_scores: Array of cross-validation scores.
        metric_name: Metric name.

    Returns:
        fig: Matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Bar chart.
    ax1.bar(range(1, len(cv_scores) + 1), cv_scores, color='steelblue', edgecolor='black')
    ax1.axhline(y=cv_scores.mean(), color='r', linestyle='--', lw=2, label=f'Mean = {cv_scores.mean():.4f}')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel(metric_name, fontsize=12)
    ax1.set_title('Cross-Validation Scores by Fold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Box plot.
    ax2.boxplot([cv_scores], labels=['CV Scores'], widths=0.5)
    ax2.set_ylabel(metric_name, fontsize=12)
    ax2.set_title(f'CV Score Distribution\nMean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig


# ============================================================================
# MODEL PERSISTENCE FUNCTIONS
# ============================================================================

def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict] = None,
    feature_names: Optional[List[str]] = None,
    scaler: Optional[Any] = None,
    encoder: Optional[Any] = None,
    task_type: Optional[str] = None
) -> bool:
    """
    Save a model and associated information.

    Args:
        model: Trained model.
        filepath: Output path (.pkl or .joblib).
        metadata: Metadata such as performance metrics and training time.
        feature_names: List of feature names.
        scaler: Feature scaler.
        encoder: Encoder.
        task_type: 'classification' or 'regression'

    Returns:
        success: Whether the model was saved successfully.
    """
    try:
        logger.info(f"Saving model to {filepath}")

        # Create the model package.
        model_package = {
            'model': model,
            'task_type': task_type,
            'feature_names': feature_names,
            'scaler': scaler,
            'encoder': encoder,
            'metadata': metadata or {},
            'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Ensure the directory exists.
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save the model package.
        joblib.dump(model_package, filepath, compress=3)

        logger.info(f"Model saved successfully to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        return False


def load_model(filepath: str) -> Tuple[Any, Dict]:
    """
    Load a model and its metadata.

    Args:
        filepath: Path to the model file.

    Returns:
        model: Loaded model.
        model_package: Complete model package dictionary.
    """
    try:
        logger.info(f"Loading model from {filepath}")

        model_package = joblib.load(filepath)

        model = model_package.get('model')
        if model is None:
            raise ValueError("Model not found in package")

        logger.info(f"Model loaded successfully from {filepath}")

        return model, model_package

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def get_model_info(filepath: str) -> Dict:
    """
    Read model information from the serialized model package.

    Args:
        filepath: Path to the model file.

    Returns:
        info: Dictionary of model information.
    """
    try:
        model_package = joblib.load(filepath)

        info = {
            'task_type': model_package.get('task_type'),
            'feature_names': model_package.get('feature_names'),
            'n_features': len(model_package.get('feature_names', [])),
            'has_scaler': model_package.get('scaler') is not None,
            'has_encoder': model_package.get('encoder') is not None,
            'save_date': model_package.get('save_date'),
            'metadata': model_package.get('metadata', {})
        }

        return info

    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise


# ============================================================================
# HIGH-LEVEL API FUNCTIONS
# ============================================================================

def train_supervised_model(
    data_df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_type: str = 'auto',
    test_size: float = 0.2,
    model_name: str = 'random_forest',
    feature_selection: Optional[str] = None,
    feature_scaling: str = 'standard',
    hyperparameter_tuning: bool = False,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    Train a supervised model using the high-level API.

    Args:
        data_df: DataFrame containing features and a target, with the last column used by default.
        target_column: Target column name; None selects the last column.
        task_type: 'classification', 'regression', or 'auto' for automatic detection.
        test_size: Fraction of samples reserved for testing.
        model_name: Model name.
        feature_selection: Selection method (None, 'correlation', 'mutual_info', 'rfe', 'tree_based').
        feature_scaling: Scaling method ('standard', 'minmax', 'robust', or None).
        hyperparameter_tuning: Enable hyperparameter tuning.
        cv_folds: Number of cross-validation folds.
        random_state: Random seed.

    Returns:
        model: Trained model.
        results: Dictionary containing evaluation metrics, predictions, and metadata.
    """
    logger.info("Starting supervised learning pipeline")

    # Determine the target column.
    if target_column is None:
        target_column = data_df.columns[-1]
        logger.info(f"Using last column as target: {target_column}")

    # Separate features from the target.
    X = data_df.drop(columns=[target_column])
    y = data_df[target_column]

    # Detect the task type automatically.
    if task_type == 'auto':
        n_unique = y.nunique()
        if n_unique <= 20:
            task_type = 'classification'
            logger.info(f"Auto-detected task type: classification ({n_unique} classes)")
        else:
            task_type = 'regression'
            logger.info(f"Auto-detected task type: regression")

    # Feature selection.
    if feature_selection is not None:
        logger.info(f"Applying feature selection: {feature_selection}")
        if feature_selection == 'correlation':
            X, fs_info = select_features_correlation(X, y, threshold=0.1)
        elif feature_selection == 'mutual_info':
            X, fs_info = select_features_mutual_info(X, y, task_type=task_type)
        elif feature_selection == 'rfe':
            X, fs_info = select_features_rfe(X, y, task_type=task_type)
        elif feature_selection == 'tree_based':
            X, fs_info = select_features_tree_based(X, y, task_type=task_type)
        else:
            fs_info = None
    else:
        fs_info = None

    feature_names = X.columns.tolist()

    # Split into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Scale features.
    scaler = None
    if feature_scaling is not None:
        X_train_scaled, scaler = scale_features(X_train, method=feature_scaling, fit=True)
        X_test_scaled, _ = scale_features(X_test, method=feature_scaling, scaler=scaler, fit=False)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # Train the model.
    if hyperparameter_tuning:
        logger.info("Training with hyperparameter tuning")
        model, train_info = hyperparameter_tuning(
            X_train_scaled, y_train, model_name, task_type, cv_folds=cv_folds
        )
    else:
        logger.info("Training with default parameters")
        model, train_info = train_single_model(
            X_train_scaled, y_train, model_name, task_type, cv_folds=cv_folds
        )

    # Evaluate the model.
    if task_type == 'classification':
        metrics = evaluate_classification(model, X_test_scaled, y_test)
    else:
        metrics = evaluate_regression(model, X_test_scaled, y_test)

    # Generate predictions.
    y_pred = model.predict(X_test_scaled)

    # Compute feature statistics for virtual sample generation.
    feature_stats = {}
    for col in feature_names:
        col_data = X[col]
        n_unique = col_data.nunique()
        # Treat features as categorical if they have at most 10 unique values or fewer than 5% of the sample count.
        is_categorical = n_unique <= 10 or n_unique < len(col_data) * 0.05

        if is_categorical:
            feature_stats[col] = {
                'type': 'categorical',
                'unique_values': sorted(col_data.unique().tolist()),
                'n_unique': n_unique
            }
        else:
            feature_stats[col] = {
                'type': 'continuous',
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'n_unique': n_unique
            }

    # Assemble the results.
    results = {
        'model': model,
        'task_type': task_type,
        'model_name': model_name,
        'feature_names': feature_names,
        'feature_stats': feature_stats,
        'scaler': scaler,
        'metrics': metrics,
        'predictions': y_pred,
        'y_test': y_test,
        'train_info': train_info,
        'feature_selection_info': fs_info,
        'n_features': len(feature_names),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test)
    }

    logger.info("Supervised learning pipeline completed successfully")

    return model, results


def compare_models_automl(
    data_df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_type: str = 'auto',
    models_to_compare: Optional[List[str]] = None,
    test_size: float = 0.2,
    feature_selection: Optional[str] = None,
    feature_scaling: str = 'standard',
    cv_folds: int = 5,
    tune_best: bool = True,
    random_state: int = 42
) -> Tuple[Any, pd.DataFrame, Dict]:
    """
    Compare multiple models automatically and select the best performer.

    Args:
        data_df: DataFrame containing features and the target.
        target_column: Target column name; None selects the last column.
        task_type: 'classification', 'regression', 'auto'
        models_to_compare: List of models to compare; None includes all supported models.
        test_size: Fraction of samples reserved for testing.
        feature_selection: Feature selection method.
        feature_scaling: Feature scaling method.
        cv_folds: Number of cross-validation folds.
        tune_best: Tune the best model's hyperparameters.
        random_state: Random seed.

    Returns:
        best_model: Best-performing model.
        comparison_df: DataFrame of model comparison results.
        all_results: Dictionary of detailed results for all models.
    """
    logger.info("Starting AutoML pipeline")

    # Determine the target column.
    if target_column is None:
        target_column = data_df.columns[-1]
        logger.info(f"Using last column as target: {target_column}")

    # Separate features from the target.
    X = data_df.drop(columns=[target_column])
    y = data_df[target_column]

    # Detect the task type automatically.
    if task_type == 'auto':
        n_unique = y.nunique()
        if n_unique <= 20:
            task_type = 'classification'
            logger.info(f"Auto-detected task type: classification ({n_unique} classes)")
        else:
            task_type = 'regression'
            logger.info(f"Auto-detected task type: regression")

    # Feature selection.
    if feature_selection is not None:
        logger.info(f"Applying feature selection: {feature_selection}")
        if feature_selection == 'correlation':
            X, fs_info = select_features_correlation(X, y, threshold=0.1)
        elif feature_selection == 'mutual_info':
            X, fs_info = select_features_mutual_info(X, y, task_type=task_type)
        elif feature_selection == 'rfe':
            X, fs_info = select_features_rfe(X, y, task_type=task_type)
        elif feature_selection == 'tree_based':
            X, fs_info = select_features_tree_based(X, y, task_type=task_type)
    else:
        fs_info = None

    feature_names = X.columns.tolist()

    # Split into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Scale features.
    scaler = None
    if feature_scaling is not None:
        X_train_scaled, scaler = scale_features(X_train, method=feature_scaling, fit=True)
        X_test_scaled, _ = scale_features(X_test, method=feature_scaling, scaler=scaler, fit=False)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # Train multiple models.
    models_dict, comparison_df, cv_scores_dict = train_multiple_models(
        X_train_scaled, y_train, X_test_scaled, y_test,
        task_type=task_type, models_to_try=models_to_compare, cv_folds=cv_folds
    )

    # Select the best model.
    if task_type == 'classification':
        best_model_name = comparison_df.iloc[0]['model']
    else:
        best_model_name = comparison_df.iloc[0]['model']

    best_model = models_dict[best_model_name]

    logger.info(f"Best model: {best_model_name}")

    # Tune hyperparameters of the best model.
    if tune_best:
        logger.info(f"Tuning best model: {best_model_name}")
        best_model, tune_info = hyperparameter_tuning(
            X_train_scaled, y_train, best_model_name, task_type, cv_folds=cv_folds
        )
    else:
        tune_info = None

    # Evaluate the best model.
    if task_type == 'classification':
        metrics = evaluate_classification(best_model, X_test_scaled, y_test)
    else:
        metrics = evaluate_regression(best_model, X_test_scaled, y_test)

    # Generate prediction results.
    predictions = best_model.predict(X_test_scaled)

    # Compute feature statistics for virtual sample generation.
    feature_stats = {}
    for col in feature_names:
        col_data = X[col]
        n_unique = col_data.nunique()
        is_categorical = n_unique <= 10 or n_unique < len(col_data) * 0.05

        if is_categorical:
            feature_stats[col] = {
                'type': 'categorical',
                'unique_values': sorted(col_data.unique().tolist()),
                'n_unique': n_unique
            }
        else:
            feature_stats[col] = {
                'type': 'continuous',
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'n_unique': n_unique
            }

    # Assemble the results.
    all_results = {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'task_type': task_type,
        'feature_names': feature_names,
        'feature_stats': feature_stats,
        'scaler': scaler,
        'metrics': metrics,
        'comparison_df': comparison_df,
        'all_models': models_dict,
        'tune_info': tune_info,
        'feature_selection_info': fs_info,
        'n_features': len(feature_names),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'y_test': y_test,
        'predictions': predictions,
        'cv_scores_dict': cv_scores_dict,
        'X_train_scaled': X_train_scaled
    }

    logger.info("AutoML pipeline completed successfully")

    return best_model, comparison_df, all_results


def plot_shap_analysis(
    model: Any,
    X_data: np.ndarray,
    feature_names: List[str],
    max_display: int = 20
) -> plt.Figure:
    """
    Generate a SHAP explanation plot.

    Args:
        model: Trained model.
        X_data: Feature data used to calculate SHAP values.
        feature_names: List of feature names.
        max_display: Maximum number of features to display.

    Returns:
        fig: Matplotlib Figure object.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP library not installed. Please install with: pip install shap")
        # Return a figure explaining the unavailable dependency.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'SHAP library not installed\nPlease install with: pip install shap',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    try:
        # Create a SHAP explainer.
        explainer = shap.Explainer(model, X_data)
        shap_values = explainer(X_data)

        # Create the figure.
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        # Generate the summary plot.
        shap.summary_plot(shap_values, X_data, feature_names=feature_names,
                         max_display=max_display, show=False)

        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Failed to generate SHAP analysis: {str(e)}")
        # Return a figure containing the error message.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Failed to generate SHAP analysis\nError: {str(e)}',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig


def plot_cv_scores(
    cv_scores_dict: Dict[str, np.ndarray],
    metric_name: str = 'Score'
) -> plt.Figure:
    """
    Visualize the score for each cross-validation fold.

    Args:
        cv_scores_dict: Dictionary mapping model names to cross-validation score arrays.
        metric_name: Metric name.

    Returns:
        fig: Matplotlib Figure object.
    """
    if not cv_scores_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No cross-validation scores available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    # Prepare the data.
    models = list(cv_scores_dict.keys())
    n_models = len(models)
    n_folds = len(cv_scores_dict[models[0]])

    # Set x-axis positions.
    x = np.arange(n_folds)
    width = 0.8 / n_models

    # Draw a bar series for each model.
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))

    for i, (model_name, scores) in enumerate(cv_scores_dict.items()):
        offset = (i - n_models/2) * width + width/2
        bars = ax.bar(x + offset, scores, width, label=model_name,
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

        # Display values above the bars.
        for j, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=8, rotation=0)

    # Configure the plot.
    ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'Cross-Validation {metric_name} by Fold', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    return fig

