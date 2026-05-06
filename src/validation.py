"""
Validation Module
=================
Cross-validation strategies and evaluation metrics for wellbore geology
prediction. Uses GroupKFold to ensure well-level data leakage prevention.

Author: Md Ashraf
"""

import logging
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE score (lower is better).
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAE score (lower is better).
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R².

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        R² score (higher is better, max 1.0).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary mapping metric names to scores.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def get_cv_splits(
    df: pd.DataFrame,
    strategy: str = "group_kfold",
    n_folds: int = 5,
    group_col: Optional[str] = "well_id",
    target_col: Optional[str] = None,
    shuffle: bool = True,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate cross-validation fold indices.

    Args:
        df: Input dataframe.
        strategy: CV strategy - 'group_kfold', 'kfold', or 'stratified_kfold'.
        n_folds: Number of folds.
        group_col: Column used as group identifier (required for 'group_kfold').
        target_col: Target column (required for 'stratified_kfold').
        shuffle: Whether to shuffle the data (used for kfold/stratified).
        random_state: Random seed.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    if strategy == "group_kfold":
        if group_col is None or group_col not in df.columns:
            raise ValueError("group_col is required for GroupKFold strategy.")
        cv = GroupKFold(n_splits=n_folds)
        groups = df[group_col].values
        splits = list(cv.split(df, groups=groups))
        wells = df[group_col].unique()
        logger.info(
            "GroupKFold: %d folds across %d unique wells.", n_folds, len(wells)
        )
    elif strategy == "kfold":
        cv = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        splits = list(cv.split(df))
        logger.info("KFold: %d folds.", n_folds)
    elif strategy == "stratified_kfold":
        if target_col is None or target_col not in df.columns:
            raise ValueError("target_col is required for StratifiedKFold strategy.")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        splits = list(cv.split(df, df[target_col]))
        logger.info("StratifiedKFold: %d folds.", n_folds)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")

    return splits


def cross_validate_model(
    model,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    fit_kwargs: Optional[dict] = None,
) -> Tuple[List[float], np.ndarray]:
    """Run cross-validation for any sklearn-compatible model.

    Args:
        model: An sklearn-compatible estimator (must implement fit/predict).
        df: Input dataframe.
        feature_cols: Feature column names.
        target_col: Target column name.
        cv_splits: List of (train_idx, val_idx) tuples from get_cv_splits().
        fit_kwargs: Extra keyword arguments passed to model.fit().

    Returns:
        Tuple of (fold_rmse_list, oof_predictions_array).
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    oof_preds = np.zeros(len(df))
    fold_scores: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col]
        X_val = df.iloc[val_idx][feature_cols]
        y_val = df.iloc[val_idx][target_col]

        model.fit(X_train, y_train, **fit_kwargs)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        score = rmse(y_val.values, preds)
        fold_scores.append(score)
        logger.info("Fold %d | RMSE: %.6f", fold, score)

    mean_rmse = float(np.mean(fold_scores))
    std_rmse = float(np.std(fold_scores))
    logger.info("CV RMSE: %.6f ± %.6f", mean_rmse, std_rmse)

    return fold_scores, oof_preds


def print_cv_summary(fold_scores: List[float]) -> None:
    """Print a formatted cross-validation summary table.

    Args:
        fold_scores: List of per-fold RMSE scores.
    """
    header = f"{'Fold':<6} {'RMSE':>10}"
    separator = "-" * 20
    print(separator)
    print(header)
    print(separator)
    for i, score in enumerate(fold_scores, start=1):
        print(f"{i:<6} {score:>10.6f}")
    print(separator)
    print(f"{'Mean':<6} {np.mean(fold_scores):>10.6f}")
    print(f"{'Std':<6} {np.std(fold_scores):>10.6f}")
    print(separator)
