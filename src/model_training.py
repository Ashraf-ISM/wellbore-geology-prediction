"""
Model Training Module
=====================
Training pipelines for LightGBM, XGBoost, CatBoost, and neural network
models with Optuna hyperparameter optimisation support.

Author: Md Ashraf
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LightGBM trainer
# ---------------------------------------------------------------------------

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 100,
    verbose_eval: int = 200,
) -> Any:
    """Train a single LightGBM model fold.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        params: LightGBM hyperparameters. Uses sensible defaults if None.
        early_stopping_rounds: Number of rounds without improvement before stopping.
        verbose_eval: Logging frequency (0 to disable).

    Returns:
        Fitted LightGBM Booster.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    callbacks = [lgb.log_evaluation(period=verbose_eval)]
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    logger.info("LightGBM best iteration: %d", model.best_iteration_)
    return model


def train_lightgbm_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 100,
    save_dir: Optional[str] = None,
) -> Tuple[List[Any], np.ndarray, List[float]]:
    """Train LightGBM with cross-validation.

    Args:
        df: Full training dataframe.
        feature_cols: Feature column names.
        target_col: Target column name.
        cv_splits: List of (train_idx, val_idx) from get_cv_splits().
        params: Optional LightGBM hyperparameters.
        early_stopping_rounds: Early stopping patience.
        save_dir: Directory to save fold models. If None, models are not saved.

    Returns:
        Tuple of (models_list, oof_predictions, fold_rmse_scores).
    """
    from .validation import rmse

    models: List[Any] = []
    oof_preds = np.zeros(len(df))
    fold_scores: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        logger.info("=" * 50)
        logger.info("Training Fold %d / %d", fold, len(cv_splits))
        logger.info("=" * 50)

        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col]
        X_val = df.iloc[val_idx][feature_cols]
        y_val = df.iloc[val_idx][target_col]

        model = train_lightgbm(
            X_train, y_train, X_val, y_val,
            params=params,
            early_stopping_rounds=early_stopping_rounds,
        )

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        score = rmse(y_val.values, preds)
        fold_scores.append(score)
        logger.info("Fold %d RMSE: %.6f", fold, score)

        models.append(model)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            model_path = Path(save_dir) / f"lgbm_fold{fold}.pkl"
            joblib.dump(model, model_path)
            logger.info("Model saved to %s", model_path)

    logger.info("OOF RMSE: %.6f ± %.6f", np.mean(fold_scores), np.std(fold_scores))
    return models, oof_preds, fold_scores


# ---------------------------------------------------------------------------
# XGBoost trainer
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 100,
    verbose_eval: int = 200,
) -> Any:
    """Train a single XGBoost model fold.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        params: XGBoost hyperparameters.
        early_stopping_rounds: Early stopping patience.
        verbose_eval: Logging frequency.

    Returns:
        Fitted XGBoost model.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    default_params: Dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose_eval,
    )
    return model


# ---------------------------------------------------------------------------
# CatBoost trainer
# ---------------------------------------------------------------------------

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 100,
) -> Any:
    """Train a single CatBoost model fold.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        params: CatBoost hyperparameters.
        early_stopping_rounds: Early stopping patience.

    Returns:
        Fitted CatBoost model.
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("catboost is not installed. Run: pip install catboost")

    default_params: Dict[str, Any] = {
        "loss_function": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "od_type": "Iter",
        "od_wait": early_stopping_rounds,
        "verbose": 200,
    }
    if params:
        default_params.update(params)

    model = CatBoostRegressor(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    return model


# ---------------------------------------------------------------------------
# Optuna hyperparameter optimisation
# ---------------------------------------------------------------------------

def run_optuna_lgbm(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 50,
    timeout: int = 3600,
    study_name: str = "lgbm_optuna",
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter search for LightGBM.

    Args:
        df: Training dataframe.
        feature_cols: Feature column names.
        target_col: Target column name.
        cv_splits: Cross-validation splits.
        n_trials: Number of Optuna trials.
        timeout: Maximum search time in seconds.
        study_name: Optuna study name.
        storage: Optional database URI for persistent storage.

    Returns:
        Dictionary of best hyperparameters found.
    """
    try:
        import optuna
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(f"Required package not installed: {e}")

    from .validation import rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        fold_scores: List[float] = []
        for train_idx, val_idx in cv_splits:
            X_tr = df.iloc[train_idx][feature_cols]
            y_tr = df.iloc[train_idx][target_col]
            X_vl = df.iloc[val_idx][feature_cols]
            y_vl = df.iloc[val_idx][target_col]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            preds = model.predict(X_vl)
            fold_scores.append(rmse(y_vl.values, preds))

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best_params = study.best_params
    logger.info("Best RMSE: %.6f", study.best_value)
    logger.info("Best params: %s", best_params)
    return best_params
