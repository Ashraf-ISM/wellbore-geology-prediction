"""
Ensemble Module
===============
Stacking, blending, and weighted averaging of multiple model predictions
for improved accuracy in wellbore geology prediction.

Author: Md Ashraf
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weighted average blending
# ---------------------------------------------------------------------------

def weighted_average(
    predictions: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Compute weighted average of multiple model predictions.

    Args:
        predictions: Dict mapping model names to prediction arrays.
        weights: Dict mapping model names to weight values.
                 If None, equal weights are used.

    Returns:
        Weighted average prediction array.
    """
    model_names = list(predictions.keys())

    if weights is None:
        weights = {name: 1.0 / len(model_names) for name in model_names}

    total_weight = sum(weights.values())
    blended = np.zeros(len(next(iter(predictions.values()))))

    for name, preds in predictions.items():
        w = weights.get(name, 0.0) / total_weight
        blended += w * np.asarray(preds)
        logger.debug("Model '%s' | weight=%.4f", name, w)

    logger.info("Blended %d models with weights: %s", len(model_names), weights)
    return blended


def rmse_weighted_blend(
    predictions: Dict[str, np.ndarray],
    oof_scores: Dict[str, float],
    power: float = 2.0,
) -> np.ndarray:
    """Blend predictions by weighting inversely proportional to OOF RMSE.

    Models with lower OOF RMSE receive higher weight.

    Args:
        predictions: Dict mapping model names to prediction arrays.
        oof_scores: Dict mapping model names to OOF RMSE scores.
        power: Exponent applied to inverse RMSE for weight sharpening.

    Returns:
        Inverse-RMSE-weighted prediction array.
    """
    inv_scores = {name: 1.0 / (score ** power + 1e-8) for name, score in oof_scores.items()}
    return weighted_average(predictions, weights=inv_scores)


# ---------------------------------------------------------------------------
# Stacking (meta-learner)
# ---------------------------------------------------------------------------

def build_stack_features(
    oof_predictions: Dict[str, np.ndarray],
    test_predictions: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build level-1 feature matrices from OOF and test predictions.

    Args:
        oof_predictions: Dict mapping model names to OOF prediction arrays.
        test_predictions: Dict mapping model names to test prediction arrays.

    Returns:
        Tuple of (oof_features_df, test_features_df).
    """
    oof_df = pd.DataFrame(oof_predictions)
    test_df = pd.DataFrame(test_predictions)
    logger.info("Stack OOF features shape: %s | Test: %s", oof_df.shape, test_df.shape)
    return oof_df, test_df


def train_meta_learner(
    oof_features: pd.DataFrame,
    y_train: np.ndarray,
    test_features: pd.DataFrame,
    meta_model: Optional[Any] = None,
    cv: int = 5,
) -> Tuple[np.ndarray, Any]:
    """Train a meta-learner on OOF predictions (stacking).

    Args:
        oof_features: Level-1 OOF feature matrix.
        y_train: Ground truth training labels.
        test_features: Level-1 test feature matrix.
        meta_model: A fitted/unfitted sklearn estimator. Defaults to Ridge.
        cv: Number of cross-validation folds for meta-learning.

    Returns:
        Tuple of (final_test_predictions, fitted_meta_model).
    """
    if meta_model is None:
        meta_model = Ridge(alpha=1.0)

    meta_model.fit(oof_features, y_train)
    test_preds = meta_model.predict(test_features)

    logger.info(
        "Meta-learner: %s | Test prediction shape: %s",
        meta_model.__class__.__name__,
        test_preds.shape,
    )
    return test_preds, meta_model


# ---------------------------------------------------------------------------
# Rank averaging
# ---------------------------------------------------------------------------

def rank_average(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """Average the rank-transformed predictions across models.

    Rank averaging is rank-invariant and can outperform raw value averaging
    when models have different prediction scales.

    Args:
        predictions: Dict mapping model names to prediction arrays.

    Returns:
        Rank-averaged prediction array.
    """
    from scipy.stats import rankdata

    ranked = [rankdata(preds) for preds in predictions.values()]
    avg_ranks = np.mean(ranked, axis=0)
    logger.info("Rank-averaged %d models.", len(predictions))
    return avg_ranks
