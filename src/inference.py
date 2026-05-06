"""
Inference Module
================
Generates predictions on test data using trained models and creates
the final Kaggle submission CSV.

Author: Md Ashraf
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    models: List[Any],
    X_test: pd.DataFrame,
    aggregation: str = "mean",
) -> np.ndarray:
    """Generate predictions from an ensemble of trained models.

    Args:
        models: List of fitted model objects (must implement .predict()).
        X_test: Test feature dataframe.
        aggregation: How to aggregate predictions - 'mean' or 'median'.

    Returns:
        Aggregated prediction array of shape (n_samples,).
    """
    if not models:
        raise ValueError("No models provided for prediction.")

    preds_list = [model.predict(X_test) for model in models]
    preds_array = np.vstack(preds_list)  # shape: (n_models, n_samples)

    if aggregation == "mean":
        return preds_array.mean(axis=0)
    elif aggregation == "median":
        return np.median(preds_array, axis=0)
    else:
        raise ValueError(f"aggregation must be 'mean' or 'median', got '{aggregation}'")


def predict_with_uncertainty(
    models: List[Any],
    X_test: pd.DataFrame,
) -> tuple:
    """Predict with uncertainty estimates from fold ensemble variance.

    Args:
        models: List of fitted model objects.
        X_test: Test feature dataframe.

    Returns:
        Tuple of (mean_predictions, std_predictions).
    """
    preds_list = [model.predict(X_test) for model in models]
    preds_array = np.vstack(preds_list)
    return preds_array.mean(axis=0), preds_array.std(axis=0)


# ---------------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------------

def make_submission(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    id_col: str,
    target_col: str,
    output_path: Union[str, Path],
    round_decimals: Optional[int] = 4,
) -> pd.DataFrame:
    """Create and save a Kaggle submission CSV.

    Args:
        test_df: Test dataframe containing the ID column.
        predictions: Predicted values.
        id_col: Name of the ID column in test_df.
        target_col: Name of the target column in the submission.
        output_path: Path (including filename) to save the submission CSV.
        round_decimals: Number of decimal places to round predictions to.
                        Pass None to skip rounding.

    Returns:
        Submission dataframe.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({
        id_col: test_df[id_col].values,
        target_col: predictions,
    })

    if round_decimals is not None:
        submission[target_col] = submission[target_col].round(round_decimals)

    submission.to_csv(output_path, index=False)
    logger.info("Submission saved to %s | Shape: %s", output_path, submission.shape)
    return submission


def load_and_predict(
    model_paths: List[Union[str, Path]],
    X_test: pd.DataFrame,
    aggregation: str = "mean",
) -> np.ndarray:
    """Load saved models from disk and run predictions.

    Args:
        model_paths: List of paths to saved model files (.pkl or .joblib).
        X_test: Test feature dataframe.
        aggregation: Aggregation strategy ('mean' or 'median').

    Returns:
        Aggregated prediction array.
    """
    import joblib

    models = []
    for path in model_paths:
        model = joblib.load(path)
        models.append(model)
        logger.info("Loaded model from %s", path)

    return predict(models, X_test, aggregation=aggregation)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def clip_predictions(
    predictions: np.ndarray,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> np.ndarray:
    """Clip predictions to a valid range.

    Args:
        predictions: Raw prediction array.
        lower: Minimum allowed value.
        upper: Maximum allowed value.

    Returns:
        Clipped prediction array.
    """
    return np.clip(predictions, a_min=lower, a_max=upper)
