"""
Preprocessing Module
====================
Handles data loading, cleaning, and preprocessing for wellbore geology data.

Author: Md Ashraf
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    data_dir: Union[str, Path],
    train_file: str = "train.csv",
    test_file: str = "test.csv",
    sample_submission_file: str = "sample_submission.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample submission CSVs.

    Args:
        data_dir: Path to the directory containing the data files.
        train_file: Name of the training CSV file.
        test_file: Name of the test CSV file.
        sample_submission_file: Name of the sample submission CSV file.

    Returns:
        Tuple of (train_df, test_df, sample_submission_df).
    """
    data_dir = Path(data_dir)
    logger.info("Loading data from %s", data_dir)

    train_path = data_dir / train_file
    test_path = data_dir / test_file
    sub_path = data_dir / sample_submission_file

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sub_path)

    logger.info("Train shape: %s | Test shape: %s", train.shape, test.shape)
    return train, test, sample_sub


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    fill_value: Optional[float] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Handle missing values in the dataframe.

    Args:
        df: Input dataframe.
        strategy: Imputation strategy - 'median', 'mean', 'mode', 'ffill',
                  'bfill', 'interpolate', or 'constant'.
        fill_value: Value to use when strategy is 'constant'.
        group_col: Optional column to group by before imputation (e.g., well_id).

    Returns:
        Dataframe with missing values handled.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    missing_before = df[numeric_cols].isnull().sum().sum()
    logger.info("Missing values before imputation: %d", missing_before)

    if strategy == "interpolate":
        if group_col:
            df[numeric_cols] = (
                df.groupby(group_col)[numeric_cols]
                .transform(lambda x: x.interpolate(method="linear", limit_direction="both"))
            )
        else:
            df[numeric_cols] = df[numeric_cols].interpolate(
                method="linear", limit_direction="both"
            )
    elif strategy in ("ffill", "bfill"):
        if group_col:
            if strategy == "ffill":
                df[numeric_cols] = df.groupby(group_col)[numeric_cols].transform(
                    lambda x: x.ffill()
                )
            else:
                df[numeric_cols] = df.groupby(group_col)[numeric_cols].transform(
                    lambda x: x.bfill()
                )
        else:
            if strategy == "ffill":
                df[numeric_cols] = df[numeric_cols].ffill()
            else:
                df[numeric_cols] = df[numeric_cols].bfill()
    elif strategy == "constant":
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)
    else:
        if group_col:
            for col in numeric_cols:
                if strategy == "median":
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                elif strategy == "mean":
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
        else:
            for col in numeric_cols:
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])

    missing_after = df[numeric_cols].isnull().sum().sum()
    logger.info("Missing values after imputation: %d", missing_after)
    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list,
    factor: float = 3.0,
) -> pd.DataFrame:
    """Clip outliers using the IQR method.

    Args:
        df: Input dataframe.
        columns: List of column names to process.
        factor: IQR multiplier for clipping bounds (default 3.0 for wellbore data).

    Returns:
        Dataframe with outliers clipped.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found, skipping outlier removal.", col)
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list,
    scaler_type: str = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fit a scaler on train data and transform both train and test.

    Args:
        train: Training dataframe.
        test: Test dataframe.
        feature_cols: Columns to scale.
        scaler_type: Type of scaler - 'standard', 'robust', or 'minmax'.

    Returns:
        Tuple of (scaled_train, scaled_test, fitted_scaler).
    """
    scaler_map = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
    }
    if scaler_type not in scaler_map:
        raise ValueError(f"scaler_type must be one of {list(scaler_map.keys())}")

    scaler = scaler_map[scaler_type]
    train = train.copy()
    test = test.copy()

    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    logger.info("Scaled %d features using %s scaler.", len(feature_cols), scaler_type)
    return train, test, scaler


# ---------------------------------------------------------------------------
# Type casting and memory reduction
# ---------------------------------------------------------------------------

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce dataframe memory usage by downcasting numeric types.

    Args:
        df: Input dataframe.
        verbose: Whether to log the memory reduction.

    Returns:
        Memory-optimized dataframe.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    df = df.copy()

    for col in df.columns:
        col_type = df[col].dtype

        if not pd.api.types.is_numeric_dtype(col_type):
            continue

        c_min = float(df[col].min())
        c_max = float(df[col].max())

        if pd.api.types.is_integer_dtype(col_type):
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if c_min > float(np.finfo(np.float32).min) and c_max < float(np.finfo(np.float32).max):
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        logger.info(
            "Memory reduced from %.2f MB to %.2f MB (%.1f%% reduction)",
            start_mem,
            end_mem,
            100 * (start_mem - end_mem) / start_mem,
        )
    return df
