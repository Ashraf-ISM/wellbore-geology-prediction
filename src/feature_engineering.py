"""
Feature Engineering Module
===========================
Generates domain-driven features for wellbore geology prediction,
including rolling statistics, lag features, depth-based features,
drilling mechanics derived quantities, and petrophysical ratios.

Author: Md Ashraf
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rolling / window features
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    windows: List[int],
    group_col: Optional[str] = None,
    stats: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add rolling mean, std, min, and max for specified features and windows.

    Args:
        df: Input dataframe (sorted by depth within each well).
        feature_cols: Columns to compute rolling statistics for.
        windows: List of window sizes (number of depth rows).
        group_col: Optional group column (e.g., well_id) for per-well rolling.
        stats: List of statistics to compute. Defaults to ['mean', 'std', 'min', 'max'].

    Returns:
        Dataframe with new rolling feature columns appended.
    """
    if stats is None:
        stats = ["mean", "std", "min", "max"]

    df = df.copy()
    logger.info("Adding rolling features for windows=%s", windows)

    for col in feature_cols:
        if col not in df.columns:
            logger.warning("Column '%s' not in dataframe, skipping.", col)
            continue
        for w in windows:
            for stat in stats:
                new_col = f"{col}_rolling{w}_{stat}"
                if group_col:
                    df[new_col] = (
                        df.groupby(group_col)[col]
                        .transform(lambda x: getattr(x.rolling(w, min_periods=1), stat)())
                    )
                else:
                    df[new_col] = getattr(df[col].rolling(w, min_periods=1), stat)()
    return df


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    lag_steps: List[int],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Add lagged versions of the specified feature columns.

    Args:
        df: Input dataframe (sorted by depth within each well).
        feature_cols: Columns to create lag features for.
        lag_steps: List of lag steps (positive = backward, negative = forward).
        group_col: Optional group column for per-well lagging.

    Returns:
        Dataframe with new lag columns appended.
    """
    df = df.copy()
    logger.info("Adding lag features for steps=%s", lag_steps)

    for col in feature_cols:
        if col not in df.columns:
            logger.warning("Column '%s' not in dataframe, skipping.", col)
            continue
        for step in lag_steps:
            new_col = f"{col}_lag{step}"
            if group_col:
                df[new_col] = df.groupby(group_col)[col].transform(lambda x: x.shift(step))
            else:
                df[new_col] = df[col].shift(step)
    return df


# ---------------------------------------------------------------------------
# Depth-based features
# ---------------------------------------------------------------------------

def add_depth_features(
    df: pd.DataFrame,
    depth_col: str = "md",
    tvd_col: Optional[str] = "tvd",
    well_col: Optional[str] = "well_id",
) -> pd.DataFrame:
    """Add depth-related features.

    Args:
        df: Input dataframe.
        depth_col: Measured depth column name.
        tvd_col: True vertical depth column name.
        well_col: Well identifier column name.

    Returns:
        Dataframe with additional depth-based feature columns.
    """
    df = df.copy()

    # Normalised depth within each well
    if well_col and well_col in df.columns:
        df["depth_normalized"] = df.groupby(well_col)[depth_col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
        df["depth_from_well_top"] = df.groupby(well_col)[depth_col].transform(
            lambda x: x - x.min()
        )
    else:
        df["depth_normalized"] = (
            (df[depth_col] - df[depth_col].min())
            / (df[depth_col].max() - df[depth_col].min() + 1e-8)
        )

    # TVD ratio (horizontal departure proxy)
    if tvd_col and tvd_col in df.columns and depth_col in df.columns:
        df["tvd_md_ratio"] = df[tvd_col] / (df[depth_col] + 1e-8)
        df["horizontal_departure"] = np.sqrt(
            np.maximum(df[depth_col] ** 2 - df[tvd_col] ** 2, 0)
        )

    return df


# ---------------------------------------------------------------------------
# Drilling mechanics derived features
# ---------------------------------------------------------------------------

def add_drilling_features(
    df: pd.DataFrame,
    rop_col: str = "rop",
    wob_col: str = "wob",
    rpm_col: str = "rpm",
    torque_col: str = "torque",
    bit_diameter: float = 8.5,
) -> pd.DataFrame:
    """Add derived drilling mechanics features.

    Computed quantities:
    - Mechanical Specific Energy (MSE)
    - WOB/RPM ratio (torque proxy)
    - Bit revolutions per foot (rev/ft)

    Args:
        df: Input dataframe.
        rop_col: Rate of penetration column.
        wob_col: Weight on bit column.
        rpm_col: Rotary speed column.
        torque_col: Torque column.
        bit_diameter: Bit diameter in inches (used for MSE calculation).

    Returns:
        Dataframe with additional drilling feature columns.
    """
    df = df.copy()
    bit_area = np.pi * (bit_diameter / 2) ** 2  # in²

    required = {rop_col, wob_col, rpm_col, torque_col}
    available = required.intersection(set(df.columns))

    if rop_col in available and wob_col in available and rpm_col in available and torque_col in available:
        # MSE = WOB/A + (120π * RPM * Torque) / (A * ROP)
        df["mse"] = (
            df[wob_col] / bit_area
            + (120 * np.pi * df[rpm_col] * df[torque_col])
            / (bit_area * (df[rop_col] + 1e-8))
        )
        df["mse"] = df["mse"].clip(lower=0)

    if rop_col in available and rpm_col in available:
        df["rev_per_ft"] = df[rpm_col] / (df[rop_col] + 1e-8)

    if wob_col in available and rpm_col in available:
        df["wob_rpm_ratio"] = df[wob_col] / (df[rpm_col] + 1e-8)

    if torque_col in available and rop_col in available:
        df["specific_torque"] = df[torque_col] / (df[rop_col] + 1e-8)

    return df


# ---------------------------------------------------------------------------
# Petrophysical derived features
# ---------------------------------------------------------------------------

def add_petrophysical_features(
    df: pd.DataFrame,
    gr_col: str = "gr",
    resistivity_col: str = "resistivity",
    neutron_col: str = "neutron",
    density_col: str = "density",
    sonic_col: str = "sonic",
) -> pd.DataFrame:
    """Add petrophysical ratio features commonly used in formation evaluation.

    Computed quantities:
    - Shale volume from Gamma Ray (Vsh)
    - Neutron-density separation
    - Acoustic impedance
    - Log(Rt) for resistivity normalisation

    Args:
        df: Input dataframe.
        gr_col: Gamma Ray column name.
        resistivity_col: Deep resistivity column name.
        neutron_col: Neutron porosity column name.
        density_col: Bulk density column name.
        sonic_col: Sonic (DT) column name.

    Returns:
        Dataframe with additional petrophysical columns.
    """
    df = df.copy()

    if gr_col in df.columns:
        gr_min = df[gr_col].quantile(0.05)
        gr_max = df[gr_col].quantile(0.95)
        df["vsh_gr"] = (df[gr_col] - gr_min) / (gr_max - gr_min + 1e-8)
        df["vsh_gr"] = df["vsh_gr"].clip(0, 1)

    if resistivity_col in df.columns:
        df["log_resistivity"] = np.log1p(df[resistivity_col].clip(lower=0))

    if neutron_col in df.columns and density_col in df.columns:
        df["neutron_density_separation"] = df[neutron_col] - df[density_col]

    if density_col in df.columns and sonic_col in df.columns:
        df["acoustic_impedance"] = df[density_col] * (1e6 / (df[sonic_col] + 1e-8))

    return df


# ---------------------------------------------------------------------------
# Directional survey features
# ---------------------------------------------------------------------------

def add_directional_features(
    df: pd.DataFrame,
    inclination_col: str = "inclination",
    azimuth_col: str = "azimuth",
) -> pd.DataFrame:
    """Add trigonometric features from directional survey data.

    Args:
        df: Input dataframe.
        inclination_col: Inclination angle column (degrees).
        azimuth_col: Azimuth angle column (degrees).

    Returns:
        Dataframe with sin/cos encoded directional features.
    """
    df = df.copy()

    if inclination_col in df.columns:
        inc_rad = np.deg2rad(df[inclination_col])
        df["sin_inclination"] = np.sin(inc_rad)
        df["cos_inclination"] = np.cos(inc_rad)

    if azimuth_col in df.columns:
        az_rad = np.deg2rad(df[azimuth_col])
        df["sin_azimuth"] = np.sin(az_rad)
        df["cos_azimuth"] = np.cos(az_rad)

    if inclination_col in df.columns and azimuth_col in df.columns:
        df["sin_inc_cos_az"] = np.sin(np.deg2rad(df[inclination_col])) * np.cos(
            np.deg2rad(df[azimuth_col])
        )

    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_all_features(
    df: pd.DataFrame,
    feature_cfg: dict,
    well_col: str = "well_id",
    depth_col: str = "md",
) -> pd.DataFrame:
    """Run all feature engineering steps based on a config dictionary.

    Args:
        df: Input dataframe.
        feature_cfg: Feature engineering configuration dictionary.
        well_col: Well identifier column.
        depth_col: Depth column.

    Returns:
        Feature-enriched dataframe.
    """
    df = df.copy()

    drilling_cols = [c for c in feature_cfg.get("drilling_features", []) if c in df.columns]
    petro_cols = [c for c in feature_cfg.get("petrophysical_features", []) if c in df.columns]
    rolling_windows = feature_cfg.get("rolling_windows", [3, 5, 10])
    lag_steps = feature_cfg.get("lag_steps", [1, 2, 3])

    all_numeric_features = drilling_cols + petro_cols

    if all_numeric_features:
        df = add_rolling_features(df, all_numeric_features, rolling_windows, group_col=well_col)
        df = add_lag_features(df, all_numeric_features, lag_steps, group_col=well_col)

    df = add_depth_features(df, depth_col=depth_col, well_col=well_col)
    df = add_drilling_features(df)
    df = add_petrophysical_features(df)
    df = add_directional_features(df)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df
