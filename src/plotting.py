"""
Plotting Utilities Module
=========================
Reusable visualisation functions for wellbore geology data exploration,
model diagnostics, and result presentation using Matplotlib and Plotly.

Author: Md Ashraf
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default color palette for formations / lithologies
FORMATION_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#8338EC", "#FB5607",
    "#3A86FF", "#06D6A0", "#EF476F", "#FFD166",
]


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_cols: int = 3,
    figsize_per_col: float = 4.0,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot histograms for a list of features.

    Args:
        df: Input dataframe.
        feature_cols: Columns to plot.
        n_cols: Number of columns in the subplot grid.
        figsize_per_col: Figure size per column in inches.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure object.
    """
    n_rows = int(np.ceil(len(feature_cols) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figsize_per_col, n_rows * 3),
        squeeze=False,
    )
    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")

    for idx, col in enumerate(feature_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        df[col].dropna().hist(ax=ax, bins=40, color="#457B9D", edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    # Hide empty subplots
    for idx in range(len(feature_cols), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature distribution plot to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    figsize: tuple = (14, 12),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot correlation heatmap using seaborn-style coloring.

    Args:
        df: Input dataframe.
        feature_cols: Columns to include. Defaults to all numeric columns.
        figsize: Figure size in inches.
        save_path: Optional save path.

    Returns:
        Matplotlib figure object.
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn is required for correlation heatmap. Run: pip install seaborn")

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdYlGn",
        center=0,
        annot=len(feature_cols) <= 20,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        square=True,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved correlation heatmap to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Log curve plots (wellbore-style)
# ---------------------------------------------------------------------------

def plot_log_curves(
    df: pd.DataFrame,
    depth_col: str = "md",
    log_cols: Optional[List[str]] = None,
    well_col: Optional[str] = None,
    well_id: Optional[Any] = None,
    figsize_per_track: float = 2.0,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot standard petrophysical log curves in vertical well-log style.

    Args:
        df: Input dataframe with depth and log data.
        depth_col: Column name for measured depth.
        log_cols: List of log columns to plot (one track per column).
        well_col: Column name for well identifier.
        well_id: Specific well to filter on. If None, first well is used.
        figsize_per_track: Width per track in inches.
        save_path: Optional save path.

    Returns:
        Matplotlib figure object.
    """
    if log_cols is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        log_cols = [c for c in numeric if c != depth_col][:6]

    if well_col and well_id is not None:
        df = df[df[well_col] == well_id].copy()
    elif well_col:
        well_id = df[well_col].iloc[0]
        df = df[df[well_col] == well_id].copy()

    df = df.sort_values(depth_col)
    n_tracks = len(log_cols)

    fig, axes = plt.subplots(
        1, n_tracks,
        figsize=(n_tracks * figsize_per_track + 1, 12),
        sharey=True,
    )
    if n_tracks == 1:
        axes = [axes]

    fig.suptitle(f"Well Log Curves — Well: {well_id}", fontsize=13, fontweight="bold")

    for ax, col in zip(axes, log_cols):
        ax.plot(df[col], df[depth_col], lw=0.8, color="#2A9D8F")
        ax.set_xlabel(col, fontsize=9)
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="both")
        ax.set_title(col, fontsize=9)

    axes[0].set_ylabel(f"Measured Depth ({depth_col})", fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved log curves plot to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    top_n: int = 30,
    title: str = "Feature Importance",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importances.

    Args:
        feature_names: List of feature names.
        importance_values: Array of importance values.
        top_n: Number of top features to display.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        Matplotlib figure object.
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importance_df)))  # type: ignore
    ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color=colors[::-1],
        edgecolor="white",
    )
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# OOF prediction diagnostics
# ---------------------------------------------------------------------------

def plot_oof_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "OOF Prediction Diagnostics",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot actual vs predicted and residuals for OOF predictions.

    Args:
        y_true: Ground truth values.
        y_pred: OOF predicted values.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        Matplotlib figure object.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Actual vs Predicted scatter
    ax = axes[0]
    ax.scatter(y_pred, y_true, alpha=0.3, s=10, color="#457B9D")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(alpha=0.3)

    # Residuals scatter
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, color="#E63946")
    ax.axhline(0, color="black", lw=1.5, linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")
    ax.grid(alpha=0.3)

    # Residuals histogram
    ax = axes[2]
    ax.hist(residuals, bins=50, color="#2A9D8F", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", lw=1.5, linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved OOF diagnostics plot to %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# CV fold scores
# ---------------------------------------------------------------------------

def plot_cv_scores(
    fold_scores: List[float],
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot per-fold CV RMSE scores.

    Args:
        fold_scores: List of per-fold RMSE values.
        model_name: Name of the model for the title.
        save_path: Optional save path.

    Returns:
        Matplotlib figure object.
    """
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    fig, ax = plt.subplots(figsize=(8, 4))
    folds = list(range(1, len(fold_scores) + 1))
    ax.bar(folds, fold_scores, color="#457B9D", edgecolor="white", alpha=0.85)
    ax.axhline(mean_score, color="#E63946", lw=2, linestyle="--", label=f"Mean={mean_score:.5f}")
    ax.fill_between(
        [0.5, len(fold_scores) + 0.5],
        mean_score - std_score,
        mean_score + std_score,
        alpha=0.2,
        color="#E63946",
        label=f"±1 Std ({std_score:.5f})",
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{model_name} — Cross-Validation RMSE by Fold", fontweight="bold")
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved CV scores plot to %s", save_path)
    return fig
