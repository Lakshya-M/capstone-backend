"""
Preprocessing utilities for time-series sensor data.

This includes:
- Normalization (mean/std scaling).
- Sliding window creation for LSTM input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["power", "temperature", "occupancy"]


@dataclass
class NormalizationStats:
    """
    Simple container for normalization statistics.
    """

    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to a serializable dictionary."""
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> "NormalizationStats":
        """Reconstruct from dictionary."""
        return cls(mean=data["mean"], std=data["std"])


def compute_normalization_stats(values: np.ndarray) -> NormalizationStats:
    """
    Compute mean and standard deviation for each feature.

    Args:
        values (np.ndarray): Array of shape (N, F) with raw feature values.

    Returns:
        NormalizationStats: Mean and std for each feature.
    """
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0.0] = 1.0  # avoid division by zero
    return NormalizationStats(mean=mean, std=std)


def apply_normalization(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """
    Normalize values using pre-computed statistics.

    Args:
        values (np.ndarray): Raw values of shape (N, F).
        stats (NormalizationStats): Normalization statistics.

    Returns:
        np.ndarray: Normalized values with same shape.
    """
    return (values - stats.mean) / stats.std


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int,
    feature_columns: Tuple[str, ...] = tuple(FEATURE_COLUMNS),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a time-series DataFrame into sliding windows for LSTM input.

    Args:
        df (pd.DataFrame): DataFrame sorted by timestamp.
        window_size (int): Number of time steps per window.
        feature_columns (tuple[str, ...]): Columns to use as features.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: shape (num_windows, window_size, num_features)
            - timestamps: shape (num_windows,) as the timestamp of the last step
    """
    values = df.loc[:, list(feature_columns)].values.astype("float32")
    timestamps = df["timestamp"].values

    num_samples = len(df) - window_size + 1
    if num_samples <= 0:
        raise ValueError(
            "Not enough rows to create at least one window. "
            f"Need at least {window_size} rows."
        )

    num_features = values.shape[1]
    X = np.zeros((num_samples, window_size, num_features), dtype="float32")
    last_timestamps = np.zeros((num_samples,), dtype="datetime64[ns]")

    for i in range(num_samples):
        X[i] = values[i : i + window_size]
        last_timestamps[i] = timestamps[i + window_size - 1]

    return X, last_timestamps

