"""
Anomaly detection utilities based on reconstruction error.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_reconstruction_errors(
    X_true: np.ndarray,
    X_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute reconstruction errors for each time window.

    Args:
        X_true (np.ndarray): Ground truth sequences, shape (N, T, F).
        X_pred (np.ndarray): Reconstructed sequences, shape (N, T, F).

    Returns:
        np.ndarray: Reconstruction error per sequence, shape (N,).
    """
    # Mean squared error per sequence
    mse = np.mean(np.mean((X_true - X_pred) ** 2, axis=-1), axis=-1)
    return mse


def compute_threshold_from_errors(
    errors: np.ndarray,
    method: str = "mean_std",
    k: float = 3.0,
    percentile: float = 95.0,
) -> float:
    """
    Compute an anomaly threshold from reconstruction errors.

    Two methods are supported:
    - 'mean_std': threshold = mean + k * std
    - 'percentile': threshold = given percentile of the error distribution

    Args:
        errors (np.ndarray): Reconstruction errors from training data.
        method (str): 'mean_std' or 'percentile'.
        k (float): Multiplier for std if method == 'mean_std'.
        percentile (float): Percentile for method == 'percentile'.

    Returns:
        float: Threshold value.
    """
    if method == "mean_std":
        mu = float(np.mean(errors))
        sigma = float(np.std(errors))
        return mu + k * sigma
    elif method == "percentile":
        return float(np.percentile(errors, percentile))
    else:
        raise ValueError("Unsupported method. Use 'mean_std' or 'percentile'.")


def flag_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
    """
    Flag anomalies given reconstruction errors and a threshold.

    Args:
        errors (np.ndarray): Reconstruction errors.
        threshold (float): Threshold above which a point is anomalous.

    Returns:
        np.ndarray: Boolean array where True indicates an anomaly.
    """
    return errors > threshold

