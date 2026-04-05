"""
Inference utilities for running anomaly detection with a trained LSTM autoencoder.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from . import anomaly_detection, data_loader, preprocessing


MODEL_DIR = "models"
MODEL_NAME_PREFIX = "lstm_autoencoder_delhi_household"


@dataclass
class LoadedModelBundle:
    """
    Container for a loaded model and its associated metadata.
    """

    model_path: str
    norm_path: str
    threshold: float
    window_size: int
    feature_columns: List[str]
    model: object  # keras Model


def _find_latest_model_prefix() -> Tuple[str, str]:
    """
    Find the latest model and normalization files based on timestamp in filename.

    Returns:
        Tuple[str, str]: (model_path, norm_path)
    """
    pattern_model = os.path.join(MODEL_DIR, f"{MODEL_NAME_PREFIX}_*.keras")
    model_files = glob.glob(pattern_model)
    if not model_files:
        raise FileNotFoundError("No trained model found in 'models' directory.")

    # Sort by modification time (latest last)
    model_files.sort(key=os.path.getmtime)
    latest_model = model_files[-1]

    # Derive norm file path from model filename
    base, _ = os.path.splitext(latest_model)
    norm_path = f"{base}_norm.npz"
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Normalization file not found for model: expected '{norm_path}'."
        )

    return latest_model, norm_path


def load_latest_model_bundle() -> LoadedModelBundle:
    """
    Load the most recently trained model and its normalization / threshold metadata.

    Returns:
        LoadedModelBundle: Loaded Keras model and metadata.
    """
    model_path, norm_path = _find_latest_model_prefix()
    data = np.load(norm_path, allow_pickle=True)

    mean = data["mean"]
    std = data["std"]
    threshold = float(data["threshold"])
    window_size = int(data["window_size"])
    feature_columns = list(data["feature_columns"])

    stats = preprocessing.NormalizationStats(mean=mean, std=std)
    keras_model = load_model(model_path)

    bundle = LoadedModelBundle(
        model_path=model_path,
        norm_path=norm_path,
        threshold=threshold,
        window_size=window_size,
        feature_columns=feature_columns,
        model=keras_model,
    )
    # Attach stats for convenience
    bundle.norm_stats = stats  # type: ignore[attr-defined]
    return bundle


def run_anomaly_detection_on_dataframe(
    df: pd.DataFrame,
    bundle: LoadedModelBundle,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Run anomaly detection on a DataFrame using a loaded model bundle.

    Args:
        df (pd.DataFrame): Data with columns matching `bundle.feature_columns`
            plus a `timestamp` column.
        bundle (LoadedModelBundle): Loaded model and metadata.

    Returns:
        Tuple:
            - result_df (pd.DataFrame): DataFrame containing timestamps, raw features,
              reconstruction error, and anomaly flag for each window.
            - errors (np.ndarray): Reconstruction errors.
            - is_anomaly (np.ndarray): Boolean mask for anomalies.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Normalize using stored stats
    values = df[bundle.feature_columns].values.astype("float32")
    values_norm = preprocessing.apply_normalization(values, bundle.norm_stats)

    df_norm = df.copy()
    df_norm[bundle.feature_columns] = values_norm

    # Build windows
    X_windows, last_timestamps = preprocessing.create_sliding_windows(
        df_norm,
        window_size=bundle.window_size,
        feature_columns=tuple(bundle.feature_columns),
    )

    # Predict reconstruction
    X_pred = bundle.model.predict(X_windows, verbose=0)
    errors = anomaly_detection.compute_reconstruction_errors(X_windows, X_pred)
    is_anomaly = anomaly_detection.flag_anomalies(errors, bundle.threshold)

    # For reporting, also capture the raw (unnormalized) values for the last step
    # in each window (aligned with last_timestamps).
    raw_values = df.loc[
        df["timestamp"].isin(last_timestamps)
    ].reset_index(drop=True)[bundle.feature_columns]

    result_df = pd.DataFrame(
        {
            "timestamp": last_timestamps,
            **{
                col: raw_values[col].values
                for col in bundle.feature_columns
            },
            "reconstruction_error": errors,
            "is_anomaly": is_anomaly.astype(bool),
        }
    )
    return result_df, errors, is_anomaly

