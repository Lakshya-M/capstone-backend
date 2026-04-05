"""
Training script for the LSTM autoencoder on synthetic (or real) data.

For your current requirement:
- It can generate synthetic Delhi household data with a *high number*
  of entries.
- It then trains the LSTM autoencoder on this data.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from . import anomaly_detection, data_loader, model as model_module, preprocessing


DEFAULT_DATA_PATH = "data/delhi_household_synthetic.csv"
MODEL_DIR = "models"
MODEL_NAME = "lstm_autoencoder_delhi_household"


def prepare_training_data(
    csv_path: str,
    window_size: int = 48,
    feature_columns: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, preprocessing.NormalizationStats, np.ndarray, list]:
    """
    Load data from CSV, normalize it, and convert into sliding windows.
    Feature columns are derived from the CSV (all numeric except timestamp) if not provided.
    """
    from . import data_loader as dl

    df = dl.load_sensor_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if feature_columns is None:
        feature_columns = dl.get_feature_columns(df)
    if not feature_columns:
        raise ValueError(
            "CSV must have at least one numeric feature column besides timestamp "
            "(e.g. power, energy or power, temperature, occupancy)."
        )

    values = df[feature_columns].values.astype("float32")
    norm_stats = preprocessing.compute_normalization_stats(values)
    values_norm = preprocessing.apply_normalization(values, norm_stats)

    df_norm = df.copy()
    df_norm[feature_columns] = values_norm

    X_windows, timestamps = preprocessing.create_sliding_windows(
        df_norm,
        window_size=window_size,
        feature_columns=tuple(feature_columns),
    )
    return X_windows, timestamps, norm_stats, values, list(feature_columns)


def train_lstm_autoencoder(
    csv_path: str = DEFAULT_DATA_PATH,
    window_size: int = 48,
    latent_dim: int = 32,
    epochs: int = 30,
    batch_size: int = 128,
    validation_split: float = 0.1,
) -> dict:
    """
    Train an LSTM autoencoder on the specified CSV dataset.

    Args:
        csv_path (str): Path to the CSV file with sensor data.
        window_size (int): Number of time steps per window.
        latent_dim (int): Dimension of the latent representation.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of data used for validation.

    Returns:
        dict: Training summary including threshold and model paths.
    """
    # Ensure data exists, generate synthetic if not
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        print(f"[train] CSV not found, generating synthetic data at: {csv_path}")
        df_generated = data_loader.generate_synthetic_delhi_household(csv_path)
        print(f"[train] Generated {len(df_generated)} rows.")

    # Prepare training windows (feature_columns derived from CSV)
    X_train, timestamps, norm_stats, _, feature_columns = prepare_training_data(
        csv_path=csv_path,
        window_size=window_size,
    )

    timesteps, num_features = X_train.shape[1], X_train.shape[2]
    print(f"[train] Feature columns: {feature_columns}")
    print(f"[train] Training data shape: {X_train.shape} (N, T, F)")

    # Build model
    autoencoder = model_module.build_lstm_autoencoder(
        input_shape=(timesteps, num_features),
        latent_dim=latent_dim,
    )
    autoencoder.summary(print_fn=lambda x: print("[model] " + x))

    # Prepare directories
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_{timestamp_str}.keras")
    norm_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_{timestamp_str}_norm.npz")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # Train
    history = autoencoder.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # Compute reconstruction errors on training data
    X_pred = autoencoder.predict(X_train, batch_size=batch_size, verbose=0)
    errors = anomaly_detection.compute_reconstruction_errors(X_train, X_pred)

    # Compute threshold (mean + 3*std by default)
    threshold = anomaly_detection.compute_threshold_from_errors(
        errors,
        method="mean_std",
        k=3.0,
    )

    # Save normalization statistics, threshold, and feature column names
    np.savez(
        norm_path,
        mean=norm_stats.mean,
        std=norm_stats.std,
        threshold=threshold,
        window_size=window_size,
        feature_columns=np.array(feature_columns),
    )

    print(f"[train] Model saved to: {model_path}")
    print(f"[train] Normalization + threshold saved to: {norm_path}")
    print(
        f"[train] Threshold (mean+3*std) = {threshold:.6f}, "
        f"train errors mean={errors.mean():.6f}, std={errors.std():.6f}"
    )

    return {
        "model_path": model_path,
        "norm_path": norm_path,
        "threshold": float(threshold),
        "trained_on_windows": int(len(X_train)),
        "trained_on_points": int(len(timestamps)),
    }


if __name__ == "__main__":
    summary = train_lstm_autoencoder(csv_path=DEFAULT_DATA_PATH)
    print("[train] Training summary:", summary)

