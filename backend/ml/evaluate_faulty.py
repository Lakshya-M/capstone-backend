"""
Offline evaluation script for a faulty (abnormal) dataset.

Usage (from project root, with venv activated):

    python -m backend.ml.evaluate_faulty

It will:
- Load the latest trained LSTM autoencoder model and its normalization stats.
- Load a faulty CSV (default: data/smart_building_faulty.csv).
- Run anomaly detection over the full file.
- Save a detailed results CSV with reconstruction errors and anomaly flags.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import data_loader, inference


DEFAULT_FAULTY_CSV = "data/smart_building_faulty.csv"
DEFAULT_OUTPUT_CSV = "data/smart_building_faulty_anomaly_results.csv"


def evaluate_faulty_dataset(
    faulty_csv_path: str = DEFAULT_FAULTY_CSV,
    output_csv_path: str = DEFAULT_OUTPUT_CSV,
) -> None:
    """
    Run anomaly detection on a faulty dataset and save detailed results.

    Args:
        faulty_csv_path (str): Path to the faulty dataset CSV.
        output_csv_path (str): Path where the results CSV will be written.
    """
    if not os.path.exists(faulty_csv_path):
        raise FileNotFoundError(
            f"Faulty CSV not found at '{faulty_csv_path}'. "
            "Please place your faulty dataset there or pass a different path."
        )

    print(f"[evaluate] Loading latest model bundle...")
    bundle = inference.load_latest_model_bundle()
    print(f"[evaluate] Using model: {bundle.model_path}")
    print(f"[evaluate] Using norm/threshold file: {bundle.norm_path}")
    print(f"[evaluate] Threshold: {bundle.threshold:.6f}")

    print(f"[evaluate] Loading faulty dataset from: {faulty_csv_path}")
    df_faulty = data_loader.load_sensor_csv(faulty_csv_path)
    print(f"[evaluate] Loaded {len(df_faulty)} rows.")

    result_df, errors, is_anomaly = inference.run_anomaly_detection_on_dataframe(
        df_faulty, bundle
    )

    num_anomalies = int(is_anomaly.sum())
    print(f"[evaluate] Detected {num_anomalies} anomalous windows.")

    # Ensure directory exists
    Path(os.path.dirname(output_csv_path) or ".").mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_csv_path, index=False)
    print(f"[evaluate] Detailed results saved to: {output_csv_path}")


if __name__ == "__main__":
    evaluate_faulty_dataset()

