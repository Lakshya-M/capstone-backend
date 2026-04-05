"""
Energy Monitoring & ML Anomaly Detection — FastAPI backend.

Uses the existing trained LSTM autoencoder from backend/ml/.
Trains on data/delhi_household_synthetic.csv on first run, then loads
the saved model from models/ on subsequent runs.

Run from the project root:
    pip install fastapi uvicorn numpy tensorflow
    uvicorn energy_api:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import sys
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Make sure backend package is importable (script runs from project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ml.train import train_lstm_autoencoder
from backend.ml.inference import load_latest_model_bundle
from backend.ml.preprocessing import apply_normalization, create_sliding_windows, NormalizationStats
from backend.ml.anomaly_detection import compute_reconstruction_errors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "delhi_household_synthetic.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ---------------------------------------------------------------------------
# Train (if needed) and load model
# ---------------------------------------------------------------------------

def _ensure_trained():
    """Train the model if no saved model exists in models/."""
    import glob
    pattern = os.path.join(MODEL_DIR, "lstm_autoencoder_delhi_household_*.keras")
    if glob.glob(pattern):
        print("[energy_api] Found existing trained model.")
        return
    print("[energy_api] No trained model found — training on delhi_household_synthetic.csv …")
    print("[energy_api] This will take a minute or two on first run only.")
    summary = train_lstm_autoencoder(csv_path=CSV_PATH)
    print(f"[energy_api] Training complete. Threshold = {summary['threshold']:.6f}")

_ensure_trained()
bundle = load_latest_model_bundle()
THRESHOLD = bundle.threshold

print(f"[energy_api] Model loaded: {bundle.model_path}")
print(f"[energy_api] Features: {bundle.feature_columns}")
print(f"[energy_api] Window size: {bundle.window_size}")
print(f"[energy_api] Threshold: {THRESHOLD:.10f}")

# ---------------------------------------------------------------------------
# Inference helper — run a single (power, energy) reading through the model
# ---------------------------------------------------------------------------

def _run_inference(power: float, energy: float) -> float:
    """
    Build a window of identical readings, normalize, predict, return
    reconstruction error for this reading.
    """
    feature_values = {}
    for col in bundle.feature_columns:
        if col == "power":
            feature_values[col] = power
        elif col == "energy":
            feature_values[col] = energy
        else:
            feature_values[col] = 0.0

    row = np.array([[feature_values[c] for c in bundle.feature_columns]], dtype="float32")
    row_norm = apply_normalization(row, bundle.norm_stats)

    window = np.tile(row_norm, (bundle.window_size, 1))
    X = window.reshape(1, bundle.window_size, len(bundle.feature_columns))

    X_pred = bundle.model.predict(X, verbose=0)
    errors = compute_reconstruction_errors(X, X_pred)
    return float(errors[0])


def _classify(error: float):
    deviation = max(((error - THRESHOLD) / THRESHOLD) * 100, 0)
    if deviation < 20:
        return round(deviation, 2), "NORMAL", "none"
    elif deviation < 40:
        return round(deviation, 2), "ANOMALY", "moderate"
    else:
        return round(deviation, 2), "HIGH ALERT", "high"


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="Energy Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/energy-data")
def get_energy_data():
    """Simulate normal sensor readings and run through the trained ML model."""
    voltage = round(random.uniform(220.0, 235.0), 1)
    power = round(random.uniform(22.0, 26.0), 2)
    energy = round(power * (5 / 60) / 1000, 5)

    error = _run_inference(power, energy)
    deviation, status, severity = _classify(error)

    return {
        "voltage": voltage,
        "power": power,
        "energy": energy,
        "reconstruction_error": round(error, 6),
        "threshold": round(THRESHOLD, 6),
        "deviation_percent": deviation,
        "status": status,
        "severity": severity,
    }


class ManualTest(BaseModel):
    power: float
    energy: float


@app.post("/api/test-anomaly")
def test_anomaly(data: ManualTest):
    """Run user-supplied power/energy through the trained ML model."""
    error = _run_inference(data.power, data.energy)
    deviation, status, severity = _classify(error)

    return {
        "power": data.power,
        "energy": data.energy,
        "reconstruction_error": round(error, 6),
        "threshold": round(THRESHOLD, 6),
        "deviation_percent": deviation,
        "status": status,
        "severity": severity,
    }
