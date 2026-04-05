"""
API routes for energy monitoring and ML anomaly test (dashboard).

These routes are used by the web dashboard:
- GET  /api/energy-data
- POST /api/test-anomaly
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from .. import schemas
from ..ml import inference


router = APIRouter(tags=["Energy & ML"])


def _classify_from_deviation(deviation_percent: float) -> tuple[str, str]:
    """
    deviation < 20 → NORMAL / none
    20–40 → ANOMALY / moderate
    > 40 → HIGH ALERT / high
    """
    if deviation_percent < 20:
        return "NORMAL", "none"
    if deviation_percent < 40:
        return "ANOMALY", "moderate"
    return "HIGH ALERT", "high"


def _build_single_reading_df(reading, feature_columns: list, window_size: int) -> pd.DataFrame:
    """Build a DataFrame of window_size rows equal to the given reading for ML inference."""
    base_ts = reading.get("timestamp") if isinstance(reading, dict) else getattr(reading, "timestamp", datetime.now(timezone.utc))
    # inference.py later matches timestamps using numpy datetime64 (timezone-naive),
    # so ensure we pass timezone-naive datetimes here.
    if getattr(base_ts, "tzinfo", None) is not None:
        base_ts = base_ts.astimezone(timezone.utc).replace(tzinfo=None)
    rows = []
    for i in range(window_size):
        ts = base_ts - timedelta(minutes=5 * (window_size - 1 - i))
        row = {"timestamp": ts}
        for col in feature_columns:
            if isinstance(reading, dict):
                row[col] = reading.get(col, 0.0) or 0.0
            else:
                row[col] = getattr(reading, col, 0.0) or 0.0
        rows.append(row)
    return pd.DataFrame(rows)


@router.get(
    "/energy-data",
    response_model=schemas.EnergyDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Get live energy metrics and ML anomaly status",
)
def get_energy_data() -> schemas.EnergyDataResponse:
    """
    Simulate NORMAL operating energy values and run the ML model.

    IMPORTANT: This endpoint is constrained to always return NORMAL.
    We do that by sampling (power, energy) until the model reconstruction
    error is strictly below the trained threshold.
    """
    try:
        bundle = inference.load_latest_model_bundle()
    except FileNotFoundError:
        # No model trained yet: return safe defaults.
        return schemas.EnergyDataResponse(
            voltage=0.0,
            power=0.0,
            energy=0.0,
            reconstruction_error=0.0,
            threshold=0.0,
            deviation_percent=0.0,
            status="NORMAL",
            severity="none",
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    threshold = float(bundle.threshold)

    # Try a few samples to ensure "never exceeds anomaly threshold"
    last_error = None
    voltage = None
    power = None
    energy = None
    for _ in range(30):
        voltage = round(random.uniform(220.0, 235.0), 1)
        power = round(random.uniform(22.0, 26.0), 2)
        energy = round(random.uniform(0.025, 0.035), 5)

        reading = {
            "timestamp": datetime.utcnow(),
            "power": power,
            "energy": energy,
            # extras the model might require (only used if feature_columns includes them)
            "temperature": 0.0,
            "occupancy": 0,
            "voltage": voltage,
        }

        df = _build_single_reading_df(reading, bundle.feature_columns, bundle.window_size)
        result_df, _, _ = inference.run_anomaly_detection_on_dataframe(df, bundle)
        last = result_df.iloc[-1]
        last_error = float(last["reconstruction_error"])

        # Constrain to NORMAL: enforce error below threshold
        if threshold <= 0 or last_error < threshold:
            break

    if last_error is None:
        last_error = 0.0
    if voltage is None:
        voltage = 0.0
    if power is None:
        power = 0.0
    if energy is None:
        energy = 0.0

    deviation = max(((last_error - threshold) / threshold) * 100.0, 0.0) if threshold > 0 else 0.0
    status, severity = _classify_from_deviation(float(deviation))

    return schemas.EnergyDataResponse(
        voltage=float(voltage),
        power=float(power),
        energy=float(energy),
        reconstruction_error=float(last_error),
        threshold=float(threshold),
        deviation_percent=float(deviation),
        status=status,
        severity=severity,
    )


@router.post(
    "/test-anomaly",
    response_model=schemas.TestAnomalyResponse,
    status_code=status.HTTP_200_OK,
    summary="Run ML anomaly check on provided power and energy",
)
def test_anomaly(data: schemas.TestAnomalyRequest) -> schemas.TestAnomalyResponse:
    """
    Run the trained ML model on the given power (W) and energy (kWh).
    Returns reconstruction_error, threshold, deviation percentage, and severity.
    """
    try:
        bundle = inference.load_latest_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No trained model found. Train a model first via POST /api/train.",
        ) from exc

    now = datetime.now(timezone.utc)
    reading = {
        "timestamp": datetime.utcnow(),
        "power": float(data.power),
        "energy": float(data.energy),
        "temperature": 0.0,
        "occupancy": 0,
        "voltage": 0.0,
    }

    df = _build_single_reading_df(reading, bundle.feature_columns, bundle.window_size)
    try:
        result_df, _, _ = inference.run_anomaly_detection_on_dataframe(df, bundle)
        last = result_df.iloc[-1]
        reconstruction_error = float(last["reconstruction_error"])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {exc}",
        ) from exc

    threshold = float(bundle.threshold)
    deviation = max(((reconstruction_error - threshold) / threshold) * 100.0, 0.0) if threshold > 0 else 0.0
    status, severity = _classify_from_deviation(float(deviation))

    return schemas.TestAnomalyResponse(
        voltage=None,
        power=float(data.power),
        energy=float(data.energy),
        reconstruction_error=float(reconstruction_error),
        threshold=float(threshold),
        deviation_percent=float(deviation),
        status=status,
        severity=severity,
    )
