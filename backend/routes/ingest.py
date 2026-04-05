"""
API routes for sensor data ingestion from the ESP32 (or any HTTP client).

Endpoint:
    POST /api/ingest

Expected JSON payload:
    {
        "timestamp": "ISO-8601 string",
        "power": float,
        "temperature": float,
        "occupancy": 0 or 1
    }
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import crud, schemas
from ..database import get_db
from ..ml import inference


router = APIRouter(tags=["Ingest sensor data"])


def _single_reading_df(created, feature_columns: list, window_size: int):
    """
    Build a DataFrame of window_size rows all equal to the just-ingested reading.
    This answers "is this single reading anomalous?" instead of mixing with old DB rows.
    """
    base_ts = created.timestamp
    rows = []
    for i in range(window_size):
        ts = base_ts - timedelta(minutes=5 * (window_size - 1 - i))
        rows.append(
            {
                "timestamp": ts,
                **{
                    col: getattr(created, col, 0.0) or 0.0
                    for col in feature_columns
                },
            }
        )
    return pd.DataFrame(rows)


@router.post(
    "/ingest",
    response_model=schemas.IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit one sensor reading (see anomaly result below)",
)
def ingest_sensor_data(
    payload: schemas.SensorDataCreate,
    db: Session = Depends(get_db),
):
    """
    **Submit a sensor reading** (power, energy, etc.).

    The reading is saved to the database. If a trained model is available,
    the response includes an **anomaly check for this reading only** (not the
    last 48 rows): the value is compared to the model's "normal" range.
    - **reconstruction_error**: how far this value is from normal
    - **is_anomaly**: `true` = anomaly, `false` = normal
    - **threshold**: error above this is considered anomaly
    """
    try:
        created = crud.create_sensor_data(db=db, data=payload)
        anomaly_result = None

        try:
            bundle = inference.load_latest_model_bundle()
            # Use only the reading we just inserted (repeated to form a window).
            # So the answer is "is this value anomalous?" not "is the last 48 DB rows anomalous?"
            df = _single_reading_df(
                created, bundle.feature_columns, bundle.window_size
            )
            result_df, _, _ = inference.run_anomaly_detection_on_dataframe(
                df, bundle
            )
            last = result_df.iloc[-1]
            anomaly_result = schemas.AnomalyCheckResult(
                reconstruction_error=float(last["reconstruction_error"]),
                is_anomaly=bool(last["is_anomaly"]),
                threshold=bundle.threshold,
            )
        except (FileNotFoundError, Exception):
            pass

        return schemas.IngestResponse(
            id=created.id,
            timestamp=created.timestamp,
            power=created.power,
            temperature=created.temperature,
            occupancy=created.occupancy,
            energy=created.energy,
            created_at=created.created_at,
            anomaly=anomaly_result,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store sensor data.",
        ) from exc

