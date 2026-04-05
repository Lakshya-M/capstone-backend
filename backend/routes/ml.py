"""
API routes for ML training and anomaly detection.

Endpoints:
- POST /api/train
- GET  /api/anomalies
"""

from __future__ import annotations

import os
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import crud, schemas
from ..database import get_db
from ..ml import inference, train

# Project root (parent of 'backend') so CSV paths work when cwd varies
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


router = APIRouter(tags=["ml"])


@router.post(
    "/train",
    response_model=schemas.TrainResponse,
    status_code=status.HTTP_200_OK,
    summary="Train LSTM autoencoder model",
)
def train_model(request: schemas.TrainRequest) -> schemas.TrainResponse:
    """
    Trigger training of the LSTM autoencoder model.

    For your current setup, this primarily uses synthetic Delhi household
    data stored in a CSV. If `use_database` is True, the service will dump
    all sensor readings from the DB to a temporary CSV and train on that.
    """
    try:
        csv_path = request.csv_path
        # Resolve relative paths from project root so "data/foo.csv" works
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(_PROJECT_ROOT, csv_path)

        if request.use_database:
            # For now, export all DB data to CSV and reuse the same training pipeline.
            # This keeps the ML code focused on CSV-based workflows.
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=(
                    "Training from database is not yet implemented in this demo. "
                    "Set use_database=false to train from CSV (synthetic data)."
                ),
            )

        summary = train.train_lstm_autoencoder(csv_path=csv_path)
        return schemas.TrainResponse(
            status="success",
            message="Model trained successfully.",
            trained_on_records=summary["trained_on_windows"],
            model_version=summary["model_path"],
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - generic safeguard
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {exc}",
        ) from exc


@router.get(
    "/anomalies",
    response_model=schemas.AnomalySummary,
    status_code=status.HTTP_200_OK,
    summary="Run anomaly detection on latest sensor data",
)
def get_anomalies(
    limit: int = 500,
    db: Session = Depends(get_db),
) -> schemas.AnomalySummary:
    """
    Run anomaly detection on the latest sensor readings stored in the database.

    Args:
        limit (int): Number of latest records to consider for anomaly detection.
        db (Session): Database session.

    Returns:
        AnomalySummary: Anomaly scores and flags for each evaluated time step.
    """
    if limit < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit should be at least 50 to form meaningful time windows.",
        )

    # Load recent data from DB
    rows = crud.get_latest_sensor_data(db, limit=limit)
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No sensor data available in the database.",
        )

    try:
        bundle = inference.load_latest_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Build DataFrame with only the columns the model expects
    df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                **{
                    col: getattr(r, col, 0.0) or 0.0
                    for col in bundle.feature_columns
                },
            }
            for r in rows
        ]
    )

    try:
        result_df, errors, is_anomaly = inference.run_anomaly_detection_on_dataframe(
            df, bundle
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - generic safeguard
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {exc}",
        ) from exc

    # Build response (features depend on model's feature_columns)
    results: List[schemas.AnomalyResult] = []
    for _, row in result_df.iterrows():
        results.append(
            schemas.AnomalyResult(
                timestamp=row["timestamp"],
                power=float(row.get("power", 0.0)),
                temperature=float(row.get("temperature", 0.0)),
                occupancy=int(row.get("occupancy", 0)),
                energy=float(row["energy"]) if "energy" in row else None,
                reconstruction_error=float(row["reconstruction_error"]),
                is_anomaly=bool(row["is_anomaly"]),
            )
        )

    return schemas.AnomalySummary(
        model_version=bundle.model_path,
        threshold=bundle.threshold,
        results=results,
    )

