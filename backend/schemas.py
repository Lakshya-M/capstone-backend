"""
Pydantic schemas (data models) for request and response validation.

These schemas define the shapes of:
- Sensor ingestion payloads from the ESP32.
- Database-backed sensor readings returned by the API.
- ML training and anomaly detection responses (for later routes).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sensor data schemas
# ---------------------------------------------------------------------------


class SensorDataBase(BaseModel):
    """
    Base schema for a single sensor reading.
    Supports 3-field (timestamp, power, energy) or 5-field (+ temperature, occupancy) payloads.
    """

    timestamp: datetime = Field(
        ...,
        description="Timestamp of the reading in ISO-8601 format (preferably UTC).",
    )
    power: float = Field(..., description="Power consumption in Watts.")
    temperature: float = Field(0.0, description="Temperature in °C (optional, default 0).")
    occupancy: int = Field(
        0,
        ge=0,
        le=1,
        description="Occupancy flag: 0 = unoccupied, 1 = occupied (optional, default 0).",
    )
    energy: Optional[float] = Field(
        None,
        description="Energy in kWh (optional; required when model uses power+energy).",
    )


class SensorDataCreate(SensorDataBase):
    """
    Schema used for data ingestion from the ESP32.
    """

    pass


class SensorDataRead(SensorDataBase):
    """
    Schema used when returning stored sensor readings from the API.
    """

    id: int = Field(..., description="Database primary key.")
    created_at: datetime = Field(
        ...,
        description="Server-side insertion time of the record.",
    )

    class Config:
        orm_mode = True


class AnomalyCheckResult(BaseModel):
    """
    Anomaly result for a single ingested point (returned with POST /api/ingest).
    """

    reconstruction_error: float = Field(
        ...,
        description="Reconstruction error from the autoencoder for the window ending at this point.",
    )
    is_anomaly: bool = Field(
        ...,
        description="True if this reading was flagged as an anomaly.",
    )
    threshold: float = Field(
        ...,
        description="Threshold used (readings with error above this are anomalies).",
    )


class IngestResponse(SensorDataRead):
    """
    Response after ingesting a sensor reading. Includes stored record plus
    optional anomaly check so you can see immediately if it was flagged.
    """

    anomaly: Optional[AnomalyCheckResult] = Field(
        default=None,
        description="Anomaly check result. Present only if a trained model exists and enough recent data is available.",
    )


# ---------------------------------------------------------------------------
# ML-related schemas (for future routes)
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """
    Request schema for triggering model training.

    Note:
        For your capstone, you mentioned that sensor entries will also
        be stored in a CSV file for training. The `csv_path` field
        allows the backend to load training data from such a file.
    """

    csv_path: Optional[str] = Field(
        default="data/training_data.csv",
        description="Path to the CSV file containing historical sensor data.",
    )
    use_database: bool = Field(
        default=False,
        description="If True, load historical data from the database instead of CSV.",
    )


class TrainResponse(BaseModel):
    """
    Response schema for model training.
    """

    status: str = Field(..., description="Status of the training job (e.g., 'success').")
    message: str = Field(..., description="Additional information about the training run.")
    trained_on_records: int = Field(
        ...,
        description="Number of records used for training.",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Identifier for the trained model version (e.g., timestamp).",
    )


class AnomalyResult(BaseModel):
    """
    Schema representing the anomaly detection result for a single time step.
    Features (power, temperature, occupancy, energy) depend on the trained model.
    """

    timestamp: datetime = Field(..., description="Timestamp of the evaluated point.")
    power: float = Field(0.0, description="Power at this time step.")
    temperature: float = Field(0.0, description="Temperature at this time step.")
    occupancy: int = Field(0, description="Occupancy at this time step.")
    energy: Optional[float] = Field(None, description="Energy in kWh (when model uses it).")
    reconstruction_error: float = Field(
        ...,
        description="Reconstruction error from the autoencoder.",
    )
    is_anomaly: bool = Field(
        ...,
        description="True if the point is flagged as anomalous.",
    )


class AnomalySummary(BaseModel):
    """
    Summary response for an anomaly detection query.
    """

    model_version: Optional[str] = Field(
        default=None,
        description="Identifier of the model used for inference.",
    )
    threshold: float = Field(
        ...,
        description="Anomaly threshold applied to reconstruction error.",
    )
    results: list[AnomalyResult] = Field(
        default_factory=list,
        description="List of anomaly detection results.",
    )


# ---------------------------------------------------------------------------
# Energy monitoring & ML test (dashboard API)
# ---------------------------------------------------------------------------


class EnergyDataResponse(BaseModel):
    """
    Live energy metrics and ML anomaly status for the dashboard.
    """

    voltage: float = Field(..., description="Voltage in Volts (V).")
    power: float = Field(..., description="Power in Watts.")
    energy: float = Field(..., description="Energy in kWh.")
    reconstruction_error: float = Field(
        ...,
        description="Reconstruction error from the autoencoder.",
    )
    threshold: float = Field(
        ...,
        description="Threshold above which readings are flagged as anomaly.",
    )
    deviation_percent: float = Field(
        ...,
        description="Deviation percentage computed from (error - threshold) / threshold.",
    )
    status: str = Field(
        ...,
        description="Status string: NORMAL | ANOMALY | HIGH ALERT.",
    )
    severity: str = Field(
        ...,
        description="Severity string: none | moderate | high.",
    )


class TestAnomalyRequest(BaseModel):
    """Request body for manual ML anomaly test (power + energy only)."""

    power: float = Field(..., description="Power in Watts.")
    energy: float = Field(..., description="Energy in kWh.")


class TestAnomalyResponse(BaseModel):
    """Response from POST /api/test-anomaly with anomaly result."""

    voltage: Optional[float] = Field(
        default=None,
        description="Optional voltage included for parity with GET responses (not required for manual tests).",
    )
    power: float = Field(..., description="Power in Watts.")
    energy: float = Field(..., description="Energy in kWh.")
    reconstruction_error: float = Field(
        ...,
        description="Reconstruction error from the autoencoder.",
    )
    threshold: float = Field(
        ...,
        description="Anomaly threshold used.",
    )
    deviation_percent: float = Field(
        ...,
        description="Deviation percentage computed from (error - threshold) / threshold.",
    )
    status: str = Field(
        ...,
        description="Status string: NORMAL | ANOMALY | HIGH ALERT.",
    )
    severity: str = Field(
        ...,
        description="Severity string: none | moderate | high.",
    )

