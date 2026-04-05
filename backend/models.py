"""
SQLAlchemy ORM models for the backend service.

These models define how sensor data is stored in the relational database.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer

from .database import Base


class SensorData(Base):
    """
    ORM model representing a single sensor reading from the ESP32.

    Attributes:
        id (int): Primary key.
        timestamp (datetime): Timestamp of the reading (UTC, ISO-8601 compatible).
        power (float): Power consumption in Watts.
        temperature (float): Temperature in degrees Celsius (optional, default 0).
        occupancy (int): Occupancy flag 0 or 1 (optional, default 0).
        energy (float): Energy in kWh (optional; used when model is trained on power+energy).
        created_at (datetime): Server-side insertion time.
    """

    __tablename__ = "sensor_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    power = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False, default=0.0)
    occupancy = Column(Integer, nullable=False, default=0)
    energy = Column(Float, nullable=True)  # kWh; used for power+energy models
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

