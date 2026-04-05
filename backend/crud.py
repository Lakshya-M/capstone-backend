"""
CRUD (Create, Read, Update, Delete) operations for database entities.

This module keeps all low-level database logic separated from
FastAPI route handlers, which improves testability and clarity.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from . import models, schemas


def create_sensor_data(db: Session, data: schemas.SensorDataCreate) -> models.SensorData:
    """
    Persist a new sensor reading in the database.

    Args:
        db (Session): SQLAlchemy database session.
        data (schemas.SensorDataCreate): Validated sensor data payload.

    Returns:
        models.SensorData: The newly created ORM object.
    """
    db_obj = models.SensorData(
        timestamp=data.timestamp,
        power=data.power,
        temperature=data.temperature,
        occupancy=data.occupancy,
        energy=data.energy,
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_all_sensor_data(db: Session):
    """
    Retrieve all sensor readings from the database ordered by timestamp.

    Args:
        db (Session): SQLAlchemy database session.

    Returns:
        list[models.SensorData]: All sensor readings.
    """
    return db.query(models.SensorData).order_by(models.SensorData.timestamp.asc()).all()


def get_latest_sensor_data(db: Session, limit: int = 100):
    """
    Retrieve the most recent sensor readings.

    Args:
        db (Session): SQLAlchemy database session.
        limit (int): Maximum number of records to return.

    Returns:
        list[models.SensorData]: List of sensor readings ordered by timestamp descending.
    """
    return (
        db.query(models.SensorData)
        .order_by(models.SensorData.timestamp.desc())
        .limit(limit)
        .all()
    )

