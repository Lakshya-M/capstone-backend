"""
Latest reading for hardware twin nodes (e.g. ESP32 Room 4 / D1).
In-memory store: POST from device, GET for dashboard polling.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["twin"])

_latest: dict[str, Any] = {
    "room_id": "roomD",
    "node_id": "D1",
    "aqi": None,
    "temperature": None,
    "humidity": None,
    "voltage": None,
    "updated_at": None,
}


class TwinReadingIn(BaseModel):
    aqi: float = Field(..., ge=0, le=500)
    temperature: float = Field(...)
    humidity: float = Field(..., ge=0, le=100)
    voltage: float = Field(220.0)


class TwinReadingOut(BaseModel):
    room_id: str
    node_id: str
    aqi: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    voltage: Optional[float] = None
    updated_at: Optional[str] = None


@router.post(
    "/twin/reading",
    response_model=TwinReadingOut,
    summary="Store latest hardware reading (ESP32 → Room 4 D1)",
)
def post_twin_reading(body: TwinReadingIn) -> TwinReadingOut:
    _latest["aqi"] = float(body.aqi)
    _latest["temperature"] = float(body.temperature)
    _latest["humidity"] = float(body.humidity)
    _latest["voltage"] = float(body.voltage)
    _latest["updated_at"] = datetime.now(timezone.utc).isoformat()
    return TwinReadingOut(**_latest)


@router.get(
    "/twin/latest",
    response_model=TwinReadingOut,
    summary="Get latest hardware reading for dashboard",
)
def get_twin_latest() -> TwinReadingOut:
    return TwinReadingOut(**_latest)
