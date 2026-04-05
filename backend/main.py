"""
FastAPI application entry point for the Smart Building Energy Anomaly
Detection backend.

This app is responsible for:
- Exposing HTTP APIs for sensor data ingestion and ML operations.
- Managing database connections and lifecycle.

At this stage, only the core application and database initialization
are set up. In subsequent steps, we will add:
- `/api/ingest` for ESP32 data ingestion.
- `/api/train` for training the LSTM autoencoder.
- `/api/anomalies` for anomaly detection and querying.
"""

from __future__ import annotations

from sqlalchemy import text
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse

from .database import Base, engine
from .routes import energy, ingest, ml, twin


def create_app() -> FastAPI:
    """
    Application factory for the FastAPI app.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    # Create database tables if they do not exist yet.
    Base.metadata.create_all(bind=engine)

    # Add `energy` column to existing tables (no-op if already present)
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE sensor_data ADD COLUMN energy REAL"))
            conn.commit()
        except Exception:
            conn.rollback()

    app = FastAPI(
        title="Smart Building API",
        version="0.1.0",
        description=(
            "Ingest sensor data (power, temperature, occupancy) and get **anomaly** results. "
            "Train the LSTM autoencoder on CSV data, then use **POST /api/ingest** to submit "
            "readings—the response shows whether each reading was flagged as an anomaly."
        ),
        docs_url=None,
        redoc_url=None,
        openapi_tags=[
            {"name": "Ingest sensor data", "description": "Submit readings; response includes anomaly (normal vs anomaly)."},
            {"name": "ml", "description": "Train model on CSV and run anomaly detection on stored data."},
            {"name": "system", "description": "Health check."},
        ],
    )

    # Allow dashboard (browser) to call the API cross-origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(ingest.router, prefix="/api")
    app.include_router(ml.router, prefix="/api")
    app.include_router(energy.router, prefix="/api")
    app.include_router(twin.router, prefix="/api")

    # Explicit /openapi.json URL so Swagger works behind Render/proxies (avoids root_path mismatch).
    @app.get("/docs", include_in_schema=False)
    async def swagger_ui() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{app.title} - Swagger UI",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc_ui() -> HTMLResponse:
        return get_redoc_html(
            openapi_url="/openapi.json",
            title=f"{app.title} - ReDoc",
        )

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """
        Simple health check endpoint to verify that the API is running.

        Returns:
            dict[str, str]: A JSON object indicating service health.
        """

        return {"status": "ok"}

    return app


app = create_app()

