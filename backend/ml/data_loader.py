"""
Data loading and synthetic data generation utilities.

For your capstone, you requested dummy data resembling a
typical Delhi (India) urban household's daily electricity
usage pattern. This module can:

- Generate a high-volume synthetic dataset with realistic
  diurnal patterns, occupancy, and temperature variations.
- Save it to CSV for training the LSTM autoencoder.
- Load existing CSV files into Pandas DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    """
    Configuration for synthetic Delhi household data generation.
    """

    days: int = 90
    freq_minutes: int = 10
    base_power_night: float = 80.0  # W
    base_power_day: float = 250.0  # W
    base_power_peak_morning: float = 900.0  # W
    base_power_peak_evening: float = 1200.0  # W
    noise_std: float = 80.0  # W
    temp_mean: float = 30.0  # °C
    temp_daily_amp: float = 5.0  # °C
    temp_noise_std: float = 1.5  # °C


def generate_synthetic_delhi_household(
    csv_path: str,
    start_date: str = "2024-01-01",
    config: Optional[SyntheticConfig] = None,
) -> pd.DataFrame:
    """
    Generate synthetic sensor data for a typical Delhi household and save to CSV.

    The generated data includes:
    - `timestamp`: ISO-8601 timestamps at a fixed frequency.
    - `power`: power consumption in Watts with realistic daily cycles.
    - `temperature`: outdoor/indoor temperature pattern for Delhi climate.
    - `occupancy`: 0/1 occupancy pattern based on common working hours.

    Args:
        csv_path (str): Destination CSV file path.
        start_date (str): Start date for the time series (YYYY-MM-DD).
        config (SyntheticConfig, optional): Configuration parameters. If None,
            sensible defaults are used.

    Returns:
        pd.DataFrame: Generated dataset as a DataFrame.
    """
    if config is None:
        config = SyntheticConfig()

    # Create a datetime index covering the specified number of days
    freq_str = f"{config.freq_minutes}T"
    date_index = pd.date_range(
        start=start_date,
        periods=int((24 * 60 / config.freq_minutes) * config.days),
        freq=freq_str,
        tz="Asia/Kolkata",
    )

    hours = date_index.hour
    day_of_week = date_index.dayofweek  # 0 = Monday, 6 = Sunday

    # Occupancy pattern:
    # - Weekdays: 1 early morning, 0 in working hours, 1 in evenings and late night.
    # - Weekends: mostly 1 during the day and evening.
    occupancy = np.zeros_like(hours, dtype=int)

    for i, (h, dow) in enumerate(zip(hours, day_of_week)):
        if dow < 5:
            # Weekday pattern
            if 6 <= h < 9:
                occupancy[i] = 1
            elif 9 <= h < 17:
                occupancy[i] = 0
            elif 17 <= h < 23:
                occupancy[i] = 1
            else:
                occupancy[i] = 1 if h < 2 else 0
        else:
            # Weekend pattern
            if 7 <= h < 23:
                occupancy[i] = 1
            else:
                occupancy[i] = 0

    # Base power profile by hour of day
    base_power = np.zeros_like(hours, dtype=float)
    for i, h in enumerate(hours):
        if 0 <= h < 5:
            base_power[i] = config.base_power_night
        elif 5 <= h < 9:
            base_power[i] = config.base_power_peak_morning
        elif 9 <= h < 17:
            base_power[i] = config.base_power_day
        elif 17 <= h < 23:
            base_power[i] = config.base_power_peak_evening
        else:
            base_power[i] = config.base_power_night

    # Modulate power slightly by occupancy (more appliances when occupied)
    occupancy_factor = 1.0 + 0.3 * occupancy
    power = base_power * occupancy_factor

    # Add Gaussian noise to power
    rng = np.random.default_rng(seed=42)
    power += rng.normal(loc=0.0, scale=config.noise_std, size=power.shape)
    power = np.clip(power, a_min=20.0, a_max=None)

    # Temperature: simple sinusoidal daily pattern with noise
    seconds_in_day = 24 * 60 * 60
    timestamp_seconds = (
        (date_index.view("int64") // 10**9) % seconds_in_day
    ).astype(float)
    temp_daily_cycle = np.sin(2 * np.pi * timestamp_seconds / seconds_in_day)

    temperature = (
        config.temp_mean
        + config.temp_daily_amp * temp_daily_cycle
        + rng.normal(loc=0.0, scale=config.temp_noise_std, size=temp_daily_cycle.shape)
    )

    df = pd.DataFrame(
        {
            "timestamp": date_index,
            "power": power,
            "temperature": temperature,
            "occupancy": occupancy,
        }
    )

    # Save to CSV
    df.to_csv(csv_path, index=False)
    return df


def load_sensor_csv(csv_path: str) -> pd.DataFrame:
    """
    Load sensor data from a CSV file.

    Supports two formats:
    - 3-field: timestamp, power, energy (e.g. Delhi synthetic).
    - 5-field: timestamp, power, temperature, occupancy, (and optionally energy).

    Column name mappings:
    - `power_watts` -> `power`
    - `temperature_c` -> `temperature`
    - `energy_kwh` -> `energy`

    At least `timestamp` and `power` (or `power_watts`) are required.
    Other numeric columns (temperature, occupancy, energy) are used as features
    when present; missing ones are not required.
    """
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Map power column (accept power, power_watts, power(W), etc.)
    if "power" not in df.columns:
        for c in df.columns:
            if c != "timestamp" and "power" in c.lower():
                df = df.rename(columns={c: "power"})
                break
        if "power" not in df.columns:
            raise ValueError("CSV must contain a power column (e.g. 'power', 'power_watts', 'power(W)').")

    # Map temperature (optional)
    if "temperature" not in df.columns:
        if "temperature_c" in df.columns:
            df = df.rename(columns={"temperature_c": "temperature"})
        else:
            for c in df.columns:
                if c != "timestamp" and "temperature" in c.lower():
                    df = df.rename(columns={c: "temperature"})
                    break

    # Map energy column (accept energy, energy_kwh, energy(kWh), etc.)
    if "energy" not in df.columns:
        for c in df.columns:
            if c != "timestamp" and "energy" in c.lower():
                df = df.rename(columns={c: "energy"})
                break

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return list of numeric column names to use as model features (excludes timestamp).

    Use this after load_sensor_csv so training/inference use the same columns
    as in the CSV (e.g. ["power", "energy"] or ["power", "temperature", "occupancy"]).
    """
    return [
        c for c in df.columns
        if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])
    ]


if __name__ == "__main__":
    # Convenience entry point to quickly generate a large dummy dataset.
    output_path = "data/delhi_household_synthetic.csv"
    print(f"[data_loader] Generating synthetic Delhi household data to: {output_path}")
    df_generated = generate_synthetic_delhi_household(csv_path=output_path)
    print(
        f"[data_loader] Generated {len(df_generated)} rows from "
        f"{df_generated['timestamp'].min()} to {df_generated['timestamp'].max()}"
    )

