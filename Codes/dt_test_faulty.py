import pandas as pd
import numpy as np

np.random.seed(42)

# ---------- FILE 1 (TXT ; separated) ----------
f1 = pd.read_csv(
    r"C:\Users\laksh\OneDrive\Documents\Capstone\Codes\file1.txt",
    sep=";",
    low_memory=False
)

f1["timestamp"] = pd.to_datetime(
    f1["Date"] + " " + f1["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

f1["Global_active_power"] = pd.to_numeric(
    f1["Global_active_power"], errors="coerce"
)

f1["power_watts"] = f1["Global_active_power"] * 1000
f1 = f1[["timestamp", "power_watts"]].dropna()

# ---------- FILE 2 (TXT , separated) ----------
f2 = pd.read_csv(
    r"C:\Users\laksh\OneDrive\Documents\Capstone\Codes\file2.txt",
    sep=","
)

f2["timestamp"] = pd.to_datetime(f2["timestamp"], errors="coerce")

f2["airTemperature"] = pd.to_numeric(
    f2["airTemperature"], errors="coerce"
)

f2["dewTemperature"] = pd.to_numeric(
    f2["dewTemperature"], errors="coerce"
)

f2 = f2.rename(columns={
    "airTemperature": "temperature_out",
    "dewTemperature": "air_quality"
})

f2 = f2[["timestamp", "temperature_out", "air_quality"]].dropna()

# ---------- FILE 3 (CSV) ----------
f3 = pd.read_csv(
    r"C:\Users\laksh\OneDrive\Documents\Capstone\Codes\file3.csv"
)

f3["timestamp"] = pd.to_datetime(f3["date"], errors="coerce")

numeric_cols = [
    "T1","T2","T3","T4",
    "RH_1","RH_2","RH_3","RH_4",
    "lights"
]

for col in numeric_cols:
    f3[col] = pd.to_numeric(f3[col], errors="coerce")

f3["temperature_c"] = f3[["T1","T2","T3","T4"]].mean(axis=1)
f3["humidity_percent"] = f3[["RH_1","RH_2","RH_3","RH_4"]].mean(axis=1)
f3["light_level"] = f3["lights"] * 25

f3 = f3[[
    "timestamp",
    "temperature_c",
    "humidity_percent",
    "light_level"
]].dropna()

# ---------- CREATE 6-MONTH TIMELINE ----------
start = pd.Timestamp("2024-12-21 00:00")
end = start + pd.DateOffset(months=6)

timeline = pd.date_range(start, end, freq="5min")
df = pd.DataFrame({"timestamp": timeline})

# ---------- MERGE DATA ----------
df = pd.merge_asof(
    df,
    f3.sort_values("timestamp"),
    on="timestamp",
    direction="nearest"
)

df = pd.merge_asof(
    df,
    f2.sort_values("timestamp"),
    on="timestamp",
    direction="nearest"
)

df = pd.merge_asof(
    df,
    f1.sort_values("timestamp"),
    on="timestamp",
    direction="nearest"
)

# ---------- OCCUPANCY LOGIC ----------
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

df["occupancy"] = (
    ((df["hour"] >= 8) & (df["hour"] <= 20) & (df["weekday"] < 5)) |
    ((df["hour"] >= 10) & (df["hour"] <= 18) & (df["weekday"] >= 5))
).astype(int)

# ---------- CLEAN NUMERIC TYPES ----------
for col in [
    "temperature_c",
    "humidity_percent",
    "light_level",
    "air_quality",
    "power_watts"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["power_watts"] = df["power_watts"].interpolate()

# ---------- SEASONAL DRIFT ----------
seasonal = 2 * np.sin(2 * np.pi * df.index / len(df))
df["temperature_c"] += seasonal
df["humidity_percent"] += np.random.normal(0, 1.5, len(df))
df["air_quality"] += np.random.normal(0, 8, len(df))

# ---------- FAULT INJECTION ----------
df["is_anomaly"] = 0

# 1% temperature dropout
fault_idx = np.random.choice(df.index, int(0.01 * len(df)), replace=False)
df.loc[fault_idx, "temperature_c"] = np.nan
df.loc[fault_idx, "is_anomaly"] = 1

# 0.5% humidity stuck fault
stuck_idx = np.random.choice(df.index, int(0.005 * len(df)), replace=False)
df.loc[stuck_idx, "humidity_percent"] = df.loc[stuck_idx[0], "humidity_percent"]
df.loc[stuck_idx, "is_anomaly"] = 1

# 1% power spikes
noise_idx = np.random.choice(df.index, int(0.01 * len(df)), replace=False)
df.loc[noise_idx, "power_watts"] *= np.random.uniform(1.5, 2.5)
df.loc[noise_idx, "is_anomaly"] = 1

df["temperature_c"] = df["temperature_c"].interpolate()

# ---------- ENERGY CALCULATION ----------
df["energy_kwh"] = df["power_watts"] * (5 / 60) / 1000

# ---------- FINAL SELECT ----------
df = df[[
    "timestamp",
    "temperature_c",
    "humidity_percent",
    "occupancy",
    "light_level",
    "air_quality",
    "power_watts",
    "energy_kwh",
    "is_anomaly"
]]

df = df.iloc[:51840]

df.to_csv(
    r"C:\Users\laksh\OneDrive\Documents\Capstone\Codes\dt_test_faulty.csv",
    index=False
)
