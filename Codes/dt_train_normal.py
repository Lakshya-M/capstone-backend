import pandas as pd
import numpy as np

np.random.seed(42)

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

f1["Global_active_power"] = pd.to_numeric(f1["Global_active_power"], errors="coerce")
f1["power_watts"] = f1["Global_active_power"] * 1000

f1 = f1[["timestamp", "power_watts"]].dropna()

f2 = pd.read_csv(
    r"C:\Users\laksh\OneDrive\Documents\Capstone\Codes\file2.txt",
    sep=","
)

f2["timestamp"] = pd.to_datetime(f2["timestamp"], errors="coerce")
f2 = f2.rename(columns={
    "airTemperature": "temperature_c",
    "dewTemperature": "air_quality"
})
f2 = f2[["timestamp", "temperature_c", "air_quality"]].dropna()

f3 = pd.read_csv(
    r"C:\Users\laksh\OneDrive\Documents\Capstone\Codes\file3.csv"
)

f3["timestamp"] = pd.to_datetime(f3["date"], errors="coerce")
f3["temperature_c"] = f3[["T1","T2","T3","T4"]].mean(axis=1)
f3["humidity_percent"] = f3[["RH_1","RH_2","RH_3","RH_4"]].mean(axis=1)
f3["light_level"] = f3["lights"] * 25
f3 = f3[[
    "timestamp",
    "temperature_c",
    "humidity_percent",
    "light_level"
]].dropna()

start = pd.Timestamp("2024-12-21 00:00")
end = start + pd.DateOffset(months=6)
timeline = pd.date_range(start, end, freq="5min")

df = pd.DataFrame({"timestamp": timeline})

df = pd.merge_asof(df, f3.sort_values("timestamp"), on="timestamp", direction="nearest")
df = pd.merge_asof(df, f2.sort_values("timestamp"), on="timestamp", direction="nearest")
df = pd.merge_asof(df, f1.sort_values("timestamp"), on="timestamp", direction="nearest")

if "temperature_c_x" in df.columns and "temperature_c_y" in df.columns:
    df["temperature_c"] = df["temperature_c_x"]
elif "temperature_c" not in df.columns:
    raise RuntimeError("temperature column missing after merge")

df = df.drop(columns=[c for c in df.columns if c.startswith("temperature_c_")])


df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

df["occupancy"] = (
    ((df["hour"] >= 8) & (df["hour"] <= 20) & (df["weekday"] < 5)) |
    ((df["hour"] >= 10) & (df["hour"] <= 18) & (df["weekday"] >= 5))
).astype(int)

seasonal = 2 * np.sin(2 * np.pi * df.index / len(df))
df["temperature_c"] += seasonal
df["humidity_percent"] += np.random.normal(0, 1, len(df))
df["air_quality"] += np.random.normal(0, 5, len(df))

df["power_watts"] = pd.to_numeric(df["power_watts"], errors="coerce")
df["power_watts"] = df["power_watts"].interpolate()
df["power_watts"] = df["power_watts"].infer_objects(copy=False)

df["energy_kwh"] = df["power_watts"] * (5 / 60) / 1000

df = df[[
    "timestamp",
    "temperature_c",
    "humidity_percent",
    "occupancy",
    "light_level",
    "air_quality",
    "power_watts",
    "energy_kwh"
]]

df = df.iloc[:51840]

df.to_csv(
    r"C:\Users\laksh\OneDrive\Documents\Capstone\Codes\dt_train_normal.csv",
    index=False
)
