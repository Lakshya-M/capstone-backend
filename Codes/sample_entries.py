import csv
import random
from datetime import datetime, timedelta

start_date = datetime(2025, 1, 1, 0, 0, 0)
end_date = start_date + timedelta(days=90)
interval = timedelta(minutes=5)

power_base = 24
energy = 0
current_time = start_date

with open("24w_bulb_power_energy_3months.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "power(W)", "energy(kWh)"])

    while current_time < end_date:
        power = power_base + random.uniform(-1.2, 1.2)

        if random.random() < 0.01:
            power += random.uniform(-2, 2)

        energy += (power * 5 / 60) / 1000

        writer.writerow([
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            round(power, 2),
            round(energy, 5)
        ])

        current_time += interval
