# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 07:15:04 2025
@author: Ahmed Abdelaziz Elsayed
"""

import psutil
import time
import csv
from pynvml import *
from datetime import datetime
import os

# Initialize NVML for GPU monitoring
nvmlInit()
gpu_count = nvmlDeviceGetCount()

# Output CSV file
filename = "Add_File_Name.csv"

# Experiment setup
report_interval = 0.02  # 20 ms (Report Interval)
duration = 15 * 60      # (Report Duration)
iterations = int(duration / report_interval)

# CSV header
headers = [
    "timestamp",
    "cpu_utilization_percent",
    "cpu_freq_MHz",
    "cpu_power_W",
    "cpu_temp_C"
]

for i in range(gpu_count):
    headers.extend([
        f"gpu{i}_utilization_percent",
        f"gpu{i}_mem_utilization",
        f"gpu{i}_Power_TDP",
        f"gpu{i}_mem_used_MB",
        f"gpu{i}_mem_total_MB",
        f"gpu{i}_power_W",
        f"gpu{i}_temp_C"
    ])

# ---- CPU power support (Linux RAPL) ----
rapl_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
rapl_prev_time = None
rapl_prev_energy = None

def read_cpu_power():
    global rapl_prev_time, rapl_prev_energy
    if not os.path.exists(rapl_path):
        return None  # Not supported
    try:
        with open(rapl_path, "r") as f:
            energy = int(f.read().strip())  # microjoules
        now = time.time()
        if rapl_prev_time is not None:
            delta_e = energy - rapl_prev_energy
            delta_t = now - rapl_prev_time
            if delta_e < 0:  # handle counter wrap
                return None
            power = (delta_e / 1e6) / delta_t  # W = J/s
        else:
            power = None
        rapl_prev_energy = energy
        rapl_prev_time = now
        return power
    except:
        return None

# ---- CPU info ----
def get_cpu_info():
    utilization = psutil.cpu_percent(interval=None)
    freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
    cpu_temp = None
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            cpu_temp = temps["coretemp"][0].current
        elif "k10temp" in temps:
            cpu_temp = temps["k10temp"][0].current
    except:
        pass
    cpu_power = read_cpu_power()
    return utilization, freq, cpu_power, cpu_temp

# ---- GPU info ----
def get_gpu_info():
    gpu_data = []
    for i in range(gpu_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        util = nvmlDeviceGetUtilizationRates(handle).gpu
        mem_util = nvmlDeviceGetUtilizationRates(handle).memory
        GPU_Power_TDP = (nvmlDeviceGetPowerUsage(handle) /
                         nvmlDeviceGetEnforcedPowerLimit(handle)) * 100
        mem_used = nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        mem_total = nvmlDeviceGetMemoryInfo(handle).total / 1024**2
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0
        temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        gpu_data.extend([util, mem_util, GPU_Power_TDP,
                         mem_used, mem_total, power, temp])
    return gpu_data

# ---- Main loop ----
data_records = []
start_time = time.time()

for i in range(iterations):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    cpu_util, cpu_freq, cpu_power, cpu_temp = get_cpu_info()
    row = [timestamp, cpu_util, cpu_freq, cpu_power, cpu_temp]
    row.extend(get_gpu_info())
    data_records.append(row)

    # Progress bar
    progress = (i + 1) / iterations
    bar_length = 40
    filled = int(bar_length * progress)
    bar = "█" * filled + "-" * (bar_length - filled)
    remaining_time = duration - (time.time() - start_time)
    print(f"\r[{bar}] {progress*100:5.1f}% | {remaining_time:6.1f}s left", end="")

    elapsed = time.time() - start_time
    expected = (i + 1) * report_interval
    sleep_time = expected - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(data_records)

nvmlShutdown()
print(f"\nRecording finished. Data saved to {filename}")
