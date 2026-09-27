# ☀️ Solar PV Health Monitor & Diagnostics Engine

A real-time IoT and Machine Learning telemetry pipeline designed to monitor photovoltaic panel health, calculate environmental degradation, and distinguish dust accumulation from physical hardware faults.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Render](https://img.shields.io/badge/Hosted_on-Render-46E3B7.svg)](https://render.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 System Architecture

```text
[ Physical Sensors ]  INA219 / DHT11 / LDR
         │
         ▼
[ Microcontroller  ]  ESP8266 / ESP32 (HTTPS Client)
         │
         ▼ (HTTP POST /api/telemetry)
[ Cloud Backend    ]  FastAPI on Render (Dynamic Port & ML Inference)
         │
         ▼ (WSS Broadcast /ws/live)
[ Mobile App       ]  Flutter Client (Real-time Radial Gauges)

##✨ Features

Real-time Telemetry Ingestion: Processes voltage (V), current (mA), irradiance (Lux), and panel temperature (°C) at sub-second intervals.

Predictive ML Clean Baseline: Compares real-time physical wattage against expected empirical yield under current atmospheric conditions.

Intelligent Fault vs. Dust Classification:

Normal / Clean: System generation operates within nominal thresholds (Loss < 15%).

Dust / Soiling: Current output drops significantly while voltage remains within normal operational limits (Loss >= 15%).

Hardware Fault: Panel generation collapses alongside critical voltage drops (Loss >= 25%, V < 4.5V).

Night Standby: Automatic low-light detection preserves system status without false alarms.

Bidirectional WebSocket Feed: Instantly streams processed diagnostic states directly to cross-platform mobile apps.

Built-in Scenario Simulation: Dedicated endpoints allowing engineers to simulate panel conditions without waiting for specific weather events.

## 🛠️ Tech Stack

Backend Framework: FastAPI, Uvicorn (ASGI)

ML / Analytics: Scikit-Learn, NumPy, Joblib

Data Schemas: Pydantic v2

Storage: SQLite

Deployment: Render Cloud Platform
