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
