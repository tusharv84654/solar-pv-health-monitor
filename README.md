# Solar PV Health Monitor

Real-time IoT & ML backend for solar PV telemetry, degradation loss tracking, and fault diagnostics via WebSockets.

## Features
* **Real-Time Telemetry:** Streams solar PV metrics via WebSockets.
* **Fault Diagnostics & ML:** Analyzes degradation loss and operational anomalies using `model.py`.
* **Static Dashboard:** Front-end assets served via `/static`.

## Project Structure
* `main.py` — WebSocket server and telemetry backend
* `model.py` — Machine learning fault diagnostic routines
* `static/` — Web dashboard files
* `requirements.txt` — Python dependencies

## Setup & Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
