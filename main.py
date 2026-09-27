import os
import json
import sqlite3
import asyncio
from datetime import datetime
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try importing the model functions; fallback to physical physics heuristic if model file isn't present
try:
    from model import predict_clean_power
except ImportError:
    def predict_clean_power(lux: float, temp: float) -> float:
        # Standard PV empirical approximation: P = (Lux / 1000) * NominalPower * TempDerating
        base_power = (lux / 1000.0) * 2.0
        temp_loss = 1.0 - max(0.0, (temp - 25.0) * 0.004)
        return max(0.0, round(float(base_power * temp_loss), 2))

# --- Database Setup ---
DB_FILE = os.environ.get("DB_PATH", "solar_telemetry.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            lux REAL,
            temperature REAL,
            voltage REAL,
            current_ma REAL,
            p_actual REAL,
            p_predicted REAL,
            loss_percentage REAL,
            status TEXT,
            badge TEXT,
            color TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- App & Middleware Initialization ---
app = FastAPI(title="Solar PV IoT Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Data Schemas ---
class TelemetryInput(BaseModel):
    lux: float
    temperature: float
    voltage: float
    current_ma: float

class SimulationRequest(BaseModel):
    scenario: str  # 'clean', 'dust', 'hardware_fault', 'night'

# In-memory latest state cache
latest_telemetry_cache = {
    "timestamp": datetime.utcnow().isoformat(),
    "lux": 850.0,
    "temperature": 27.0,
    "voltage": 5.85,
    "current_ma": 304.0,
    "p_actual": 1.78,
    "p_predicted": 1.75,
    "loss_percentage": 0.0,
    "status": "NORMAL_CLEAN",
    "badge": "Panel Clean - Optimal Efficiency",
    "color": "green",
}

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead_connections = []
        payload = json.dumps(message)
        for conn in self.active_connections:
            try:
                await conn.send_text(payload)
            except Exception:
                dead_connections.append(conn)
        for dead in dead_connections:
            self.disconnect(dead)

manager = ConnectionManager()

# --- Core Telemetry & Diagnostics Processing ---
def process_telemetry_data(lux: float, temp: float, voltage: float, current_ma: float) -> dict:
    # 1. Calculate measured physical wattage
    p_actual = round(max(0.0, (voltage * current_ma) / 1000.0), 2)

    # 2. Estimate expected theoretical clean wattage
    p_predicted = round(float(predict_clean_power(lux, temp)), 2)

    # 3. Calculate Loss Percentage
    if p_predicted > 0.05:
        loss_percentage = round(max(0.0, ((p_predicted - p_actual) / p_predicted) * 100.0), 1)
    else:
        loss_percentage = 0.0

    # 4. Diagnostic State Engine
    # Night/Idle Condition
    if lux < 30.0:
        status = "NIGHT_IDLE"
        badge = "Night Mode - Idle Standby"
        color = "slate"
        loss_percentage = 0.0
    # Fault: Low power, nominal sun, and critical voltage drop (below 4.5V on a nominal 6V panel)
    elif loss_percentage >= 25.0 and voltage < 4.5:
        status = "HARDWARE_FAULT"
        badge = "Hardware Fault - Voltage Failure"
        color = "amber"
    # Dust: Noticeable generation loss, but voltage stays healthy (current reduced by dust cover)
    elif loss_percentage >= 15.0:
        status = "DUST_DETECTED"
        badge = "Dust Detected - Maintenance Required"
        color = "red"
    # Optimal
    else:
        status = "NORMAL_CLEAN"
        badge = "Panel Clean - Optimal Efficiency"
        color = "green"

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "lux": round(lux, 1),
        "temperature": round(temp, 1),
        "voltage": round(voltage, 2),
        "current_ma": round(current_ma, 1),
        "p_actual": p_actual,
        "p_predicted": p_predicted,
        "loss_percentage": loss_percentage,
        "status": status,
        "badge": badge,
        "color": color,
    }

    # Store in database
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO telemetry 
            (timestamp, lux, temperature, voltage, current_ma, p_actual, p_predicted, loss_percentage, status, badge, color)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["timestamp"], result["lux"], result["temperature"], result["voltage"],
            result["current_ma"], result["p_actual"], result["p_predicted"],
            result["loss_percentage"], result["status"], result["badge"], result["color"]
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database write error: {e}")

    # Update cache
    global latest_telemetry_cache
    latest_telemetry_cache = result

    return result

# --- API Endpoints ---
@app.get("/")
def health_check():
    return {"status": "online", "service": "Solar Telemetry Backend", "connections": len(manager.active_connections)}

@app.get("/api/telemetry/latest")
def get_latest_telemetry():
    return latest_telemetry_cache

@app.post("/api/telemetry")
async def receive_hardware_telemetry(payload: TelemetryInput):
    """
    Endpoint targeted by physical microcontrollers (ESP8266 / ESP32)
    """
    processed = process_telemetry_data(
        lux=payload.lux,
        temp=payload.temperature,
        voltage=payload.voltage,
        current_ma=payload.current_ma
    )
    await manager.broadcast(processed)
    return {"status": "recorded", "data": processed}

@app.post("/api/telemetry/simulate")
async def simulate_scenario(req: SimulationRequest):
    """
    Testing endpoint for UI scenario triggers
    """
    scenario = req.scenario.lower()
    if scenario == "clean":
        lux, temp, volt, current = 880.0, 27.0, 5.85, 304.0
    elif scenario == "dust":
        lux, temp, volt, current = 880.0, 28.0, 5.75, 175.0
    elif scenario == "hardware_fault":
        lux, temp, volt, current = 880.0, 31.0, 3.20, 110.0
    elif scenario == "night":
        lux, temp, volt, current = 5.0, 19.0, 0.20, 0.0
    else:
        raise HTTPException(status_code=400, detail="Invalid scenario")

    processed = process_telemetry_data(lux=lux, temp=temp, voltage=volt, current_ma=current)
    await manager.broadcast(processed)
    return {"status": "simulated", "data": processed}

# --- WebSocket Streaming ---
@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    await manager.connect(websocket)
    # Send the latest known frame immediately upon connection
    await websocket.send_text(json.dumps(latest_telemetry_cache))
    try:
        while True:
            # Keeps the socket connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

# --- Dynamic Port Execution (Compatible with Render / Cloud & Local) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)