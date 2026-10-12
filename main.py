import os
import sqlite3
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

# Attempt to load ML model helper if available, with graceful fallback
try:
    from model import predict_clean_power
except ImportError:
    def predict_clean_power(lux: float, temp: float) -> float:
        """Fallback empirical baseline yield calculation"""
        if lux <= 30.0:
            return 0.0
        # Reference: ~1.0W nominal output at 1000 Lux standard benchmark
        temp_derate = 1.0 - (max(0.0, temp - 25.0) * 0.004)
        expected_w = (lux / 1000.0) * 1.0 * temp_derate
        return max(0.0001, expected_w)

# ---------------------------------------------------------
# App Initialization & CORS
# ---------------------------------------------------------
app = FastAPI(
    title="Solar PV Health Monitor API",
    description="Real-time IoT Telemetry Ingestion, Fault Classification, and Diagnostics Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# SQLite Database Persistence
# ---------------------------------------------------------
DB_FILE = "solar_telemetry.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            voltage REAL,
            current_ma REAL,
            p_actual REAL,
            p_predicted REAL,
            loss_percentage REAL,
            status TEXT,
            temperature REAL,
            humidity REAL,
            lux REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_telemetry_to_db(record: Dict[str, Any]):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO telemetry_history 
            (voltage, current_ma, p_actual, p_predicted, loss_percentage, status, temperature, humidity, lux)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["voltage"],
            record["current_ma"],
            record["p_actual"],
            record["p_predicted"],
            record["loss_percentage"],
            record["status"],
            record["temperature"],
            record["humidity"],
            record["lux"]
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB Error] Failed to log telemetry: {e}")

# ---------------------------------------------------------
# WebSocket Broadcast Manager
# ---------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# ---------------------------------------------------------
# Schemas & In-Memory State
# ---------------------------------------------------------
class TelemetryIn(BaseModel):
    voltage: float
    current_ma: Optional[float] = None
    current: Optional[float] = None  # Compatible with either naming convention
    power: Optional[float] = None
    temperature: Optional[float] = 25.0
    humidity: Optional[float] = 50.0
    lux: Optional[float] = 500.0

class SimulateScenarioRequest(BaseModel):
    scenario: str  # Options: 'clean', 'dust', 'fault', 'night'

latest_telemetry: Dict[str, Any] = {
    "voltage": 0.0,
    "current_ma": 0.0,
    "p_actual": 0.0,
    "p_predicted": 0.0,
    "loss_percentage": 0.0,
    "status": "System Initializing",
    "temperature": 25.0,
    "humidity": 50.0,
    "lux": 0.0,
}

# ---------------------------------------------------------
# Diagnostic Classifier Logic
# ---------------------------------------------------------
def evaluate_diagnostics(voltage: float, current_ma: float, p_actual: float, lux: float, temp: float) -> tuple[float, float, str]:
    """
    Evaluates empirical yield, degradation loss, and detects hardware faults.
    Includes safeguards for low-voltage mini solar panel bench testing.
    """
    # 1. Night / Standby Check
    if lux <= 35.0 and voltage < 0.3 and current_ma < 0.2:
        return 0.0, 0.0, "Night Standby"

    # 2. Predicted baseline output based on light and temperature
    raw_predicted = predict_clean_power(lux, temp)
    p_predicted = round(raw_predicted, 4)

    # 3. Voltage Fault condition
    # Threshold lowered from 4.5V to 0.2V to support small test panels (e.g., 0.7V)
    # Only flags as hardware failure when voltage completely collapses with zero current
    if voltage < 0.2 and current_ma <= 0.1:
        loss_percentage = 100.0
        status = "Hardware Fault - Voltage Failure"
        return p_predicted, loss_percentage, status

    # 4. Standard Degradation Loss percentage calculation
    if p_predicted > 0:
        raw_loss = ((p_predicted - p_actual) / p_predicted) * 100.0
        loss_percentage = round(max(0.0, min(100.0, raw_loss)), 2)
    else:
        loss_percentage = 0.0

    # 5. Fault vs. Dust Classification
    if loss_percentage < 15.0:
        status = "Normal / Clean"
    elif 15.0 <= loss_percentage < 60.0:
        status = "Dust / Soiling Detected"
    elif loss_percentage >= 60.0 and voltage >= 0.5:
        # High loss but voltage is still healthy points to dust, shading, or soiling
        status = "Heavy Soiling / Partial Shading"
    else:
        status = "Hardware Fault - Open Circuit / Cell Degradation"

    return p_predicted, loss_percentage, status

# ---------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Solar Telemetry Backend",
        "connections": len(manager.active_connections)
    }

@app.get("/api/telemetry/latest")
def get_latest_telemetry():
    return latest_telemetry

@app.get("/api/telemetry/history")
def get_telemetry_history(limit: int = 50):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, voltage, current_ma, p_actual, p_predicted, loss_percentage, status, temperature, humidity, lux 
            FROM telemetry_history 
            ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()

        keys = ["id", "timestamp", "voltage", "current_ma", "p_actual", "p_predicted", "loss_percentage", "status", "temperature", "humidity", "lux"]
        return [dict(zip(keys, row)) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/telemetry")
async def ingest_telemetry(payload: TelemetryIn):
    global latest_telemetry

    v = round(float(payload.voltage), 3)
    
    # Accept current either under current_ma or current
    i_ma = payload.current_ma if payload.current_ma is not None else (payload.current or 0.0)
    i_ma = round(float(i_ma), 3)

    # Compute actual power in Watts (P = V * I_A = V * (I_mA / 1000.0))
    # Preserve 4 decimal places so sub-watt test levels (e.g., 0.0006 W) are never zeroed
    if payload.power is not None and payload.power > 0:
        p_actual = round(float(payload.power), 4)
    else:
        p_actual = round((v * (i_ma / 1000.0)), 4)

    lux = float(payload.lux if payload.lux is not None else 0.0)
    temp = float(payload.temperature if payload.temperature is not None else 25.0)
    hum = float(payload.humidity if payload.humidity is not None else 50.0)

    p_predicted, loss_percentage, status = evaluate_diagnostics(v, i_ma, p_actual, lux, temp)

    record = {
        "voltage": v,
        "current_ma": i_ma,
        "p_actual": p_actual,
        "p_predicted": p_predicted,
        "loss_percentage": loss_percentage,
        "status": status,
        "temperature": temp,
        "humidity": hum,
        "lux": lux
    }

    latest_telemetry = record
    log_telemetry_to_db(record)

    # Stream out instantly to all connected Flutter apps
    await manager.broadcast(record)

    return {"status": "recorded", "data": record}

@app.post("/api/telemetry/simulate")
async def simulate_scenario(payload: SimulateScenarioRequest):
    """
    Scenario injector used by the interactive buttons on the mobile application dashboard.
    """
    global latest_telemetry
    scen = payload.scenario.lower().strip()

    if scen == "clean":
        sim_data = {
            "voltage": 5.10,
            "current_ma": 190.0,
            "p_actual": 0.969,
            "p_predicted": 1.000,
            "loss_percentage": 3.10,
            "status": "Normal / Clean",
            "temperature": 27.5,
            "humidity": 45.0,
            "lux": 1000.0
        }
    elif scen == "dust":
        sim_data = {
            "voltage": 5.02,
            "current_ma": 130.0,
            "p_actual": 0.6526,
            "p_predicted": 1.000,
            "loss_percentage": 34.74,
            "status": "Dust / Soiling Detected",
            "temperature": 29.0,
            "humidity": 42.0,
            "lux": 1000.0
        }
    elif scen in ["fault", "hardware_fault"]:
        sim_data = {
            "voltage": 0.05,
            "current_ma": 0.0,
            "p_actual": 0.0000,
            "p_predicted": 0.980,
            "loss_percentage": 100.0,
            "status": "Hardware Fault - Voltage Failure",
            "temperature": 31.0,
            "humidity": 55.0,
            "lux": 980.0
        }
    elif scen == "night":
        sim_data = {
            "voltage": 0.08,
            "current_ma": 0.0,
            "p_actual": 0.0000,
            "p_predicted": 0.000,
            "loss_percentage": 0.0,
            "status": "Night Standby",
            "temperature": 22.0,
            "humidity": 70.0,
            "lux": 5.0
        }
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown scenario '{payload.scenario}'. Use 'clean', 'dust', 'fault', or 'night'."
        )

    latest_telemetry = sim_data
    log_telemetry_to_db(sim_data)
    await manager.broadcast(sim_data)

    return {"status": "simulated", "scenario": scen, "data": sim_data}

# ---------------------------------------------------------
# WebSocket Connection Handler
# ---------------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Deliver current cached frame immediately upon connection
        await websocket.send_json(latest_telemetry)
        while True:
            # Keep heartbeat connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)