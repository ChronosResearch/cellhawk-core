"""FastAPI Ground Control Station application (§2.2).

Endpoints:
  GET  /health                    — system health
  GET  /fleet                     — connected drone IDs
  POST /mission/{drone_id}/waypoint — dispatch waypoint command
  WS   /telemetry/{drone_id}      — real-time telemetry stream
  GET  /danger_grid               — query Danger Grid within radius
  GET  /metrics                   — Prometheus metrics
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from .telemetry import TelemetryHub
from .danger_grid import DangerGrid, DangerEntry
from .crypto import CommandCipher, derive_key

log = structlog.get_logger()

# ── Prometheus metrics ────────────────────────────────────────────────────────
TELEMETRY_FRAMES   = Counter("cellhawk_telemetry_frames_total", "Total telemetry frames received", ["drone_id"])
ACTIVE_DRONES      = Gauge("cellhawk_active_drones", "Currently connected drones")
TIER_GAUGE         = Gauge("cellhawk_navigation_tier", "Active navigation tier", ["drone_id"])
JNR_GAUGE          = Gauge("cellhawk_jnr_db", "Current JNR (dB)", ["drone_id"])

# ── Shared state ──────────────────────────────────────────────────────────────
# Cipher key: in production load from env / secrets manager.
# Fallback to a deterministic dev key so the server starts without config.
import os as _os
_PSK = _os.environb.get(b"CELLHAWK_COMMAND_PSK", b"dev-insecure-psk-change-in-prod")
_cipher = CommandCipher(derive_key(_PSK))

hub   = TelemetryHub(cipher=_cipher)
grid  = DangerGrid()


async def _on_telemetry(drone_id: int, frame: dict) -> None:
    did = str(drone_id)
    TELEMETRY_FRAMES.labels(drone_id=did).inc()
    if "jnr_db" in frame:
        JNR_GAUGE.labels(drone_id=did).set(frame["jnr_db"])
    if "tier" in frame:
        TIER_GAUGE.labels(drone_id=did).set(frame["tier"])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    hub.subscribe(_on_telemetry)
    log.info("cellhawk_gcs_started")
    yield
    await grid.close()
    log.info("cellhawk_gcs_stopped")


app = FastAPI(
    title="CellHawk GCS",
    version="0.1.0",
    description="Ground Control Station for CellHawk triply-redundant navigation",
    lifespan=lifespan,
)

# ── Models ────────────────────────────────────────────────────────────────────

class WaypointRequest(BaseModel):
    east_m:           float = Field(..., description="Target east position (m ENU)")
    north_m:          float = Field(..., description="Target north position (m ENU)")
    altitude_m:       float = Field(..., ge=0.0, description="Target altitude AGL (m)")
    speed_m_s:        float = Field(5.0, ge=0.0, le=30.0)


class DangerQueryRequest(BaseModel):
    east_m:    float
    north_m:   float
    radius_m:  float = Field(500.0, ge=1.0, le=10_000.0)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "connected_drones": hub.connected_drones,
        "timestamp": time.time(),
    }


@app.get("/fleet")
async def fleet() -> dict:
    ACTIVE_DRONES.set(len(hub.connected_drones))
    return {"drone_ids": hub.connected_drones}


@app.post("/mission/{drone_id}/waypoint")
async def dispatch_waypoint(drone_id: int, req: WaypointRequest) -> dict:
    import json
    payload = json.dumps({
        "type": "waypoint",
        "drone_id": drone_id,
        "east_m": req.east_m,
        "north_m": req.north_m,
        "altitude_m": req.altitude_m,
        "speed_m_s": req.speed_m_s,
    }).encode()
    sent = await hub.send_command(drone_id, payload)
    if not sent:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not connected")
    return {"status": "dispatched", "drone_id": drone_id}


@app.websocket("/telemetry/{drone_id}")
async def telemetry_ws(drone_id: int, ws: WebSocket) -> None:
    await hub.connect(drone_id, ws)
    ACTIVE_DRONES.inc()
    try:
        await hub.receive_loop(drone_id, ws)
    finally:
        ACTIVE_DRONES.dec()


@app.post("/danger_grid/query")
async def query_danger_grid(req: DangerQueryRequest) -> dict:
    entries = await grid.query_radius(req.east_m, req.north_m, req.radius_m)
    return {
        "count": len(entries),
        "entries": [
            {
                "drone_id": e.drone_id,
                "east_m": e.east_m,
                "north_m": e.north_m,
                "severity": e.severity,
                "threat_type": e.threat_type,
            }
            for e in entries
        ],
    }


@app.post("/danger_grid/broadcast")
async def broadcast_hazard(entry: DangerEntry) -> dict:
    await grid.broadcast(entry)
    return {"status": "broadcast_ok"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return generate_latest().decode()
