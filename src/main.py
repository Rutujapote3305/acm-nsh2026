"""
ACM — Autonomous Constellation Manager
FastAPI backend exposing all required API endpoints on port 8000.
Compatible with Python 3.10, 3.11, 3.12, 3.13 on Windows & Linux.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    TelemetryRequest, TelemetryResponse,
    ManeuverRequest, ManeuverResponse, ManeuverValidation,
    SimStepRequest, SimStepResponse,
)
from .constellation import ConstellationManager

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Autonomous Constellation Manager",
    description="NSH 2026 — Orbital Debris Avoidance & Constellation Management",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Resolve paths (cross-platform: works on Windows + Linux) ────────────────
_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_GS_CSV       = os.environ.get("GS_CSV",
                    os.path.join(_PROJECT_ROOT, "data", "ground_stations.csv"))

# ─── Global constellation manager singleton ───────────────────────────────────
manager = ConstellationManager(ground_station_csv=_GS_CSV)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/telemetry
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/telemetry", response_model=TelemetryResponse)
async def ingest_telemetry(payload: TelemetryRequest):
    """Ingest high-frequency ECI state vector updates for satellites and debris."""
    objects_raw = [
        {
            "id": obj.id,
            "type": obj.type,
            "r": {"x": obj.r.x, "y": obj.r.y, "z": obj.r.z},
            "v": {"x": obj.v.x, "y": obj.v.y, "z": obj.v.z},
        }
        for obj in payload.objects
    ]
    count = manager.ingest_telemetry(payload.timestamp, objects_raw)
    active_warnings = sum(1 for e in manager.cdm_events if e.is_warning)
    return TelemetryResponse(
        status="ACK",
        processed_count=count,
        active_cdm_warnings=active_warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/maneuver/schedule
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/maneuver/schedule", response_model=ManeuverResponse)
async def schedule_maneuver(payload: ManeuverRequest):
    """Schedule a validated burn sequence for a satellite."""
    burns_raw = [
        {
            "burn_id": b.burn_id,
            "burnTime": b.burnTime,
            "deltaV_vector": {
                "x": b.deltaV_vector.x,
                "y": b.deltaV_vector.y,
                "z": b.deltaV_vector.z,
            },
        }
        for b in payload.maneuver_sequence
    ]

    if payload.satelliteId not in manager.satellites:
        raise HTTPException(
            status_code=404,
            detail=f"Satellite '{payload.satelliteId}' not found in constellation."
        )

    los_ok, fuel_ok, proj_mass = manager.schedule_burn_external(
        payload.satelliteId, burns_raw
    )

    return ManeuverResponse(
        status="SCHEDULED",
        validation=ManeuverValidation(
            ground_station_los=los_ok,
            sufficient_fuel=fuel_ok,
            projected_mass_remaining_kg=round(proj_mass, 3),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/simulate/step
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/simulate/step", response_model=SimStepResponse)
async def simulate_step(payload: SimStepRequest):
    """Advance simulation by step_seconds, propagate all objects, execute burns."""
    if payload.step_seconds <= 0 or payload.step_seconds > 86400:
        raise HTTPException(
            status_code=400,
            detail="step_seconds must be in range (0, 86400]."
        )
    collisions, maneuvers = manager.step(payload.step_seconds)
    return SimStepResponse(
        status="STEP_COMPLETE",
        new_timestamp=manager.sim_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        collisions_detected=collisions,
        maneuvers_executed=maneuvers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/visualization/snapshot
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/visualization/snapshot")
async def get_snapshot():
    """Optimised payload for Orbital Insight dashboard (flat debris tuples)."""
    return JSONResponse(content=manager.snapshot())


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    """Fleet-wide health metrics, active CDMs, and maneuver log."""
    return JSONResponse(content=manager.fleet_status())


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/health  (liveness probe used by Docker healthcheck)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "sim_time": manager.sim_time.isoformat()}


# ─────────────────────────────────────────────────────────────────────────────
# Static frontend — mount LAST so /api/* routes always win
# Uses absolute path so it works regardless of working directory (Windows-safe)
# ─────────────────────────────────────────────────────────────────────────────
_frontend_dir = os.path.join(_PROJECT_ROOT, "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True),
              name="frontend")
