"""Pydantic models for all API request/response bodies."""

from __future__ import annotations
from typing import List, Optional, Tuple, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# Telemetry
# ─────────────────────────────────────────────────────────────
class Vec3(BaseModel):
    x: float
    y: float
    z: float

    def to_list(self):
        return [self.x, self.y, self.z]


class TelemetryObject(BaseModel):
    id: str
    type: str          # "SATELLITE" | "DEBRIS"
    r: Vec3
    v: Vec3


class TelemetryRequest(BaseModel):
    timestamp: str
    objects: List[TelemetryObject]


class TelemetryResponse(BaseModel):
    status: str = "ACK"
    processed_count: int
    active_cdm_warnings: int


# ─────────────────────────────────────────────────────────────
# Maneuver scheduling
# ─────────────────────────────────────────────────────────────
class BurnCommand(BaseModel):
    burn_id: str
    burnTime: str
    deltaV_vector: Vec3


class ManeuverRequest(BaseModel):
    satelliteId: str
    maneuver_sequence: List[BurnCommand]


class ManeuverValidation(BaseModel):
    ground_station_los: bool
    sufficient_fuel: bool
    projected_mass_remaining_kg: float


class ManeuverResponse(BaseModel):
    status: str = "SCHEDULED"
    validation: ManeuverValidation


# ─────────────────────────────────────────────────────────────
# Simulation step
# ─────────────────────────────────────────────────────────────
class SimStepRequest(BaseModel):
    step_seconds: float


class SimStepResponse(BaseModel):
    status: str = "STEP_COMPLETE"
    new_timestamp: str
    collisions_detected: int
    maneuvers_executed: int


# ─────────────────────────────────────────────────────────────
# Visualization snapshot
# ─────────────────────────────────────────────────────────────
class SatelliteSnapshot(BaseModel):
    id: str
    lat: float
    lon: float
    alt: float
    fuel_kg: float
    status: str
    active_conjunctions: int = 0


class VisualizationSnapshot(BaseModel):
    timestamp: str
    satellites: List[SatelliteSnapshot]
    # Flattened tuples: [id, lat, lon, alt]
    debris_cloud: List[List[Any]]


# ─────────────────────────────────────────────────────────────
# Constellation status (extra endpoint)
# ─────────────────────────────────────────────────────────────
class CDMInfo(BaseModel):
    satellite_id: str
    debris_id: str
    tca_offset_s: float
    miss_distance_km: float
    risk_level: str


class ManeuverLog(BaseModel):
    satellite_id: str
    burn_id: str
    scheduled_at_offset: float
    dv_mag_ms: float
    maneuver_type: str
    executed: bool


class FleetStatus(BaseModel):
    sim_time: str
    total_satellites: int
    total_debris: int
    active_cdms: int
    critical_cdms: int
    scheduled_maneuvers: int
    executed_maneuvers: int
    total_dv_consumed_ms: float
    conjunctions: List[CDMInfo]
    maneuver_log: List[ManeuverLog]
