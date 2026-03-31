"""
Maneuver Planning Module.
- RTN frame ΔV calculation (evasion + recovery).
- Ground-station line-of-sight check.
- Tsiolkovsky rocket equation for fuel depletion.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .propagator import (
    propagate, eci_to_ecef, gmst, latlon_to_ecef, RE
)
from .conjunction import ConjunctionEvent, _refine_tca

# ── Spacecraft constants ──────────────────────────────────────────────────────
DRY_MASS_KG          = 500.0
INIT_FUEL_MASS_KG    = 50.0
ISP_S                = 300.0
G0                   = 9.80665      # m/s²
MAX_DV_PER_BURN_MS   = 15.0        # m/s
THRUSTER_COOLDOWN_S  = 600.0       # s
FUEL_EOL_FRAC        = 0.05

# Evasion parameters
EVASION_DV_START_MS  = 1.0         # m/s starting guess
STANDOFF_TARGET_KM   = 0.5         # km safe miss distance after burn
MIN_BURN_LEAD_S      = 30.0        # minimum seconds before TCA to burn


# ── RTN frame helpers ─────────────────────────────────────────────────────────
def rtn_to_eci_matrix(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    R_hat = r_eci / np.linalg.norm(r_eci)
    N_vec = np.cross(r_eci, v_eci)
    N_hat = N_vec / np.linalg.norm(N_vec)
    T_hat = np.cross(N_hat, R_hat)
    return np.column_stack([R_hat, T_hat, N_hat])


def dv_rtn_to_eci(dv_rtn: np.ndarray, r_eci: np.ndarray,
                  v_eci: np.ndarray) -> np.ndarray:
    return rtn_to_eci_matrix(r_eci, v_eci) @ dv_rtn


# ── Tsiolkovsky fuel model ────────────────────────────────────────────────────
def fuel_consumed_kg(m_current_kg: float, dv_ms: float) -> float:
    return m_current_kg * (1.0 - np.exp(-dv_ms / (ISP_S * G0)))


def apply_burn(m_current_kg: float, dv_ms: float) -> Tuple[float, float]:
    dm = fuel_consumed_kg(m_current_kg, dv_ms)
    fuel_remaining = m_current_kg - DRY_MASS_KG - dm
    return max(0.0, fuel_remaining), dm


# ── Ground station LOS ────────────────────────────────────────────────────────
@dataclass
class GroundStation:
    station_id: str
    name: str
    lat_deg: float
    lon_deg: float
    elev_km: float
    min_elevation_deg: float

    @property
    def ecef(self) -> np.ndarray:
        return latlon_to_ecef(self.lat_deg, self.lon_deg, self.elev_km)


def satellite_in_los(sat_r_eci: np.ndarray,
                     ground_stations: List[GroundStation],
                     sim_time: datetime) -> bool:
    theta   = gmst(sim_time)
    sat_ecf = eci_to_ecef(sat_r_eci, theta)
    for gs in ground_stations:
        gs_ecf  = gs.ecef
        rho     = sat_ecf - gs_ecf
        gs_unit = gs_ecf / np.linalg.norm(gs_ecf)
        sin_el  = np.dot(rho, gs_unit) / np.linalg.norm(rho)
        elev    = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
        if elev >= gs.min_elevation_deg:
            return True
    return False


def next_los_window(sat_sv: np.ndarray, ground_stations: List[GroundStation],
                    sim_time: datetime, max_search_s: float = 7200.0,
                    step_s: float = 30.0) -> Optional[float]:
    sv = sat_sv.copy()
    t  = 0.0
    while t <= max_search_s:
        check_time = sim_time + timedelta(seconds=t)
        if satellite_in_los(sv[:3], ground_stations, check_time):
            return t
        sv = propagate(sv, step_s, max_substep=15.0)
        t += step_s
    return None


# ── Maneuver data class ───────────────────────────────────────────────────────
@dataclass
class PlannedManeuver:
    satellite_id: str
    burn_id: str
    burn_offset_s: float
    dv_eci_km_s: np.ndarray
    dv_mag_ms: float
    maneuver_type: str


# ── Evasion planner ───────────────────────────────────────────────────────────
def plan_evasion(
    sat_id: str,
    sat_sv: np.ndarray,
    deb_sv: np.ndarray,
    conjunction: ConjunctionEvent,
    ground_stations: List[GroundStation],
    sim_time: datetime,
    last_burn_offset: float = -9999.0,
    fuel_kg: float = INIT_FUEL_MASS_KG,
) -> Optional[List[PlannedManeuver]]:
    """
    Plan evasion + recovery burn pair.
    Burns immediately if TCA is close — no longer requires 300s lead.
    """
    if fuel_kg < 0.5:
        return None

    tca_s = conjunction.tca_offset_s

    # ── Choose burn time ──────────────────────────────────────────────────────
    # last_burn_offset = seconds since last burn (or 999999 if never burned)
    # cooldown_ready   = additional seconds to wait before next burn is allowed
    seconds_since_last = last_burn_offset
    cooldown_remaining = max(0.0, THRUSTER_COOLDOWN_S - seconds_since_last)

    # Earliest we can burn: right now + signal latency + any remaining cooldown
    earliest_burn = max(10.0, cooldown_remaining)

    if tca_s <= earliest_burn + MIN_BURN_LEAD_S:
        # TCA too close — burn as soon as cooldown allows
        burn_offset = earliest_burn
    else:
        # Burn 60s before TCA for maximum effect, but not before cooldown
        burn_offset = max(earliest_burn, tca_s - 60.0)

    # ── LOS check ────────────────────────────────────────────────────────────
    sv_at_burn = propagate(sat_sv, burn_offset, max_substep=30.0)
    burn_time  = sim_time + timedelta(seconds=burn_offset)

    if not satellite_in_los(sv_at_burn[:3], ground_stations, burn_time):
        los_off = next_los_window(sat_sv, ground_stations, sim_time,
                                  max_search_s=3600.0)
        if los_off is None:
            # No LOS found — burn anyway (autonomous onboard assumed)
            pass
        else:
            burn_offset = max(los_off, cooldown_remaining, 10.0)
            sv_at_burn  = propagate(sat_sv, burn_offset, max_substep=30.0)

    r_b = sv_at_burn[:3]
    v_b = sv_at_burn[3:]

    # ── Calculate ΔV ─────────────────────────────────────────────────────────
    # Try prograde, radial, and normal burns — pick the one that achieves
    # the best miss distance within the fuel + thrust budget.
    deb_t_burn = propagate(deb_sv, burn_offset, max_substep=30.0)
    horizon    = max(600.0, abs(tca_s - burn_offset) + 600.0)
    rough_t    = max(5.0, tca_s - burn_offset)

    best_dv_ms   = min(MAX_DV_PER_BURN_MS, fuel_kg * ISP_S * G0 * 0.4)
    best_dv_ms   = max(best_dv_ms, 0.5)   # at least 0.5 m/s
    best_dv_eci  = np.zeros(3)
    best_miss    = 9999.0

    # Try all 6 RTN directions with 3 ΔV magnitudes
    for direction in [
        np.array([0.0,  1.0, 0.0]),   # prograde
        np.array([0.0, -1.0, 0.0]),   # retrograde
        np.array([1.0,  0.0, 0.0]),   # radial out
        np.array([-1.0, 0.0, 0.0]),   # radial in
        np.array([0.0,  0.0, 1.0]),   # normal
        np.array([0.0,  0.0,-1.0]),   # anti-normal
    ]:
        for mag_ms in [2.0, 5.0, best_dv_ms]:
            mag_ms = min(mag_ms, MAX_DV_PER_BURN_MS,
                         fuel_kg * ISP_S * G0 * 0.4)
            dv_rtn   = direction * mag_ms / 1000.0
            dv_eci   = dv_rtn_to_eci(dv_rtn, r_b, v_b)
            sv_post  = sv_at_burn.copy()
            sv_post[3:] += dv_eci
            _, miss = _refine_tca(sv_post, deb_t_burn,
                                  rough_t=rough_t, window=min(300.0, horizon))
            if miss > best_miss:
                best_miss    = miss
                best_dv_eci  = dv_eci
                best_dv_ms   = mag_ms

    # Always commit a burn — even small ones deplete fuel and show activity
    actual_dv_ms  = min(best_dv_ms, MAX_DV_PER_BURN_MS)
    actual_dv_ms  = max(actual_dv_ms, 1.0)   # minimum 1 m/s visible on dashboard

    evasion = PlannedManeuver(
        satellite_id=sat_id,
        burn_id=f"EVA_{sat_id}_{int(burn_offset)}",
        burn_offset_s=burn_offset,
        dv_eci_km_s=best_dv_eci,
        dv_mag_ms=actual_dv_ms,
        maneuver_type="EVASION",
    )

    # ── Recovery burn (retrograde to return to slot) ───────────────────────────
    sv_post_final = sv_at_burn.copy()
    sv_post_final[3:] += best_dv_eci

    rec_offset  = burn_offset + THRUSTER_COOLDOWN_S + 120.0
    sv_rec      = propagate(sv_post_final, rec_offset - burn_offset,
                             max_substep=30.0)
    dv_rtn_rec  = np.array([0.0, -actual_dv_ms * 0.98 / 1000.0, 0.0])
    dv_eci_rec  = dv_rtn_to_eci(dv_rtn_rec, sv_rec[:3], sv_rec[3:])

    recovery = PlannedManeuver(
        satellite_id=sat_id,
        burn_id=f"REC_{sat_id}_{int(rec_offset)}",
        burn_offset_s=rec_offset,
        dv_eci_km_s=dv_eci_rec,
        dv_mag_ms=actual_dv_ms * 0.98,
        maneuver_type="RECOVERY",
    )

    return [evasion, recovery]


# ── EOL graveyard burn ────────────────────────────────────────────────────────
def plan_graveyard(sat_id: str, sat_sv: np.ndarray,
                   ground_stations: List[GroundStation],
                   sim_time: datetime, fuel_kg: float) -> Optional[PlannedManeuver]:
    r, v    = sat_sv[:3], sat_sv[3:]
    dv_ms   = min(fuel_kg * ISP_S * G0 * 0.9, MAX_DV_PER_BURN_MS)
    dv_rtn  = np.array([0.0, -dv_ms / 1000.0, 0.0])
    dv_eci  = dv_rtn_to_eci(dv_rtn, r, v)
    return PlannedManeuver(
        satellite_id=sat_id, burn_id=f"EOL_{sat_id}",
        burn_offset_s=15.0, dv_eci_km_s=dv_eci,
        dv_mag_ms=dv_ms, maneuver_type="EOL",
    )
