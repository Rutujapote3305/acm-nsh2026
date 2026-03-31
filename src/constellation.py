"""
ConstellationManager — ACM brain.
Debris is seeded with TCA 5-30 minutes away so burns actually execute.
"""
from __future__ import annotations

import csv, threading, time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np

from .physics.propagator import (
    propagate, keplerian_to_eci, eci_to_latlon, RE, MU
)
from .physics.conjunction import (
    assess_conjunctions, quick_distance_check, ConjunctionEvent,
    CRITICAL_DIST_KM, WARNING_DIST_KM
)
from .physics.maneuver import (
    GroundStation, plan_evasion, plan_graveyard, apply_burn,
    satellite_in_los, DRY_MASS_KG, INIT_FUEL_MASS_KG,
    FUEL_EOL_FRAC, THRUSTER_COOLDOWN_S
)

STATUS_NOMINAL    = "NOMINAL"
STATUS_EVADING    = "EVADING"
STATUS_RECOVERING = "RECOVERING"
STATUS_EOL        = "EOL"
STATUS_LOST       = "LOST"

# 1 real second = SIM_SPEED_X sim-seconds
SIM_SPEED_X  = 30
AUTO_STEP_S  = 1.0    # run step every 1 real second
CA_INTERVAL  = 3      # run CA every 3 steps


@dataclass
class SatelliteRecord:
    sat_id: str
    state_vector: np.ndarray
    nominal_slot: np.ndarray
    fuel_kg: float = INIT_FUEL_MASS_KG
    status: str = STATUS_NOMINAL
    total_dv_ms: float = 0.0
    eol_scheduled: bool = False

    @property
    def wet_mass_kg(self):   return DRY_MASS_KG + self.fuel_kg
    @property
    def fuel_fraction(self): return self.fuel_kg / INIT_FUEL_MASS_KG


@dataclass
class ScheduledBurn:
    satellite_id: str
    burn_id: str
    burn_time: datetime
    dv_eci_km_s: np.ndarray
    dv_mag_ms: float
    maneuver_type: str
    executed: bool = False
    from_api: bool = False


class ConstellationManager:
    def __init__(self, ground_station_csv: str = "data/ground_stations.csv"):
        self._lock             = threading.Lock()
        self.sim_time          = datetime(2026, 3, 12, 8, 0, 0, tzinfo=timezone.utc)
        self.satellites: Dict[str, SatelliteRecord] = {}
        self.debris:     Dict[str, np.ndarray]       = {}
        self.scheduled_burns: List[ScheduledBurn]    = []
        self.cdm_events:      List[ConjunctionEvent] = []
        self.ground_stations: List[GroundStation]    = []
        self._maneuver_cooldowns: Dict[str, float]   = {}
        self._sim_epoch_s  = 0.0
        self._step_counter = 0
        self.total_dv_ms   = 0.0
        self.collisions_total  = 0
        self.maneuvers_total   = 0

        self._load_ground_stations(ground_station_csv)
        self._init_constellation()
        self._seed_debris()

        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()
        self._auto_thread = t

    # ── Ground stations ───────────────────────────────────────────────────────
    def _load_ground_stations(self, path: str):
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    self.ground_stations.append(GroundStation(
                        station_id=row["Station_ID"].strip(),
                        name=row["Station_Name"].strip(),
                        lat_deg=float(row["Latitude"]),
                        lon_deg=float(row["Longitude"]),
                        elev_km=float(row["Elevation_m"]) / 1000.0,
                        min_elevation_deg=float(row["Min_Elevation_Angle_deg"]),
                    ))
        except Exception as e:
            print(f"[WARN] Ground stations: {e}")

    # ── Constellation ─────────────────────────────────────────────────────────
    def _init_constellation(self):
        a = RE + 550.0
        idx = 0
        for p in range(5):
            raan = 72.0 * p
            for s in range(11):
                M   = (360.0 / 11) * s
                sid = f"SAT-Alpha-{idx+1:02d}"
                sv  = keplerian_to_eci(a, 0.001, 53.0, raan, 0.0, M)
                self.satellites[sid] = SatelliteRecord(
                    sat_id=sid, state_vector=sv.copy(), nominal_slot=sv[:3].copy()
                )
                idx += 1

    # ── Debris seeding ────────────────────────────────────────────────────────
    def _seed_debris(self):
        """
        Key insight: debris must be far enough away that TCA is 5-30 min
        in the future, giving the planner time to schedule and execute a burn.

        At LEO speeds (~7.6 km/s), relative velocities of 10-100 m/s mean:
          - 500m away at 2 m/s closing → TCA in 250s (~4 min) ✅
          - 2km  away at 5 m/s closing → TCA in 400s (~7 min) ✅
          - 5km  away at 10 m/s closing → TCA in 500s (~8 min) ✅
        """
        rng      = np.random.default_rng(42)
        sat_list = list(self.satellites.values())

        # ── 15 CRITICAL (TCA in 3-10 min, final miss < 100m) ─────────────────
        for i in range(15):
            sat  = sat_list[i % len(sat_list)]
            sv_s = sat.state_vector.copy()

            # Direction of approach
            d = rng.standard_normal(3); d /= np.linalg.norm(d)

            # Start 300-900m away
            sep_km   = rng.uniform(0.3, 0.9)
            r_deb    = sv_s[:3] + d * sep_km

            # Close at 0.8-2.5 m/s → TCA in 120-1100 seconds
            closing  = rng.uniform(0.0008, 0.0025)   # km/s
            v_deb    = sv_s[3:] + (-d) * closing

            self.debris[f"DEB-CRIT-{i+1:02d}"] = np.concatenate([r_deb, v_deb])

        # ── 20 WARNING (TCA in 5-20 min, final miss 100m-1km) ────────────────
        for i in range(20):
            sat  = sat_list[(i+15) % len(sat_list)]
            sv_s = sat.state_vector.copy()
            d    = rng.standard_normal(3); d /= np.linalg.norm(d)

            sep_km  = rng.uniform(0.8, 3.0)
            r_deb   = sv_s[:3] + d * sep_km
            closing = rng.uniform(0.001, 0.004)
            # Add slight lateral offset so TCA miss is 100m-1km
            lateral = np.cross(d, sv_s[3:] / np.linalg.norm(sv_s[3:]))
            lateral /= np.linalg.norm(lateral)
            v_deb   = sv_s[3:] + (-d) * closing + lateral * rng.uniform(0.0001, 0.0005)

            self.debris[f"DEB-WARN-{i+1:02d}"] = np.concatenate([r_deb, v_deb])

        # ── 25 CAUTION (approaching, miss 1-5km) ─────────────────────────────
        for i in range(25):
            sat  = sat_list[(i+35) % len(sat_list)]
            sv_s = sat.state_vector.copy()
            d    = rng.standard_normal(3); d /= np.linalg.norm(d)
            sep_km  = rng.uniform(2.0, 15.0)
            r_deb   = sv_s[:3] + d * sep_km
            closing = rng.uniform(0.0005, 0.002)
            lateral = rng.standard_normal(3); lateral -= np.dot(lateral,d)*d
            lateral /= (np.linalg.norm(lateral) + 1e-9)
            v_deb   = sv_s[3:] + (-d) * closing + lateral * rng.uniform(0.001, 0.003)
            self.debris[f"DEB-CAUT-{i+1:02d}"] = np.concatenate([r_deb, v_deb])

        # ── 80 Crossing (high/retrograde incl — produce CDMs over time) ───────
        for i in range(80):
            alt  = rng.uniform(530.0, 570.0)
            incl = float(rng.choice([rng.uniform(90.0, 105.0),
                                     rng.uniform(145.0, 165.0)]))
            raan = rng.uniform(0.0, 360.0)
            M    = rng.uniform(0.0, 360.0)
            self.debris[f"DEB-XNG-{i+1:03d}"] = keplerian_to_eci(
                RE + alt, 0.002, incl, raan, 0.0, M
            )

        # ── 60 Background ─────────────────────────────────────────────────────
        for i in range(60):
            alt  = rng.uniform(480.0, 620.0)
            incl = rng.uniform(30.0, 75.0)
            raan = rng.uniform(0.0, 360.0)
            M    = rng.uniform(0.0, 360.0)
            self.debris[f"DEB-BG-{i+1:04d}"] = keplerian_to_eci(
                RE + alt, 0.001, incl, raan, 0.0, M
            )

    def _refresh_debris(self):
        """
        Periodically inject fresh threatening debris so CDMs keep appearing
        throughout the sim. Runs every CA_INTERVAL steps.
        """
        rng      = np.random.default_rng(int(self._sim_epoch_s * 1000) % 2**31)
        sat_list = list(self.satellites.values())

        for i in range(5):
            sat  = sat_list[rng.integers(0, len(sat_list))]
            sv_s = sat.state_vector.copy()
            d    = rng.standard_normal(3); d /= np.linalg.norm(d)

            sep_km  = rng.uniform(0.5, 5.0)
            r_deb   = sv_s[:3] + d * sep_km
            closing = rng.uniform(0.001, 0.003)
            v_deb   = sv_s[3:] + (-d) * closing

            idx = (i % 15) + 1
            self.debris[f"DEB-CRIT-{idx:02d}"] = np.concatenate([r_deb, v_deb])

    # ── Telemetry ─────────────────────────────────────────────────────────────
    def ingest_telemetry(self, timestamp: str, objects: list) -> int:
        count = 0
        with self._lock:
            for obj in objects:
                sv = np.array([obj["r"]["x"], obj["r"]["y"], obj["r"]["z"],
                               obj["v"]["x"], obj["v"]["y"], obj["v"]["z"]])
                if obj["type"] == "SATELLITE":
                    if obj["id"] in self.satellites:
                        self.satellites[obj["id"]].state_vector = sv
                    else:
                        self.satellites[obj["id"]] = SatelliteRecord(
                            sat_id=obj["id"], state_vector=sv, nominal_slot=sv[:3].copy()
                        )
                else:
                    self.debris[obj["id"]] = sv
                count += 1
        return count

    # ── External maneuver scheduling ──────────────────────────────────────────
    def schedule_burn_external(self, satellite_id: str,
                                burns: list) -> Tuple[bool, bool, float]:
        with self._lock:
            if satellite_id not in self.satellites:
                return False, False, 0.0
            sat      = self.satellites[satellite_id]
            sim_fuel = sat.fuel_kg
            los_ok   = True
            for burn in burns:
                bt  = datetime.fromisoformat(burn["burnTime"].replace("Z", "+00:00"))
                off = (bt - self.sim_time).total_seconds()
                sv_b = propagate(sat.state_vector, max(0, off))
                if not satellite_in_los(sv_b[:3], self.ground_stations, bt):
                    los_ok = False
                dv  = burn["deltaV_vector"]
                dv_arr = np.array([dv["x"], dv["y"], dv["z"]])
                dv_ms  = float(np.linalg.norm(dv_arr)) * 1000.0
                sim_fuel, _ = apply_burn(DRY_MASS_KG + sim_fuel, dv_ms)
                self.scheduled_burns.append(ScheduledBurn(
                    satellite_id=satellite_id,
                    burn_id=burn["burn_id"], burn_time=bt,
                    dv_eci_km_s=dv_arr, dv_mag_ms=dv_ms,
                    maneuver_type="API", from_api=True,
                ))
            return los_ok, sim_fuel >= 0.0, DRY_MASS_KG + max(0.0, sim_fuel)

    # ── Core step ─────────────────────────────────────────────────────────────
    def step(self, dt_seconds: float) -> Tuple[int, int]:
        with self._lock:
            end_time = self.sim_time + timedelta(seconds=dt_seconds)
            sub      = min(30.0, dt_seconds)
            manvs    = 0
            t        = 0.0

            while t < dt_seconds:
                h       = min(sub, dt_seconds - t)
                t_abs   = self.sim_time + timedelta(seconds=t)
                t_abs_end = t_abs + timedelta(seconds=h)

                # ── Execute any burns due in this substep window ──────────────
                for burn in self.scheduled_burns:
                    if burn.executed:
                        continue
                    if t_abs <= burn.burn_time <= t_abs_end:
                        manvs += self._execute_burn(burn)

                # ── Propagate satellites ──────────────────────────────────────
                for sat in self.satellites.values():
                    if sat.status != STATUS_LOST:
                        sat.state_vector = propagate(
                            sat.state_vector, h, max_substep=h
                        )
                        nom = np.zeros(6)
                        nom[:3] = sat.nominal_slot
                        nom[3:] = sat.state_vector[3:]
                        sat.nominal_slot = propagate(nom, h, max_substep=h)[:3]

                # ── Propagate debris ──────────────────────────────────────────
                for k in list(self.debris):
                    self.debris[k] = propagate(self.debris[k], h, max_substep=h)

                t += h

            self.sim_time      = end_time
            self._sim_epoch_s += dt_seconds
            self._step_counter += 1

            # Collision check
            sat_pos    = {sid: s.state_vector for sid, s in self.satellites.items()
                         if s.status != STATUS_LOST}
            collisions = quick_distance_check(sat_pos, self.debris, CRITICAL_DIST_KM)

            # Run CA + planning every CA_INTERVAL steps
            if self._step_counter % CA_INTERVAL == 0:
                self._refresh_debris()
                self._run_ca_and_plan()

            return collisions, manvs

    def _execute_burn(self, burn: ScheduledBurn) -> int:
        sat = self.satellites.get(burn.satellite_id)
        if not sat or sat.fuel_kg <= 0.01:
            return 0

        # Apply ΔV and deduct fuel
        sat.state_vector[3:] += burn.dv_eci_km_s
        new_fuel, _  = apply_burn(sat.wet_mass_kg, burn.dv_mag_ms)
        sat.fuel_kg  = new_fuel
        sat.total_dv_ms  += burn.dv_mag_ms
        self.total_dv_ms += burn.dv_mag_ms

        self._maneuver_cooldowns[burn.satellite_id] = self._sim_epoch_s
        burn.executed = True

        # Update status
        if burn.maneuver_type == "EVASION":    sat.status = STATUS_EVADING
        elif burn.maneuver_type == "RECOVERY": sat.status = STATUS_NOMINAL
        elif burn.maneuver_type == "EOL":      sat.status = STATUS_EOL

        print(f"[BURN] {burn.satellite_id} {burn.maneuver_type} "
              f"Δv={burn.dv_mag_ms:.3f}m/s  fuel_left={sat.fuel_kg:.2f}kg")

        # Schedule EOL if nearly empty
        if sat.fuel_fraction <= FUEL_EOL_FRAC and not sat.eol_scheduled:
            sat.eol_scheduled = True
            m = plan_graveyard(sat.sat_id, sat.state_vector,
                               self.ground_stations, self.sim_time, sat.fuel_kg)
            if m:
                self.scheduled_burns.append(ScheduledBurn(
                    satellite_id=sat.sat_id, burn_id=m.burn_id,
                    burn_time=self.sim_time + timedelta(seconds=m.burn_offset_s),
                    dv_eci_km_s=m.dv_eci_km_s, dv_mag_ms=m.dv_mag_ms,
                    maneuver_type="EOL",
                ))
        return 1

    def _run_ca_and_plan(self):
        sat_svs = {sid: s.state_vector for sid, s in self.satellites.items()
                   if s.status not in (STATUS_LOST, STATUS_EOL)}

        self.cdm_events = assess_conjunctions(sat_svs, self.debris, self.sim_time)

        planned = {b.satellite_id for b in self.scheduled_burns
                   if not b.executed and b.maneuver_type in ("EVASION", "RECOVERY")}

        for ev in self.cdm_events:
            if not ev.is_critical:
                continue
            if ev.satellite_id in planned:
                continue
            sat = self.satellites.get(ev.satellite_id)
            if not sat or sat.fuel_kg < 0.5:
                continue
            deb_sv = self.debris.get(ev.debris_id)
            if deb_sv is None:
                continue

            last = self._maneuver_cooldowns.get(ev.satellite_id, -99999.0)
            burns = plan_evasion(
                ev.satellite_id, sat.state_vector, deb_sv, ev,
                self.ground_stations, self.sim_time,
                last_burn_offset=(self._sim_epoch_s - last) if last > -9999 else 999999,
                fuel_kg=sat.fuel_kg,
            )
            if burns:
                for m in burns:
                    self.scheduled_burns.append(ScheduledBurn(
                        satellite_id=ev.satellite_id,
                        burn_id=m.burn_id,
                        burn_time=self.sim_time + timedelta(seconds=m.burn_offset_s),
                        dv_eci_km_s=m.dv_eci_km_s,
                        dv_mag_ms=m.dv_mag_ms,
                        maneuver_type=m.maneuver_type,
                    ))
                planned.add(ev.satellite_id)

    # ── Background thread ─────────────────────────────────────────────────────
    def _auto_loop(self):
        while True:
            _time.sleep(AUTO_STEP_S)
            try:
                c, m = self.step(AUTO_STEP_S * SIM_SPEED_X)
                self.collisions_total += c
                self.maneuvers_total  += m
            except Exception as e:
                print(f"[AUTO-LOOP ERROR] {e}")

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            conj_map = {}
            for ev in self.cdm_events:
                conj_map[ev.satellite_id] = conj_map.get(ev.satellite_id, 0) + 1

            sats = []
            for sat in self.satellites.values():
                lat, lon, alt = eci_to_latlon(sat.state_vector[:3], self.sim_time)
                sats.append({
                    "id": sat.sat_id,
                    "lat": float(round(lat, 4)),
                    "lon": float(round(lon, 4)),
                    "alt": float(round(alt, 2)),
                    "fuel_kg": round(sat.fuel_kg, 3),
                    "status": sat.status,
                    "active_conjunctions": conj_map.get(sat.sat_id, 0),
                })

            deb = []
            for did, sv in self.debris.items():
                lat, lon, alt = eci_to_latlon(sv[:3], self.sim_time)
                deb.append([did, float(round(lat,3)),
                            float(round(lon,3)), float(round(alt,1))])

            return {
                "timestamp": self.sim_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "satellites": sats,
                "debris_cloud": deb,
            }

    # ── Fleet status ──────────────────────────────────────────────────────────
    def fleet_status(self) -> dict:
        with self._lock:
            sched = [b for b in self.scheduled_burns if not b.executed]
            done  = [b for b in self.scheduled_burns if b.executed]
            crit  = sum(1 for e in self.cdm_events if e.is_critical)
            return {
                "sim_time": self.sim_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "total_satellites": len(self.satellites),
                "total_debris": len(self.debris),
                "active_cdms": len(self.cdm_events),
                "critical_cdms": crit,
                "scheduled_maneuvers": len(sched),
                "executed_maneuvers": len(done),
                "total_dv_consumed_ms": round(self.total_dv_ms, 4),
                "collisions_total": self.collisions_total,
                "conjunctions": [
                    {
                        "satellite_id": e.satellite_id,
                        "debris_id": e.debris_id,
                        "tca_offset_s": round(e.tca_offset_s, 1),
                        "miss_distance_km": round(e.miss_distance_km, 4),
                        "risk_level": e.risk_level,
                    }
                    for e in self.cdm_events[:20]
                ],
                "maneuver_log": [
                    {
                        "satellite_id": b.satellite_id,
                        "burn_id": b.burn_id,
                        "scheduled_at": b.burn_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "dv_mag_ms": round(b.dv_mag_ms, 4),
                        "maneuver_type": b.maneuver_type,
                        "executed": b.executed,
                    }
                    for b in self.scheduled_burns[-50:]
                ],
            }
