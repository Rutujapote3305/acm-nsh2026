"""
Orbital Propagator: RK4 numerical integration with J2 perturbation.
All distances in km, velocities in km/s, time in seconds.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Tuple

# ── Physical constants ────────────────────────────────────────────────────────
MU = 398600.4418          # km³/s²  Earth gravitational parameter
RE = 6378.137             # km      Earth equatorial radius
J2 = 1.08263e-3           # J2 zonal harmonic
OMEGA_EARTH = 7.2921150e-5  # rad/s  Earth rotation rate
J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ── J2 perturbed equations of motion ─────────────────────────────────────────
def _j2_accel(r: np.ndarray) -> np.ndarray:
    """J2 oblateness perturbation acceleration in ECI (km/s²)."""
    x, y, z = r
    r_mag = np.linalg.norm(r)
    factor = (3.0 / 2.0) * J2 * MU * RE ** 2 / r_mag ** 5
    ratio = 5.0 * z ** 2 / r_mag ** 2
    return factor * np.array([
        x * (ratio - 1.0),
        y * (ratio - 1.0),
        z * (ratio - 3.0),
    ])


def _eom(state: np.ndarray) -> np.ndarray:
    """State derivative: [vx,vy,vz, ax,ay,az]."""
    r = state[:3]
    v = state[3:]
    r_mag = np.linalg.norm(r)
    a_grav = -(MU / r_mag ** 3) * r
    a_j2 = _j2_accel(r)
    return np.concatenate([v, a_grav + a_j2])


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """Single RK4 integration step."""
    k1 = _eom(state)
    k2 = _eom(state + 0.5 * dt * k1)
    k3 = _eom(state + 0.5 * dt * k2)
    k4 = _eom(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate(state: np.ndarray, dt_seconds: float,
              max_substep: float = 30.0) -> np.ndarray:
    """Propagate state vector forward by dt_seconds using RK4."""
    t = 0.0
    while t < dt_seconds:
        h = min(max_substep, dt_seconds - t)
        state = rk4_step(state, h)
        t += h
    return state


def propagate_many(states: np.ndarray, dt_seconds: float,
                   max_substep: float = 30.0) -> np.ndarray:
    """Vectorised propagation for a (N, 6) array of state vectors."""
    result = states.copy()
    t = 0.0
    while t < dt_seconds:
        h = min(max_substep, dt_seconds - t)
        for i in range(len(result)):
            result[i] = rk4_step(result[i], h)
        t += h
    return result


# ── Coordinate helpers ────────────────────────────────────────────────────────
def gmst(sim_time: datetime) -> float:
    """Greenwich Mean Sidereal Time (radians) for a given UTC datetime."""
    days = (sim_time - J2000).total_seconds() / 86400.0
    theta_deg = 280.46061837 + 360.98564736629 * days
    return np.radians(theta_deg % 360.0)


def eci_to_ecef(r_eci: np.ndarray, theta_gmst: float) -> np.ndarray:
    """Rotate ECI position to ECEF using GMST angle (rad)."""
    c, s = np.cos(theta_gmst), np.sin(theta_gmst)
    Rz = np.array([[c, s, 0.0],
                   [-s, c, 0.0],
                   [0.0, 0.0, 1.0]])
    return Rz @ r_eci


def ecef_to_latlon(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    """ECEF → (lat_deg, lon_deg, alt_km)."""
    x, y, z = r_ecef
    r_mag = np.linalg.norm(r_ecef)
    lat = np.degrees(np.arcsin(np.clip(z / r_mag, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    alt = r_mag - RE
    return lat, lon, alt


def eci_to_latlon(r_eci: np.ndarray,
                  sim_time: datetime) -> Tuple[float, float, float]:
    """ECI position → (lat_deg, lon_deg, alt_km)."""
    theta = gmst(sim_time)
    r_ecef = eci_to_ecef(r_eci, theta)
    return ecef_to_latlon(r_ecef)


def latlon_to_ecef(lat_deg: float, lon_deg: float,
                   alt_km: float = 0.0) -> np.ndarray:
    """Geographic (deg, deg, km) → ECEF (km)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = RE + alt_km
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat),
    ])


# ── Keplerian → ECI (for initial condition generation) ───────────────────────
def keplerian_to_eci(a: float, e: float, i_deg: float,
                     raan_deg: float, omega_deg: float,
                     M_deg: float) -> np.ndarray:
    """Convert Keplerian elements to ECI state vector [r(km), v(km/s)]."""
    i = np.radians(i_deg)
    raan = np.radians(raan_deg)
    omega = np.radians(omega_deg)
    M = np.radians(M_deg)

    # Solve Kepler's equation: M = E - e*sin(E)
    E = M
    for _ in range(100):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break

    # True anomaly
    nu = 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2.0),
        np.sqrt(1 - e) * np.cos(E / 2.0)
    )

    # Distance
    r_scalar = a * (1.0 - e * np.cos(E))

    # Perifocal frame
    r_pf = r_scalar * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = (np.sqrt(MU / a) / (1.0 - e * np.cos(E))) * np.array(
        [-np.sin(E), np.sqrt(1 - e ** 2) * np.cos(E), 0.0]
    )

    # Rotation matrices
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(omega), np.sin(omega)

    R = np.array([
        [cos_raan * cos_w - sin_raan * sin_w * cos_i,
         -cos_raan * sin_w - sin_raan * cos_w * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_w + cos_raan * sin_w * cos_i,
         -sin_raan * sin_w + cos_raan * cos_w * cos_i,
         -cos_raan * sin_i],
        [sin_w * sin_i, cos_w * sin_i, cos_i],
    ])

    r_eci = R @ r_pf
    v_eci = R @ v_pf
    return np.concatenate([r_eci, v_eci])


def orbital_period(a: float) -> float:
    """Orbital period in seconds for semi-major axis a (km)."""
    return 2.0 * np.pi * np.sqrt(a ** 3 / MU)
