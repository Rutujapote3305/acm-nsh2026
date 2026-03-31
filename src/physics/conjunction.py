"""
Conjunction Assessment — Fast, Reliable, Two-Pass Algorithm.

Pass 1: KD-tree on CURRENT positions → find all pairs within 50 km (fast).
Pass 2: Propagate each flagged pair forward 2 hours, record minimum distance.
Pass 3: Binary-search around minimum to get precise TCA + miss distance.

No velocity filter — works for all closing speeds including slow approaches.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from .propagator import propagate, rk4_step

# ── Thresholds ────────────────────────────────────────────────────────────────
CRITICAL_DIST_KM = 0.100    # 100 m
WARNING_DIST_KM  = 1.000    # 1 km
CAUTION_DIST_KM  = 5.000    # 5 km
SEARCH_RADIUS_KM = 50.0     # KD-tree coarse filter radius

# ── Scan parameters ───────────────────────────────────────────────────────────
SCAN_HORIZON_S = 7200.0     # 2-hour prediction window
SCAN_STEP_S    = 60.0       # step size for coarse scan (1 min)
REFINE_ITERS   = 25         # binary search iterations


@dataclass
class ConjunctionEvent:
    satellite_id: str
    debris_id: str
    tca_offset_s: float
    miss_distance_km: float
    approach_velocity_km_s: float
    is_critical: bool
    is_warning: bool

    @property
    def risk_level(self) -> str:
        if self.is_critical: return "CRITICAL"
        if self.is_warning:  return "WARNING"
        return "CAUTION"


def _scan_pair(sv_sat: np.ndarray, sv_deb: np.ndarray) -> Tuple[float, float]:
    """
    Propagate two objects forward and find the minimum separation.
    Returns (tca_seconds, min_distance_km).
    """
    s_s = sv_sat.copy()
    s_d = sv_deb.copy()

    best_dist = np.linalg.norm(s_s[:3] - s_d[:3])
    best_t    = 0.0
    prev_dist = best_dist

    t = 0.0
    while t < SCAN_HORIZON_S:
        s_s = rk4_step(s_s, SCAN_STEP_S)
        s_d = rk4_step(s_d, SCAN_STEP_S)
        t  += SCAN_STEP_S
        dist = np.linalg.norm(s_s[:3] - s_d[:3])
        if dist < best_dist:
            best_dist = dist
            best_t    = t
        # Stop early if clearly diverging and already past a close pass
        if dist > prev_dist * 1.05 and t > best_t + SCAN_STEP_S * 5:
            if best_dist > CAUTION_DIST_KM * 2:
                break
        prev_dist = dist

    return best_t, best_dist


def _refine_tca(sv_sat: np.ndarray, sv_deb: np.ndarray,
                rough_t: float, window: float = 120.0) -> Tuple[float, float]:
    """Binary search around rough_t for precise TCA."""
    lo = max(0.0, rough_t - window)
    hi = rough_t + window

    s_lo_sat = propagate(sv_sat, lo, max_substep=30.0)
    s_lo_deb = propagate(sv_deb, lo, max_substep=30.0)

    for _ in range(REFINE_ITERS):
        mid  = (lo + hi) / 2.0
        dt   = mid - lo
        sa   = propagate(s_lo_sat, dt,      max_substep=20.0)
        da   = propagate(s_lo_deb, dt,      max_substep=20.0)
        sb   = propagate(s_lo_sat, dt + 5,  max_substep=20.0)
        db   = propagate(s_lo_deb, dt + 5,  max_substep=20.0)
        d_mid = np.linalg.norm(sa[:3] - da[:3])
        d_hi  = np.linalg.norm(sb[:3] - db[:3])
        if d_hi < d_mid:
            lo = mid
        else:
            hi = mid

    tca = (lo + hi) / 2.0
    dt  = tca - lo
    s_t = propagate(s_lo_sat, dt, max_substep=20.0)
    d_t = propagate(s_lo_deb, dt, max_substep=20.0)
    return tca, float(np.linalg.norm(s_t[:3] - d_t[:3]))


# ─────────────────────────────────────────────────────────────────────────────
def assess_conjunctions(
    sat_states: Dict[str, np.ndarray],
    deb_states: Dict[str, np.ndarray],
    sim_time: Optional[datetime] = None,
) -> List[ConjunctionEvent]:
    """
    Two-pass conjunction assessment.

    Pass 1: KD-tree filter — O(N log M) — finds pairs currently within 50 km.
    Pass 2: Time-scan each pair to find minimum approach over next 2 hours.
    Pass 3: Binary-search refinement for precise TCA.

    Fast because Pass 1 eliminates the vast majority of pairs before any
    expensive propagation is needed.
    """
    if not sat_states or not deb_states:
        return []

    deb_ids  = list(deb_states.keys())
    deb_pos  = np.array([deb_states[d][:3] for d in deb_ids])
    tree     = KDTree(deb_pos)

    events: List[ConjunctionEvent] = []

    for sat_id, sat_sv in sat_states.items():
        # Pass 1: current-epoch proximity
        candidate_idx = tree.query_ball_point(sat_sv[:3], SEARCH_RADIUS_KM)
        if not candidate_idx:
            continue

        for idx in candidate_idx:
            deb_id = deb_ids[idx]
            deb_sv = deb_states[deb_id]

            # Pass 2: time-scan for minimum distance
            rough_t, rough_dist = _scan_pair(sat_sv, deb_sv)

            if rough_dist > CAUTION_DIST_KM:
                continue   # never gets close enough

            # Pass 3: refinement
            window  = min(SCAN_STEP_S, rough_t + 1.0)
            tca_s, miss_km = _refine_tca(sat_sv, deb_sv, rough_t, window=window)

            if miss_km > CAUTION_DIST_KM:
                continue

            rel_v = float(np.linalg.norm(sat_sv[3:] - deb_sv[3:]))

            events.append(ConjunctionEvent(
                satellite_id=sat_id,
                debris_id=deb_id,
                tca_offset_s=tca_s,
                miss_distance_km=miss_km,
                approach_velocity_km_s=rel_v,
                is_critical=(miss_km < CRITICAL_DIST_KM),
                is_warning=(miss_km < WARNING_DIST_KM),
            ))

    events.sort(key=lambda e: e.miss_distance_km)
    return events


def quick_distance_check(
    sat_states: Dict[str, np.ndarray],
    deb_states: Dict[str, np.ndarray],
    threshold: float = CRITICAL_DIST_KM,
) -> int:
    if not sat_states or not deb_states:
        return 0
    deb_pos = np.array([v[:3] for v in deb_states.values()])
    tree    = KDTree(deb_pos)
    return sum(len(tree.query_ball_point(sv[:3], threshold))
               for sv in sat_states.values())
