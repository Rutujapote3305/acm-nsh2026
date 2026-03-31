# 🛰 Autonomous Constellation Manager (ACM)
### National Space Hackathon 2026 — IIT Delhi

> Orbital Debris Avoidance & Constellation Management System

---

## 🔧 Requirements Fix (Read This First!)

The original `numpy==1.26.4` does **not** support Python 3.13 on Windows —
it fails to build from source because it needs GCC ≥ 8.4 and there is no
pre-built wheel.  The `requirements.txt` in this version is already fixed:

```
fastapi==0.115.12
uvicorn[standard]==0.34.0
numpy>=2.1.0          ← pre-built wheel for Py 3.10–3.13 ✅
scipy>=1.14.1         ← pre-built wheel for Py 3.10–3.13 ✅
pydantic==2.10.6
python-dateutil==2.9.0.post0
```

---

## Prerequisites

| Tool | Min version | Check |
|------|-------------|-------|
| Python | **3.10+** | `python --version` |
| pip | any recent | `pip --version` |
| Docker (for submission) | 20.10+ | `docker --version` |

---

## Installation on Windows

Open **PowerShell** inside the project folder:

```powershell
# 1 — Create a virtual environment
python -m venv .venv

# 2 — Activate it  (you should see (.venv) in the prompt)
.venv\Scripts\activate

# 3 — Upgrade pip first (prevents metadata build errors)
python -m pip install --upgrade pip

# 4 — Install all dependencies  (pre-built wheels, no GCC needed)
pip install -r requirements.txt
```

> If Step 4 still complains about numpy, force binary-only:
> ```powershell
> pip install "numpy>=2.1.0" --only-binary=:all:
> pip install -r requirements.txt
> ```

---

## Running Locally

### ✅ Method 1 — `python run.py`  (Recommended)

```powershell
# (.venv) must be active
python run.py
```

Expected output:
```
============================================================
  Autonomous Constellation Manager — NSH 2026
  Dashboard → http://localhost:8000
  API docs  → http://localhost:8000/docs
  Health    → http://localhost:8000/api/health
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Open **http://localhost:8000** in your browser.

---

### Method 2 — `python -m uvicorn`  (avoids PATH issue)

If typing `uvicorn` in PowerShell gives *"not recognized as a cmdlet"*,
use the module form — it always works when the venv is active:

```powershell
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Docker Build & Run  (required for submission)

```powershell
# Build  (uses ubuntu:22.04 as required by §8)
docker build -t acm-nsh2026 .

# Run
docker run -d --name acm -p 8000:8000 acm-nsh2026

# Verify
curl http://localhost:8000/api/health
# → {"status":"ok", ...}

# Stop
docker stop acm && docker rm acm
```

Or with Compose:
```powershell
docker compose up --build      # Ctrl+C to stop
docker compose down
```

---

## API Quick-Test (PowerShell)

```powershell
# 1. Health
Invoke-RestMethod http://localhost:8000/api/health

# 2. Inject telemetry
$body = '{"timestamp":"2026-03-12T08:00:00.000Z","objects":[{"id":"DEB-001","type":"DEBRIS","r":{"x":6928.0,"y":0,"z":0},"v":{"x":0,"y":7.61,"z":0}}]}'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/telemetry -ContentType "application/json" -Body $body

# 3. Advance 1 hour
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/simulate/step -ContentType "application/json" -Body '{"step_seconds":3600}'

# 4. Snapshot
Invoke-RestMethod http://localhost:8000/api/visualization/snapshot

# 5. Fleet status
Invoke-RestMethod http://localhost:8000/api/status
```

Interactive docs also at: **http://localhost:8000/docs**

---

## File Structure

```
acm-nsh2026/
├── run.py                      ← START HERE: python run.py
├── Dockerfile                  ← ubuntu:22.04, port 8000 ✅
├── docker-compose.yml
├── requirements.txt            ← Fixed for Python 3.10–3.13 ✅
├── README.md
├── data/
│   └── ground_stations.csv
├── frontend/
│   └── index.html              ← Orbital Insight dashboard
└── src/
    ├── main.py                 ← FastAPI (5 endpoints)
    ├── constellation.py        ← ACM brain
    ├── models.py               ← Pydantic schemas
    └── physics/
        ├── propagator.py       ← RK4 + J2 + ECI↔geo
        ├── conjunction.py      ← KD-tree CA + TCA
        └── maneuver.py         ← RTN ΔV + Tsiolkovsky + LOS
```

---

## Physics Reference

**Propagation** — RK4 with J2 oblateness:
```
d²r/dt² = −(μ/|r|³)r + a_J2
μ = 398600.4418 km³/s²  |  RE = 6378.137 km  |  J₂ = 1.08263×10⁻³
```

**Conjunction Assessment** — KD-tree (O(N log M) not O(N²)):
1. KD-tree over debris positions → 50 km coarse filter per satellite
2. Close pairs → TCA binary search over 24-hour horizon
3. CRITICAL < 100 m | WARNING < 1 km | CAUTION < 5 km

**Maneuver** — RTN frame → ECI rotation:
- Evasion: prograde burn (+T̂), Recovery: retrograde burn
- Tsiolkovsky: Δm = m_current × (1 − e^(−|Δv|/Isp·g₀)), Isp = 300 s
- EOL graveyard burn at fuel < 5%

---

## Submission Checklist

- [ ] GitHub repo is **public**
- [ ] `Dockerfile` at root using `ubuntu:22.04` ✅
- [ ] Port `8000` exposed and bound to `0.0.0.0` ✅
- [ ] All 5 endpoints pass grader tests ✅
- [ ] Technical report PDF
- [ ] Video demo ≤ 5 min

```powershell
git init
git add .
git commit -m "feat: ACM submission NSH 2026"
git remote add origin https://github.com/YOUR_TEAM/acm-nsh2026.git
git branch -M main
git push -u origin main
```
