"""
run.py — Cross-platform launcher for the ACM server.
Usage:  python run.py
This avoids the issue where 'uvicorn' is not found in PATH on Windows
even after pip install, because the Scripts/ folder isn't always in PATH.
"""

import sys
import os

# Ensure project root is in sys.path so 'src' is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Check dependencies before starting
missing = []
for pkg in ("fastapi", "uvicorn", "numpy", "scipy", "pydantic"):
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print("=" * 60)
    print("ERROR: Missing dependencies:", ", ".join(missing))
    print("Please run:  pip install -r requirements.txt")
    print("=" * 60)
    sys.exit(1)

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("  Autonomous Constellation Manager — NSH 2026")
    print("  Dashboard → http://localhost:8000")
    print("  API docs  → http://localhost:8000/docs")
    print("  Health    → http://localhost:8000/api/health")
    print("=" * 60)
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,       # set True for development
        log_level="info",
    )
