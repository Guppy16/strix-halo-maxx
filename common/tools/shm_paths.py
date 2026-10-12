"""Repo root and scratch dir for host-side scripts (the Python twin of setup.sh).

    sys.path.insert(0, <repo>/common/tools); from shm_paths import REPO, SCRATCH
"""
import os
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "setup.sh").exists())
SCRATCH = Path(os.environ.get("SHM_SCRATCH") or REPO / ".scratch").resolve()
