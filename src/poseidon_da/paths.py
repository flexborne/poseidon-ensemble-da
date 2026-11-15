"""Repo-relative data locations: a dev/test fallback. Working runs get paths from the case."""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

BIN_DIR = PROJECT_ROOT / 'bin'
CONFIG_DIR = PROJECT_ROOT / 'config'
MEASUREMENTS_DIR = PROJECT_ROOT / 'measurements.exp2'
RUNS_ROOT = PROJECT_ROOT / 'DAruns' / '01.RunKFS'


def box_surfareas_with_centers_path() -> Path:
    return CONFIG_DIR / 'Box_surfareas_with_centers.xlsx'


def poseidon_exe_path() -> Path:
    return BIN_DIR / 'Poseidon4.exe'
