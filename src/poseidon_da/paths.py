"""Repo-relative dataset locations for tests and dev tools only. Working runs get every path
(including the model executable) from the case config."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG_DIR = PROJECT_ROOT / 'config'
MEASUREMENTS_DIR = PROJECT_ROOT / 'measurements.exp2'
RUNS_ROOT = PROJECT_ROOT / 'DAruns' / '01.RunKFS'


def box_surfareas_with_centers_path() -> Path:
    return CONFIG_DIR / 'Box_surfareas_with_centers.xlsx'
