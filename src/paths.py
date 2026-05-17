from __future__ import annotations

from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

BIN_DIR = PROJECT_ROOT / 'bin'
CONFIG_DIR = PROJECT_ROOT / 'config'
MEASUREMENTS_DIR = PROJECT_ROOT / 'measurements.exp2'

RUNS_ROOT = PROJECT_ROOT / 'DAruns' / '01.RunKFS'
TEMPLATES_DIR = RUNS_ROOT / 'templates'
OUTPUTS_DIR = RUNS_ROOT / 'outputs'


def ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def date_dir(strdate: str) -> Path:
    return RUNS_ROOT / strdate


def template_hydrol_dir(name: str) -> Path:
    return TEMPLATES_DIR / name


def output_xlsx_path(strdate: str) -> Path:
    return ensure_outputs_dir() / f'out_{strdate}.xlsx'


def obs2pos_mat_path() -> Path:
    return ensure_outputs_dir() / 'obs2pos.mat'


def obs_pos_xls_path() -> Path:
    return ensure_outputs_dir() / 'obs_pos.xls'


def box_surfareas_path() -> Path:
    return CONFIG_DIR / 'Box_surfareas.xlsx'


def box_surfareas_with_centers_path() -> Path:
    return CONFIG_DIR / 'Box_surfareas_with_centers.xlsx'


def poseidon_exe_path() -> Path:
    return BIN_DIR / 'Poseidon4.exe'
