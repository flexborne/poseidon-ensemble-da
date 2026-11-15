"""Case configuration: load and validate a TOML file into a typed Config."""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_SCHEMA = 1

# Single source of truth for analysis defaults.
DEFAULT_GSD = 1.3
DEFAULT_BDEPTHS = [25, 100, 500, 2000]
DEFAULT_LOCALIZATION = {'horizontal_m': 100000.0, 'vertical_m': 100.0,
                        'temporal_years': 1.0, 'exclude': [1, 2, 3, 49, 50]}


@dataclass
class Config:
    case_id: str
    config_path: Path
    measurements_dir: Path
    geometry: Path
    runs_root: Path
    target_time: str
    observation_times: list
    gsd: float = DEFAULT_GSD
    bdepths: list = field(default_factory=lambda: list(DEFAULT_BDEPTHS))
    localization: dict = field(default_factory=lambda: dict(DEFAULT_LOCALIZATION))
    model_executable: Path | None = None
    launcher: list = field(default_factory=list)
    use_pty: bool = False
    schema_version: int = 1
    cycle_mode: str = 'filter'
    cycle_dates: list = field(default_factory=list)
    cycle_window_years: float = 0.25


def protected_inputs(config: Config) -> list:
    """Input paths that an output directory must never overlap (dataset, measurements, geometry)."""
    return [config.runs_root, config.measurements_dir, config.geometry]


def _resolve(base: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path) -> Config:
    """Read a case TOML; paths are resolved relative to the config file's directory."""
    path = Path(path).resolve()
    with open(path, 'rb') as fh:
        raw = tomllib.load(fh)

    base = path.parent
    dataset = raw.get('dataset', {})
    analysis = raw.get('analysis', {})
    model = raw.get('model', {})
    cycle = raw.get('cycle', {})

    target = analysis.get('target_time')
    times = list(analysis.get('observation_times', [target] if target else []))
    exe = model.get('executable')

    return Config(
        case_id=raw.get('case', {}).get('case_id', path.stem),
        config_path=path,
        measurements_dir=_resolve(base, dataset.get('measurements_dir', '.')),
        geometry=_resolve(base, dataset.get('geometry', 'Box_surfareas_with_centers.xlsx')),
        runs_root=_resolve(base, dataset.get('runs_root', '.')),
        target_time=target,
        observation_times=times,
        gsd=float(analysis.get('gsd', DEFAULT_GSD)),
        bdepths=list(analysis['bdepths']) if 'bdepths' in analysis else list(DEFAULT_BDEPTHS),
        localization={**DEFAULT_LOCALIZATION, **raw.get('localization', {})},
        model_executable=_resolve(base, exe) if exe else None,
        launcher=list(model.get('launcher', [])),
        use_pty=bool(model.get('use_pty', False)),
        schema_version=int(raw.get('case', {}).get('schema_version', 1)),
        cycle_mode=cycle.get('mode', 'filter'),
        cycle_dates=list(cycle.get('dates', [])),
        cycle_window_years=float(cycle.get('window_years', 0.25)),
    )


def validate(config: Config) -> list:
    """Return a list of human-readable problems; empty means the case is runnable."""
    problems = []
    if config.schema_version != SUPPORTED_SCHEMA:
        problems.append(f'unsupported schema_version {config.schema_version} (supported: {SUPPORTED_SCHEMA})')
    if not config.target_time:
        problems.append('analysis.target_time is required')
    if not config.observation_times:
        problems.append('analysis.observation_times is empty')
    elif config.observation_times[-1] != config.target_time:
        problems.append('analysis.target_time must be the last of observation_times')
    if len(set(config.observation_times)) != len(config.observation_times):
        problems.append('analysis.observation_times contains duplicates')
    if not (math.isfinite(config.gsd) and config.gsd > 1.0):
        problems.append(f'analysis.gsd must be a finite value > 1 (got {config.gsd})')
    loc = config.localization
    for key in ('horizontal_m', 'vertical_m', 'temporal_years'):
        if key in loc and not (math.isfinite(loc[key]) and loc[key] > 0):
            problems.append(f'localization.{key} must be a finite positive value')
    bd = config.bdepths
    if not bd:
        problems.append('analysis.bdepths must be a non-empty list')
    elif not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bd):
        problems.append('analysis.bdepths must be finite numbers')
    elif any(bd[i] >= bd[i + 1] for i in range(len(bd) - 1)):
        problems.append(f'analysis.bdepths must be strictly increasing (got {bd})')
    else:
        from .geometry import box_levels
        need = int(np.max(box_levels()))
        if len(bd) < need:
            problems.append(f'analysis.bdepths needs at least {need} entries (got {len(bd)})')
    if not config.geometry.exists():
        problems.append(f'geometry file not found: {config.geometry}')
    else:
        from .geometry import validate_geometry
        problems += validate_geometry(config.geometry)
    if not config.measurements_dir.is_dir():
        problems.append(f'measurements_dir not found: {config.measurements_dir}')
    for d in config.observation_times:
        f = config.measurements_dir / f'{d}.dat'
        if not f.exists():
            problems.append(f'missing measurement file: {f}')
        ddir = config.runs_root / d
        if not ddir.is_dir():
            problems.append(f'run directory not found for observation date {d}: {ddir}')
        elif not sorted(ddir.glob('*hydrol*dep*')):
            problems.append(f'no ensemble member folders in {ddir}')
    return problems
