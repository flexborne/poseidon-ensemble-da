"""Case configuration: load and validate a TOML file into a typed Config."""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_SCHEMA = 1

# Recognised tables and keys; anything else is reported (catches typos like `gds`).
SCHEMA = {
    'case': {'schema_version', 'case_id'},
    'dataset': {'measurements_dir', 'geometry', 'runs_root'},
    'analysis': {'target_time', 'observation_times', 'gsd', 'bdepths'},
    'localization': {'horizontal_m', 'vertical_m', 'temporal_years', 'exclude'},
    'model': {'executable', 'launcher', 'use_pty'},
    'cycle': {'mode', 'dates', 'window_years'},
}


def schema_problems(raw: dict) -> list:
    """Unknown tables/keys in a parsed case TOML, so a mistyped key is reported, not ignored."""
    problems = []
    for table, value in raw.items():
        if table not in SCHEMA:
            problems.append(f'unknown config table [{table}]')
        elif not isinstance(value, dict):
            problems.append(f'config table [{table}] must be a table')
        else:
            problems += [f'unknown config key {table}.{k}' for k in value if k not in SCHEMA[table]]
    return problems

# Single source of truth for analysis defaults.
DEFAULT_GSD = 1.3
DEFAULT_BDEPTHS = [25, 100, 500, 2000]
DEFAULT_LOCALIZATION = {'horizontal_m': 100000.0, 'vertical_m': 100.0,
                        'temporal_years': 1.0, 'exclude': (1, 2, 3, 49, 50)}


def normalize_localization(overrides=None) -> dict:
    """Localization settings with defaults applied and `exclude` as an immutable tuple.

    Returns a fresh dict every call so independently loaded configs never share mutable
    state (the `exclude` list in particular).
    """
    loc = dict(DEFAULT_LOCALIZATION)
    loc.update(overrides or {})
    loc['exclude'] = tuple(loc.get('exclude', ()))
    return loc


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
    localization: dict = field(default_factory=normalize_localization)
    model_executable: Path | None = None
    launcher: list = field(default_factory=list)
    use_pty: bool = False
    schema_version: int = 1
    cycle_mode: str = 'filter'
    cycle_dates: list = field(default_factory=list)
    cycle_window_years: float = 0.25


def protected_inputs(config: Config) -> list:
    """Input paths an output directory must never overlap (dataset, measurements, geometry,
    the case file and the model executable)."""
    paths = [config.runs_root, config.measurements_dir, config.geometry, config.config_path]
    if config.model_executable is not None:
        paths.append(config.model_executable)
    return paths


def raise_if_invalid(problems) -> None:
    """Raise a single ValueError listing all problems (shared by every public operation)."""
    if problems:
        raise ValueError('invalid case:\n  - ' + '\n  - '.join(problems))


def resolve_perturbation(seed, rng, perturb):
    """Return (rng_or_None, manifest_metadata) for a run.

    When perturbing without an explicit seed/rng, materialize a seed so the manifest records
    something reproducible instead of silently drawing OS entropy.
    """
    if rng is not None and seed is not None:
        raise ValueError('pass either seed or rng, not both')
    if not perturb:
        return None, {'enabled': False, 'seed': None, 'rng': None}
    if rng is not None:
        return rng, {'enabled': True, 'seed': None, 'rng': 'external (not reconstructible)'}
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    return np.random.default_rng(seed), {'enabled': True, 'seed': seed, 'rng': None}


def _resolve(base: Path, value) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else base / p).resolve()   # normalize both for the overlap guard


def load_config(path) -> Config:
    """Read a case TOML; paths are resolved relative to the config file's directory."""
    path = Path(path).resolve()
    with open(path, 'rb') as fh:
        raw = tomllib.load(fh)

    problems = schema_problems(raw)
    if problems:                                      # reject mistyped keys at load time
        raise ValueError('invalid case file: ' + '; '.join(problems))

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
        localization=normalize_localization(raw.get('localization', {})),
        model_executable=_resolve(base, exe) if exe else None,
        launcher=list(model.get('launcher', [])),
        use_pty=bool(model.get('use_pty', False)),
        schema_version=int(raw.get('case', {}).get('schema_version', 1)),
        cycle_mode=cycle.get('mode', 'filter'),
        cycle_dates=list(cycle.get('dates', [])),
        cycle_window_years=float(cycle.get('window_years', 0.25)),
    )


def _check_bdepths(bd) -> list:
    if not bd:
        return ['analysis.bdepths must be a non-empty list']
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bd):
        return ['analysis.bdepths must be finite numbers']
    if bd[0] <= 0:
        return [f'analysis.bdepths must start with a positive depth boundary (got {bd[0]})']
    if any(bd[i] >= bd[i + 1] for i in range(len(bd) - 1)):
        return [f'analysis.bdepths must be strictly increasing (got {bd})']
    from .geometry import box_levels
    need = int(np.max(box_levels()))
    return [f'analysis.bdepths needs at least {need} entries (got {len(bd)})'] if len(bd) < need else []


def _check_date_inputs(config, date) -> list:
    """Measurement file and ensemble directory for one date exist (cheap stats)."""
    problems = []
    if not (config.measurements_dir / f'{date}.dat').exists():
        problems.append(f'missing measurement file: {config.measurements_dir / f"{date}.dat"}')
    ddir = config.runs_root / date
    if not ddir.is_dir():
        problems.append(f'run directory not found for date {date}: {ddir}')
    elif not sorted(ddir.glob('*hydrol*dep*')):
        problems.append(f'no ensemble member folders in {ddir}')
    return problems


def validate_parameters(config: Config) -> list:
    """Analysis parameters + input existence (cheap; file *content* is checked when read).

    Covers schema version, gsd, localization, bdepths, and that geometry / measurements exist.
    Shared by the analysis (analyze/run) and the cycle, which both run the Kalman update.
    """
    problems = []
    if config.schema_version != SUPPORTED_SCHEMA:
        problems.append(f'unsupported schema_version {config.schema_version} (supported: {SUPPORTED_SCHEMA})')
    if not (math.isfinite(config.gsd) and config.gsd > 1.0):
        problems.append(f'analysis.gsd must be a finite value > 1 (got {config.gsd})')
    loc = config.localization
    for key in ('horizontal_m', 'vertical_m', 'temporal_years'):
        if key in loc and not (math.isfinite(loc[key]) and loc[key] > 0):
            problems.append(f'localization.{key} must be a finite positive value')
    problems += _check_bdepths(config.bdepths)
    if not config.geometry.exists():
        problems.append(f'geometry file not found: {config.geometry}')
    if not config.measurements_dir.is_dir():
        problems.append(f'measurements_dir not found: {config.measurements_dir}')
    return problems


def validate(config: Config) -> list:
    """Full single-window validation (the `validate` command, analyze/run entry); empty = runnable.

    Geometry *content* is validated when load_geometry reads it, not here.
    """
    problems = validate_parameters(config)
    if not config.target_time:
        problems.append('analysis.target_time is required')
    if not config.observation_times:
        problems.append('analysis.observation_times is empty')
    elif config.observation_times[-1] != config.target_time:
        problems.append('analysis.target_time must be the last of observation_times')
    if len(set(config.observation_times)) != len(config.observation_times):
        problems.append('analysis.observation_times contains duplicates')
    for d in config.observation_times:
        problems += _check_date_inputs(config, d)
    return problems


def validate_diagnose(config: Config, date) -> list:
    """Only what diagnose needs: bdepths, geometry/measurements and the chosen date's inputs."""
    problems = []
    if config.schema_version != SUPPORTED_SCHEMA:
        problems.append(f'unsupported schema_version {config.schema_version} (supported: {SUPPORTED_SCHEMA})')
    problems += _check_bdepths(config.bdepths)
    if not config.geometry.exists():
        problems.append(f'geometry file not found: {config.geometry}')
    if not config.measurements_dir.is_dir():
        problems.append(f'measurements_dir not found: {config.measurements_dir}')
    problems += _check_date_inputs(config, date)
    return problems
