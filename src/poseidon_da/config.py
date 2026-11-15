"""Case configuration: load and validate a TOML file into a typed Config."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    case_id: str
    config_path: Path
    measurements_dir: Path
    geometry: Path
    runs_root: Path
    target_time: str
    observation_times: list
    gsd: float = 1.3
    bdepths: list = field(default_factory=lambda: [25, 100, 500, 2000])
    localization: dict = field(default_factory=dict)
    model_executable: Path | None = None
    launcher: list = field(default_factory=list)
    schema_version: int = 1


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
        gsd=float(analysis.get('gsd', 1.3)),
        bdepths=list(analysis.get('bdepths', [25, 100, 500, 2000])),
        localization=dict(raw.get('localization', {})),
        model_executable=_resolve(base, exe) if exe else None,
        launcher=list(model.get('launcher', [])),
        schema_version=int(raw.get('case', {}).get('schema_version', 1)),
    )


def validate(config: Config) -> list:
    """Return a list of human-readable problems; empty means the case is runnable."""
    problems = []
    if not config.target_time:
        problems.append('analysis.target_time is required')
    if not config.observation_times:
        problems.append('analysis.observation_times is empty')
    elif config.observation_times[-1] != config.target_time:
        problems.append('analysis.target_time must be the last of observation_times')
    if len(set(config.observation_times)) != len(config.observation_times):
        problems.append('analysis.observation_times contains duplicates')
    if not (config.gsd > 1.0):
        problems.append(f'analysis.gsd must be > 1 (got {config.gsd})')
    loc = config.localization
    for key in ('horizontal_m', 'vertical_m', 'temporal_years'):
        if key in loc and not (loc[key] > 0):
            problems.append(f'localization.{key} must be positive')
    if not config.geometry.exists():
        problems.append(f'geometry file not found: {config.geometry}')
    if not config.measurements_dir.is_dir():
        problems.append(f'measurements_dir not found: {config.measurements_dir}')
    for d in config.observation_times:
        f = config.measurements_dir / f'{d}.dat'
        if not f.exists():
            problems.append(f'missing measurement file: {f}')
    if config.target_time:
        tdir = config.runs_root / config.target_time
        if not tdir.is_dir():
            problems.append(f'target run directory not found: {tdir}')
        elif not sorted(tdir.glob('*hydrol*dep*')):
            problems.append(f'no ensemble member folders in {tdir}')
    return problems
