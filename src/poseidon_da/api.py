"""Stable public API: load a case, run the analysis, materialize a run."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import workflow
from .config import Config, load_config, validate  # load_config re-exported
from .domain import AnalysisResult, AnalysisWindow, RunResult
from .results import build_manifest, export_result, write_manifest
from .runner import RunContext

__all__ = ['load_config', 'analyze', 'run', 'Config', 'AnalysisResult', 'RunResult']


def _require_valid(config: Config) -> None:
    problems = validate(config)
    if problems:
        raise ValueError('invalid case:\n  - ' + '\n  - '.join(problems))


def _window(config: Config) -> AnalysisWindow:
    return AnalysisWindow(target_time=config.target_time,
                          observation_times=tuple(config.observation_times))


def _effective_localization(config: Config) -> dict:
    return {**workflow.DEFAULT_LOCALIZATION, **config.localization}


def _rng(seed, rng):
    if rng is not None:
        return rng
    if seed is not None:
        return np.random.default_rng(seed)
    return None


def _analyze(config: Config, ctx: RunContext, perturb, rng):
    return workflow.analyze(
        _window(config), ctx,
        geometry_path=str(config.geometry), measurements_dir=str(config.measurements_dir),
        gsd=config.gsd, localization=config.localization, bdepths=config.bdepths,
        perturb=perturb, rng=rng)


def analyze(config: Config, *, perturb=True, rng=None, seed=None, output_dir=None) -> AnalysisResult:
    """Validate the case, compute the analysis, and return the result (no files written)."""
    _require_valid(config)
    base = Path(output_dir) if output_dir else config.runs_root
    ctx = RunContext.isolated(base, runs_root=config.runs_root)
    result, _ = _analyze(config, ctx, perturb, _rng(seed, rng))
    return result


def run(config: Config, output_dir, *, perturb=True, rng=None, seed=None, force=False) -> RunResult:
    """Validate, analyze, and materialize restarts + outputs + manifest under output_dir.

    Refuses a non-empty output_dir unless force=True, and writes the manifest last, so a
    failure never leaves a previous run's successful-looking manifest behind.
    """
    from . import __version__
    _require_valid(config)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f'output directory is not empty: {output_dir} (use force=True to overwrite)')

    # Build the whole run in a private staging directory and publish it atomically only
    # on success, so a failure never leaves a partial or mixed-provenance output behind.
    staging = Path(tempfile.mkdtemp(prefix='poseidon-run-'))
    try:
        ctx = RunContext.isolated(staging, runs_root=config.runs_root)
        steps = []
        result, payload = _analyze(config, ctx, perturb, _rng(seed, rng))
        steps.append({'step': 'analyze', 'status': 'ok', 'diagnostics': result.diagnostics})
        restarts = workflow.write_restarts(result, payload, ctx)
        steps.append({'step': 'write_restarts', 'status': 'ok', 'count': len(restarts)})
        workflow.write_outputs(result, payload, ctx)
        steps.append({'step': 'write_outputs', 'status': 'ok'})
        export_result(result, staging)
        steps.append({'step': 'export', 'status': 'ok'})
        manifest = build_manifest(config, result, __version__, steps,
                                  localization_used=_effective_localization(config),
                                  perturb=perturb, seed=seed)
        write_manifest(manifest, staging)

        # publish: move the completed staging tree into the (empty) output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in staging.iterdir():
            dest = output_dir / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(dest))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return RunResult(
        status='ok', steps=steps,
        outputs={'restarts_dir': str(output_dir / 'templates'),
                 'outputs_dir': str(output_dir / 'outputs'),
                 'npz': str(output_dir / 'analysis.npz'),
                 'manifest': str(output_dir / 'manifest.json')},
        manifest=manifest,
    )
