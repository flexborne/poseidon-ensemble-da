"""Stable public API: load a case, run the analysis, materialize a run."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import workflow
from .config import Config, load_config, protected_inputs, validate  # load_config re-exported
from .domain import AnalysisResult, AnalysisWindow, RunResult
from .publish import assert_safe_dest, staged_publish
from .results import build_manifest, export_result, write_manifest
from .runner import RunContext

__all__ = ['load_config', 'analyze', 'write_analysis', 'run', 'diagnose', 'run_cycle',
           'Config', 'AnalysisResult', 'RunResult']


def _require_valid(config: Config) -> None:
    problems = validate(config)
    if problems:
        raise ValueError('invalid case:\n  - ' + '\n  - '.join(problems))


def _window(config: Config) -> AnalysisWindow:
    return AnalysisWindow(target_time=config.target_time,
                          observation_times=tuple(config.observation_times))


def _rng(seed, rng):
    if rng is not None and seed is not None:
        raise ValueError('pass either seed or rng, not both')
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


def analyze(config: Config, *, perturb=True, rng=None, seed=None) -> AnalysisResult:
    """Validate the case, compute the analysis, and return the result (no files written)."""
    _require_valid(config)
    ctx = RunContext.isolated(config.runs_root, runs_root=config.runs_root)
    result, _ = _analyze(config, ctx, perturb, _rng(seed, rng))
    return result


def write_analysis(config: Config, output_dir, *, perturb=True, rng=None, seed=None, force=False) -> RunResult:
    """Validate, analyze, and publish the analysis export (NPZ/JSON + manifest) atomically.

    The same output-directory safety, non-empty refusal (unless force) and atomic publication
    as `run`, so an analyze output directory is never a mix of two runs.
    """
    from . import __version__
    assert_safe_dest(output_dir, protected_inputs(config))
    _require_valid(config)
    rng_obj = _rng(seed, rng)
    perturbation = {'enabled': bool(perturb), 'seed': seed,
                    'rng': 'external (not reconstructible)' if rng is not None else None}

    with staged_publish(output_dir, force) as staging:
        ctx = RunContext.isolated(staging, runs_root=config.runs_root)
        result, _ = _analyze(config, ctx, perturb, rng_obj)
        export_result(result, staging)
        steps = [{'step': 'analyze', 'status': 'ok', 'diagnostics': result.diagnostics},
                 {'step': 'export', 'status': 'ok'}]
        manifest = build_manifest(config, result, __version__, steps,
                                  localization_used=config.localization, perturbation=perturbation)
        write_manifest(manifest, staging)

    out = Path(output_dir)
    return RunResult(status='ok', steps=steps,
                     outputs={'npz': str(out / 'analysis.npz'), 'json': str(out / 'result.json'),
                              'manifest': str(out / 'manifest.json')},
                     manifest=manifest)


def run(config: Config, output_dir, *, perturb=True, rng=None, seed=None, force=False) -> RunResult:
    """Validate, analyze, and materialize restarts + outputs + manifest under output_dir.

    The whole run is built in a staging directory and published atomically on success, so a
    failure (or an aborted force-run) leaves the previous complete output intact, never a mix.
    """
    from . import __version__
    assert_safe_dest(output_dir, protected_inputs(config))          # never overwrite the dataset (2.1)
    _require_valid(config)
    rng_obj = _rng(seed, rng)
    perturbation = {'enabled': bool(perturb), 'seed': seed,
                    'rng': 'external (not reconstructible)' if rng is not None else None}

    with staged_publish(output_dir, force) as staging:
        ctx = RunContext.isolated(staging, runs_root=config.runs_root)
        steps = []
        result, payload = _analyze(config, ctx, perturb, rng_obj)
        steps.append({'step': 'analyze', 'status': 'ok', 'diagnostics': result.diagnostics})
        restarts = workflow.write_restarts(result, payload, ctx)
        steps.append({'step': 'write_restarts', 'status': 'ok', 'count': len(restarts)})
        workflow.write_outputs(result, payload, ctx)
        steps.append({'step': 'write_outputs', 'status': 'ok'})
        export_result(result, staging)
        steps.append({'step': 'export', 'status': 'ok'})
        manifest = build_manifest(config, result, __version__, steps,
                                  localization_used=config.localization, perturbation=perturbation)
        write_manifest(manifest, staging)

    out = Path(output_dir)
    return RunResult(
        status='ok', steps=steps,
        outputs={'restarts_dir': str(out / 'templates'), 'outputs_dir': str(out / 'outputs'),
                 'npz': str(out / 'analysis.npz'), 'manifest': str(out / 'manifest.json')},
        manifest=manifest,
    )


def diagnose(config: Config, output_dir, *, date=None, force=False) -> RunResult:
    """Validate, compare the forecast at `date` (default target) against observations, publish atomically."""
    from . import __version__
    assert_safe_dest(output_dir, protected_inputs(config))
    _require_valid(config)
    date = date or config.target_time

    with staged_publish(output_dir, force) as staging:
        ctx = RunContext.isolated(staging, runs_root=config.runs_root)
        obs, members = workflow.write_forecast_diagnostics(config, ctx, date)
        steps = [{'step': 'diagnose', 'date': date, 'status': 'ok',
                  'n_obs': int(obs.shape[0]), 'n_members': len(members)}]
        manifest = build_manifest(config, None, __version__, steps,
                                  localization_used=config.localization,
                                  perturbation={'enabled': False, 'seed': None},
                                  dates=[date], member_order=members, mode='diagnose')
        write_manifest(manifest, staging)

    out = Path(output_dir)
    return RunResult(status='ok', steps=steps,
                     outputs={'outputs_dir': str(out / 'outputs'), 'manifest': str(out / 'manifest.json')},
                     manifest=manifest)


from .cycle import run_cycle  # noqa: E402  (re-export: the public API includes the cycle)
