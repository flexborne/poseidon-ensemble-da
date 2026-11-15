"""Stable public API: load a case, run the analysis, materialize a run."""
from __future__ import annotations

import tempfile
from pathlib import Path


from . import workflow
from .config import (Config, load_config, normalize_localization,  # load_config re-exported
                     protected_inputs, raise_if_invalid, resolve_perturbation, validate, validate_diagnose)
from .domain import AnalysisResult, AnalysisWindow, RunResult
from .publish import assert_safe_dest, staged_publish
from .results import build_manifest, export_result, write_manifest
from .runner import RunContext

__all__ = ['load_config', 'analyze', 'write_analysis', 'run', 'diagnose', 'run_cycle',
           'Config', 'AnalysisResult', 'RunResult']


def _prepare(config: Config) -> None:
    """Finalize the config for an operation in place (complete the localization dict)."""
    config.localization = normalize_localization(config.localization)


def _analyze(config: Config, ctx: RunContext, perturb, rng):
    window = AnalysisWindow(config.target_time, tuple(config.observation_times))
    return workflow.analyze(
        window, ctx, geometry_path=str(config.geometry), measurements_dir=str(config.measurements_dir),
        gsd=config.gsd, localization=config.localization, bdepths=config.bdepths, perturb=perturb, rng=rng)


def analyze(config: Config, *, perturb=True, rng=None, seed=None) -> AnalysisResult:
    """Validate the case, compute the analysis, and return the result (no files written)."""
    _prepare(config)
    raise_if_invalid(validate(config))
    rng_obj, _ = resolve_perturbation(seed, rng, perturb)
    ctx = RunContext.isolated(tempfile.mkdtemp(prefix='poseidon-analyze-'), runs_root=config.runs_root)
    result, _ = _analyze(config, ctx, perturb, rng_obj)
    return result


def _publish_analysis(config: Config, output_dir, *, include_model_outputs,
                      perturb, rng, seed, force) -> RunResult:
    """Shared body of write_analysis/run: validate, analyze, publish atomically.

    With include_model_outputs it also writes the restart files and out_<date>.xlsx; otherwise it
    publishes only the analysis export. Everything is staged and published in one atomic step.
    """
    from . import __version__
    assert_safe_dest(output_dir, protected_inputs(config))          # never overwrite the dataset
    _prepare(config)
    raise_if_invalid(validate(config))
    rng_obj, perturbation = resolve_perturbation(seed, rng, perturb)

    with staged_publish(output_dir, force) as staging:
        ctx = RunContext.isolated(staging, runs_root=config.runs_root)
        result, payload = _analyze(config, ctx, perturb, rng_obj)
        steps = [{'step': 'analyze', 'status': 'ok', 'diagnostics': result.diagnostics}]
        if include_model_outputs:
            restarts = workflow.write_restarts(result, payload, ctx)
            steps.append({'step': 'write_restarts', 'status': 'ok', 'count': len(restarts)})
            workflow.write_outputs(result, payload, ctx)
            steps.append({'step': 'write_outputs', 'status': 'ok'})
        export_result(result, staging)
        steps.append({'step': 'export', 'status': 'ok'})
        manifest = build_manifest(
            config, __version__, steps, operation='run' if include_model_outputs else 'analyze',
            target_time=result.target, dates=list(config.observation_times), member_ids=result.member_ids,
            perturbation=perturbation, diagnostics=result.diagnostics)
        write_manifest(manifest, staging)

    out = Path(output_dir)
    outputs = {'npz': str(out / 'analysis.npz'), 'json': str(out / 'result.json'),
               'manifest': str(out / 'manifest.json')}
    if include_model_outputs:
        outputs.update({'restarts_dir': str(out / 'templates'), 'outputs_dir': str(out / 'outputs')})
    return RunResult(status='ok', steps=steps, outputs=outputs, manifest=manifest)


def write_analysis(config: Config, output_dir, *, perturb=True, rng=None, seed=None, force=False) -> RunResult:
    """Validate, analyze, and publish the analysis export (NPZ/JSON + manifest) atomically."""
    return _publish_analysis(config, output_dir, include_model_outputs=False,
                             perturb=perturb, rng=rng, seed=seed, force=force)


def run(config: Config, output_dir, *, perturb=True, rng=None, seed=None, force=False) -> RunResult:
    """Validate, analyze, and materialize restarts + outputs + manifest under output_dir (atomic)."""
    return _publish_analysis(config, output_dir, include_model_outputs=True,
                             perturb=perturb, rng=rng, seed=seed, force=force)


def diagnose(config: Config, output_dir, *, date=None, force=False) -> RunResult:
    """Validate, compare the forecast at `date` (default target) against observations, publish atomically."""
    from . import __version__
    assert_safe_dest(output_dir, protected_inputs(config))
    _prepare(config)
    date = date or config.target_time
    raise_if_invalid(validate_diagnose(config, date))

    with staged_publish(output_dir, force) as staging:
        ctx = RunContext.isolated(staging, runs_root=config.runs_root)
        obs, members = workflow.write_forecast_diagnostics(config, ctx, date)
        steps = [{'step': 'diagnose', 'date': date, 'status': 'ok',
                  'n_obs': int(obs.shape[0]), 'n_members': len(members)}]
        manifest = build_manifest(
            config, __version__, steps, operation='diagnose', target_time=date,
            dates=[date], member_ids=members, perturbation={'enabled': False, 'seed': None}, mode='diagnose')
        write_manifest(manifest, staging)

    out = Path(output_dir)
    return RunResult(status='ok', steps=steps,
                     outputs={'outputs_dir': str(out / 'outputs'), 'manifest': str(out / 'manifest.json')},
                     manifest=manifest)


from .cycle import run_cycle  # noqa: E402  (re-export: the public API includes the cycle)
