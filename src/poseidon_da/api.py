"""Stable public API: load a case, run the analysis, materialize a run."""
from __future__ import annotations

from pathlib import Path

from . import workflow
from .config import Config, load_config  # re-exported
from .domain import AnalysisResult, AnalysisWindow, RunResult
from .results import build_manifest, export_result, write_manifest
from .runner import RunContext

__all__ = ['load_config', 'analyze', 'run', 'Config', 'AnalysisResult', 'RunResult']


def _window(config: Config) -> AnalysisWindow:
    return AnalysisWindow(target_time=config.target_time,
                          observation_times=tuple(config.observation_times))


def _context(config: Config, output_dir=None) -> RunContext:
    if output_dir is None:
        return RunContext.default() if config.runs_root is None else \
            RunContext(runs_root=config.runs_root,
                       templates_dir=config.runs_root / 'templates',
                       outputs_dir=config.runs_root / 'outputs')
    return RunContext.isolated(output_dir, runs_root=config.runs_root)


def _analyze(config: Config, ctx: RunContext, perturb, rng):
    return workflow.analyze(
        _window(config), ctx,
        geometry_path=str(config.geometry), measurements_dir=str(config.measurements_dir),
        gsd=config.gsd, localization=config.localization, bdepths=config.bdepths,
        perturb=perturb, rng=rng)


def analyze(config: Config, *, perturb=True, rng=None, output_dir=None) -> AnalysisResult:
    """Compute the analysis and return the structured result (no files written)."""
    ctx = _context(config, output_dir)
    result, _payload = _analyze(config, ctx, perturb, rng)
    return result


def run(config: Config, output_dir, *, perturb=True, rng=None) -> RunResult:
    """Compute the analysis, write restart files + outputs + a manifest under output_dir."""
    from . import __version__
    output_dir = Path(output_dir)
    ctx = RunContext.isolated(output_dir, runs_root=config.runs_root)

    steps = []
    result, payload = _analyze(config, ctx, perturb, rng)
    steps.append({'step': 'analyze', 'status': 'ok', 'diagnostics': result.diagnostics})

    restarts = workflow.write_restarts(result, payload, ctx)
    steps.append({'step': 'write_restarts', 'status': 'ok', 'count': len(restarts)})

    workflow.write_outputs(result, payload, ctx)
    steps.append({'step': 'write_outputs', 'status': 'ok'})

    exported = export_result(result, output_dir)
    steps.append({'step': 'export', 'status': 'ok'})

    manifest = build_manifest(config, result, __version__, steps)
    write_manifest(manifest, output_dir)

    return RunResult(
        status='ok', steps=steps,
        outputs={'restarts_dir': str(ctx.templates_dir), 'outputs_dir': str(ctx.outputs_dir),
                 **{k: str(v) for k, v in exported.items()}},
        manifest=manifest,
    )
