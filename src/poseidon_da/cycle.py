"""Sequential assimilation-forecast cycle (NODA / filter) with explicit provenance.

The cycle never mutates the dataset: it copies the needed date directories into an
isolated work area and runs there. The external model is injected as `model_runner`
so the orchestration can be tested without POSEIDON. The smoother cycle is gated until
its information-reuse contract is defined.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import workflow
from .domain import AnalysisWindow, RunResult
from .geometry import box_levels
from .publish import staged_publish
from .results import export_result, write_manifest
from .runner import (RunContext, copy_lastconc_to_initconc, find_members,
                     publish_initconc, run_poseidon_all)


@dataclass
class CycleStep:
    kind: str                       # 'analyze' | 'forecast' | 'diagnose'
    date: str
    time: float
    source: str | None = None       # date whose state seeds a forecast
    provenance: dict = field(default_factory=dict)


def plan_cycle(mode: str, dates, measurements_dir) -> list:
    """Deterministic step plan with per-step provenance (what seeds/uses what)."""
    if mode == 'smoother':
        raise NotImplementedError(
            "smoother cycle needs a defined information-reuse contract; "
            "use mode 'filter' or 'noda'")
    if mode not in ('noda', 'filter'):
        raise ValueError(f'unknown cycle mode {mode!r}')
    dates = list(dates)
    steps = []
    for i in range(len(dates) - 1):
        cur, nxt = dates[i], dates[i + 1]
        if mode == 'filter':
            steps.append(CycleStep('analyze', cur, float(cur), provenance={'observations': [cur]}))
        steps.append(CycleStep('forecast', nxt, float(nxt), source=cur,
                               provenance={'restart_from': cur, 'mode': mode}))
        has_obs = (Path(measurements_dir) / f'{nxt}.dat').exists()
        steps.append(CycleStep('diagnose', nxt, float(nxt),
                               provenance={'observations': [nxt] if has_obs else [],
                                           **({} if has_obs else {'skipped': 'no measurement file'})}))
    return steps


def _default_model_runner(config, members):
    def runner(date_dir, expected_time, bnlev):
        run_poseidon_all(date_dir, members=members, expected_members=members,
                         expected_time=expected_time, bnlev=bnlev,
                         exe_path=config.model_executable, launcher=config.launcher)
    return runner


def _validate_date_path(d, runs_root: Path) -> None:
    """A cycle date must be numeric AND a safe single path component inside runs_root."""
    try:
        float(d)
    except (TypeError, ValueError):
        raise ValueError(f'cycle date {d!r} is not numeric')
    if d in ('', '.', '..') or '/' in d or '\\' in d or os.sep in d:
        raise ValueError(f'cycle date {d!r} is not a safe path component')
    if (runs_root / d).resolve().parent != runs_root.resolve():
        raise ValueError(f'cycle date {d!r} escapes the dataset root {runs_root}')


def run_cycle(config, output_dir, *, mode=None, dates=None, model_runner=None,
              perturb=True, seed=None, force=False) -> RunResult:
    """Run a NODA or filter cycle over a date sequence, producing a provenance manifest.

    The whole plan (mode, numeric and path-safe dates, existing directories, an identical
    ordered member set on every date, input files) is validated BEFORE anything is written,
    and the run is built in a staging directory and published atomically on success.
    """
    from . import __version__
    mode = mode or config.cycle_mode
    dates = list(dates or config.cycle_dates)
    if len(dates) < 2:
        raise ValueError('cycle requires at least two dates')
    runs_root = Path(config.runs_root)

    # ---- validate the full plan before any write ----
    for d in dates:
        _validate_date_path(d, runs_root)                        # numeric + safe path component (R2)
    plan = plan_cycle(mode, dates, config.measurements_dir)     # rejects unsupported mode
    member_set = None
    for d in dates:
        ddir = runs_root / d
        if not ddir.is_dir():
            raise FileNotFoundError(f'cycle: run directory not found for date {d}: {ddir}')
        ms = find_members(ddir)
        if not ms:
            raise FileNotFoundError(f'cycle: no ensemble members for date {d}')
        if member_set is None:
            member_set = ms
        elif set(ms) != set(member_set):                         # identical ensemble on every date (R3)
            raise ValueError(f'cycle: member set at {d} differs from {dates[0]}')

    bnlev = box_levels()
    rng = np.random.default_rng(seed) if seed is not None else None
    runner = model_runner or _default_model_runner(config, member_set)

    with staged_publish(output_dir, force) as staging:
        work = staging / 'work'
        for d in dates:
            shutil.copytree(runs_root / d, work / d, dirs_exist_ok=True)
        ctx = RunContext(runs_root=work, templates_dir=staging / 'templates', outputs_dir=staging / 'outputs')

        executed = []
        for step in plan:
            if step.kind == 'analyze':
                window = AnalysisWindow(step.date, (step.date,))
                result, payload = workflow.analyze(
                    window, ctx, geometry_path=str(config.geometry),
                    measurements_dir=str(config.measurements_dir), gsd=config.gsd,
                    localization=config.localization, bdepths=config.bdepths, perturb=perturb, rng=rng)
                workflow.write_restarts(result, payload, ctx)
                export_result(result, staging / 'analyses' / step.date)   # per-step artifact
                executed.append({'step': 'analyze', 'date': step.date, 'status': 'ok',
                                 'provenance': step.provenance, 'diagnostics': result.diagnostics})
            elif step.kind == 'forecast':
                if mode == 'filter':
                    publish_initconc(ctx.templates_dir, ctx.date_dir(step.date))
                else:
                    copy_lastconc_to_initconc(ctx.date_dir(step.source), ctx.date_dir(step.date), member_set)
                runner(ctx.date_dir(step.date), step.time, bnlev)
                executed.append({'step': 'forecast', 'date': step.date, 'source': step.source,
                                 'status': 'ok', 'provenance': step.provenance})
            elif step.kind == 'diagnose':
                if step.provenance.get('observations'):
                    workflow.write_forecast_diagnostics(config, ctx, step.date, bnlev)
                    executed.append({'step': 'diagnose', 'date': step.date, 'status': 'ok',
                                     'provenance': step.provenance})
                else:
                    executed.append({'step': 'diagnose', 'date': step.date, 'status': 'skipped',
                                     'provenance': step.provenance})

        manifest = {'package_version': __version__, 'case_id': config.case_id, 'mode': mode,
                    'dates': list(dates), 'member_order': list(member_set),
                    'gsd': config.gsd, 'bdepths': config.bdepths, 'localization_used': config.localization,
                    'perturbation': {'enabled': bool(perturb), 'seed': seed},
                    'steps': executed}
        write_manifest(manifest, staging)

    out = Path(output_dir)
    return RunResult(status='ok', steps=executed,
                     outputs={'work': str(out / 'work'), 'outputs_dir': str(out / 'outputs')},
                     manifest=manifest)
