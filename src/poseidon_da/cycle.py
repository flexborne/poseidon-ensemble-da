"""Sequential assimilation-forecast cycle (NODA / filter / smoother) with provenance.

The cycle never mutates the dataset: it copies the needed date directories into an isolated
work area and runs there, publishing atomically on success. The external model is injected as
`model_runner` so the orchestration can be tested without POSEIDON.

`smoother` is a faithful port of the original run_LN_KSF3: for each analysis date it uses a
+/- window of neighbouring dates (a single-date window degenerates to the filter), then
forecasts far enough ahead to feed the next window. NOTE (method property of the original):
overlapping windows re-assimilate observations already present in the forecast background, and
the window ensemble mixes members with different observation histories; this is inherited from
the original method, not introduced by the port.
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


def derive_windows(dates, window_years=0.25):
    """+/- window_years assimilation windows as 1-based (minind, maxind) over sorted dates.

    minind[i]/maxind[i] are the earliest/latest date indices whose physical time lies within
    window_years of date i. With window_years=0.25 this reproduces the original run_LN_KSF3
    hardcoded arrays for the standard 10-date sequence.
    """
    t = [float(d) for d in dates]
    n = len(t)
    tol = 1e-9
    minind = [1 + min(j for j in range(n) if t[j] >= t[i] - window_years - tol) for i in range(n)]
    maxind = [1 + max(j for j in range(n) if t[j] <= t[i] + window_years + tol) for i in range(n)]
    return minind, maxind


def plan_cycle(mode: str, dates, measurements_dir) -> list:
    """Deterministic step plan for noda/filter (smoother is planned per-window in run_cycle)."""
    if mode not in ('noda', 'filter'):
        raise ValueError(f"plan_cycle supports 'noda'/'filter' (got {mode!r})")
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
                         exe_path=config.model_executable, launcher=config.launcher,
                         use_pty=config.use_pty)
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


# ---- shared step primitives ----------------------------------------------------

def _analyze_at(window, config, ctx, perturb, rng, staging, provenance):
    result, payload = workflow.analyze(
        window, ctx, geometry_path=str(config.geometry),
        measurements_dir=str(config.measurements_dir), gsd=config.gsd,
        localization=config.localization, bdepths=config.bdepths, perturb=perturb, rng=rng)
    workflow.write_restarts(result, payload, ctx)
    export_result(result, staging / 'analyses' / window.target_time)
    return {'step': 'analyze', 'date': window.target_time, 'status': 'ok',
            'provenance': provenance, 'diagnostics': result.diagnostics}


def _diagnose_at(config, ctx, date, bnlev):
    if (config.measurements_dir / f'{date}.dat').exists():
        workflow.write_forecast_diagnostics(config, ctx, date, bnlev)
        return {'step': 'diagnose', 'date': date, 'status': 'ok', 'provenance': {'observations': [date]}}
    return {'step': 'diagnose', 'date': date, 'status': 'skipped',
            'provenance': {'skipped': 'no measurement file'}}


def _run_simple(mode, plan, config, ctx, member_set, perturb, rng, runner, bnlev, staging):
    executed = []
    for step in plan:
        if step.kind == 'analyze':
            executed.append(_analyze_at(AnalysisWindow(step.date, (step.date,)),
                                        config, ctx, perturb, rng, staging, step.provenance))
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
                executed.append(_diagnose_at(config, ctx, step.date, bnlev))
            else:
                executed.append({'step': 'diagnose', 'date': step.date, 'status': 'skipped',
                                 'provenance': step.provenance})
    return executed


def _run_smoother(dates, minind, maxind, config, ctx, member_set, perturb, rng, runner, bnlev, staging):
    """Faithful run_LN_KSF3: per-date filter/smoother choice + forecast lookahead to maxind(i+1)."""
    executed = []
    Nd = len(dates)
    for i in range(1, Nd):                                  # 1-based analysis index, i = 1..Nd-1
        cur, nxt = dates[i - 1], dates[i]
        if minind[i - 1] == i and maxind[i - 1] == i:       # trivial window -> filter
            window = AnalysisWindow(cur, (cur,))
            prov = {'observations': [cur], 'kind': 'filter'}
        else:                                               # smoother over the +/- window
            meas = [dates[j - 1] for j in range(minind[i - 1], i)]          # past dates
            meas += [dates[j - 1] for j in range(i + 1, maxind[i - 1] + 1)]  # future dates
            meas += [cur]                                                    # target is always last
            window = AnalysisWindow(cur, tuple(meas))
            prov = {'observations': meas, 'kind': 'smoother', 'window_index': [minind[i - 1], maxind[i - 1]]}
        executed.append(_analyze_at(window, config, ctx, perturb, rng, staging, prov))

        # forecast lookahead: j = i+1 .. maxind(i+1), seeding the chain forward
        publish_initconc(ctx.templates_dir, ctx.date_dir(nxt))      # analysis of cur seeds the first leg
        lo, hi = i + 1, maxind[i]                                   # maxind[i] (0-based) == maxind(i+1)
        for j in range(lo, hi + 1):
            fdate = dates[j - 1]
            source = cur if j == lo else dates[j - 2]
            if j > lo:
                copy_lastconc_to_initconc(ctx.date_dir(dates[j - 2]), ctx.date_dir(fdate), member_set)
            runner(ctx.date_dir(fdate), float(fdate), bnlev)
            executed.append({'step': 'forecast', 'date': fdate, 'source': source,
                             'status': 'ok', 'provenance': {'restart_from': source}})
            if j == lo:
                executed.append(_diagnose_at(config, ctx, fdate, bnlev))
    return executed


def run_cycle(config, output_dir, *, mode=None, dates=None, model_runner=None,
              perturb=True, seed=None, force=False) -> RunResult:
    """Run a NODA / filter / smoother cycle over a date sequence, producing a provenance manifest.

    The whole plan (mode, numeric and path-safe dates, existing directories, an identical ordered
    member set on every date) is validated BEFORE anything is written, and the run is built in a
    staging directory and published atomically on success.
    """
    from . import __version__
    mode = mode or config.cycle_mode
    dates = list(dates or config.cycle_dates)
    if len(dates) < 2:
        raise ValueError('cycle requires at least two dates')
    if mode not in ('noda', 'filter', 'smoother'):
        raise ValueError(f'unknown cycle mode {mode!r}')
    runs_root = Path(config.runs_root)

    # ---- validate everything before any write ----
    for d in dates:
        _validate_date_path(d, runs_root)                           # numeric + safe path component (R2)
    plan = plan_cycle(mode, dates, config.measurements_dir) if mode in ('noda', 'filter') else None
    minind = maxind = None
    if mode == 'smoother':
        minind, maxind = derive_windows(dates, config.cycle_window_years)
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
        elif set(ms) != set(member_set):                            # identical ensemble on every date (R3)
            raise ValueError(f'cycle: member set at {d} differs from {dates[0]}')

    bnlev = box_levels()
    rng = np.random.default_rng(seed) if seed is not None else None
    runner = model_runner or _default_model_runner(config, member_set)

    with staged_publish(output_dir, force) as staging:
        work = staging / 'work'
        for d in dates:
            shutil.copytree(runs_root / d, work / d, dirs_exist_ok=True)
        ctx = RunContext(runs_root=work, templates_dir=staging / 'templates', outputs_dir=staging / 'outputs')

        if mode == 'smoother':
            executed = _run_smoother(dates, minind, maxind, config, ctx, member_set,
                                     perturb, rng, runner, bnlev, staging)
        else:
            executed = _run_simple(mode, plan, config, ctx, member_set, perturb, rng, runner, bnlev, staging)

        manifest = {'package_version': __version__, 'case_id': config.case_id, 'mode': mode,
                    'dates': list(dates), 'member_order': list(member_set),
                    'window_years': config.cycle_window_years if mode == 'smoother' else None,
                    'gsd': config.gsd, 'bdepths': config.bdepths, 'localization_used': config.localization,
                    'perturbation': {'enabled': bool(perturb), 'seed': seed},
                    'steps': executed}
        write_manifest(manifest, staging)

    out = Path(output_dir)
    return RunResult(status='ok', steps=executed,
                     outputs={'work': str(out / 'work'), 'outputs_dir': str(out / 'outputs')},
                     manifest=manifest)
