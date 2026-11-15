"""Sequential assimilation-forecast cycle (NODA / filter / smoother) with provenance.

The cycle never mutates the dataset: it copies the needed date directories into an isolated
work area and runs there, publishing atomically on success. The external model is injected as
`model_runner` so the orchestration can be tested without POSEIDON.

In `smoother` mode each analysis date uses a +/- window of neighbouring dates (a single-date
window degenerates to the filter), then forecasts far enough ahead to feed the next window.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import workflow
from .domain import AnalysisWindow, RunResult
from .geometry import box_levels
from .config import protected_inputs, validate_parameters
from .publish import assert_safe_dest, staged_publish
from .results import build_manifest, export_result, write_manifest
from .runner import (RunContext, copy_lastconc_to_initconc, find_members,
                     publish_initconc, run_poseidon_all)


@dataclass
class CycleStep:
    kind: str                        # 'analyze' | 'forecast' | 'diagnose'
    date: str                        # analyze: target date; forecast/diagnose: that date
    window: tuple | None = None      # analyze: observation dates (target last)
    seed: str | None = None          # forecast: 'analysis' (restart file) | 'restart' (prior lastconc)
    source: str | None = None        # forecast: date whose state seeds this one
    provenance: dict = field(default_factory=dict)


def derive_windows(dates, window_years=0.25):
    """+/- window_years assimilation windows as 1-based (minind, maxind) over sorted dates.

    minind[i]/maxind[i] are the earliest/latest date indices whose physical time lies within
    window_years of date i.
    """
    t = [float(d) for d in dates]
    n = len(t)
    tol = 1e-9
    minind = [1 + min(j for j in range(n) if t[j] >= t[i] - window_years - tol) for i in range(n)]
    maxind = [1 + max(j for j in range(n) if t[j] <= t[i] + window_years + tol) for i in range(n)]
    return minind, maxind


def _diagnose_step(date, measurements_dir) -> CycleStep:
    has_obs = (Path(measurements_dir) / f'{date}.dat').exists()
    return CycleStep('diagnose', date,
                     provenance={'observations': [date]} if has_obs else {'skipped': 'no measurement file'})


def plan_cycle(mode: str, dates, measurements_dir, window_years=0.25) -> list:
    """Deterministic step plan (analyze/forecast/diagnose records) for any of the three modes.

    One plan shape for every mode, so a single executor can run it and the full sequence of
    states, windows and restart sources is inspectable before the model runs.
    """
    dates = list(dates)
    if mode == 'noda':
        return _plan_sequential(dates, measurements_dir, analyze=False)
    if mode == 'filter':
        return _plan_sequential(dates, measurements_dir, analyze=True)
    if mode == 'smoother':
        return _plan_smoother(dates, measurements_dir, window_years)
    raise ValueError(f'unknown cycle mode {mode!r}')


def _plan_sequential(dates, measurements_dir, analyze: bool) -> list:
    """NODA (analyze=False) / filter (analyze=True): analyse each date, forecast to the next."""
    steps = []
    for i in range(len(dates) - 1):
        cur, nxt = dates[i], dates[i + 1]
        if analyze:
            steps.append(CycleStep('analyze', cur, window=(cur,),
                                   provenance={'observations': [cur], 'kind': 'filter'}))
        steps.append(CycleStep('forecast', nxt, seed='analysis' if analyze else 'restart', source=cur,
                               provenance={'restart_from': cur}))
        steps.append(_diagnose_step(nxt, measurements_dir))
    return steps


def _plan_smoother(dates, measurements_dir, window_years) -> list:
    """Per analysis date: assimilate its +/- window, then forecast ahead to maxind(i+1)."""
    minind, maxind = derive_windows(dates, window_years)
    steps = []
    for i in range(1, len(dates)):                           # 1-based analysis index, i = 1..Nd-1
        cur = dates[i - 1]
        if minind[i - 1] == i and maxind[i - 1] == i:        # trivial window -> filter
            window, prov = (cur,), {'observations': [cur], 'kind': 'filter'}
        else:                                                # smoother over the +/- window
            meas = [dates[j - 1] for j in range(minind[i - 1], i)]           # past dates
            meas += [dates[j - 1] for j in range(i + 1, maxind[i - 1] + 1)]  # future dates
            meas += [cur]                                                    # target is always last
            window = tuple(meas)
            prov = {'observations': meas, 'kind': 'smoother', 'window_index': [minind[i - 1], maxind[i - 1]]}
        steps.append(CycleStep('analyze', cur, window=window, provenance=prov))

        lo, hi = i + 1, maxind[i]                             # forecast lookahead: maxind(i+1)
        for j in range(lo, hi + 1):
            fdate = dates[j - 1]
            if j == lo:
                steps.append(CycleStep('forecast', fdate, seed='analysis', source=cur,
                                       provenance={'restart_from': cur}))
                steps.append(_diagnose_step(fdate, measurements_dir))
            else:
                src = dates[j - 2]
                steps.append(CycleStep('forecast', fdate, seed='restart', source=src,
                                       provenance={'restart_from': src}))
    return steps


def _default_model_runner(config, members):
    def runner(date_dir, expected_time, bnlev):
        run_poseidon_all(date_dir, members=members, expected_time=expected_time, bnlev=bnlev,
                         exe_path=config.model_executable, launcher=config.launcher,
                         use_pty=config.use_pty)
    return runner


def _cycle_time(d, runs_root: Path) -> float:
    """Parse a cycle date to a finite physical time and confirm it is a direct child of runs_root.

    A non-numeric value (including path separators, '.', '..') fails float(); the resolved-parent
    check then rejects anything that would escape the dataset root.
    """
    try:
        t = float(d)
    except (TypeError, ValueError):
        raise ValueError(f'cycle date {d!r} is not numeric')
    if not np.isfinite(t):
        raise ValueError(f'cycle date {d!r} is not a finite physical time')
    if (runs_root / d).resolve().parent != runs_root.resolve():
        raise ValueError(f'cycle date {d!r} escapes the dataset root {runs_root}')
    return t


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


def _execute(plan, config, ctx, member_set, perturb, rng, runner, bnlev, staging):
    """Run one plan: analyse windows, seed + forecast each date, diagnose against observations.

    Each record gets a sequential `id`, so repeated forecasts of the same date (overlapping
    smoother windows) are individually identifiable in the manifest.
    """
    executed = []
    for i, step in enumerate(plan):
        if step.kind == 'analyze':
            rec = _analyze_at(AnalysisWindow(step.date, step.window),
                              config, ctx, perturb, rng, staging, step.provenance)
        elif step.kind == 'forecast':
            if step.seed == 'analysis':
                publish_initconc(ctx.templates_dir, ctx.date_dir(step.date))   # from the analysis restart
            else:
                copy_lastconc_to_initconc(ctx.date_dir(step.source), ctx.date_dir(step.date), member_set)
            runner(ctx.date_dir(step.date), float(step.date), bnlev)
            rec = {'step': 'forecast', 'date': step.date, 'source': step.source,
                   'status': 'ok', 'provenance': step.provenance}
        else:  # diagnose
            rec = _diagnose_at(config, ctx, step.date, bnlev)
        executed.append({'id': i, **rec})
    return executed


def run_cycle(config, output_dir, *, mode=None, dates=None, model_runner=None,
              perturb=True, seed=None, force=False) -> RunResult:
    """Run a NODA / filter / smoother cycle over a date sequence, producing a provenance manifest.

    Everything (mode, finite path-safe ascending dates, existing dirs, an identical member set on
    every date) is validated before any write; the run is staged and published atomically.
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
    assert_safe_dest(output_dir, protected_inputs(config))          # never overwrite the dataset (R2.1)
    problems = validate_parameters(config)                          # shared case/parameter checks
    if problems:
        raise ValueError('invalid case:\n  - ' + '\n  - '.join(problems))
    times = [_cycle_time(d, runs_root) for d in dates]              # finite time + safe path component (R2)
    if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
        raise ValueError(f'cycle dates must be strictly increasing in physical time (got {dates})')
    plan = plan_cycle(mode, dates, config.measurements_dir, config.cycle_window_years)
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

    # each analysis window state must either be produced by an earlier forecast or already exist
    # (a smoother cycle started mid-archive needs its first window's future states prepared)
    produced = set()
    for step in plan:
        if step.kind == 'analyze':
            for wd in step.window:
                if wd in produced:
                    continue
                for m in member_set:
                    f = runs_root / wd / m / 'lastconc.xls'
                    if not f.exists():
                        raise FileNotFoundError(
                            f'cycle: analysis {step.date} needs the state for date {wd}, member {m} '
                            f'({f}); it is neither present nor produced earlier in the plan')
        elif step.kind == 'forecast':
            produced.add(step.date)

    bnlev = box_levels()
    rng = np.random.default_rng(seed) if seed is not None else None
    runner = model_runner or _default_model_runner(config, member_set)

    with staged_publish(output_dir, force) as staging:
        work = staging / 'work'
        for d in dates:
            shutil.copytree(runs_root / d, work / d, dirs_exist_ok=True)
        ctx = RunContext(runs_root=work, templates_dir=staging / 'templates', outputs_dir=staging / 'outputs')

        executed = _execute(plan, config, ctx, member_set, perturb, rng, runner, bnlev, staging)

        manifest = build_manifest(
            config, None, __version__, executed,
            localization_used=config.localization, perturbation={'enabled': bool(perturb), 'seed': seed},
            dates=dates, member_order=member_set, mode=mode,
            window_years=config.cycle_window_years if mode == 'smoother' else None)
        write_manifest(manifest, staging)

    out = Path(output_dir)
    return RunResult(status='ok', steps=executed,
                     outputs={'work': str(out / 'work'), 'outputs_dir': str(out / 'outputs')},
                     manifest=manifest)
