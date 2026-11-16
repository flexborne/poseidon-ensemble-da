"""Sequential assimilation-forecast cycle (NODA / filter / smoother) with provenance.

The cycle never mutates the dataset: it copies the needed date directories into an isolated
work area and runs there, publishing atomically on success. The external model is injected as
`model_runner` so the orchestration can be tested without POSEIDON.

In `smoother` mode each analysis date uses a +/- window of neighbouring dates (a single-date
window degenerates to the filter), then forecasts far enough ahead to feed the next window.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import workflow
from .domain import AnalysisWindow, RunResult
from .geometry import box_levels
from .config import (normalize_localization, protected_inputs, raise_if_invalid,
                     resolve_perturbation, validate_parameters)
from .publish import assert_safe_dest, staged_publish
from .results import _checksum, build_manifest, export_result, write_manifest
from .runner import (RunContext, copy_lastconc_to_initconc, find_members,
                     publish_initconc, run_poseidon_all)


@dataclass
class CycleStep:
    """A single planned step. The fields are the one source of truth; `provenance` derives the
    manifest record from them, so no fact is stored twice."""

    kind: str                          # 'analyze' | 'forecast' | 'diagnose'
    date: str                          # analyze: target date; forecast/diagnose: that date
    window: tuple | None = None        # analyze: observation dates (target last; len 1 == filter)
    window_index: list | None = None   # analyze: [first, last] 0-based inclusive smoother-window bounds
    restart_kind: str | None = None    # forecast: 'analysis' (restart file) | 'restart' (prior lastconc)
    source: str | None = None          # forecast: date whose state seeds this one
    has_measurement: bool = False      # diagnose: a measurement file exists for this date

    @property
    def provenance(self) -> dict:
        if self.kind == 'analyze':
            prov = {'observations': list(self.window),
                    'kind': 'filter' if len(self.window) == 1 else 'smoother'}
            if self.window_index is not None:
                prov['window_index'] = self.window_index
            return prov
        if self.kind == 'forecast':
            return {'restart_from': self.source}
        return {'observations': [self.date]} if self.has_measurement else {'skipped': 'no measurement file'}


def derive_windows(dates, window_years=0.25):
    """+/- window_years assimilation windows as 0-based inclusive (first, last) index lists.

    first[i]/last[i] are the earliest/latest date indices whose physical time lies within
    window_years of date i.
    """
    t = [float(d) for d in dates]
    n = len(t)
    tol = 1e-9
    first = [min(j for j in range(n) if t[j] >= t[i] - window_years - tol) for i in range(n)]
    last = [max(j for j in range(n) if t[j] <= t[i] + window_years + tol) for i in range(n)]
    return first, last


def _diagnose_step(date, measurements_dir) -> CycleStep:
    """The planner is the sole owner of the 'is there a measurement for this date?' decision."""
    return CycleStep('diagnose', date,
                     has_measurement=(Path(measurements_dir) / f'{date}.dat').exists())


def _analyze_step(mode, dates, t, first, last) -> CycleStep:
    """Analysis step for target date index t: a single-date (filter) or +/- neighbour (smoother) window."""
    cur = dates[t]
    if mode == 'filter' or (first[t] == t and last[t] == t):     # trivial window -> filter
        return CycleStep('analyze', cur, window=(cur,))
    lo, hi = first[t], last[t]                                   # 0-based inclusive window bounds
    meas = [dates[j] for j in range(lo, t)]                      # past dates
    meas += [dates[j] for j in range(t + 1, hi + 1)]            # future dates
    meas += [cur]                                                # target is always last
    return CycleStep('analyze', cur, window=tuple(meas), window_index=[lo, hi])


def plan_cycle(mode: str, dates, measurements_dir, window_years=0.25) -> list:
    """Deterministic analyze/forecast/diagnose plan for any mode, built in one pass over the dates.

    For each target date: optionally analyse it, then forecast forward to the right edge of the
    horizon (the next date for noda/filter, the next window's last date for smoother), seeding the
    first leg from the analysis (or from the previous state for noda) and diagnosing that forecast.
    """
    if mode not in ('noda', 'filter', 'smoother'):
        raise ValueError(f'unknown cycle mode {mode!r}')
    dates = list(dates)
    first, last = derive_windows(dates, window_years) if mode == 'smoother' else (None, None)

    steps = []
    for t in range(len(dates) - 1):                          # target date index
        cur = dates[t]
        if mode != 'noda':
            steps.append(_analyze_step(mode, dates, t, first, last))
        horizon_end = last[t + 1] + 1 if mode == 'smoother' else t + 2    # exclusive date index
        for f in range(t + 1, horizon_end):
            if f == t + 1:
                steps.append(CycleStep('forecast', dates[f],
                                       restart_kind='restart' if mode == 'noda' else 'analysis', source=cur))
                steps.append(_diagnose_step(dates[f], measurements_dir))
            else:
                steps.append(CycleStep('forecast', dates[f], restart_kind='restart', source=dates[f - 1]))
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


# ---- execution -----------------------------------------------------------------

def _window_state_hashes(ctx, window, members) -> dict:
    """Per-date sha256 over the ensemble states actually read for this window (the work-dir copies,
    including any produced by earlier forecasts), so each analysis links to its real inputs — not
    just the manifest's archive hashes."""
    out = {}
    for d in window:
        h = hashlib.sha256()
        for m in sorted(members):
            h.update(_checksum(ctx.date_dir(d) / m / 'lastconc.xls').encode())
        out[d] = h.hexdigest()
    return out


def _run_analyze(step, config, ctx, perturb, rng, staging):
    result, data = workflow.analyze(AnalysisWindow(step.date, step.window), ctx, config,
                                    perturb=perturb, rng=rng)
    workflow.write_restarts(result, data, ctx)
    export_result(result, staging / 'analyses' / step.date)
    return {'step': 'analyze', 'date': step.date, 'status': 'ok',
            'provenance': step.provenance, 'diagnostics': result.diagnostics,
            'inputs_sha256': _window_state_hashes(ctx, step.window, result.member_ids)}


def _execute(plan, config, ctx, member_set, perturb, rng, runner, bnlev, staging):
    """Run one plan: analyse windows, seed + forecast each date, diagnose against observations.

    Each record gets a sequential `id`, so repeated forecasts of the same date (overlapping
    smoother windows) are individually identifiable in the manifest.
    """
    executed = []
    for i, step in enumerate(plan):
        if step.kind == 'analyze':
            rec = _run_analyze(step, config, ctx, perturb, rng, staging)
        elif step.kind == 'forecast':
            if step.restart_kind == 'analysis':
                publish_initconc(ctx.templates_dir, ctx.date_dir(step.date))   # from the analysis restart
            else:
                copy_lastconc_to_initconc(ctx.date_dir(step.source), ctx.date_dir(step.date), member_set)
            runner(ctx.date_dir(step.date), float(step.date), bnlev)
            rec = {'step': 'forecast', 'date': step.date, 'source': step.source,
                   'status': 'ok', 'provenance': step.provenance}
        else:  # diagnose
            if step.has_measurement:
                workflow.write_forecast_diagnostics(config, ctx, step.date, bnlev)
            rec = {'step': 'diagnose', 'date': step.date,
                   'status': 'ok' if step.has_measurement else 'skipped', 'provenance': step.provenance}
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
    runs_root = Path(config.runs_root)                              # mode is validated by plan_cycle below

    # ---- validate everything before any write ----
    assert_safe_dest(output_dir, protected_inputs(config))          # never overwrite the dataset (R2.1)
    config.localization = normalize_localization(config.localization)
    times = [_cycle_time(d, runs_root) for d in dates]              # finite time + safe path component (R2)
    if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
        raise ValueError(f'cycle dates must be strictly increasing in physical time (got {dates})')
    plan = plan_cycle(mode, dates, config.measurements_dir, config.cycle_window_years)
    if any(s.kind == 'analyze' for s in plan):                     # assimilation params only when analysing
        raise_if_invalid(validate_parameters(config))
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
    rng, perturbation = resolve_perturbation(seed, None, perturb)   # one generator for the whole cycle
    runner = model_runner or _default_model_runner(config, member_set)

    with staged_publish(output_dir, force) as staging:
        work = staging / 'work'
        for d in dates:
            shutil.copytree(runs_root / d, work / d, dirs_exist_ok=True)
        ctx = RunContext(runs_root=work, templates_dir=staging / 'templates', outputs_dir=staging / 'outputs')

        executed = _execute(plan, config, ctx, member_set, perturb, rng, runner, bnlev, staging)

        manifest = build_manifest(
            config, __version__, executed, operation='cycle', target_time=None,
            dates=dates, member_ids=member_set, perturbation=perturbation, mode=mode,
            window_years=config.cycle_window_years if mode == 'smoother' else None)
        write_manifest(manifest, staging)

    out = Path(output_dir)
    return RunResult(status='ok', steps=executed,
                     outputs={'work': str(out / 'work'), 'outputs_dir': str(out / 'outputs')},
                     manifest=manifest)
