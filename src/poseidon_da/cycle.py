"""Sequential assimilation-forecast cycle (NODA / filter) with explicit provenance.

The cycle never mutates the dataset: it copies the needed date directories into an
isolated work area and runs there. The external model is injected as `model_runner`
so the orchestration can be tested without POSEIDON. The smoother cycle is gated until
its information-reuse contract is decided (review 4.1).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import workflow
from .domain import AnalysisWindow, RunResult
from .geometry import box_levels
from .observations import aggregate_observations
from .poseidon_io import read_poseidon, xlswrite
from .results import write_manifest
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
            "smoother cycle needs a decided information-reuse contract (review 4.1); "
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


def _default_model_runner(config):
    def runner(date_dir, expected_time, bnlev):
        run_poseidon_all(date_dir, expected_time=expected_time, bnlev=bnlev,
                         exe_path=config.model_executable, launcher=config.launcher)
    return runner


def _diagnose(config, ctx, date, bnlev):
    batch = aggregate_observations(str(config.measurements_dir / f'{date}.dat'),
                                   str(config.geometry), config.bdepths, expected_time=float(date))
    obs, brn, brl = batch['values'], batch['brefnum'], batch['breflev']
    members = find_members(ctx.date_dir(date))
    No = obs.shape[0]
    ens = np.zeros((No, 1))
    xlswrite(str(ctx.output_xlsx(date)), np.concatenate((obs, brn), axis=1), 'obs')
    for m in members:
        arr, _, _, _ = read_poseidon(str(ctx.date_dir(date) / m / 'lastconc.xls'), bnlev)
        sim = np.array([[arr[int(brn[L, 0]) - 1, int(brl[L, 0]) - 1]] for L in range(No)])
        ens += sim
        xlswrite(str(ctx.output_xlsx(date)), sim, m)
    xlswrite(str(ctx.output_xlsx(date)), ens / np.single(len(members)), 'arithmetic_mean')


def run_cycle(config, output_dir, *, mode=None, dates=None, model_runner=None,
              perturb=True, seed=None, force=False) -> RunResult:
    """Run a NODA or filter cycle over a date sequence, producing a provenance manifest."""
    from . import __version__
    mode = mode or config.cycle_mode
    dates = list(dates or config.cycle_dates)
    if len(dates) < 2:
        raise ValueError('cycle requires at least two dates')
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f'output directory is not empty: {output_dir} (use force=True)')
    for d in dates:
        if not (config.runs_root / d).is_dir():
            raise FileNotFoundError(f'cycle: run directory not found for date {d}: {config.runs_root / d}')
    if model_runner is None:
        model_runner = _default_model_runner(config)

    bnlev = box_levels()
    work = output_dir / 'work'
    for d in dates:
        shutil.copytree(config.runs_root / d, work / d, dirs_exist_ok=True)
    ctx = RunContext(runs_root=work, templates_dir=output_dir / 'templates',
                     outputs_dir=output_dir / 'outputs')
    rng = np.random.default_rng(seed) if seed is not None else None

    executed = []
    for step in plan_cycle(mode, dates, config.measurements_dir):
        if step.kind == 'analyze':
            window = AnalysisWindow(step.date, (step.date,))
            result, payload = workflow.analyze(
                window, ctx, geometry_path=str(config.geometry),
                measurements_dir=str(config.measurements_dir), gsd=config.gsd,
                localization=config.localization, bdepths=config.bdepths, perturb=perturb, rng=rng)
            workflow.write_restarts(result, payload, ctx)
            executed.append({'step': 'analyze', 'date': step.date, 'status': 'ok',
                             'provenance': step.provenance, 'diagnostics': result.diagnostics})
        elif step.kind == 'forecast':
            if mode == 'filter':
                publish_initconc(ctx.templates_dir, ctx.date_dir(step.date))       # analysed state seeds the forecast
            else:
                copy_lastconc_to_initconc(ctx.date_dir(step.source), ctx.date_dir(step.date))
            model_runner(ctx.date_dir(step.date), step.time, bnlev)
            executed.append({'step': 'forecast', 'date': step.date, 'source': step.source,
                             'status': 'ok', 'provenance': step.provenance})
        elif step.kind == 'diagnose':
            if step.provenance.get('observations'):
                _diagnose(config, ctx, step.date, bnlev)
                executed.append({'step': 'diagnose', 'date': step.date, 'status': 'ok',
                                 'provenance': step.provenance})
            else:
                executed.append({'step': 'diagnose', 'date': step.date, 'status': 'skipped',
                                 'provenance': step.provenance})

    manifest = {'package_version': __version__, 'case_id': config.case_id, 'mode': mode,
                'dates': list(dates), 'perturbation': {'enabled': bool(perturb), 'seed': seed},
                'steps': executed}
    write_manifest(manifest, output_dir)
    return RunResult(status='ok', steps=executed, outputs={'work': str(work),
                     'outputs_dir': str(ctx.outputs_dir)}, manifest=manifest)
