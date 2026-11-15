"""Regenerate CANDIDATE reference arrays and compare to the frozen baseline.

The frozen baseline in tests/reference/*.npz is the regression ground truth (see
tests/reference/PROVENANCE.md) and is deliberately NOT overwritten here. This script
recomputes the arrays with the current package (deterministic, perturbations off),
writes them under tests/reference/candidate/, and prints the max deviation from the
baseline. Promoting a candidate to the baseline is a manual, justified step.

Usage:
    python tools/make_reference.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from poseidon_da.config import DEFAULT_BDEPTHS, DEFAULT_GSD, normalize_localization
from poseidon_da.domain import AnalysisWindow
from poseidon_da.paths import MEASUREMENTS_DIR, RUNS_ROOT, box_surfareas_with_centers_path
from poseidon_da.runner import RunContext
from poseidon_da import workflow

REF_DIR = Path(__file__).resolve().parent.parent / 'tests' / 'reference'
CAND_DIR = REF_DIR / 'candidate'
GEOM = str(box_surfareas_with_centers_path())
MEAS = str(MEASUREMENTS_DIR)
PARAMS = dict(geometry_path=GEOM, measurements_dir=MEAS, gsd=DEFAULT_GSD,
              localization=normalize_localization(), bdepths=list(DEFAULT_BDEPTHS), perturb=False)

CASES = {
    'filter_1986.75.npz': AnalysisWindow('1986.75', ('1986.75',)),
    'smoother_1988.625_1988.5.npz': AnalysisWindow('1988.5', ('1988.625', '1988.5')),
}


def _run(window):
    ctx = RunContext.isolated(CAND_DIR / '_tmp', runs_root=RUNS_ROOT)
    result, _ = workflow.analyze(window, ctx, **PARAMS)
    return result.log_states


def main():
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    for name, window in CASES.items():
        sol = _run(window)
        np.savez(CAND_DIR / name, sol=sol)
        baseline = REF_DIR / name
        if baseline.exists():
            gold = np.load(baseline)['sol']
            delta = np.max(np.abs(sol - gold)) if sol.shape == gold.shape else float('nan')
            print(f'{name}: candidate {sol.shape}, max|candidate-baseline| = {delta:.3e}')
        else:
            print(f'{name}: candidate {sol.shape} (no baseline to compare)')
    print(f'candidates written to {CAND_DIR} (baseline left untouched)')


if __name__ == '__main__':
    main()
