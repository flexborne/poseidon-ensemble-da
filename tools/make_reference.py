"""Regenerate the golden reference arrays (deterministic, perturbations off).

The committed arrays in tests/reference/ were originally captured from the previous
stacked-least-squares solver; the current observation-space solver reproduces them within
~1e-6. Running this re-derives them from the current package (so they stay
reproducible); the regression tests allow a small tolerance either way.

Usage:
    python tools/make_reference.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from poseidon_da.domain import AnalysisWindow
from poseidon_da.paths import MEASUREMENTS_DIR, RUNS_ROOT, box_surfareas_with_centers_path
from poseidon_da.runner import RunContext
from poseidon_da import workflow

REF_DIR = Path(__file__).resolve().parent.parent / 'tests' / 'reference'
GEOM = str(box_surfareas_with_centers_path())
MEAS = str(MEASUREMENTS_DIR)


def _run(window):
    ctx = RunContext.isolated(REF_DIR.parent / '_tmp_ref', runs_root=RUNS_ROOT)
    result, _ = workflow.analyze(window, ctx, geometry_path=GEOM, measurements_dir=MEAS, perturb=False)
    return result.log_states


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    sol = _run(AnalysisWindow('1986.75', ('1986.75',)))
    np.savez(REF_DIR / 'filter_1986.75.npz', sol=sol, strdate='1986.75', kind='filter')
    print(f'filter 1986.75: {sol.shape}')

    sol = _run(AnalysisWindow('1988.5', ('1988.625', '1988.5')))
    np.savez(REF_DIR / 'smoother_1988.625_1988.5.npz', sol=sol,
             strdates=np.array(['1988.625', '1988.5']), kind='smoother')
    print(f'smoother [1988.625,1988.5]: {sol.shape}')


if __name__ == '__main__':
    main()
