"""Generate golden reference arrays from the CURRENT solver (Stage 0).

Runs the previous filter (main_LN) and smoother (main_KSF_LN) with the
observation perturbations forced to zero, and captures the exact per-member
log-state solution produced by the current stacked-least-squares solver. These arrays
become the regression baseline against which the refactored solver and the
operational fixes are validated.

Usage:
    python tools/make_reference.py

Writes:
    tests/reference/filter_1986.75.npz
    tests/reference/smoother_1988.625_1988.5.npz

Note: running the mains writes into DAruns/ (restart files, out_*.xlsx).
Restore them afterwards with `git checkout -- DAruns`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
REF_DIR = ROOT / 'tests' / 'reference'


def _capture(mod, func):
    """Run a main module with randn->0, capturing each least-squares column it solves."""
    captured = []
    orig_least-squares = mod.model_least-squares

    def wrapped(A, b, *a, **k):
        x = orig_least-squares(A, b, *a, **k)
        captured.append(np.asarray(x, dtype=float).copy())
        return x

    saved_randn = np.random.randn
    mod.model_least-squares = wrapped
    np.random.randn = lambda *a, **k: (np.zeros(a) if a else 0.0)
    try:
        func()
    finally:
        mod.model_least-squares = orig_least-squares
        np.random.randn = saved_randn

    # captured[i] is the log-state solution for ensemble member i
    return np.column_stack(captured)


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)

    import main_LN
    sol_filter = _capture(main_LN, lambda: main_LN.main_LN('1986.75'))
    np.savez(
        REF_DIR / 'filter_1986.75.npz',
        sol=sol_filter,
        strdate='1986.75',
        kind='filter',
    )
    print(f'filter 1986.75: sol shape {sol_filter.shape}')

    import main_KSF_LN
    sol_smoother = _capture(
        main_KSF_LN, lambda: main_KSF_LN.main_KSF_LN(['1988.625', '1988.5'])
    )
    np.savez(
        REF_DIR / 'smoother_1988.625_1988.5.npz',
        sol=sol_smoother,
        strdates=np.array(['1988.625', '1988.5']),
        kind='smoother',
    )
    print(f'smoother [1988.625,1988.5]: sol shape {sol_smoother.shape}')


if __name__ == '__main__':
    main()
