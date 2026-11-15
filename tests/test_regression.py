"""Regression: refactored solver reproduces the previous result within tolerance.

Golden arrays in tests/reference/ were captured from the previous stacked-least-squares
solver with perturbations zeroed (tools/make_reference.py). The refactored
observation-space solver is the exact least-squares minimizer, so it agrees with
the previous least-squares output up to least-squares's convergence tolerance (~1e-6), not bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import HAS_DATA, REF

requires_data = pytest.mark.skipif(not HAS_DATA, reason='experiment archive not present')

# least-squares (atol=1e-8) leaves a small residual vs the exact solver.
TOL = 5e-6


@requires_data
def test_filter_matches_golden(isolated_ctx):
    import main_LN
    gold = np.load(REF / 'filter_1986.75.npz')['sol']
    res = main_LN.main_LN('1986.75', run_ctx=isolated_ctx, perturb=False)
    assert res.log_states.shape == gold.shape
    assert np.max(np.abs(res.log_states - gold)) < TOL


@requires_data
def test_smoother_matches_golden(isolated_ctx):
    import main_KSF_LN
    gold = np.load(REF / 'smoother_1988.625_1988.5.npz')['sol']
    res = main_KSF_LN.main_KSF_LN(['1988.625', '1988.5'], run_ctx=isolated_ctx, perturb=False)
    assert res.log_states.shape == gold.shape
    assert np.max(np.abs(res.log_states - gold)) < TOL


@requires_data
def test_perturbations_are_deterministic_when_disabled(isolated_ctx):
    import main_LN
    a = main_LN.main_LN('1986.75', run_ctx=isolated_ctx, perturb=False)
    b = main_LN.main_LN('1986.75', run_ctx=isolated_ctx, perturb=False)
    assert np.array_equal(a.log_states, b.log_states)
