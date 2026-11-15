"""Regression: the solver reproduces the stored reference arrays within tolerance.

Reference arrays in tests/reference/ were produced deterministically (perturbations off).
The comparison allows a small numerical tolerance (~1e-6) rather than bit-for-bit equality.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import HAS_DATA, REF

from poseidon_da.config import DEFAULT_BDEPTHS, DEFAULT_GSD, normalize_localization
from poseidon_da.domain import AnalysisWindow
from poseidon_da import workflow
from poseidon_da.paths import box_surfareas_with_centers_path, MEASUREMENTS_DIR

requires_data = pytest.mark.skipif(not HAS_DATA, reason='experiment archive not present')
TOL = 5e-6
GEOM = str(box_surfareas_with_centers_path())
MEAS = str(MEASUREMENTS_DIR)
PARAMS = dict(geometry_path=GEOM, measurements_dir=MEAS, gsd=DEFAULT_GSD,
              localization=normalize_localization(), bdepths=list(DEFAULT_BDEPTHS), perturb=False)


@requires_data
def test_filter_matches_golden(isolated_ctx):
    gold = np.load(REF / 'filter_1986.75.npz')['sol']
    w = AnalysisWindow('1986.75', ('1986.75',))
    res, _ = workflow.analyze(w, isolated_ctx, **PARAMS)
    assert res.log_states.shape == gold.shape
    assert np.max(np.abs(res.log_states - gold)) < TOL


@requires_data
def test_smoother_matches_golden(isolated_ctx):
    gold = np.load(REF / 'smoother_1988.625_1988.5.npz')['sol']
    w = AnalysisWindow('1988.5', ('1988.625', '1988.5'))
    res, _ = workflow.analyze(w, isolated_ctx, **PARAMS)
    assert res.log_states.shape == gold.shape
    assert np.max(np.abs(res.log_states - gold)) < TOL


@requires_data
def test_deterministic_when_perturbation_disabled(isolated_ctx):
    w = AnalysisWindow('1986.75', ('1986.75',))
    a, _ = workflow.analyze(w, isolated_ctx, **PARAMS)
    b, _ = workflow.analyze(w, isolated_ctx, **PARAMS)
    assert np.array_equal(a.log_states, b.log_states)
