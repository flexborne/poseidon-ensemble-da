"""Independent algebraic properties of the analysis core (no disk, no POSEIDON)."""
from __future__ import annotations

import numpy as np
import pytest

from poseidon_da.analysis import analysis_update
from poseidon_da.geometry import box_levels
from poseidon_da.state import build_layout, observation_indices


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def test_scalar_gaussian_update_matches_closed_form():
    B = np.array([[4.0, 1.0], [1.0, 2.0]])
    obs_idx = np.array([0])
    R = np.array([[0.5]])
    Xf = np.array([[1.0], [2.0]])
    Y = np.array([[3.0]])
    K = B[:, obs_idx] @ np.linalg.inv(B[np.ix_(obs_idx, obs_idx)] + R)
    expected = Xf + K @ (Y - Xf[obs_idx, :])
    assert np.allclose(analysis_update(B, obs_idx, Y, R, Xf), expected, atol=1e-12)


def test_unobserved_decoupled_coordinate_is_unchanged():
    B = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]])
    obs_idx = np.array([0, 1])
    R = 0.3 * np.eye(2)
    Xf = np.array([[1.0], [1.0], [7.0]])
    Y = np.array([[2.0], [0.0]])
    Xa = analysis_update(B, obs_idx, Y, R, Xf)
    assert Xa[2, 0] == pytest.approx(Xf[2, 0], abs=1e-12)


def test_full_equals_reduced_window():
    n = 12
    B = _spd(n, seed=1)
    obs_idx = np.array([1, 4, 7])
    target = np.array([0, 1, 4, 7, 11])
    M = 5
    rng = np.random.default_rng(2)
    Xf = rng.standard_normal((n, M))
    R = 0.2 * np.eye(len(obs_idx))
    Y = Xf[obs_idx, :] + rng.standard_normal((len(obs_idx), M))
    Xa_full = analysis_update(B, obs_idx, Y, R, Xf)

    keep = np.array(sorted(set(target) | set(obs_idx)))
    pos = {c: i for i, c in enumerate(keep)}
    Br = B[np.ix_(keep, keep)]
    obs_idx_r = np.array([pos[c] for c in obs_idx])
    Xa_red = analysis_update(Br, obs_idx_r, Y, R, Xf[keep, :])
    for c in target:
        assert np.allclose(Xa_full[c, :], Xa_red[pos[c], :], atol=1e-10)


def test_observation_indices_rejects_nonexistent_layer():
    layout = build_layout(box_levels())
    with pytest.raises(ValueError):
        observation_indices([1], [2], layout)       # box 1 has a single level
    idx = observation_indices([1], [1], layout)
    assert idx.shape == (1,) and idx[0] >= 0
