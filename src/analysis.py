"""Pure analysis core: observation operator indices and the Kalman update.

No file or process side effects. Functions take validated NumPy arrays and
return NumPy arrays, so they can be tested without disk or POSEIDON.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class AnalysisResult:
    """Structured result of one analysis step (no file side effects)."""

    target: str
    member_ids: list
    obs_indices: np.ndarray       # (p,) 0-based observed state rows
    observations: np.ndarray      # (p,) assimilated values, log-space
    log_states: np.ndarray        # (n, M) per-member analysis, log-space
    concentrations: np.ndarray    # (n, M) = exp(log_states)
    geometric_mean: np.ndarray    # (n,) = exp(mean(log_states, axis=1))
    diagnostics: dict = field(default_factory=dict)


def observation_indices(brefnum, breflev, Lind, Tind=None):
    """0-based state indices observed by H; validates box/layer (and date) existence.

    brefnum/breflev are 1-based physical box/level IDs. Lind maps (box, level[, date])
    to a 1-based state index, with 0 marking a non-existent component. Raises
    ValueError on any observation that points at a non-existent state component,
    instead of silently wrapping to another valid index.
    """
    brefnum = np.asarray(brefnum).ravel()
    breflev = np.asarray(breflev).ravel()
    n_obs = int(brefnum.size)
    if Tind is not None:
        Tind = np.asarray(Tind).ravel()

    nb = int(Lind.shape[0])
    ml = int(Lind.shape[1])
    idx = np.empty(n_obs, dtype=int)
    for i in range(n_obs):
        b = int(brefnum[i])
        lev = int(breflev[i])
        if not (1 <= b <= nb):
            raise ValueError(f'observation {i}: box {b} out of range 1..{nb}')
        if not (1 <= lev <= ml):
            raise ValueError(f'observation {i}: level {lev} out of range 1..{ml}')
        if Tind is None:
            lind_val = int(Lind[b - 1, lev - 1])
            where = f'box {b}, level {lev}'
        else:
            t = int(Tind[i])
            lind_val = int(Lind[b - 1, lev - 1, t])
            where = f'box {b}, level {lev}, date index {t}'
        if lind_val < 1:
            raise ValueError(f'observation {i}: no state component for {where}')
        idx[i] = lind_val - 1
    return idx


def analysis_update(B, obs_indices, Y, R, Xf):
    """Ensemble Kalman analysis computed in observation space.

    For every ensemble member m this minimizes
        J(x) = (x - xf_m)^T B^-1 (x - xf_m) + (H x - y_m)^T R^-1 (H x - y_m),
    where H selects the rows ``obs_indices``. Closed form:
        Xa = Xf + B H^T (H B H^T + R)^-1 (Y - H Xf).
    This is the normal-equations solution of the previous stacked least-squares
    problem, computed with a single Cholesky factorization of the p x p system
    instead of an explicit B^-1, an LDL factorization and one least-squares per member.

    Parameters
    ----------
    B : (n, n) localized background covariance, symmetric positive definite.
    obs_indices : (p,) 0-based state rows observed by H.
    Y : (p, M) observations per ensemble member (already perturbed if desired).
    R : (p, p) observation-error covariance, symmetric positive definite.
    Xf : (n, M) background (forecast) log-states per member.

    Returns
    -------
    Xa : (n, M) analysis log-states per member.
    """
    B = np.asarray(B, dtype=float)
    Xf = np.asarray(Xf, dtype=float)
    Y = np.asarray(Y, dtype=float)
    R = np.asarray(R, dtype=float)
    obs_indices = np.asarray(obs_indices, dtype=int)

    if Xf.ndim != 2:
        Xf = Xf.reshape(-1, 1)
    if Y.ndim != 2:
        Y = Y.reshape(-1, 1)

    cross = B[:, obs_indices]                      # n x p  (B H^T)
    S = B[np.ix_(obs_indices, obs_indices)] + R    # p x p  (H B H^T + R)
    innovation = Y - Xf[obs_indices, :]            # p x M
    factor = cho_factor(S, lower=True)
    Xa = Xf + cross @ cho_solve(factor, innovation)
    if not np.all(np.isfinite(Xa)):
        raise FloatingPointError('analysis_update produced non-finite values')
    return Xa
