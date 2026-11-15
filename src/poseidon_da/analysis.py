"""The analysis update: an ensemble Kalman step solved in observation space."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def analysis_update(B, obs_indices, Y, R, Xf):
    """Ensemble Kalman analysis computed in observation space.

    For every ensemble member m this minimizes
        J(x) = (x - xf_m)^T B^-1 (x - xf_m) + (H x - y_m)^T R^-1 (H x - y_m),
    where H selects the rows obs_indices. Closed form:
        Xa = Xf + B H^T (H B H^T + R)^-1 (Y - H Xf),
    computed with a single Cholesky factorization of the p x p system.

    B : (n, n) localized background covariance, symmetric positive definite.
    obs_indices : (p,) 0-based state rows observed by H.
    Y : (p, M) observations per ensemble member (already perturbed if desired).
    R : (p, p) observation-error covariance, symmetric positive definite.
    Xf : (n, M) background log-states per member. Returns Xa : (n, M).
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
    Xa = Xf + cross @ cho_solve(cho_factor(S, lower=True), innovation)
    if not np.all(np.isfinite(Xa)):
        raise FloatingPointError('analysis_update produced non-finite values')
    return Xa
