"""Ensemble covariance estimation and distance-based localization (pure math)."""
from __future__ import annotations

import math

import numpy as np


def geo_distance(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points (degrees), in metres."""
    R = 6373.0
    fi1, fi2 = lat1 * 3.1416 / 180.0, lat2 * 3.1416 / 180.0
    psi1, psi2 = lon1 * 3.1416 / 180.0, lon2 * 3.1416 / 180.0
    inside = math.sin(0.5 * (fi2 - fi1)) ** 2 + math.cos(fi2) * math.cos(fi1) * math.sin(0.5 * (psi2 - psi1)) ** 2
    return 1000.0 * (2.0 * R * math.asin(math.sqrt(inside)))


def estimate_covariance(log_states) -> np.ndarray:
    """Sample covariance of the ensemble (variables = state rows, 1/(M-1) normalization)."""
    return np.cov(np.asarray(log_states, dtype=float), bias=False)


def localize_covariance(CC, Iind, Jind, centers, layer_depths, exclude, RcorH, RcorV,
                        Tind=None, obstimes=None, RcorT=1.0):
    """Element-wise localization of CC by horizontal, vertical (and optional temporal) distance.

    Damping is exp(-DistH/RcorH) * exp(-DistV/RcorV) [* exp(-DistT/RcorT)]. layer_depths
    are the representative depths used for the vertical distance. Boxes in `exclude`
    (1-based ids) are decoupled from all others. Returns (B, LocM).
    """
    exclude = set(int(x) for x in exclude)
    N = CC.shape[0]
    LocM = np.ones((N, N), dtype=float)
    B = np.array(CC, dtype=float, copy=True)
    for ii in range(N):
        for jj in range(ii + 1, N):
            box1, box2 = int(Iind[ii]), int(Iind[jj])
            if (box1 + 1) in exclude or (box2 + 1) in exclude:
                LocM[ii, jj] = LocM[jj, ii] = 0.0
                B[ii, jj] = B[jj, ii] = 0.0
                continue
            lat1, lon1 = centers[box1]
            lat2, lon2 = centers[box2]
            DistH = geo_distance(lat1, lon1, lat2, lon2)
            DistV = abs(layer_depths[int(Jind[ii])] - layer_depths[int(Jind[jj])])
            factor = math.exp(-DistH / RcorH) * math.exp(-DistV / RcorV)
            if Tind is not None:
                DistT = abs(float(obstimes[int(Tind[ii])]) - float(obstimes[int(Tind[jj])]))
                factor *= math.exp(-DistT / RcorT)
            LocM[ii, jj] = LocM[jj, ii] = factor
            B[ii, jj] = B[jj, ii] = B[ii, jj] * factor
    return B, LocM
