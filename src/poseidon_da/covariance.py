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
    (1-based ids) are decoupled from all others. Returns the localized covariance B.
    """
    exclude = set(int(x) for x in exclude)
    N = CC.shape[0]
    B = np.array(CC, dtype=float, copy=True)

    dist_cache = {}
    def _horizontal(a, b):                       # geo_distance depends only on the box pair; memoize it
        if a == b:
            return 0.0
        key = (a, b) if a < b else (b, a)
        d = dist_cache.get(key)
        if d is None:
            d = dist_cache[key] = geo_distance(*centers[a], *centers[b])
        return d

    for ii in range(N):
        for jj in range(ii + 1, N):
            box1, box2 = int(Iind[ii]), int(Iind[jj])
            if (box1 + 1) in exclude or (box2 + 1) in exclude:
                B[ii, jj] = B[jj, ii] = 0.0
                continue
            dist = _horizontal(box1, box2) / RcorH
            dist += abs(layer_depths[int(Jind[ii])] - layer_depths[int(Jind[jj])]) / RcorV
            if Tind is not None:
                dist += abs(float(obstimes[int(Tind[ii])]) - float(obstimes[int(Tind[jj])])) / RcorT
            B[ii, jj] = B[jj, ii] = B[ii, jj] * math.exp(-dist)
    return B
