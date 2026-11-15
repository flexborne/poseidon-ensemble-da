"""Box grid: vertical level counts, localization depths, and box centres of mass."""
from __future__ import annotations

import numpy as np

from .poseidon_io import load_numeric_excel

def layer_depths_from_bdepths(bdepths):
    """Representative mid-depth of each layer from the depth-bin upper bounds.

    One source of truth for the vertical coordinate: the same bdepths that bin the
    measurements also define the depths used for vertical localization.
    """
    edges = [0.0] + [float(b) for b in bdepths]
    return [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(bdepths))]


def box_levels() -> np.ndarray:
    """Number of vertical levels per box. Index 0 is an unused slot; boxes are 1..50."""
    bnlev = np.zeros(51, dtype=int)
    bnlev[1:51] = 4
    bnlev[1:3] = 1
    bnlev[34] = 1
    bnlev[44:46] = 1
    bnlev[48:51] = 1
    bnlev[33] = 2
    bnlev[35:37] = 2
    bnlev[46:48] = 2
    bnlev[3] = 3
    bnlev[6] = 3
    bnlev[14] = 3
    bnlev[37] = 3
    bnlev[39] = 3
    return bnlev


def load_box_centers(geometry_path) -> dict:
    """Return {0-based box index -> (lat, lon)} from the geometry workbook (Sheet1)."""
    AAA = load_numeric_excel(geometry_path, sheet_name='Sheet1')
    boxn = AAA[:, 0]
    lat = AAA[:, 9]
    lon = AAA[:, 10]
    return {int(boxn[i]) - 1: (float(lat[i]), float(lon[i]))
            for i in range(len(boxn)) if not np.isnan(boxn[i])}
