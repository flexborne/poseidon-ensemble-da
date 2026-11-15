"""Box grid: vertical level counts, localization depths, and box centres of mass."""
from __future__ import annotations

import numpy as np

from .poseidon_io import load_numeric_excel

# Representative mid-depths of the four fixed layers, used for vertical localization.
LAYER_DEPTHS = [25 / 2, (25 + 100) / 2, (100 + 500) / 2, (500 + 2000) / 2]


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
