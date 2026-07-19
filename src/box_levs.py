from __future__ import annotations

import numpy as np


def box_levs() -> np.ndarray:
    """Number of vertical levels in each box.

    Index 0 is an unused dummy slot; valid boxes are 1..50.
    """
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
