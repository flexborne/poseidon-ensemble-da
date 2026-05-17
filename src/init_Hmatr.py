from __future__ import annotations

import numpy as np


def init_Hmatr(brefnum, breflev, Lind):
    brefnum = np.asarray(brefnum).ravel()
    breflev = np.asarray(breflev).ravel()
    Nobs = int(brefnum.size)
    Nstate = int(np.max(Lind))
    Hmatr = np.zeros((Nobs, Nstate), dtype=float)
    for i in range(Nobs):
        Hmatr[i, Lind[int(brefnum[i]) - 1, int(breflev[i]) - 1] - 1] = 1.0
    return Hmatr
