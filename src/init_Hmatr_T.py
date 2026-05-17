from __future__ import annotations

import numpy as np


def init_Hmatr_T(brefnum, breflev, Lind, Tind):
    brefnum = np.asarray(brefnum).ravel()
    breflev = np.asarray(breflev).ravel()
    Tind = np.asarray(Tind).ravel()
    No = int(brefnum.size)
    Ns = int(np.max(Lind))
    Hmatr = np.zeros((No, Ns), dtype=float)
    for i in range(No):
        Hmatr[i, Lind[int(brefnum[i]) - 1, int(breflev[i]) - 1, int(Tind[i])] - 1] = 1.0
    return Hmatr
