from __future__ import annotations

import numpy as np


def IJ_Lind(bnlev):
    Nb = len(bnlev) - 1
    MaxLev = int(np.max(bnlev))
    Lind = np.zeros((Nb, MaxLev), dtype=int)
    Iind = []
    Jind = []
    LL = 0
    for i in range(1, Nb + 1):
        for j in range(1, int(bnlev[i]) + 1):
            LL += 1
            Lind[i - 1, j - 1] = LL
            Iind.append(i - 1)
            Jind.append(j - 1)
    return Lind, np.array(Iind, dtype=int), np.array(Jind, dtype=int)
