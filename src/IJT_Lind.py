from __future__ import annotations

import numpy as np


def IJT_Lind(bnlev, brefnum, breflev, Nobs_T):
    brefnum = np.asarray(brefnum).ravel()
    breflev = np.asarray(breflev).ravel()

    Nb = len(bnlev) - 1
    MaxLev = int(np.max(bnlev))
    Ndates = len(Nobs_T)

    Lind = np.zeros((Nb, MaxLev, Ndates), dtype=int)
    Iind = []
    Jind = []
    Tind = []
    LL = 0
    obs_offset = 0

    for k in range(Ndates - 1):
        for j in range(int(Nobs_T[k])):
            LL += 1
            idx = obs_offset + j
            cur_box = int(brefnum[idx]) - 1
            cur_lev = int(breflev[idx]) - 1
            Lind[cur_box, cur_lev, k] = LL
            Iind.append(cur_box)
            Jind.append(cur_lev)
            Tind.append(k)
        obs_offset += int(Nobs_T[k])

    Lind_extract = np.zeros((Nb, MaxLev), dtype=int)
    LLL = 0
    for i in range(1, Nb + 1):
        for j in range(1, int(bnlev[i]) + 1):
            LL += 1
            LLL += 1
            Lind[i - 1, j - 1, Ndates - 1] = LL
            Iind.append(i - 1)
            Jind.append(j - 1)
            Tind.append(Ndates - 1)
            Lind_extract[i - 1, j - 1] = LLL

    return Lind, np.array(Iind, dtype=int), np.array(Jind, dtype=int), np.array(Tind, dtype=int), Lind_extract
