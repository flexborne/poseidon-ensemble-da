from __future__ import annotations

import numpy as np


def regularize_Covmatr(CC, Iind, Jind, exclude=None):
    if exclude is None:
        exclude = [1, 2, 3, 49, 40]
    exclude = set(int(x) for x in exclude)
    N = CC.shape[0]
    LocM = np.ones((N, N), dtype=float)
    CC1 = np.array(CC, dtype=float, copy=True)
    for ii in range(N):
        for jj in range(ii + 1, N):
            if int(Iind[ii]) + 1 in exclude or int(Iind[jj]) + 1 in exclude:
                LocM[ii, jj] = 0.0
                LocM[jj, ii] = 0.0
                CC1[ii, jj] = 0.0
                CC1[jj, ii] = 0.0
    return CC1, LocM
