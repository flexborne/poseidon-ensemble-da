from __future__ import annotations

import numpy as np

from Dist_geocords import Dist_geocords
from matlab_compat import load_numeric_excel


def regularize_Covmatr_analit(CC, Iind, Jind, exclude=None, RcorH=150000, RcorV=100, fname_boxes='Box_surfareas_with_centers.xlsx'):
    if exclude is None:
        exclude = [1, 2, 3, 49, 40]
    exclude = set(int(x) for x in exclude)
    depb = [25 / 2, (25 + 100) / 2, (100 + 500) / 2, (500 + 2000) / 2]

    AAA = load_numeric_excel(fname_boxes)
    boxn = AAA[:, 0].astype(int)
    boxlat = AAA[:, 9]
    boxlon = AAA[:, 10]
    centers = {int(boxn[i]) - 1: (float(boxlat[i]), float(boxlon[i])) for i in range(len(boxn)) if not np.isnan(boxn[i])}

    N = CC.shape[0]
    LocM = np.ones((N, N), dtype=float)
    CC1 = np.array(CC, dtype=float, copy=True)
    for ii in range(N):
        for jj in range(ii + 1, N):
            curbox1 = int(Iind[ii])
            curbox2 = int(Iind[jj])
            if (curbox1 + 1) in exclude or (curbox2 + 1) in exclude:
                LocM[ii, jj] = 0.0
                LocM[jj, ii] = 0.0
                CC1[ii, jj] = 0.0
                CC1[jj, ii] = 0.0
                continue

            lat1, lon1 = centers[curbox1]
            lat2, lon2 = centers[curbox2]
            DistH = Dist_geocords(lat1, lon1, lat2, lon2)
            DistV = abs(depb[int(Jind[ii])] - depb[int(Jind[jj])])
            LocM[ii, jj] = np.exp(-DistH / RcorH) * np.exp(-DistV / RcorV)
            LocM[jj, ii] = LocM[ii, jj]
            CC1[ii, jj] = CC1[ii, jj] * LocM[ii, jj]
            CC1[jj, ii] = CC1[ii, jj]
    return CC1, LocM
