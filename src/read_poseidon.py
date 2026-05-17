from __future__ import annotations

import numpy as np

from box_levs import box_levs


def read_poseidon(pfile=r'..\1986.5\01.hydrol1dep1\lastconc.xls', bnlev=None):
    if bnlev is None:
        bnlev = box_levs()
    Nbox = len(bnlev) - 1
    MaxLev = int(np.max(bnlev))
    arr = np.zeros((Nbox, MaxLev), dtype=float)
    sed = np.zeros((Nbox, 2), dtype=float)
    header = []
    footer = []

    with open(pfile, 'r', encoding='utf-8', errors='ignore') as fid:
        count = 0
        for raw_line in fid:
            tline = raw_line.rstrip('\r\n')
            count += 1
            if count <= 6:
                header.append({'tline': tline})
                continue
            stripped = tline.strip()
            if not stripped:
                continue
            C = stripped.split()
            nn = int(float(C[0]))
            if nn > Nbox:
                footer.append({'tline': tline})
                continue
            expected_cols = int(bnlev[nn]) + 3
            if len(C) != expected_cols:
                raise ValueError('Error from read_poseidon: number of levels specified in bnlev is inconsistent with the number of levels in data file')
            for j in range(1, int(bnlev[nn]) + 1):
                arr[nn - 1, j - 1] = float(C[j])
            sed[nn - 1, 0] = float(C[j + 1])
            sed[nn - 1, 1] = float(C[j + 2])
    return arr, header, footer, sed
