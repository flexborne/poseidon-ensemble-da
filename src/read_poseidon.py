from __future__ import annotations

import numpy as np

from box_levs import box_levs


def read_poseidon(pfile=r'..\1986.5\01.hydrol1dep1\lastconc.xls', bnlev=None, validate=True):
    """Read a POSEIDON output file (lastconc.xls).

    Returns arr (concentrations per box x level), header, footer (lines kept
    unchanged) and sed (sediments, 2 columns).

    With validate=True every box row 1..Nbox must be present exactly once, and
    the active water concentrations must be finite and strictly positive (so the
    subsequent log is well defined). A missing or duplicated row, or a
    non-positive active concentration, raises ValueError instead of silently
    leaving a zero or overwriting a value. Inactive padding (levels beyond
    bnlev) and sediments are not subject to the positivity check.
    """
    if bnlev is None:
        bnlev = box_levs()
    Nbox = len(bnlev) - 1
    MaxLev = int(np.max(bnlev))
    arr = np.zeros((Nbox, MaxLev), dtype=float)
    sed = np.zeros((Nbox, 2), dtype=float)
    header = []
    footer = []
    seen = set()

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
            if validate and nn in seen:
                raise ValueError(f'read_poseidon: duplicate row for box {nn} in {pfile}')
            seen.add(nn)
            expected_cols = int(bnlev[nn]) + 3
            if len(C) != expected_cols:
                raise ValueError('Error from read_poseidon: number of levels specified in bnlev is inconsistent with the number of levels in data file')
            for j in range(1, int(bnlev[nn]) + 1):
                value = float(C[j])
                if validate and not (np.isfinite(value) and value > 0.0):
                    raise ValueError(f'read_poseidon: non-positive/non-finite concentration for box {nn}, level {j} in {pfile}: {value}')
                arr[nn - 1, j - 1] = value
            sed[nn - 1, 0] = float(C[j + 1])
            sed[nn - 1, 1] = float(C[j + 2])

    if validate:
        missing = [b for b in range(1, Nbox + 1) if b not in seen]
        if missing:
            raise ValueError(f'read_poseidon: missing box rows {missing} in {pfile}')
    return arr, header, footer, sed
