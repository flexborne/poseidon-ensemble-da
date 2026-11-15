"""Build the state-vector layout and the observation gather indices."""
from __future__ import annotations

import numpy as np

from .domain import StateLayout


def validate_observation_coords(brefnum, breflev, bnlev) -> None:
    """Reject observations whose (box, level) is not an existing physical coordinate.

    Checked against the domain (bnlev) BEFORE the layout is built, so a non-existent
    coordinate can never be materialized and later mistaken for a valid one.
    """
    brefnum = np.asarray(brefnum).ravel()
    breflev = np.asarray(breflev).ravel()
    Nb = len(bnlev) - 1
    for i in range(brefnum.size):
        bf, lf = float(brefnum[i]), float(breflev[i])
        if bf != int(bf) or lf != int(lf):
            raise ValueError(f'observation {i}: non-integer box/level ({bf}, {lf})')
        b, lev = int(bf), int(lf)
        if not (1 <= b <= Nb):
            raise ValueError(f'observation {i}: box {b} out of range 1..{Nb}')
        if not (1 <= lev <= int(bnlev[b])):
            raise ValueError(f'observation {i}: level {lev} does not exist in box {b} (has {int(bnlev[b])})')


def build_layout(bnlev, brefnum=None, breflev=None, nobs_per_date=None) -> StateLayout:
    """State-vector layout for an analysis window (a single date is a one-date window).

    The state holds the observed boxes on the earlier dates, then the full box grid on
    the analysis (last) date. With no observation arguments it is a single-date layout
    (the full grid only). lind_extract maps the analysis-date grid to a 1-based index
    for writing a restart.
    """
    brefnum = np.asarray(brefnum).ravel() if brefnum is not None else np.empty(0, int)
    breflev = np.asarray(breflev).ravel() if breflev is not None else np.empty(0, int)
    nobs_per_date = list(nobs_per_date) if nobs_per_date else [0]
    Nb, MaxLev, Ndates = len(bnlev) - 1, int(np.max(bnlev)), len(nobs_per_date)

    Lind = np.zeros((Nb, MaxLev, Ndates), dtype=int)
    Iind, Jind, Tind = [], [], []
    LL = offset = 0
    for k in range(Ndates - 1):                         # earlier-date observed boxes
        for j in range(int(nobs_per_date[k])):
            LL += 1
            box, lev = int(brefnum[offset + j]) - 1, int(breflev[offset + j]) - 1
            Lind[box, lev, k] = LL
            Iind.append(box); Jind.append(lev); Tind.append(k)
        offset += int(nobs_per_date[k])

    anl_ind_start = LL
    lind_extract = np.zeros((Nb, MaxLev), dtype=int)
    LLL = 0
    for i in range(1, Nb + 1):                           # full grid on the analysis date
        for j in range(1, int(bnlev[i]) + 1):
            LL += 1; LLL += 1
            Lind[i - 1, j - 1, Ndates - 1] = LL
            Iind.append(i - 1); Jind.append(j - 1); Tind.append(Ndates - 1)
            lind_extract[i - 1, j - 1] = LLL

    return StateLayout(
        Lind=Lind, Iind=np.array(Iind, int), Jind=np.array(Jind, int),
        Tind=np.array(Tind, int), lind_extract=lind_extract, n_state=LL,
        anl_ind_start=anl_ind_start, nobs_per_date=tuple(int(n) for n in nobs_per_date),
    )


def observation_indices(brefnum, breflev, layout: StateLayout) -> np.ndarray:
    """0-based state rows observed by H (a pure gather over the validated layout).

    Returns the index of each observation's state component; raises if an observation
    maps to a non-existent component (Lind == 0), so a bad input cannot silently wrap.
    """
    brefnum = np.asarray(brefnum).ravel()
    breflev = np.asarray(breflev).ravel()
    Lind, Tind = layout.Lind, layout.Tind
    idx = np.empty(brefnum.size, dtype=int)
    for i in range(brefnum.size):
        val = int(Lind[int(brefnum[i]) - 1, int(breflev[i]) - 1, int(Tind[i])])
        if val < 1:
            raise ValueError(f'observation {i}: no state component for box {int(brefnum[i])}, '
                             f'level {int(breflev[i])}, date index {int(Tind[i])}')
        idx[i] = val - 1
    return idx
