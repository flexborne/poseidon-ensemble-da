"""Build the state-vector layout and the observation gather indices."""
from __future__ import annotations

import numpy as np

from .domain import StateLayout


def validate_observation_coords(brefnum, breflev, bnlev) -> None:
    """Reject observations whose (box, level) is not an existing physical coordinate.

    Checked against the domain (bnlev) BEFORE any layout is built, so a non-existent
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


def build_layout_single(bnlev) -> StateLayout:
    """Layout for a single date: one state component per (box, active level)."""
    Nb = len(bnlev) - 1
    MaxLev = int(np.max(bnlev))
    Lind = np.zeros((Nb, MaxLev), dtype=int)
    Iind, Jind = [], []
    LL = 0
    for i in range(1, Nb + 1):
        for j in range(1, int(bnlev[i]) + 1):
            LL += 1
            Lind[i - 1, j - 1] = LL
            Iind.append(i - 1)
            Jind.append(j - 1)
    return StateLayout(Lind=Lind, Iind=np.array(Iind, dtype=int),
                       Jind=np.array(Jind, dtype=int), n_state=LL)


def build_layout_window(bnlev, brefnum, breflev, nobs_per_date) -> StateLayout:
    """Layout for a window: observed boxes on earlier dates + full grid on the target date.

    The target-date block starts at anl_ind_start (0-based). lind_extract maps the
    target-date (box, level) grid to a 1-based index into that block for writing a restart.
    """
    brefnum = np.asarray(brefnum).ravel()
    breflev = np.asarray(breflev).ravel()
    Nb = len(bnlev) - 1
    MaxLev = int(np.max(bnlev))
    Ndates = len(nobs_per_date)

    Lind = np.zeros((Nb, MaxLev, Ndates), dtype=int)
    Iind, Jind, Tind = [], [], []
    LL = 0
    offset = 0
    for k in range(Ndates - 1):
        for j in range(int(nobs_per_date[k])):
            LL += 1
            idx = offset + j
            box = int(brefnum[idx]) - 1
            lev = int(breflev[idx]) - 1
            Lind[box, lev, k] = LL
            Iind.append(box)
            Jind.append(lev)
            Tind.append(k)
        offset += int(nobs_per_date[k])

    anl_ind_start = LL
    lind_extract = np.zeros((Nb, MaxLev), dtype=int)
    LLL = 0
    for i in range(1, Nb + 1):
        for j in range(1, int(bnlev[i]) + 1):
            LL += 1
            LLL += 1
            Lind[i - 1, j - 1, Ndates - 1] = LL
            Iind.append(i - 1)
            Jind.append(j - 1)
            Tind.append(Ndates - 1)
            lind_extract[i - 1, j - 1] = LLL

    return StateLayout(
        Lind=Lind, Iind=np.array(Iind, dtype=int), Jind=np.array(Jind, dtype=int),
        n_state=LL, Tind=np.array(Tind, dtype=int), lind_extract=lind_extract,
        anl_ind_start=anl_ind_start,
    )


def observation_indices(brefnum, breflev, layout: StateLayout) -> np.ndarray:
    """0-based state rows observed by H; validates box/level (and date) existence.

    Raises ValueError on any observation that points at a non-existent state
    component, instead of silently wrapping to another valid index.
    """
    brefnum = np.asarray(brefnum).ravel()
    breflev = np.asarray(breflev).ravel()
    Lind = layout.Lind
    Tind = layout.Tind
    nb, ml = Lind.shape[0], Lind.shape[1]
    idx = np.empty(brefnum.size, dtype=int)
    for i in range(brefnum.size):
        b, lev = int(brefnum[i]), int(breflev[i])
        if not (1 <= b <= nb):
            raise ValueError(f'observation {i}: box {b} out of range 1..{nb}')
        if not (1 <= lev <= ml):
            raise ValueError(f'observation {i}: level {lev} out of range 1..{ml}')
        if Tind is None:
            val = int(Lind[b - 1, lev - 1])
            where = f'box {b}, level {lev}'
        else:
            t = int(Tind[i])
            val = int(Lind[b - 1, lev - 1, t])
            where = f'box {b}, level {lev}, date index {t}'
        if val < 1:
            raise ValueError(f'observation {i}: no state component for {where}')
        idx[i] = val - 1
    return idx
