"""Spatial geolocation and box-aggregation of raw measurements (pure, reads .dat/.xlsx)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .poseidon_io import load_numeric_excel

SURFACE_DEPTH_M = 10.0


def locate_point3D(slon, slat, sdepth, boxnum, boxlons, boxlats, boxnlev, bdepths):
    """Box(es) and vertical level a point (lon, lat, depth) falls into.

    Returns bind (0-based box indices), bn (box numbers) and blev (level, 1-based).
    Rectangle bounds are inclusive, so a point on a shared edge lands in each
    adjacent box.
    """
    bind, bn = [], []
    for i in range(len(boxnum)):
        if (boxlons[i, 0] <= slon <= boxlons[i, 1]) and (boxlats[i, 0] <= slat <= boxlats[i, 1]):
            bind.append(i)
            bn.append(int(boxnum[i]))
    if not bind:
        raise ValueError(f'point (lon={slon}, lat={slat}) is outside the box system')

    blev = 0
    for j in range(int(boxnlev[bind[0]])):
        if sdepth < bdepths[j]:
            blev = j + 1
            break
    if blev == 0:
        blev = int(boxnlev[bind[0]])
    return bind, bn, blev


def aggregate_observations(dat_path, geometry_path, bdepths, expected_time=None, time_tol=1e-6):
    """Read one <date>.dat, map each measurement to box/level and average within a box.

    Column 0 is the physical time (year); when expected_time is given every row must
    match it, so a mislabelled file is rejected rather than silently relabelled. Raw
    measured concentrations must be finite and positive before box-averaging. Returns
    a dict with values/brefnum/breflev (p,1 arrays), sigc, and box_vs_allobs.
    """
    BBB = load_numeric_excel(geometry_path, sheet_name='Sheet1')
    boxnum = BBB[:, 0]
    boxlats = np.column_stack((BBB[:, 4], BBB[:, 5]))
    boxlons = np.column_stack((BBB[:, 6], BBB[:, 7]))
    boxnlev = BBB[:, 8].astype(int)
    Nbox = boxnum.shape[0]
    Maxlev = int(np.max(boxnlev))

    data = np.atleast_2d(np.loadtxt(str(dat_path), skiprows=1))
    stime, slon, slat, sval = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    Ns = slon.shape[0]

    if not np.all(np.isfinite(stime)):
        bad = int(np.where(~np.isfinite(stime))[0][0])
        raise ValueError(f'{dat_path}: non-finite time at row {bad + 1}')
    if expected_time is not None:
        bad = np.where(np.abs(stime - float(expected_time)) > time_tol)[0]
        if bad.size:
            raise ValueError(
                f'{dat_path}: time column does not match requested time {expected_time} '
                f'(e.g. row {int(bad[0]) + 1} has {stime[bad[0]]})')
    if not np.all(np.isfinite(sval) & (sval > 0.0)):
        bad = int(np.where(~(np.isfinite(sval) & (sval > 0.0)))[0][0])
        raise ValueError(f'{dat_path}: non-positive/non-finite measurement at row {bad + 1}: {sval[bad]}')

    nobs_box = np.zeros((Nbox, Maxlev), dtype=int)
    box_st = {(j, k): [] for j in range(Nbox) for k in range(Maxlev)}
    box_vs_allobs = np.zeros((Ns, 4), dtype=float)

    for i in range(Ns):
        bind, bn, blev = locate_point3D(slon[i], slat[i], SURFACE_DEPTH_M, boxnum, boxlons, boxlats, boxnlev, bdepths)
        if len(bind) > 4:
            raise ValueError(f'measurement {i + 1} belongs to more than 4 boxes')
        for j, bj in enumerate(bind):
            box_vs_allobs[i, j] = bn[j]
            nobs_box[bj, blev - 1] += 1
            box_st[(bj, blev - 1)].append(sval[i])

    box_vs_allobs = np.concatenate(
        (slon.reshape(-1, 1), slat.reshape(-1, 1), sval.reshape(-1, 1), box_vs_allobs), axis=1)

    values, brefnum, breflev, sigc = [], [], [], []
    for j in range(Nbox):
        for k in range(Maxlev):
            vals = box_st[(j, k)]
            if not vals:
                continue
            nn = len(vals)
            brefnum.append(boxnum[j])
            breflev.append(k + 1)
            values.append(sum(vals) / np.single(nn))
            sigc.append(0.0 if nn == 1 else 0.5 * (max(vals) - min(vals)))

    return {
        'values': np.asarray(values, dtype=float).reshape(-1, 1),
        'brefnum': np.asarray(brefnum).reshape(-1, 1),
        'breflev': np.asarray(breflev).reshape(-1, 1),
        'sigc': np.asarray(sigc, dtype=float).reshape(-1, 1),
        'box_vs_allobs': box_vs_allobs,
    }
