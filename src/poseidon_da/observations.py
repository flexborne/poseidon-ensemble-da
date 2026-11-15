"""Spatial geolocation and box-aggregation of raw measurements (pure, reads .dat)."""
from __future__ import annotations

import numpy as np

SURFACE_DEPTH_M = 10.0


def locate_point3D(slon, slat, sdepth, boxnum, boxlons, boxlats, boxnlev, bdepths):
    """0-based box indices and the 1-based vertical level a point (lon, lat, depth) falls into.

    Rectangle bounds are inclusive, so a point on a shared edge lands in each adjacent box.
    """
    bind = [i for i in range(len(boxnum))
            if (boxlons[i, 0] <= slon <= boxlons[i, 1]) and (boxlats[i, 0] <= slat <= boxlats[i, 1])]
    if not bind:
        raise ValueError(f'point (lon={slon}, lat={slat}) is outside the box system')

    blev = 0
    for j in range(int(boxnlev[bind[0]])):
        if sdepth < bdepths[j]:
            blev = j + 1
            break
    if blev == 0:
        blev = int(boxnlev[bind[0]])
    return bind, blev


def aggregate_observations(dat_path, geometry, bdepths, expected_time=None, time_tol=1e-6):
    """Read one <date>.dat, map each measurement to box/level and average within a box.

    `geometry` is a loaded Geometry (read once, reused across dates). Column 0 of the file
    is the physical time (year); when expected_time is given every row must match it, so a
    mislabelled file is rejected rather than silently relabelled. Raw measured concentrations
    must be finite and positive before box-averaging. Returns a dict with values/brefnum/
    breflev as 1-D (p,) arrays.
    """
    boxnum, boxlats, boxlons = geometry.boxnum, geometry.boxlats, geometry.boxlons
    boxnlev = geometry.boxnlev
    Nbox, Maxlev = geometry.nbox, geometry.maxlev

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

    box_st = {(j, k): [] for j in range(Nbox) for k in range(Maxlev)}
    for i in range(Ns):
        bind, blev = locate_point3D(slon[i], slat[i], SURFACE_DEPTH_M, boxnum, boxlons, boxlats, boxnlev, bdepths)
        if len(bind) > 4:
            raise ValueError(f'measurement {i + 1} belongs to more than 4 boxes')
        for bj in bind:
            box_st[(bj, blev - 1)].append(sval[i])

    values, brefnum, breflev = [], [], []
    for j in range(Nbox):
        for k in range(Maxlev):
            vals = box_st[(j, k)]
            if not vals:
                continue
            brefnum.append(boxnum[j])
            breflev.append(k + 1)
            values.append(sum(vals) / np.single(len(vals)))

    return {
        'values': np.asarray(values, dtype=float),
        'brefnum': np.asarray(brefnum),
        'breflev': np.asarray(breflev),
    }
