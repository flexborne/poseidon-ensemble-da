"""Spatial geolocation and box-aggregation of raw measurements (pure, reads .dat)."""
from __future__ import annotations

import numpy as np

from .geometry import box_levels

SURFACE_DEPTH_M = 10.0


def _vertical_level(nlev, bdepths) -> int:
    """1-based depth bin for a surface measurement, clamped to this box's own level count."""
    for j in range(int(nlev)):
        if SURFACE_DEPTH_M < bdepths[j]:
            return j + 1
    return int(nlev)


def locate_point3D(slon, slat, boxlons, boxlats):
    """0-based indices of every box a point (lon, lat) falls into (inclusive rectangle bounds)."""
    bind = [i for i in range(boxlons.shape[0])
            if (boxlons[i, 0] <= slon <= boxlons[i, 1]) and (boxlats[i, 0] <= slat <= boxlats[i, 1])]
    if not bind:
        raise ValueError(f'point (lon={slon}, lat={slat}) is outside the box system')
    return bind


def aggregate_observations(dat_path, geometry, bdepths, expected_time=None, time_tol=1e-6):
    """Read one <date>.dat, map each measurement to box/level and average within a box.

    `geometry` is a loaded Geometry (read once, reused across dates). Column 0 of the file
    is the physical time (year); when expected_time is given every row must match it, so a
    mislabelled file is rejected rather than silently relabelled. Raw measured concentrations
    must be finite and positive before box-averaging. Returns a dict with values/brefnum/
    breflev as 1-D (p,) arrays.
    """
    bnlev = box_levels()                                # single source of truth for per-box level counts
    boxnum, boxlats, boxlons = geometry.boxnum, geometry.boxlats, geometry.boxlons
    Nbox, Maxlev = geometry.nbox, int(bnlev.max())

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
        bind = locate_point3D(slon[i], slat[i], boxlons, boxlats)
        if len(bind) > 4:
            raise ValueError(f'measurement {i + 1} belongs to more than 4 boxes')
        for bj in bind:
            blev = _vertical_level(bnlev[int(boxnum[bj])], bdepths)   # per-box level count
            box_st[(bj, blev - 1)].append(sval[i])

    values, brefnum, breflev = [], [], []
    for j in range(Nbox):
        for k in range(Maxlev):
            vals = box_st[(j, k)]
            if not vals:
                continue
            brefnum.append(boxnum[j])
            breflev.append(k + 1)
            values.append(sum(vals) / len(vals))

    return {
        'values': np.asarray(values, dtype=float),
        'brefnum': np.asarray(brefnum),
        'breflev': np.asarray(breflev),
    }
