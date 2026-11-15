"""Box grid: vertical level counts, localization depths, and box centres of mass."""
from __future__ import annotations

import numpy as np

from .poseidon_io import load_numeric_excel

def layer_depths_from_bdepths(bdepths):
    """Representative mid-depth of each layer from the depth-bin upper bounds.

    One source of truth for the vertical coordinate: the same bdepths that bin the
    measurements also define the depths used for vertical localization.
    """
    edges = [0.0] + [float(b) for b in bdepths]
    return [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(bdepths))]


def box_levels() -> np.ndarray:
    """Number of vertical levels per box. Index 0 is an unused slot; boxes are 1..50."""
    bnlev = np.zeros(51, dtype=int)
    bnlev[1:51] = 4
    bnlev[1:3] = 1
    bnlev[34] = 1
    bnlev[44:46] = 1
    bnlev[48:51] = 1
    bnlev[33] = 2
    bnlev[35:37] = 2
    bnlev[46:48] = 2
    bnlev[3] = 3
    bnlev[6] = 3
    bnlev[14] = 3
    bnlev[37] = 3
    bnlev[39] = 3
    return bnlev


# Columns the readers depend on: id(0), lat/lon bounds(4..7), level count(8), centre(9,10).
GEOMETRY_COLUMNS = 11


def validate_geometry(geometry_path) -> list:
    """Problems with the geometry workbook: schema (required columns), box ids, per-box values."""
    try:
        AAA = load_numeric_excel(geometry_path, sheet_name='Sheet1')
    except Exception as exc:
        return [f'cannot read geometry {geometry_path}: {exc}']
    problems = []
    if AAA.shape[1] < GEOMETRY_COLUMNS:
        return [f'geometry: needs at least {GEOMETRY_COLUMNS} columns (has {AAA.shape[1]})']

    box_rows = AAA[np.isfinite(AAA[:, 0])]
    ids = box_rows[:, 0]
    if np.any(ids != ids.astype(int)):
        problems.append('geometry: non-integer box id(s)')
    seen, dups = set(), set()
    for v in ids.astype(int):
        (dups.add(int(v)) if v in seen else seen.add(int(v)))
    if dups:
        problems.append(f'geometry: duplicate box id(s) {sorted(dups)}')
    if not np.all(np.isfinite(box_rows[:, [4, 5, 6, 7, 8, 9, 10]])):
        problems.append('geometry: missing lat/lon bounds, level count or centre for some box')
    else:
        levels = box_rows[:, 8]
        if np.any(levels != levels.astype(int)) or np.any(levels < 1):
            problems.append('geometry: level count (column 8) must be a positive integer for every box')
    return problems


def load_box_centers(geometry_path) -> dict:
    """Return {0-based box index -> (lat, lon)} from the geometry workbook (Sheet1)."""
    AAA = load_numeric_excel(geometry_path, sheet_name='Sheet1')
    boxn = AAA[:, 0]
    lat = AAA[:, 9]
    lon = AAA[:, 10]
    return {int(boxn[i]) - 1: (float(lat[i]), float(lon[i]))
            for i in range(len(boxn)) if not np.isnan(boxn[i])}
