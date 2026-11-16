"""Box grid: vertical level counts, localization depths, and the box geometry object."""
from __future__ import annotations

from dataclasses import dataclass

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


# Columns the readers depend on: id(0), lat/lon bounds(4..7), centre(9,10). Per-box level
# counts come from box_levels() (the single source of truth), not from the workbook.
GEOMETRY_COLUMNS = 11


@dataclass(frozen=True)
class Geometry:
    """Box geometry read once from the workbook: ids, lat/lon bounds, centres."""

    boxnum: np.ndarray       # (Nbox,) 1-based box ids
    boxlats: np.ndarray      # (Nbox, 2) [south, north]
    boxlons: np.ndarray      # (Nbox, 2) [west, east]
    centers: dict            # {0-based box index -> (lat, lon)} centre of mass

    @property
    def nbox(self) -> int:
        return int(self.boxnum.shape[0])


def load_geometry(geometry_path) -> Geometry:
    """Read and validate the geometry workbook (Sheet1) in one call -> Geometry, or raise ValueError.

    Checks the required columns, integer and unique box ids, and finite per-box lat/lon bounds and
    centres — so a reader never meets an unvalidated workbook.
    """
    rows = load_numeric_excel(geometry_path, sheet_name='Sheet1')
    if rows.shape[1] < GEOMETRY_COLUMNS:
        raise ValueError(f'geometry: needs at least {GEOMETRY_COLUMNS} columns (has {rows.shape[1]})')
    rows = rows[np.isfinite(rows[:, 0])]
    ids = rows[:, 0]
    if np.any(ids != ids.astype(int)):
        raise ValueError('geometry: non-integer box id(s)')
    if len(set(ids.astype(int))) != len(ids):
        raise ValueError('geometry: duplicate box id(s)')
    if not np.all(np.isfinite(rows[:, [4, 5, 6, 7, 9, 10]])):
        raise ValueError('geometry: missing lat/lon bounds or centre for some box')

    boxnum = ids
    centers = {int(boxnum[i]) - 1: (float(rows[i, 9]), float(rows[i, 10])) for i in range(len(boxnum))}
    return Geometry(boxnum=boxnum, boxlats=rows[:, [4, 5]], boxlons=rows[:, [6, 7]], centers=centers)
