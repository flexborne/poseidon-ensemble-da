"""POSEIDON text format and Excel I/O (the only openpyxl-aware module)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook


def read_poseidon(pfile, bnlev):
    """Read a POSEIDON state file -> (arr[box,level], header, footer, sed[box,2]).

    A missing/duplicated box row or a non-positive/non-finite active concentration raises
    ValueError rather than being silently accepted.
    """
    Nbox = len(bnlev) - 1
    MaxLev = int(np.max(bnlev))
    arr = np.zeros((Nbox, MaxLev), dtype=float)
    sed = np.zeros((Nbox, 2), dtype=float)
    header, footer = [], []
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
            tok = float(C[0])
            if tok > Nbox:                      # footer rows (extra boxes, trailing physical-time line)
                footer.append({'tline': tline})
                continue
            if tok != int(tok) or int(tok) < 1:  # a body row must be a positive integer box id
                raise ValueError(f'read_poseidon: invalid box id {C[0]!r} in {pfile}')
            nn = int(tok)
            if nn in seen:
                raise ValueError(f'read_poseidon: duplicate row for box {nn} in {pfile}')
            seen.add(nn)
            nlev = int(bnlev[nn])
            if len(C) != nlev + 3:
                raise ValueError('read_poseidon: number of levels in bnlev is inconsistent with the data file')
            for j in range(1, nlev + 1):
                value = float(C[j])
                if not (np.isfinite(value) and value > 0.0):
                    raise ValueError(f'read_poseidon: non-positive/non-finite concentration for box {nn}, level {j} in {pfile}: {value}')
                arr[nn - 1, j - 1] = value
            s0, s1 = float(C[nlev + 1]), float(C[nlev + 2])
            if not (np.isfinite(s0) and np.isfinite(s1)):
                raise ValueError(f'read_poseidon: non-finite sediment value for box {nn} in {pfile}: {s0}, {s1}')
            sed[nn - 1, 0] = s0
            sed[nn - 1, 1] = s1

    missing = [b for b in range(1, Nbox + 1) if b not in seen]
    if missing:
        raise ValueError(f'read_poseidon: missing box rows {missing} in {pfile}')
    return arr, header, footer, sed


def extract_physical_time(footer):
    """Physical time stamped in a POSEIDON footer (the trailing single-token line), or None."""
    for item in reversed(footer):
        toks = item['tline'].split()
        if len(toks) == 1:
            try:
                return float(toks[0])
            except ValueError:
                continue
    return None


def read_state(pfile, bnlev, expected_time, time_tol: float = 1e-6):
    """read_poseidon plus the physical-time contract: the stamped time must be finite and
    match expected_time, so a mislabelled file can never pass as data for another date."""
    arr, header, footer, sed = read_poseidon(pfile, bnlev)
    t = extract_physical_time(footer)
    if t is None or not np.isfinite(t):
        raise ValueError(f'{pfile}: missing or non-finite physical time (expected {expected_time})')
    if abs(t - float(expected_time)) > time_tol:
        raise ValueError(f'{pfile}: physical time {t} does not match expected {expected_time}')
    return arr, header, footer, sed


def write_poseidon(fname, vect, Lind, bnlev, header, footer, sed):
    """Write a state vector as POSEIDON Initconc.dat; header/footer/sediments are carried over
    unchanged, values come from vect via the 1-based indices in Lind."""
    path = Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fid:
        for item in header:
            fid.write(f"{item['tline']}\n")
        Nbox = len(bnlev) - 1
        for i in range(1, Nbox + 1):
            fid.write(f"   {i:2d}")
            for j in range(1, int(bnlev[i]) + 1):
                fid.write(f"   {vect[Lind[i - 1, j - 1] - 1]:9.3E}")
            fid.write(' ' * ((4 - int(bnlev[i])) * 12))
            fid.write(f"   {sed[i - 1, 0]:9.3E}   {sed[i - 1, 1]:9.3E} \n")
        for item in footer:
            fid.write(f"{item['tline']}\n")


def load_numeric_excel(path_like, sheet_name=None) -> np.ndarray:
    """Read a named Excel sheet (or the active one) as a float array; empty cells -> NaN,
    a missing named sheet raises KeyError."""
    path = Path(path_like)
    wb = load_workbook(path, data_only=True, read_only=True)
    if sheet_name is not None:
        names = wb.sheetnames
        if sheet_name not in names:
            wb.close()
            raise KeyError(f"sheet '{sheet_name}' not found in {path}; available: {names}")
        ws = wb[sheet_name]
    else:
        ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    nrows = len(rows)
    ncols = max((len(r) for r in rows), default=0)
    arr = np.full((nrows, ncols), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            if value is None:
                continue
            try:
                arr[i, j] = float(value)
            except (TypeError, ValueError):
                arr[i, j] = np.nan
    wb.close()
    return arr


def _normalize_2d(data) -> np.ndarray:
    if np.isscalar(data):
        return np.array([[data]])
    arr = np.asarray(data)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def write_sheets(path_like, sheets) -> None:
    """Write {sheet_name: 2D-array-like} to one workbook in a single save (sheet order preserved)."""
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    for name, data in sheets.items():
        ws = wb.create_sheet(title=str(name).strip())
        arr = _normalize_2d(data)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                value = arr[i, j]
                if isinstance(value, np.generic):
                    value = value.item()
                ws.cell(row=i + 1, column=j + 1, value=value)
    wb.save(path)
    wb.close()
