"""POSEIDON text format and Excel I/O (the only openpyxl-aware module)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook


def read_poseidon(pfile, bnlev, validate=True):
    """Read and validate a POSEIDON output file (lastconc.xls).

    Returns arr (concentrations per box x level), header, footer (lines kept
    unchanged) and sed (sediments, 2 columns). With validate=True a missing or
    duplicated box row, or a non-positive/non-finite active concentration, raises
    ValueError instead of silently leaving a zero or overwriting a value.
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
            if validate and nn in seen:
                raise ValueError(f'read_poseidon: duplicate row for box {nn} in {pfile}')
            seen.add(nn)
            expected_cols = int(bnlev[nn]) + 3
            if len(C) != expected_cols:
                raise ValueError('read_poseidon: number of levels in bnlev is inconsistent with the data file')
            for j in range(1, int(bnlev[nn]) + 1):
                value = float(C[j])
                if validate and not (np.isfinite(value) and value > 0.0):
                    raise ValueError(f'read_poseidon: non-positive/non-finite concentration for box {nn}, level {j} in {pfile}: {value}')
                arr[nn - 1, j - 1] = value
            s0, s1 = float(C[j + 1]), float(C[j + 2])
            if validate and not (np.isfinite(s0) and np.isfinite(s1)):
                raise ValueError(f'read_poseidon: non-finite sediment value for box {nn} in {pfile}: {s0}, {s1}')
            sed[nn - 1, 0] = s0
            sed[nn - 1, 1] = s1

    if validate:
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
    """Read a POSEIDON state and enforce the physical-time contract (one path for all callers).

    Requires a finite stamped physical time that matches `expected_time`; a missing,
    non-finite or mismatched time is rejected, so a mislabelled directory/file can never
    pass as data for another date. Returns (arr, header, footer, sed).
    """
    arr, header, footer, sed = read_poseidon(pfile, bnlev)
    t = extract_physical_time(footer)
    if t is None or not np.isfinite(t):
        raise ValueError(f'{pfile}: missing or non-finite physical time (expected {expected_time})')
    if abs(t - float(expected_time)) > time_tol:
        raise ValueError(f'{pfile}: physical time {t} does not match expected {expected_time}')
    return arr, header, footer, sed


def write_poseidon(fname, vect, Lind, bnlev, header, footer, sed):
    """Write a state vector in POSEIDON format (Initconc.dat) for model restart.

    Header, footer and sediments are carried over unchanged; values come from
    vect via the 1-based indices in Lind (a single-date grid or a window
    analysis-date grid).
    """
    vect = list(vect)
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
    """Read numeric data from a named Excel sheet (or the active one); empty -> NaN.

    A missing named sheet raises KeyError instead of silently reading the wrong one.
    """
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


def xlswrite(path_like, data, sheet_name: str = 'Sheet1') -> None:
    """Write an array to a dedicated sheet of an Excel workbook (accumulating sheets)."""
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            wb = load_workbook(path)
        except Exception as exc:
            raise IOError(f'xlswrite: cannot open existing workbook {path}: {exc}') from exc
    else:
        wb = Workbook()
        wb.active.title = 'Sheet1'
    if sheet_name in wb.sheetnames:
        wb.remove(wb[sheet_name])
    ws = wb.create_sheet(title=str(sheet_name).strip())
    arr = _normalize_2d(data)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if isinstance(value, np.generic):
                value = value.item()
            ws.cell(row=i + 1, column=j + 1, value=value)
    wb.save(path)
    wb.close()
