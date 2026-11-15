from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook
from scipy.io import loadmat, savemat

from paths import SRC_DIR, poseidon_exe_path

np.set_printoptions(linewidth=160, threshold=10000)
MODULE_DIR = SRC_DIR
PROJECT_ROOT = MODULE_DIR.parent


def resolve_path(path_like: str | os.PathLike) -> Path:
    """Resolve a file by its path or relative to the module directory."""
    path = Path(path_like)
    if path.exists():
        return path
    module_relative = MODULE_DIR / path
    if module_relative.exists():
        return module_relative
    return path


def load_numeric_excel(path_like: str | os.PathLike, sheet_name=None) -> np.ndarray:
    """Read numeric data from a named Excel sheet (or the active one) into an array.

    Empty/non-numeric cells become NaN. Passing sheet_name avoids depending on
    which sheet happens to be active when the workbook was saved; a missing named
    sheet raises KeyError instead of silently reading the wrong one.
    """
    path = resolve_path(path_like)
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
            except Exception:
                arr[i, j] = np.nan
    wb.close()
    return arr


def _normalize_2d(data) -> np.ndarray:
    """Coerce scalar/1D/2D data to a 2D array for writing to Excel."""
    if np.isscalar(data):
        return np.array([[data]])
    arr = np.asarray(data)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def xlswrite_compat(path_like: str | os.PathLike, data, sheet_name: str = 'Sheet1') -> None:
    """Write an array to a dedicated sheet of an Excel workbook."""
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            wb = load_workbook(path)
        except Exception as exc:
            raise IOError(f'xlswrite_compat: cannot open existing workbook {path}: {exc}') from exc
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
    if 'Sheet' in wb.sheetnames and len(wb.sheetnames) > 1 and wb['Sheet'].max_row == 1 and wb['Sheet'].max_column == 1 and wb['Sheet']['A1'].value is None:
        wb.remove(wb['Sheet'])
    wb.save(path)
    wb.close()


def save_mat_compat(path_like: str | os.PathLike, **kwargs) -> None:
    """Save variables to a .mat file."""
    savemat(path_like, kwargs)


def load_mat_compat(path_like: str | os.PathLike):
    """Load variables from a .mat file."""
    return loadmat(path_like, squeeze_me=True, struct_as_record=False)


def model_print(name: str, value) -> None:
    """Print a variable in a labelled block format (for inspecting intermediate results)."""
    print(f"\n{name} =\n")
    print(value)
    print()


def find_poseidon_folders(date_path: str | os.PathLike) -> list[str]:
    """Return the sorted list of hydrol folders (ensemble members) for a date."""
    date_dir = Path(date_path)
    return sorted([p.name for p in date_dir.glob('*hydrol*dep*') if p.is_dir()])


def mkdir_warning(path_like: str | os.PathLike) -> None:
    """Create a directory; if it already exists, print a warning."""
    path = Path(path_like)
    if path.exists():
        print('Warning: Directory already exists.')
    else:
        path.mkdir(parents=True, exist_ok=True)


def copy_matching(pattern: str, destination: str | os.PathLike) -> None:
    """Copy files/folders matching a glob pattern to the destination."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    for match in sorted(glob.glob(pattern)):
        src = Path(match)
        target = dest / src.name
        if src.is_dir():
            shutil.copytree(src, target, dirs_exist_ok=True)
        else:
            shutil.copy2(src, target)


def copy_file(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    """Copy a single file, creating the destination directory if needed."""
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_p, dst_p)


def is_windows() -> bool:
    """Whether the current OS is Windows."""
    return os.name == 'nt'


def _poseidon_exe_path(exe_name: str = 'Poseidon4.exe') -> Path:
    """Path to the POSEIDON executable."""
    return poseidon_exe_path()


def resolve_executable_command(exe_name: str = 'Poseidon4.exe') -> list[str]:
    """Build the command to run the .exe: directly (Windows) or via wine (Linux/macOS)."""
    exe_path = _poseidon_exe_path(exe_name)
    if is_windows():
        return [str(exe_path)]
    wine = shutil.which('wine')
    if wine:
        return [wine, str(exe_path)]
    raise RuntimeError(f'Cannot run {exe_path}')


def run_executable_in_dir(directory: str | os.PathLike, exe_name: str = 'Poseidon4.exe') -> None:
    """Run Poseidon4.exe in the given directory (working dir = that directory)."""
    folder_dir = Path(directory)
    exe_path = _poseidon_exe_path(exe_name)
    if not exe_path.exists():
        raise FileNotFoundError(f'Poseidon executable not found: {exe_path}')

    print()
    print(f"{folder_dir}>{exe_name}")
    result = subprocess.run(resolve_executable_command(exe_name), cwd=folder_dir)
    if result.returncode != 0:
        raise RuntimeError(f'{exe_name} failed in {folder_dir} with exit code {result.returncode}')
    print()
