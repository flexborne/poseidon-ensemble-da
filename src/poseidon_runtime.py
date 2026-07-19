from __future__ import annotations

from pathlib import Path

from model_compat import copy_file, find_poseidon_folders, run_executable_in_dir


def run_poseidon_all(date_dir, folder_names=None):
    """Run Poseidon4.exe in every hydrol folder of the given date."""
    date_dir = Path(date_dir)
    if folder_names is None:
        folder_names = find_poseidon_folders(date_dir)
    for folder_name in folder_names:
        run_executable_in_dir(date_dir / folder_name)


def copy_lastconc_to_initconc(src_date_dir, dst_date_dir, folder_names=None):
    """Copy lastconc.xls -> Initconc.dat (restart for the next date)."""
    src_date_dir = Path(src_date_dir)
    dst_date_dir = Path(dst_date_dir)
    if folder_names is None:
        folder_names = find_poseidon_folders(src_date_dir)
    for folder_name in folder_names:
        copy_file(
            src_date_dir / folder_name / 'lastconc.xls',
            dst_date_dir / folder_name / 'Initconc.dat'
        )