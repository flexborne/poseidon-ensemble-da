from __future__ import annotations

from pathlib import Path

from matlab_compat import copy_file, find_poseidon_folders, run_executable_in_dir


def run_poseidon_all(date_dir, folder_names=None):
    date_dir = Path(date_dir)
    if folder_names is None:
        folder_names = find_poseidon_folders(date_dir)
    for folder_name in folder_names:
        run_executable_in_dir(date_dir / folder_name)


def copy_lastconc_to_initconc(src_date_dir, dst_date_dir, folder_names=None):
    src_date_dir = Path(src_date_dir)
    dst_date_dir = Path(dst_date_dir)
    if folder_names is None:
        folder_names = find_poseidon_folders(src_date_dir)
    for folder_name in folder_names:
        copy_file(
            src_date_dir / folder_name / 'lastconc.xls',
            dst_date_dir / folder_name / 'Initconc.dat'
        )