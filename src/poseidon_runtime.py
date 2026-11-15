from __future__ import annotations

from pathlib import Path

from model_compat import copy_file, find_poseidon_folders, run_executable_in_dir


def run_poseidon_all(date_dir, folder_names=None, expected_members=None):
    """Run Poseidon4.exe in every hydrol folder of the given date.

    Validates that the date directory exists and actually contains member
    folders, and that each run produced a lastconc.xls - so a missing directory
    or an empty member set raises instead of silently succeeding.
    """
    date_dir = Path(date_dir)
    if not date_dir.is_dir():
        raise FileNotFoundError(f'run_poseidon_all: date directory does not exist: {date_dir}')
    if folder_names is None:
        folder_names = find_poseidon_folders(date_dir)
    if not folder_names:
        raise FileNotFoundError(f'run_poseidon_all: no member folders (*hydrol*dep*) in {date_dir}')
    if expected_members is not None:
        missing = [m for m in expected_members if m not in folder_names]
        if missing:
            raise FileNotFoundError(f'run_poseidon_all: missing expected member folders in {date_dir}: {missing}')
    for folder_name in folder_names:
        folder = date_dir / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f'run_poseidon_all: member folder missing: {folder}')
        run_executable_in_dir(folder)
        if not (folder / 'lastconc.xls').exists():
            raise RuntimeError(f'run_poseidon_all: {folder_name} produced no lastconc.xls')


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


def publish_initconc(templates_dir, dest_date_dir):
    """Copy each member's freshly analysed Initconc.dat from templates into a date dir."""
    templates_dir = Path(templates_dir)
    dest_date_dir = Path(dest_date_dir)
    for src in sorted(templates_dir.glob('*hydrol*dep*')):
        if not src.is_dir():
            continue
        init = src / 'Initconc.dat'
        if not init.exists():
            continue
        copy_file(init, dest_date_dir / src.name / 'Initconc.dat')