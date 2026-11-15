"""Run context (directories) and the external POSEIDON executable + file plumbing."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class RunContext:
    """The single set of input/output directories every step uses."""

    runs_root: Path       # contains the per-date working directories
    templates_dir: Path   # where updated Initconc.dat files are written
    outputs_dir: Path     # where out_*.xlsx / obs_pos.xlsx / obs2pos.mat go

    @classmethod
    def default(cls) -> "RunContext":
        return cls(runs_root=paths.RUNS_ROOT, templates_dir=paths.TEMPLATES_DIR, outputs_dir=paths.OUTPUTS_DIR)

    @classmethod
    def isolated(cls, workdir, runs_root=None) -> "RunContext":
        """A context whose writable outputs live under workdir (tests / parallel runs)."""
        workdir = Path(workdir)
        return cls(
            runs_root=Path(runs_root) if runs_root is not None else paths.RUNS_ROOT,
            templates_dir=workdir / 'templates',
            outputs_dir=workdir / 'outputs',
        )

    def date_dir(self, strdate: str) -> Path:
        return self.runs_root / strdate

    def template_dir(self, name: str) -> Path:
        return self.templates_dir / name

    def _outputs(self) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        return self.outputs_dir

    def output_xlsx(self, strdate: str) -> Path:
        return self._outputs() / f'out_{strdate}.xlsx'

    def obs_pos_xlsx(self) -> Path:
        return self._outputs() / 'obs_pos.xlsx'

    def obs2pos_mat(self) -> Path:
        return self._outputs() / 'obs2pos.mat'


# ---- file plumbing -------------------------------------------------------------

def copy_file(src, dst) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(src), dst)


def find_members(date_dir) -> list:
    """Sorted member (hydrol) folder names for a date directory."""
    date_dir = Path(date_dir)
    return sorted(p.name for p in date_dir.glob('*hydrol*dep*') if p.is_dir())


def copy_lastconc_to_initconc(src_date_dir, dst_date_dir, members=None) -> None:
    src_date_dir, dst_date_dir = Path(src_date_dir), Path(dst_date_dir)
    for name in (members or find_members(src_date_dir)):
        copy_file(src_date_dir / name / 'lastconc.xls', dst_date_dir / name / 'Initconc.dat')


def publish_initconc(templates_dir, dest_date_dir) -> None:
    """Copy each member's freshly analysed Initconc.dat from templates into a date dir."""
    templates_dir, dest_date_dir = Path(templates_dir), Path(dest_date_dir)
    for src in sorted(templates_dir.glob('*hydrol*dep*')):
        init = src / 'Initconc.dat'
        if src.is_dir() and init.exists():
            copy_file(init, dest_date_dir / src.name / 'Initconc.dat')


# ---- external executable -------------------------------------------------------

def _is_windows() -> bool:
    return os.name == 'nt'


def _executable_command(exe_path: Path, launcher=None) -> list:
    if launcher:
        return [*launcher, str(exe_path)]
    if _is_windows():
        return [str(exe_path)]
    wine = shutil.which('wine')
    if wine:
        return [wine, str(exe_path)]
    raise RuntimeError(f'Cannot run {exe_path}: not on Windows and `wine` not found')


def run_executable_in_dir(directory, exe_path=None, launcher=None) -> None:
    directory = Path(directory)
    exe_path = Path(exe_path) if exe_path else paths.poseidon_exe_path()
    if not exe_path.exists():
        raise FileNotFoundError(f'POSEIDON executable not found: {exe_path}')
    result = subprocess.run(_executable_command(exe_path, launcher), cwd=directory)
    if result.returncode != 0:
        raise RuntimeError(f'{exe_path.name} failed in {directory} with exit code {result.returncode}')


def run_poseidon_all(date_dir, members=None, expected_members=None, exe_path=None,
                     launcher=None, expected_time=None, bnlev=None) -> None:
    """Run POSEIDON in every member folder of a date, verifying a FRESH output.

    Any pre-existing lastconc.xls is removed before the run, so a stale file can never
    be mistaken for the new forecast. After the run the output must reappear; when
    bnlev is given it is parsed (structure check) and, when expected_time is given, its
    stamped physical time must match.
    """
    from .poseidon_io import extract_physical_time, read_poseidon

    date_dir = Path(date_dir)
    if not date_dir.is_dir():
        raise FileNotFoundError(f'run_poseidon_all: date directory does not exist: {date_dir}')
    members = members or find_members(date_dir)
    if not members:
        raise FileNotFoundError(f'run_poseidon_all: no member folders (*hydrol*dep*) in {date_dir}')
    if expected_members is not None:
        missing = [m for m in expected_members if m not in members]
        if missing:
            raise FileNotFoundError(f'run_poseidon_all: missing expected members in {date_dir}: {missing}')
    for name in members:
        folder = date_dir / name
        if not folder.is_dir():
            raise FileNotFoundError(f'run_poseidon_all: member folder missing: {folder}')
        out = folder / 'lastconc.xls'
        if out.exists():
            out.unlink()                       # never treat a previous output as this run's result
        run_executable_in_dir(folder, exe_path=exe_path, launcher=launcher)
        if not out.exists():
            raise RuntimeError(f'run_poseidon_all: {name} produced no lastconc.xls')
        if bnlev is not None:
            _, _, footer, _ = read_poseidon(str(out), bnlev)
            if expected_time is not None:
                ptime = extract_physical_time(footer)
                if ptime is not None and abs(ptime - float(expected_time)) > 1e-6:
                    raise RuntimeError(
                        f'run_poseidon_all: {name} output has physical time {ptime}, expected {expected_time}')
