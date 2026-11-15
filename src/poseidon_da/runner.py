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


def _pty_supported() -> bool:
    """PTYs exist only on POSIX; Windows console apps get a real console via subprocess."""
    return os.name == 'posix' and hasattr(os, 'fork')


def _run_in_pty(argv, cwd, log_path=None) -> int:
    """Run argv in cwd under a pseudo-terminal, returning the exit code.

    A console app writing to the terminal device (e.g. a Fortran app using CONOUT$) needs a real
    console, which headless `wine` lacks; pty.fork() gives the child its own tty. The master fd is
    always closed and the child reaped (finally), and drained output goes to log_path.
    """
    import pty
    pid, fd = pty.fork()
    if pid == 0:                                   # child
        try:
            os.chdir(cwd)
            os.execvp(argv[0], argv)
        except BaseException:
            os._exit(127)
    chunks = []
    try:                                           # parent: drain the pty until the child exits
        while True:
            try:
                data = os.read(fd, 4096)
            except OSError:
                break
            if not data:
                break
            chunks.append(data)
    finally:
        os.close(fd)
        _, status = os.waitpid(pid, 0)
    if log_path is not None and chunks:
        try:
            Path(log_path).write_bytes(b''.join(chunks))
        except OSError:
            pass
    return os.waitstatus_to_exitcode(status)


def run_executable_in_dir(directory, exe_path=None, launcher=None, use_pty=False) -> None:
    directory = Path(directory)
    exe_path = Path(exe_path) if exe_path else paths.poseidon_exe_path()
    if not exe_path.exists():
        raise FileNotFoundError(f'POSEIDON executable not found: {exe_path}')
    cmd = _executable_command(exe_path, launcher)
    if use_pty and _pty_supported():               # headless console apps (e.g. wine) need a tty
        rc = _run_in_pty(cmd, str(directory), log_path=directory / 'poseidon.log')
    else:
        rc = subprocess.run(cmd, cwd=directory).returncode
    if rc != 0:
        raise RuntimeError(f'{exe_path.name} failed in {directory} with exit code {rc}')


def run_poseidon_all(date_dir, members=None, expected_members=None, exe_path=None,
                     launcher=None, expected_time=None, bnlev=None, use_pty=False) -> None:
    """Run POSEIDON in every member folder of a date, verifying a FRESH output.

    Any pre-existing lastconc.xls is deleted first (a stale file can't pass as the forecast);
    the output must reappear and, when bnlev/expected_time are given, parse and carry that time.
    """
    from .poseidon_io import read_poseidon, read_state

    date_dir = Path(date_dir)
    if not date_dir.is_dir():
        raise FileNotFoundError(f'run_poseidon_all: date directory does not exist: {date_dir}')
    found = find_members(date_dir)
    if expected_members is not None:
        # exact set: reject BOTH missing and extra members, run in the declared order
        if set(found) != set(expected_members):
            missing = sorted(set(expected_members) - set(found))
            extra = sorted(set(found) - set(expected_members))
            raise FileNotFoundError(
                f'run_poseidon_all: member set mismatch in {date_dir}: missing={missing}, extra={extra}')
        members = list(expected_members)
    else:
        members = members or found
    if not members:
        raise FileNotFoundError(f'run_poseidon_all: no member folders (*hydrol*dep*) in {date_dir}')
    for name in members:
        folder = date_dir / name
        if not folder.is_dir():
            raise FileNotFoundError(f'run_poseidon_all: member folder missing: {folder}')
        out = folder / 'lastconc.xls'
        if out.exists():
            out.unlink()                       # never treat a previous output as this run's result
        run_executable_in_dir(folder, exe_path=exe_path, launcher=launcher, use_pty=use_pty)
        if not out.exists():
            raise RuntimeError(f'run_poseidon_all: {name} produced no lastconc.xls')
        if bnlev is not None and expected_time is not None:
            read_state(str(out), bnlev, expected_time=expected_time)   # fresh output must carry the right time
        elif bnlev is not None:
            read_poseidon(str(out), bnlev)                             # at least structurally valid
