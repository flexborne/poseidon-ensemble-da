"""End-to-end smoke test for the real POSEIDON model via the package runner.

Runs Poseidon4.exe in a copy of a COMPLETE project directory (one that contains proj.dat,
modeles.dat, the forcing files, Initconc.dat and an archived lastconc.xls) through exactly the
code path a run/cycle uses: runner.run_executable_in_dir with use_pty on POSIX. The archived
lastconc.xls is removed first, so only a fresh model run can recreate it; the result is then
compared byte-for-byte to the archived output.

The executable and a full project are environment-specific and live OUTSIDE the repo (the repo
keeps only the DA-relevant Initconc.dat/lastconc.xls per member), so pass their paths in. On
macOS/Linux also export WINEPREFIX for a prepared wine prefix, e.g.:

    export WINEPREFIX=~/.wine_poseidon
    python tools/smoke_model.py \
        --project "/path/to/DAruns.Sasha/1986.75.noda/01.hydrol1dep1" \
        --exe bin/Poseidon4.exe
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
import tempfile
from pathlib import Path

from poseidon_da.runner import run_executable_in_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Run the real POSEIDON model and reproduce an archived output.')
    p.add_argument('--project', required=True, help='complete project dir (proj.dat + archived lastconc.xls)')
    p.add_argument('--exe', required=True, help='path to Poseidon4.exe')
    p.add_argument('--launcher', nargs='*', default=['wine'],
                   help='command prefix to run the exe (default: wine; use "" on Windows)')
    p.add_argument('--no-pty', action='store_true', help='run without a pseudo-terminal')
    args = p.parse_args(argv)

    project, exe = Path(args.project), Path(args.exe).resolve()   # absolute: the runner cd's into the work dir
    archived = project / 'lastconc.xls'
    if not (project / 'proj.dat').exists():
        print(f'ERROR: {project} is not a complete project (no proj.dat)', file=sys.stderr)
        return 2
    if not archived.exists():
        print(f'ERROR: no archived lastconc.xls in {project} to compare against', file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / 'member'
        shutil.copytree(project, work)
        (work / 'lastconc.xls').unlink()                 # only a FRESH run may recreate it
        print(f'running {exe.name} via the package runner (launcher={args.launcher or "[]"}, '
              f'pty={not args.no_pty})...')
        run_executable_in_dir(work, exe_path=exe, launcher=list(args.launcher), use_pty=not args.no_pty)
        out = work / 'lastconc.xls'
        if not out.exists():
            print('FAIL: the model produced no lastconc.xls', file=sys.stderr)
            return 1
        identical = filecmp.cmp(out, archived, shallow=False)
        print(f'fresh lastconc.xls: {out.stat().st_size} bytes; byte-identical to archive: {identical}')
        return 0 if identical else 1


if __name__ == '__main__':
    sys.exit(main())
