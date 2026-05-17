from __future__ import annotations

from pathlib import Path

from main_LN import main_LN
from main_obsproconly import main_obsproconly
from matlab_compat import copy_file
from paths import TEMPLATES_DIR, date_dir, output_xlsx_path
from poseidon_runtime import run_poseidon_all


def _copy_initconc_only(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for src in sorted(TEMPLATES_DIR.glob('*hydrol*dep*')):
        if not src.is_dir():
            continue

        init_src = src / 'Initconc.dat'
        if not init_src.exists():
            continue

        dst_dir = destination / src.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        copy_file(init_src, dst_dir / 'Initconc.dat')


def runall_LN():
    dates = ['1986.75', '1987.5', '1988.25', '1988.5', '1988.625', '1988.75', '1988.85', '1990.75', '1991.75', '2007.5']
    Nd = len(dates)

    for i in range(Nd - 1):
        curdate = dates[i]
        nextdate = dates[i + 1]

        main_LN(curdate)

        cur_anl_dir = date_dir(curdate) / 'ANL'
        _copy_initconc_only(cur_anl_dir)

        out_cur = output_xlsx_path(curdate)
        if out_cur.exists():
            copy_file(out_cur, cur_anl_dir / f'out_{curdate}.xlsx')

        next_dir = date_dir(nextdate)
        _copy_initconc_only(next_dir)

        run_poseidon_all(next_dir)

        if i < Nd - 2:
            main_obsproconly(nextdate)
            out_next = output_xlsx_path(nextdate)
            if out_next.exists():
                copy_file(out_next, next_dir / f'out_{nextdate}.xlsx')


if __name__ == '__main__':
    runall_LN()
