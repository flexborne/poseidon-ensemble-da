from __future__ import annotations

from pathlib import Path
import shutil

from main_obsproconly import main_obsproconly
from poseidon_runtime import copy_lastconc_to_initconc, run_poseidon_all


def run_NODA():
    script_dir = Path(__file__).resolve().parent
    dates = ['1986.75', '1987.5', '1988.25', '1988.5', '1988.625', '1988.75', '1988.85', '1990.75', '1991.75', '2007.5']
    Nd = len(dates)
    foldlist = sorted([p.name for p in (script_dir.parent / dates[0]).glob('*hydrol*dep*') if p.is_dir()])

    i = 0
    while i < Nd - 1:
        i += 1
        nextdate = dates[i]
        for j in range(i, i + 2):
            run_poseidon_all(script_dir.parent / dates[j - 1])
            if j == i + 1:
                main_obsproconly(dates[j - 1])
                shutil.copy2(script_dir / f'out_{nextdate}.xlsx', script_dir.parent / dates[j - 1] / f'out_{nextdate}.xlsx')
                break
            copy_lastconc_to_initconc(script_dir.parent / dates[j - 1], script_dir.parent / dates[j], foldlist)


if __name__ == '__main__':
    run_NODA()
