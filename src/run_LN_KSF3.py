from __future__ import annotations

import shutil
from pathlib import Path

from main_KSF_LN import main_KSF_LN
from main_LN import main_LN
from main_obsproconly import main_obsproconly
from poseidon_runtime import copy_lastconc_to_initconc, run_poseidon_all


def _copy_local_hydrols(script_dir: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(script_dir.glob('*hydrol*dep*')):
        if src.is_dir():
            shutil.copytree(src, dest_dir / src.name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest_dir / src.name)


def run_LN_KSF3():
    script_dir = Path(__file__).resolve().parent
    dates = ['1986.75', '1987.5', '1988.25', '1988.5', '1988.625', '1988.75', '1988.85', '1990.75', '1991.75', '2007.5']
    Nd = len(dates)
    minind = [1, 2, 3, 3, 4, 4, 5, 8, 9, 10]
    maxind = [1, 2, 4, 6, 7, 7, 7, 8, 9, 10]
    foldlist = sorted([p.name for p in (script_dir.parent / dates[0]).glob('*hydrol*dep*') if p.is_dir()])

    i = 0
    while i < Nd - 1:
        i += 1
        curdate = dates[i - 1]
        nextdate = dates[i]

        if minind[i - 1] == i and maxind[i - 1] == i:
            main_LN(curdate)
        else:
            measdates = []
            for j in range(minind[i - 1], i):
                measdates.append(dates[j - 1])
            for j in range(i + 1, maxind[i - 1] + 1):
                measdates.append(dates[j - 1])
            measdates.append(dates[i - 1])
            main_KSF_LN(measdates)

        _copy_local_hydrols(script_dir, script_dir.parent / curdate / 'ANL')
        shutil.copy2(script_dir / f'out_{curdate}.xlsx', script_dir.parent / curdate / 'ANL' / f'out_{curdate}.xlsx')
        _copy_local_hydrols(script_dir, script_dir.parent / nextdate)

        for j in range(i + 1, maxind[i] + 1):
            run_poseidon_all(script_dir.parent / dates[j - 1])
            if j == i + 1:
                main_obsproconly(dates[j - 1])
                shutil.copy2(script_dir / f'out_{nextdate}.xlsx', script_dir.parent / dates[j - 1] / f'out_{nextdate}.xlsx')
            if j == maxind[i]:
                break
            copy_lastconc_to_initconc(script_dir.parent / dates[j - 1], script_dir.parent / dates[j], foldlist)


if __name__ == '__main__':
    run_LN_KSF3()
