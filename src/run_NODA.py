from __future__ import annotations

from main_obsproconly import main_obsproconly
from model_compat import find_poseidon_folders
from poseidon_runtime import copy_lastconc_to_initconc, run_poseidon_all
from run_context import RunContext

DATES = ['1986.75', '1987.5', '1988.25', '1988.5', '1988.625',
         '1988.75', '1988.85', '1990.75', '1991.75', '2007.5']


def run_NODA(run_ctx=None):
    """Control cycle without data assimilation (No DA).

    Only the POSEIDON forecast over the date sequence plus a diagnostic
    comparison against observations - a baseline for assessing the effect of
    assimilation. All date directories are resolved through run_ctx.
    """
    if run_ctx is None:
        run_ctx = RunContext.default()
    dates = DATES
    Nd = len(dates)
    foldlist = find_poseidon_folders(run_ctx.date_dir(dates[0]))

    i = 0
    while i < Nd - 1:
        i += 1
        nextdate = dates[i]
        for j in range(i, i + 2):
            run_poseidon_all(run_ctx.date_dir(dates[j - 1]), expected_members=foldlist)
            if j == i + 1:
                main_obsproconly(dates[j - 1], run_ctx=run_ctx)
                break
            copy_lastconc_to_initconc(run_ctx.date_dir(dates[j - 1]),
                                      run_ctx.date_dir(dates[j]), foldlist)


if __name__ == '__main__':
    run_NODA()
