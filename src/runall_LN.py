from __future__ import annotations

from main_LN import main_LN
from main_obsproconly import main_obsproconly
from model_compat import copy_file
from poseidon_runtime import publish_initconc, run_poseidon_all
from run_context import RunContext

DATES = ['1986.75', '1987.5', '1988.25', '1988.5', '1988.625',
         '1988.75', '1988.85', '1990.75', '1991.75', '2007.5']


def runall_LN(run_ctx=None):
    """Full sequential analysis-forecast cycle over all dates (filter-only, main_LN).

    For each date: analysis -> publish Initconc.dat forward -> POSEIDON forecast
    to the next date -> diagnostic comparison against observations.
    """
    if run_ctx is None:
        run_ctx = RunContext.default()
    dates = DATES
    Nd = len(dates)

    for i in range(Nd - 1):
        curdate = dates[i]
        nextdate = dates[i + 1]

        main_LN(curdate, run_ctx=run_ctx)

        anl_dir = run_ctx.date_dir(curdate) / 'ANL'
        publish_initconc(run_ctx.templates_dir, anl_dir)
        out_cur = run_ctx.output_xlsx(curdate)
        if out_cur.exists():
            copy_file(out_cur, anl_dir / f'out_{curdate}.xlsx')

        publish_initconc(run_ctx.templates_dir, run_ctx.date_dir(nextdate))
        run_poseidon_all(run_ctx.date_dir(nextdate))

        if i < Nd - 2:
            main_obsproconly(nextdate, run_ctx=run_ctx)


if __name__ == '__main__':
    runall_LN()
