from __future__ import annotations

from main_KSF_LN import main_KSF_LN
from main_LN import main_LN
from main_obsproconly import main_obsproconly
from model_compat import copy_file, find_poseidon_folders
from poseidon_runtime import copy_lastconc_to_initconc, publish_initconc, run_poseidon_all
from run_context import RunContext

DATES = ['1986.75', '1987.5', '1988.25', '1988.5', '1988.625',
         '1988.75', '1988.85', '1990.75', '1991.75', '2007.5']
# earliest / latest date index whose measurements are relevant to date i (+-3 month window)
MININD = [1, 2, 3, 3, 4, 4, 5, 8, 9, 10]
MAXIND = [1, 2, 4, 6, 7, 7, 7, 8, 9, 10]


def run_LN_KSF3(run_ctx=None):
    """Full analysis-forecast cycle combining filter and smoother.

    For dates whose +/-3-month observation window covers neighbouring dates it
    uses the smoother main_KSF_LN, otherwise the filter main_LN; then runs the
    POSEIDON forecast and moves to the next date. All date directories and
    outputs are resolved through run_ctx.
    """
    if run_ctx is None:
        run_ctx = RunContext.default()
    dates, minind, maxind = DATES, MININD, MAXIND
    Nd = len(dates)
    foldlist = find_poseidon_folders(run_ctx.date_dir(dates[0]))

    i = 0
    while i < Nd - 1:
        i += 1
        curdate = dates[i - 1]
        nextdate = dates[i]

        if minind[i - 1] == i and maxind[i - 1] == i:
            main_LN(curdate, run_ctx=run_ctx)
        else:
            measdates = [dates[j - 1] for j in range(minind[i - 1], i)]
            measdates += [dates[j - 1] for j in range(i + 1, maxind[i - 1] + 1)]
            measdates.append(dates[i - 1])         # analysis date is always last
            main_KSF_LN(measdates, run_ctx=run_ctx)

        anl_dir = run_ctx.date_dir(curdate) / 'ANL'
        publish_initconc(run_ctx.templates_dir, anl_dir)
        out_cur = run_ctx.output_xlsx(curdate)
        if out_cur.exists():
            copy_file(out_cur, anl_dir / f'out_{curdate}.xlsx')
        publish_initconc(run_ctx.templates_dir, run_ctx.date_dir(nextdate))

        for j in range(i + 1, maxind[i] + 1):
            run_poseidon_all(run_ctx.date_dir(dates[j - 1]), expected_members=foldlist)
            if j == i + 1:
                main_obsproconly(dates[j - 1], run_ctx=run_ctx)
            if j == maxind[i]:
                break
            copy_lastconc_to_initconc(run_ctx.date_dir(dates[j - 1]),
                                      run_ctx.date_dir(dates[j]), foldlist)


if __name__ == '__main__':
    run_LN_KSF3()
