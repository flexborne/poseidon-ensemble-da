from __future__ import annotations

import sys

import numpy as np
from scipy.io import savemat

from box_levs import box_levs
from matlab_compat import xlswrite_compat
from obs_vs_poseidon import obs_vs_poseidon
from paths import (
    MEASUREMENTS_DIR,
    box_surfareas_with_centers_path,
    date_dir,
    obs2pos_mat_path,
    obs_pos_xls_path,
    output_xlsx_path,
)
from read_poseidon import read_poseidon


def main_obsproconly(*args):
    nargin = len(args)
    if nargin < 2:
        posfolds = [
            '01.hydrol1dep1', '02.hydrol1dep2', '03.hydrol1dep3', '04.hydrol1dep4',
            '05.hydrol2dep1', '06.hydrol2dep2', '07.hydrol2dep3', '08.hydrol2dep4',
            '09.hydrol3dep1', '10.hydrol3dep2', '11.hydrol3dep3', '12.hydrol3dep4',
            '13.hydrol4dep1', '14.hydrol4dep2', '15.hydrol4dep3', '16.hydrol4dep4',
            '17.hydrol5dep1', '18.hydrol5dep2', '19.hydrol5dep3', '20.hydrol5dep4',
            '21.hydrol6dep1', '22.hydrol6dep2', '23.hydrol6dep3', '24.hydrol6dep4',
        ]
    else:
        posfolds = args[1]
    if nargin < 1:
        strdate = '1987.5'
    else:
        strdate = args[0]

    boxes_with_centers = box_surfareas_with_centers_path()
    date_dir_cur = date_dir(strdate)

    obs, brefnum, breflev, box_vs_allobs, boxlats, boxlons = obs_vs_poseidon(
        f'{strdate}.dat',
        str(MEASUREMENTS_DIR.resolve()),
        str(boxes_with_centers),
        [25, 100, 500, 2000],
        str(obs_pos_xls_path()),
    )
    savemat(obs2pos_mat_path(), {
        'obs': obs,
        'brefnum': brefnum,
        'breflev': breflev,
        'box_vs_allobs': box_vs_allobs,
    })
    xlswrite_compat(str(output_xlsx_path(strdate)), np.concatenate((obs, brefnum), axis=1), 'obs')

    bnlev = box_levs()
    No = int(obs.shape[0])

    Nens = len(posfolds)
    ens = np.zeros((No, 1), dtype=float)
    sim = np.zeros((No, 1), dtype=float)
    for i in range(Nens):
        posfold = posfolds[i]
        pfile = date_dir_cur / posfold / 'lastconc.xls'
        arr, pheader, pfooter, sed = read_poseidon(str(pfile), bnlev)

        for L in range(No):
            sim[L, 0] = arr[int(brefnum[L, 0]) - 1, int(breflev[L, 0]) - 1]
            ens[L, 0] = ens[L, 0] + sim[L, 0]

        xlswrite_compat(str(output_xlsx_path(strdate)), sim, posfold)

    ens = ens / np.single(Nens)
    xlswrite_compat(str(output_xlsx_path(strdate)), ens, 'Ens')

    return obs, sim


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        main_obsproconly()
    elif len(args) == 1:
        main_obsproconly(args[0])
    else:
        main_obsproconly(args[0], args[1:])
