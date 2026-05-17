from __future__ import annotations

import sys

import numpy as np
import scipy.linalg as la
from scipy.io import loadmat, savemat

from Box_Neighbors import Box_Neighbors
from IJ_Lind import IJ_Lind
from box_levs import box_levs
from init_Hmatr import init_Hmatr
from matlab_compat import matlab_lsqr, matlab_print, mkdir_warning, xlswrite_compat
from obs_vs_poseidon import obs_vs_poseidon
from paths import (
    MEASUREMENTS_DIR,
    TEMPLATES_DIR,
    box_surfareas_with_centers_path,
    date_dir,
    obs2pos_mat_path,
    obs_pos_xls_path,
    output_xlsx_path,
)
from read_poseidon import read_poseidon
from regularize_Covmatr_analit import regularize_Covmatr_analit
from write_poseidon import write_poseidon

Lind = None
Iind = None
Jind = None


def main_LN(*args):
    global Lind, Iind, Jind
    nargin = len(args)
    if nargin == 0:
        strdate = '1986.75'
    else:
        strdate = args[0]
    if nargin < 2:
        GSD = 1.3
    else:
        GSD = args[1]

    CovPerturb = 1.0
    rel2ens = False

    RcorH = 100000
    RcorV = 100

    areg = float(np.log(GSD))
    method = 1

    boxes_with_centers = box_surfareas_with_centers_path()
    date_dir_cur = date_dir(strdate)

    d = sorted([p for p in date_dir_cur.glob('*hydrol*dep*') if p.is_dir()])
    if not d:
        raise FileNotFoundError(f'No Poseidon folders found for date {strdate}')
    posfolds = [p.name for p in d]

    obsready = False

    if obsready:
        s = loadmat(obs2pos_mat_path(), squeeze_me=True)
        obs = np.asarray(s['obs'], dtype=float).reshape(-1, 1)
        brefnum = np.asarray(s['brefnum'], dtype=int).reshape(-1, 1)
        breflev = np.asarray(s['breflev'], dtype=int).reshape(-1, 1)
        boxlats = np.asarray(s['boxlats'], dtype=float)
        boxlons = np.asarray(s['boxlons'], dtype=float)
    else:
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
            'boxlats': boxlats,
            'boxlons': boxlons,
        })
        xlswrite_compat(str(output_xlsx_path(strdate)), np.concatenate((obs, brefnum), axis=1), 'obs')
    Nobs = obs.shape[0]

    obs = np.log(obs)

    invO2 = np.zeros((Nobs, Nobs), dtype=float)
    matlab_print('invO2', invO2)
    for i in range(Nobs):
        invO2[i, i] = 1.0 / areg

    bnlev = box_levs()
    Box_Neighbors(np.size(bnlev, 1) if np.ndim(bnlev) > 1 else np.size(bnlev, 0), brefnum, boxlats, boxlons)
    Lind, Iind, Jind = IJ_Lind(bnlev)
    matlab_print('Lind', Lind)
    matlab_print('Iind', Iind)
    matlab_print('Jind', Jind)
    Np = np.size(Iind)

    Hmatr = init_Hmatr(brefnum, breflev, Lind)

    Nens = len(posfolds)
    Ens = np.zeros((Np, Nens), dtype=float)
    Sens = [None] * Nens
    for i in range(Nens):
        pfile = date_dir_cur / posfolds[i] / 'lastconc.xls'
        arr, pheader, pfooter, sed = read_poseidon(str(pfile), bnlev)
        for L in range(Np):
            ii = int(np.asarray(Iind).ravel()[L])
            jj = int(np.asarray(Jind).ravel()[L])
            Ens[L, i] = arr[ii, jj]
        Sens[i] = sed

    Ens = np.log(Ens)

    CC = np.cov(Ens, bias=False)
    CC1, LocM = regularize_Covmatr_analit(
        CC, Iind, Jind, [1, 2, 3, 49, 50], RcorH, RcorV, str(boxes_with_centers)
    )

    if rel2ens:
        CovDiag = Hmatr @ np.diag(CC1)
        for i in range(Nobs):
            invO2[i, i] = 1.0 / areg / np.sqrt(CovDiag[i])

    Covr_1 = np.linalg.inv(CC1)

    Lmatr, Dmatr, perm = la.ldl(Covr_1, lower=True, hermitian=True)
    Dsqrt = np.sqrt(Dmatr)

    Gmatr_p1 = invO2 @ Hmatr
    Gmatr_p2 = Dsqrt @ (Lmatr.T)
    Gmodif = np.concatenate((Gmatr_p1, Gmatr_p2), axis=0)

    obs_noise = np.zeros((Nobs, Nens), dtype=float)
    for ii in range(Nobs):
        for jj in range(Nens):
            obs_noise[ii, jj] = obs[ii, 0] + (1 / invO2[ii, ii]) * np.random.randn()

    sol = np.zeros((Np, Nens), dtype=float)
    for i in range(Nens):
        ymodif1 = invO2 @ obs_noise[:, i].reshape(-1, 1)
        ymodif2 = Dsqrt @ (Lmatr.T @ Ens[:, i].reshape(-1, 1))
        ymodif = np.concatenate((ymodif1, ymodif2), axis=0)

        if method == 1:
            sol[:, i] = matlab_lsqr(Gmodif, ymodif.ravel(), 1e-8, 10000)
        elif method == 2:
            raise NotImplementedError
        else:
            print('Method is unknown')
            print(method)
            raise ValueError('Method is unknown')

        out_dir = TEMPLATES_DIR / posfolds[i]
        mkdir_warning(out_dir)
        ofname = out_dir / 'Initconc.dat'
        if i == 0:
            write_poseidon(str(ofname), np.exp(sol[:, i]), Lind, bnlev, pheader, pfooter, Sens[i])
        else:
            write_poseidon(str(ofname), np.exp(Ens[:, i] + CovPerturb * (sol[:, i] - Ens[:, i])), Lind, bnlev, pheader, pfooter, Sens[i])

        xlswrite_compat(str(output_xlsx_path(strdate)), (Hmatr @ np.exp(sol[:, i])).reshape(-1, 1), posfolds[i])

    ens = np.exp(np.mean(sol, axis=1))
    xlswrite_compat(str(output_xlsx_path(strdate)), (Hmatr @ ens).reshape(-1, 1), 'Ens')

    return obs, None


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        main_LN()
    elif len(args) == 1:
        main_LN(args[0])
    else:
        main_LN(args[0], float(args[1]))
