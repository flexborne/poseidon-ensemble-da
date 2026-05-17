from __future__ import annotations

import sys

import numpy as np
import scipy.linalg as la

from Box_Neighbors import Box_Neighbors
from IJT_Lind import IJT_Lind
from Tregularize_Covmatr_analit import Tregularize_Covmatr_analit
from box_levs import box_levs
from init_Hmatr_T import init_Hmatr_T
from matlab_compat import matlab_lsqr, mkdir_warning, xlswrite_compat
from obs_vs_poseidon import obs_vs_poseidon
from paths import (
    MEASUREMENTS_DIR,
    TEMPLATES_DIR,
    box_surfareas_with_centers_path,
    date_dir,
    obs_pos_xls_path,
    output_xlsx_path,
)
from read_poseidon import read_poseidon
from write_poseidon import write_poseidon

Lind = None
Iind = None
Jind = None


def main_KSF_LN(strdates=None, GSD=1.3):
    global Lind, Iind, Jind

    if strdates is None:
        strdates = ['1988.625', '1988.5']

    CovPerturb = 1.0
    RcorH = 100000
    RcorV = 100
    RcorT = 1.0
    areg = float(np.log(GSD))

    boxes_with_centers = box_surfareas_with_centers_path()

    Ndates = len(strdates)
    anl_date = strdates[-1]
    obstimes = [float(x) for x in strdates]

    date_anl_dir = date_dir(anl_date)
    posfolds = sorted([p.name for p in date_anl_dir.glob('*hydrol*dep*') if p.is_dir()])
    if not posfolds:
        raise FileNotFoundError(f'No Poseidon folders found for date {anl_date}')

    obs = None
    brefnum = None
    breflev = None
    Nobs_T = []

    for strdate in strdates:
        obs_, brefnum_, breflev_, box_vs_allobs, boxlats, boxlons = obs_vs_poseidon(
            f'{strdate}.dat',
            str(MEASUREMENTS_DIR.resolve()),
            str(boxes_with_centers),
            [25, 100, 500, 2000],
            str(obs_pos_xls_path()),
        )

        xlswrite_compat(
            str(output_xlsx_path(strdate)),
            np.concatenate((obs_, brefnum_), axis=1),
            'obs'
        )

        if obs is None:
            obs = obs_
            brefnum = brefnum_
            breflev = breflev_
        else:
            obs = np.vstack((obs, obs_))
            brefnum = np.vstack((brefnum, brefnum_))
            breflev = np.vstack((breflev, breflev_))

        Nobs_T.append(obs_.shape[0])

    Nobs = obs.shape[0]
    anl_ind_start = sum(Nobs_T[:-1])

    obs = np.log(obs)

    invO2 = np.zeros((Nobs, Nobs), dtype=float)
    for i in range(Nobs):
        invO2[i, i] = 1.0 / areg

    bnlev = box_levs()
    Box_Neighbors(
        np.size(bnlev, 1) if np.ndim(bnlev) > 1 else np.size(bnlev, 0),
        brefnum,
        boxlats,
        boxlons,
    )

    Lind, Iind, Jind, Tind, Lind_extract = IJT_Lind(
        bnlev,
        brefnum.ravel(),
        breflev.ravel(),
        Nobs_T
    )

    Nstate = len(Iind)
    Hmatr = init_Hmatr_T(
        brefnum.ravel(),
        breflev.ravel(),
        Lind,
        Tind
    )

    Nens = len(posfolds)
    Ens = np.zeros((Nstate, Nens), dtype=float)
    Sens = [None] * Nens

    for i, posfold in enumerate(posfolds):
        arr = {}
        pheader = {}
        pfooter = {}
        sed = {}

        for k, strdate in enumerate(strdates):
            pfile = date_dir(strdate) / posfold / 'lastconc.xls'
            arr[k], pheader[k], pfooter[k], sed[k] = read_poseidon(str(pfile), bnlev)

        for L in range(Nstate):
            Ens[L, i] = arr[int(Tind[L])][int(Iind[L]), int(Jind[L])]

        Sens[i] = sed[Ndates - 1]

    Ens = np.log(Ens)

    CC = np.cov(Ens, bias=False)
    CC1, LocM = Tregularize_Covmatr_analit(
        CC,
        Iind,
        Jind,
        Tind,
        obstimes,
        [1, 2, 3, 49, 50],
        RcorH,
        RcorV,
        RcorT,
        str(boxes_with_centers),
    )

    Covr_1 = np.linalg.inv(CC1)

    Lmatr, Dmatr, perm = la.ldl(Covr_1, lower=True, hermitian=True)
    Dsqrt = np.sqrt(Dmatr)

    Gmodif = np.vstack((
        invO2 @ Hmatr,
        Dsqrt @ (Lmatr.T),
    ))

    obs_noise = np.zeros((Nobs, Nens), dtype=float)
    for ii in range(Nobs):
        for jj in range(Nens):
            obs_noise[ii, jj] = obs[ii, 0] + (1.0 / invO2[ii, ii]) * np.random.randn()

    sol = np.zeros((Nstate, Nens), dtype=float)

    for i, posfold in enumerate(posfolds):
        ymodif = np.concatenate((
            invO2 @ obs_noise[:, i],
            Dsqrt @ (Lmatr.T @ Ens[:, i]),
        ))

        sol[:, i] = matlab_lsqr(Gmodif, ymodif, 1e-8, 10000)

        out_dir = TEMPLATES_DIR / posfold
        mkdir_warning(out_dir)
        ofname = out_dir / 'Initconc.dat'

        x_writeout = sol[anl_ind_start:, i]

        if i == 0:
            write_poseidon(
                str(ofname),
                np.exp(x_writeout),
                Lind_extract,
                bnlev,
                pheader[Ndates - 1],
                pfooter[Ndates - 1],
                Sens[i],
            )
        else:
            base = Ens[anl_ind_start:, i]
            write_poseidon(
                str(ofname),
                np.exp(base + CovPerturb * (x_writeout - base)),
                Lind_extract,
                bnlev,
                pheader[Ndates - 1],
                pfooter[Ndates - 1],
                Sens[i],
            )

        xx = np.exp(Hmatr @ sol[:, i])
        outvect = xx[anl_ind_start:anl_ind_start + Nobs_T[-1]]
        xlswrite_compat(
            str(output_xlsx_path(anl_date)),
            outvect.reshape(-1, 1),
            posfold
        )

    ens = np.exp(np.mean(sol, axis=1))
    xx = Hmatr @ ens
    outvect = xx[anl_ind_start:anl_ind_start + Nobs_T[-1]]
    xlswrite_compat(
        str(output_xlsx_path(anl_date)),
        outvect.reshape(-1, 1),
        'Ens'
    )

    return obs, None


if __name__ == '__main__':
    args = sys.argv[1:]
    main_KSF_LN(args if args else None)
