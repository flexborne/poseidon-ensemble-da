from __future__ import annotations

import sys

import numpy as np

from IJT_Lind import IJT_Lind
from Tregularize_Covmatr_analit import Tregularize_Covmatr_analit
from analysis import AnalysisResult, analysis_update, observation_indices
from box_levs import box_levs
from model_compat import xlswrite_compat
from obs_vs_poseidon import obs_vs_poseidon
from paths import MEASUREMENTS_DIR, box_surfareas_with_centers_path
from read_poseidon import read_poseidon
from run_context import RunContext
from write_poseidon import write_poseidon


def main_KSF_LN(strdates=None, GSD=1.3, run_ctx=None, perturb=True, rng=None):
    """Kalman smoother analysis over several dates.

    Jointly assimilates observations from all dates in strdates with temporal
    localization; the analysis is built for the last date in the array. Writes
    updated Initconc.dat restart files and out_<date>.xlsx, and returns an
    AnalysisResult.

    perturb=False zeroes the observation perturbations (deterministic run); rng
    is an optional numpy Generator for reproducible perturbations.
    """
    if strdates is None:
        strdates = ['1988.625', '1988.5']
    if run_ctx is None:
        run_ctx = RunContext.default()
    if rng is None:
        rng = np.random

    CovPerturb = 1.0
    RcorH = 100000
    RcorV = 100
    RcorT = 1.0
    sigma = float(np.log(GSD))
    boxes_with_centers = str(box_surfareas_with_centers_path())

    Ndates = len(strdates)
    anl_date = strdates[-1]
    obstimes = [float(x) for x in strdates]

    date_anl_dir = run_ctx.date_dir(anl_date)
    posfolds = sorted(p.name for p in date_anl_dir.glob('*hydrol*dep*') if p.is_dir())
    if not posfolds:
        raise FileNotFoundError(f'No Poseidon folders found for date {anl_date}')

    obs = None
    brefnum = None
    breflev = None
    Nobs_T = []
    for strdate in strdates:
        obs_, brefnum_, breflev_, _, boxlats, boxlons = obs_vs_poseidon(
            f'{strdate}.dat', str(MEASUREMENTS_DIR.resolve()), boxes_with_centers,
            [25, 100, 500, 2000], str(run_ctx.obs_pos_xlsx()),
        )
        xlswrite_compat(str(run_ctx.output_xlsx(strdate)), np.concatenate((obs_, brefnum_), axis=1), 'obs')
        if obs is None:
            obs, brefnum, breflev = obs_, brefnum_, breflev_
        else:
            obs = np.vstack((obs, obs_))
            brefnum = np.vstack((brefnum, brefnum_))
            breflev = np.vstack((breflev, breflev_))
        Nobs_T.append(obs_.shape[0])

    Nobs = obs.shape[0]
    anl_ind_start = sum(Nobs_T[:-1])
    y = np.log(obs).reshape(-1, 1)

    bnlev = box_levs()
    Lind, Iind, Jind, Tind, Lind_extract = IJT_Lind(bnlev, brefnum.ravel(), breflev.ravel(), Nobs_T)
    Nstate = len(Iind)
    obs_idx = observation_indices(brefnum.ravel(), breflev.ravel(), Lind, Tind)

    Nens = len(posfolds)
    Ens = np.zeros((Nstate, Nens), dtype=float)
    restart_meta = [None] * Nens   # (header, footer, sed) of the analysis date, per member
    for i, posfold in enumerate(posfolds):
        arr = {}
        for k, strdate in enumerate(strdates):
            arr[k], header_k, footer_k, sed_k = read_poseidon(
                str(run_ctx.date_dir(strdate) / posfold / 'lastconc.xls'), bnlev)
            if k == Ndates - 1:                      # analysis date supplies the restart metadata
                restart_meta[i] = (header_k, footer_k, sed_k)
        for L in range(Nstate):
            Ens[L, i] = arr[int(Tind[L])][int(Iind[L]), int(Jind[L])]

    Xf = np.log(Ens)

    CC = np.cov(Xf, bias=False)
    B, LocM = Tregularize_Covmatr_analit(
        CC, Iind, Jind, Tind, obstimes, [1, 2, 3, 49, 50], RcorH, RcorV, RcorT, boxes_with_centers)
    R = sigma ** 2 * np.eye(Nobs)

    # Observation perturbations - the only source of randomness (stochastic EnKF).
    Y = np.tile(y, (1, Nens))
    if perturb:
        for ii in range(Nobs):
            for jj in range(Nens):
                Y[ii, jj] = y[ii, 0] + sigma * rng.randn()

    sol = analysis_update(B, obs_idx, Y, R, Xf)

    last_obs_slice = slice(anl_ind_start, anl_ind_start + Nobs_T[-1])
    for i, posfold in enumerate(posfolds):
        header_last, footer_last, sed_last = restart_meta[i]
        out_dir = run_ctx.template_dir(posfold)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Xf[anl_ind_start:, i]
        analysed = base + CovPerturb * (sol[anl_ind_start:, i] - base)
        write_poseidon(
            str(out_dir / 'Initconc.dat'),
            np.exp(analysed), Lind_extract, bnlev, header_last, footer_last, sed_last,
        )
        xlswrite_compat(str(run_ctx.output_xlsx(anl_date)),
                        np.exp(sol[obs_idx, i])[last_obs_slice].reshape(-1, 1), posfold)

    geo_mean = np.exp(np.mean(sol, axis=1))
    xlswrite_compat(str(run_ctx.output_xlsx(anl_date)),
                    geo_mean[obs_idx][last_obs_slice].reshape(-1, 1), 'Ens')

    return AnalysisResult(
        target=anl_date,
        member_ids=list(posfolds),
        obs_indices=obs_idx,
        observations=y.ravel(),
        log_states=sol,
        concentrations=np.exp(sol),
        geometric_mean=geo_mean,
        diagnostics={'Nobs': Nobs, 'Nstate': Nstate, 'Nens': Nens,
                     'anl_ind_start': anl_ind_start, 'Nobs_T': list(Nobs_T)},
    )


if __name__ == '__main__':
    args = sys.argv[1:]
    main_KSF_LN(args if args else None)
