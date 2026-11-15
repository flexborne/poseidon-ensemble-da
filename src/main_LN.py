from __future__ import annotations

import sys

import numpy as np
from scipy.io import savemat

from IJ_Lind import IJ_Lind
from analysis import AnalysisResult, analysis_update, observation_indices
from box_levs import box_levs
from model_compat import xlswrite_compat
from obs_vs_poseidon import obs_vs_poseidon
from paths import MEASUREMENTS_DIR, box_surfareas_with_centers_path
from read_poseidon import read_poseidon
from regularize_Covmatr_analit import regularize_Covmatr_analit
from run_context import RunContext
from write_poseidon import write_poseidon


def main_LN(strdate='1986.75', GSD=1.3, run_ctx=None, perturb=True, rng=None):
    """EnKF (filter) analysis for a single date.

    Reads the ensemble forecast, estimates and localizes the background
    covariance, and solves the analysis in observation space for every member.
    Writes updated Initconc.dat restart files and out_<date>.xlsx, and returns
    an AnalysisResult.

    perturb=False zeroes the observation perturbations (deterministic run); rng
    is an optional numpy Generator for reproducible perturbations.
    """
    if run_ctx is None:
        run_ctx = RunContext.default()
    if rng is None:
        rng = np.random

    CovPerturb = 1.0
    RcorH = 100000
    RcorV = 100
    sigma = float(np.log(GSD))              # observation error std in log-space
    boxes_with_centers = str(box_surfareas_with_centers_path())

    date_dir_cur = run_ctx.date_dir(strdate)
    posfolds = sorted(p.name for p in date_dir_cur.glob('*hydrol*dep*') if p.is_dir())
    if not posfolds:
        raise FileNotFoundError(f'No Poseidon folders found for date {strdate}')

    obs, brefnum, breflev, box_vs_allobs, boxlats, boxlons = obs_vs_poseidon(
        f'{strdate}.dat',
        str(MEASUREMENTS_DIR.resolve()),
        boxes_with_centers,
        [25, 100, 500, 2000],
        str(run_ctx.obs_pos_xlsx()),
    )
    savemat(str(run_ctx.obs2pos_mat()), {
        'obs': obs, 'brefnum': brefnum, 'breflev': breflev,
        'boxlats': boxlats, 'boxlons': boxlons,
    })
    xlswrite_compat(str(run_ctx.output_xlsx(strdate)), np.concatenate((obs, brefnum), axis=1), 'obs')

    Nobs = obs.shape[0]
    y = np.log(obs).reshape(-1, 1)          # assimilated values, log-space

    bnlev = box_levs()
    Lind, Iind, Jind = IJ_Lind(bnlev)
    Np = int(np.size(Iind))
    obs_idx = observation_indices(brefnum, breflev, Lind)

    Nens = len(posfolds)
    Ens = np.zeros((Np, Nens), dtype=float)
    headers = [None] * Nens
    footers = [None] * Nens
    sediments = [None] * Nens
    for i, posfold in enumerate(posfolds):
        arr, header, footer, sed = read_poseidon(str(date_dir_cur / posfold / 'lastconc.xls'), bnlev)
        for L in range(Np):
            Ens[L, i] = arr[int(Iind[L]), int(Jind[L])]
        headers[i], footers[i], sediments[i] = header, footer, sed

    Xf = np.log(Ens)                        # background log-states

    CC = np.cov(Xf, bias=False)
    B, LocM = regularize_Covmatr_analit(CC, Iind, Jind, [1, 2, 3, 49, 50], RcorH, RcorV, boxes_with_centers)
    R = sigma ** 2 * np.eye(Nobs)

    # Observation perturbations - the only source of randomness (stochastic EnKF).
    # perturb=False zeroes them for a deterministic, reproducible run.
    Y = np.tile(y, (1, Nens))
    if perturb:
        for ii in range(Nobs):
            for jj in range(Nens):
                Y[ii, jj] = y[ii, 0] + sigma * rng.randn()

    sol = analysis_update(B, obs_idx, Y, R, Xf)   # (Np, Nens) analysis log-states

    for i, posfold in enumerate(posfolds):
        out_dir = run_ctx.template_dir(posfold)
        out_dir.mkdir(parents=True, exist_ok=True)
        analysed = Xf[:, i] + CovPerturb * (sol[:, i] - Xf[:, i])
        write_poseidon(
            str(out_dir / 'Initconc.dat'),
            np.exp(analysed), Lind, bnlev, headers[i], footers[i], sediments[i],
        )
        xlswrite_compat(str(run_ctx.output_xlsx(strdate)), np.exp(sol[obs_idx, i]).reshape(-1, 1), posfold)

    geo_mean = np.exp(np.mean(sol, axis=1))
    xlswrite_compat(str(run_ctx.output_xlsx(strdate)), geo_mean[obs_idx].reshape(-1, 1), 'Ens')

    return AnalysisResult(
        target=strdate,
        member_ids=list(posfolds),
        obs_indices=obs_idx,
        observations=y.ravel(),
        log_states=sol,
        concentrations=np.exp(sol),
        geometric_mean=geo_mean,
        diagnostics={'Nobs': Nobs, 'Nstate': Np, 'Nens': Nens},
    )


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        main_LN()
    elif len(args) == 1:
        main_LN(args[0])
    else:
        main_LN(args[0], float(args[1]))
