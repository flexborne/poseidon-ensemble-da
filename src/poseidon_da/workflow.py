"""Orchestration: assemble inputs, run the shared analysis, write restarts/outputs."""
from __future__ import annotations

import numpy as np

from .analysis import analysis_update
from .covariance import estimate_covariance, localize_covariance
from .domain import AnalysisResult, AnalysisWindow
from .geometry import box_levels, load_box_centers
from .observations import aggregate_observations
from .poseidon_io import read_poseidon, write_poseidon, xlswrite
from .runner import RunContext, find_members, run_poseidon_all
from .state import build_layout_single, build_layout_window, observation_indices

DEFAULT_BDEPTHS = [25, 100, 500, 2000]
DEFAULT_EXCLUDE = [1, 2, 3, 49, 50]
DEFAULT_LOCALIZATION = {'horizontal_m': 100000.0, 'vertical_m': 100.0,
                        'temporal_years': 1.0, 'exclude': DEFAULT_EXCLUDE}


def _perturbed_obs(y, M, sigma, perturb, rng):
    """Replicate the stochastic-EnKF perturbation order (col-major over obs, members)."""
    Y = np.tile(np.asarray(y, dtype=float).reshape(-1, 1), (1, M))
    if perturb:
        p = Y.shape[0]
        for ii in range(p):
            for jj in range(M):
                Y[ii, jj] = y[ii] + sigma * rng.randn()
    return Y


def analyze(window: AnalysisWindow, ctx: RunContext, *, geometry_path, measurements_dir,
            gsd=1.3, localization=None, bdepths=None, perturb=True, rng=None):
    """Run one analysis (filter if single date, smoother otherwise).

    Returns (AnalysisResult, restart_payload). The payload carries everything
    needed to write restart files; writing is an explicit, separate step.
    """
    if rng is None:
        rng = np.random
    loc = {**DEFAULT_LOCALIZATION, **(localization or {})}
    bdepths = bdepths or DEFAULT_BDEPTHS
    sigma = float(np.log(gsd))
    dates = list(window.observation_times)
    target = window.target_time
    members = find_members(ctx.date_dir(target))
    if not members:
        raise FileNotFoundError(f'No ensemble members for date {target}')
    M = len(members)
    bnlev = box_levels()
    centers = load_box_centers(geometry_path)

    # --- observations (per date), stacked ---
    batches = [aggregate_observations(f'{measurements_dir}/{d}.dat', geometry_path, bdepths) for d in dates]
    nobs_per_date = [b['values'].shape[0] for b in batches]
    values = np.vstack([b['values'] for b in batches])
    brefnum = np.vstack([b['brefnum'] for b in batches])
    breflev = np.vstack([b['breflev'] for b in batches])
    y = np.log(values).ravel()

    # --- state layout + observation gather indices ---
    if window.is_filter:
        layout = build_layout_single(bnlev)
        write_index = layout.Lind
    else:
        layout = build_layout_window(bnlev, brefnum.ravel(), breflev.ravel(), nobs_per_date)
        write_index = layout.lind_extract
    obs_idx = observation_indices(brefnum.ravel(), breflev.ravel(), layout)
    anl0 = layout.anl_ind_start

    # --- ensemble background (log) + per-member restart metadata (target date) ---
    Xf = np.zeros((layout.n_state, M), dtype=float)
    metas = [None] * M
    Iind, Jind, Tind = layout.Iind, layout.Jind, layout.Tind
    for i, member in enumerate(members):
        per_date = {}
        for k, d in enumerate(dates):
            arr, header, footer, sed = read_poseidon(str(ctx.date_dir(d) / member / 'lastconc.xls'), bnlev)
            per_date[k] = arr
            if k == len(dates) - 1:
                metas[i] = (header, footer, sed)
        for L in range(layout.n_state):
            t = 0 if Tind is None else int(Tind[L])
            Xf[L, i] = per_date[t][int(Iind[L]), int(Jind[L])]
    Xf = np.log(Xf)

    # --- covariance + localization + analysis ---
    CC = estimate_covariance(Xf)
    obstimes = [float(d) for d in dates]
    B, _ = localize_covariance(
        CC, Iind, Jind, centers, loc['exclude'], loc['horizontal_m'], loc['vertical_m'],
        Tind=Tind, obstimes=obstimes, RcorT=loc['temporal_years'])
    R = sigma ** 2 * np.eye(values.shape[0])
    Y = _perturbed_obs(y, M, sigma, perturb, rng)
    sol = analysis_update(B, obs_idx, Y, R, Xf)

    geo_mean = np.exp(np.mean(sol, axis=1))
    result = AnalysisResult(
        target=target, member_ids=list(members), obs_indices=obs_idx, observations=y,
        log_states=sol, concentrations=np.exp(sol), geometric_mean=geo_mean,
        diagnostics={'Nobs': int(values.shape[0]), 'Nstate': int(layout.n_state),
                     'Nens': M, 'anl_ind_start': int(anl0), 'nobs_per_date': nobs_per_date},
    )
    payload = {'bnlev': bnlev, 'write_index': write_index, 'anl_ind_start': anl0,
               'members': members, 'metas': metas, 'batches': batches, 'dates': dates}
    return result, payload


def write_restarts(result: AnalysisResult, payload, ctx: RunContext, cov_perturb=1.0) -> list:
    """Write each member's analysed Initconc.dat into the templates directory."""
    bnlev = payload['bnlev']
    write_index = payload['write_index']
    anl0 = payload['anl_ind_start']
    sol = result.log_states
    written = []
    for i, member in enumerate(payload['members']):
        header, footer, sed = payload['metas'][i]
        out_dir = ctx.template_dir(member)
        base = sol[anl0:, i]                       # target-date block (== full state for a filter)
        analysed = base + cov_perturb * (sol[anl0:, i] - base)
        ofname = out_dir / 'Initconc.dat'
        write_poseidon(str(ofname), np.exp(analysed), write_index, bnlev, header, footer, sed)
        written.append(ofname)
    return written


def write_outputs(result: AnalysisResult, payload, ctx: RunContext) -> None:
    """Write out_<date>.xlsx: obs sheet(s), one sheet per member, and Ens (geometric mean)."""
    dates = payload['dates']
    for d, batch in zip(dates, payload['batches']):
        xlswrite(str(ctx.output_xlsx(d)),
                 np.concatenate((batch['values'], batch['brefnum']), axis=1), 'obs')
    target = result.target
    obs_idx = result.obs_indices
    anl0 = payload['anl_ind_start']
    n_last = payload['batches'][-1]['values'].shape[0]
    sl = slice(anl0, anl0 + n_last)
    for i, member in enumerate(payload['members']):
        xlswrite(str(ctx.output_xlsx(target)),
                 np.exp(result.log_states[obs_idx, i])[sl].reshape(-1, 1), member)
    xlswrite(str(ctx.output_xlsx(target)), result.geometric_mean[obs_idx][sl].reshape(-1, 1), 'Ens')
