"""Orchestration: assemble validated inputs, run the shared analysis, write outputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import analysis_update
from .covariance import estimate_covariance, localize_covariance
from .domain import AnalysisResult, AnalysisWindow
from .geometry import box_levels, layer_depths_from_bdepths, load_box_centers
from .observations import aggregate_observations
from .poseidon_io import extract_physical_time, read_poseidon, write_poseidon, xlswrite
from .runner import RunContext, find_members
from .state import (build_layout_single, build_layout_window, observation_indices,
                    validate_observation_coords)

DEFAULT_BDEPTHS = [25, 100, 500, 2000]
DEFAULT_EXCLUDE = [1, 2, 3, 49, 50]
DEFAULT_LOCALIZATION = {'horizontal_m': 100000.0, 'vertical_m': 100.0,
                        'temporal_years': 1.0, 'exclude': DEFAULT_EXCLUDE}
TIME_TOL = 1e-6


@dataclass
class RestartPayload:
    """Everything needed to write restart files; keeps member order explicit."""

    bnlev: np.ndarray
    write_index: np.ndarray            # Lind (filter) or lind_extract (window)
    anl_ind_start: int
    members: list
    metas: list                        # per member: (header, footer, sed)
    batches: list
    dates: list


def _draw_normal(rng):
    # supports np.random module / RandomState (.standard_normal) and Generator (.standard_normal)
    if hasattr(rng, 'standard_normal'):
        return float(rng.standard_normal())
    return float(rng.randn())


def _perturbed_obs(y, M, sigma, perturb, rng):
    Y = np.tile(np.asarray(y, dtype=float).reshape(-1, 1), (1, M))
    if perturb:
        for ii in range(Y.shape[0]):
            for jj in range(M):
                Y[ii, jj] = y[ii] + sigma * _draw_normal(rng)
    return Y


def analyze(window: AnalysisWindow, ctx: RunContext, *, geometry_path, measurements_dir,
            gsd=1.3, localization=None, bdepths=None, perturb=True, rng=None):
    """Run one analysis (filter if single date, smoother otherwise).

    Returns (AnalysisResult, RestartPayload). Writing files is a separate step.
    """
    if rng is None:
        rng = np.random
    loc = {**DEFAULT_LOCALIZATION, **(localization or {})}
    bdepths = list(bdepths or DEFAULT_BDEPTHS)
    depb = layer_depths_from_bdepths(bdepths)
    sigma = float(np.log(gsd))
    dates = list(window.observation_times)
    target = window.target_time
    members = find_members(ctx.date_dir(target))
    if not members:
        raise FileNotFoundError(f'No ensemble members for date {target}')
    M = len(members)
    bnlev = box_levels()
    centers = load_box_centers(geometry_path)

    # --- observations (per date), time-checked and stacked ---
    batches, times = [], []
    for d in dates:
        b = aggregate_observations(f'{measurements_dir}/{d}.dat', geometry_path, bdepths,
                                   expected_time=float(d), time_tol=TIME_TOL)
        batches.append(b)
        times.append(float(d))
    nobs_per_date = [b['values'].shape[0] for b in batches]
    values = np.vstack([b['values'] for b in batches])
    brefnum = np.vstack([b['brefnum'] for b in batches])
    breflev = np.vstack([b['breflev'] for b in batches])
    y = np.log(values).ravel()

    # --- validate coordinates against the domain BEFORE building any layout ---
    validate_observation_coords(brefnum.ravel(), breflev.ravel(), bnlev)

    if window.is_filter:
        layout = build_layout_single(bnlev)
        write_index = layout.Lind
    else:
        layout = build_layout_window(bnlev, brefnum.ravel(), breflev.ravel(), nobs_per_date)
        write_index = layout.lind_extract
    obs_idx = observation_indices(brefnum.ravel(), breflev.ravel(), layout)
    anl0 = layout.anl_ind_start
    Iind, Jind, Tind = layout.Iind, layout.Jind, layout.Tind

    # --- ensemble background (log) + per-member restart metadata, with time checks ---
    Xf = np.zeros((layout.n_state, M), dtype=float)
    metas = [None] * M
    for i, member in enumerate(members):
        per_date = {}
        for k, d in enumerate(dates):
            arr, header, footer, sed = read_poseidon(str(ctx.date_dir(d) / member / 'lastconc.xls'), bnlev)
            ptime = extract_physical_time(footer)
            if ptime is not None and abs(ptime - float(d)) > TIME_TOL:
                raise ValueError(
                    f'member {member}: forecast state for requested date {d} has physical time {ptime}')
            per_date[k] = arr
            if k == len(dates) - 1:
                metas[i] = (header, footer, sed)
        for L in range(layout.n_state):
            t = 0 if Tind is None else int(Tind[L])
            Xf[L, i] = per_date[t][int(Iind[L]), int(Jind[L])]
    Xf = np.log(Xf)

    # --- covariance + localization + analysis ---
    CC = estimate_covariance(Xf)
    B, _ = localize_covariance(
        CC, Iind, Jind, centers, depb, loc['exclude'], loc['horizontal_m'], loc['vertical_m'],
        Tind=Tind, obstimes=times, RcorT=loc['temporal_years'])
    R = sigma ** 2 * np.eye(values.shape[0])
    Y = _perturbed_obs(y, M, sigma, perturb, rng)
    sol = analysis_update(B, obs_idx, Y, R, Xf)

    # --- physical coordinates of every state row and observation ---
    box_id = np.array([int(Iind[L]) + 1 for L in range(layout.n_state)], dtype=int)
    layer_id = np.array([int(Jind[L]) + 1 for L in range(layout.n_state)], dtype=int)
    state_time = np.array([times[0 if Tind is None else int(Tind[L])] for L in range(layout.n_state)], dtype=float)
    obs_time = np.concatenate([np.full(n, t) for n, t in zip(nobs_per_date, times)])

    result = AnalysisResult(
        target=target, member_ids=list(members), obs_indices=obs_idx, observations=y,
        log_states=sol, concentrations=np.exp(sol),
        geometric_mean=np.exp(np.mean(sol, axis=1)), arithmetic_mean=np.mean(np.exp(sol), axis=1),
        box_id=box_id, layer_id=layer_id, time=state_time,
        obs_box_id=brefnum.ravel().astype(int), obs_layer_id=breflev.ravel().astype(int), obs_time=obs_time,
        diagnostics={'Nobs': int(values.shape[0]), 'Nstate': int(layout.n_state), 'Nens': M,
                     'anl_ind_start': int(anl0), 'nobs_per_date': nobs_per_date,
                     'perturb': bool(perturb)},
    )
    payload = RestartPayload(bnlev=bnlev, write_index=write_index, anl_ind_start=anl0,
                             members=list(members), metas=metas, batches=batches, dates=dates)
    return result, payload


def write_restarts(result: AnalysisResult, payload: RestartPayload, ctx: RunContext) -> list:
    """Write each member's analysed Initconc.dat into the templates directory."""
    anl0 = payload.anl_ind_start
    sol = result.log_states
    written = []
    for i, member in enumerate(payload.members):
        header, footer, sed = payload.metas[i]
        ofname = ctx.template_dir(member) / 'Initconc.dat'
        write_poseidon(str(ofname), np.exp(sol[anl0:, i]), payload.write_index,
                       payload.bnlev, header, footer, sed)
        written.append(ofname)
    return written


def write_outputs(result: AnalysisResult, payload: RestartPayload, ctx: RunContext) -> None:
    """Write out_<date>.xlsx: obs sheet(s), one sheet per member, and Ens (geometric mean)."""
    for d, batch in zip(payload.dates, payload.batches):
        xlswrite(str(ctx.output_xlsx(d)),
                 np.concatenate((batch['values'], batch['brefnum']), axis=1), 'obs')
    obs_idx = result.obs_indices
    anl0 = payload.anl_ind_start
    n_last = payload.batches[-1]['values'].shape[0]
    sl = slice(anl0, anl0 + n_last)
    for i, member in enumerate(payload.members):
        xlswrite(str(ctx.output_xlsx(result.target)),
                 np.exp(result.log_states[obs_idx, i])[sl].reshape(-1, 1), member)
    xlswrite(str(ctx.output_xlsx(result.target)), result.geometric_mean[obs_idx][sl].reshape(-1, 1), 'Ens')
