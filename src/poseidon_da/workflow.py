"""Orchestration: assemble validated inputs, run the shared analysis, write outputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import analysis_update
from .config import DEFAULT_BDEPTHS, normalize_localization
from .covariance import estimate_covariance, localize_covariance
from .domain import AnalysisResult, AnalysisWindow
from .geometry import box_levels, layer_depths_from_bdepths, load_box_centers
from .observations import aggregate_observations
from .poseidon_io import read_state, write_poseidon, xlswrite
from .runner import RunContext, find_members
from .state import build_layout, observation_indices, validate_observation_coords

TIME_TOL = 1e-6


@dataclass
class RestartPayload:
    """What write_restarts/write_outputs need, with member order kept explicit."""

    bnlev: np.ndarray
    write_index: np.ndarray
    anl_ind_start: int
    members: list
    metas: list                # per member: (header, footer, sed)
    batches: list
    dates: list


def _perturbed_obs(y, M, sigma, perturb, rng):
    """Observation ensemble Y = y (broadcast), optionally with Gaussian perturbations."""
    Y = np.tile(np.asarray(y, float).reshape(-1, 1), (1, M))
    if perturb:
        Y = Y + sigma * rng.standard_normal(Y.shape)
    return Y


def analyze(window: AnalysisWindow, ctx: RunContext, *, geometry_path, measurements_dir,
            gsd=1.3, localization=None, bdepths=None, perturb=True, rng=None):
    """Run one analysis (a single-date window is the filter). Returns (AnalysisResult, RestartPayload)."""
    rng = rng if rng is not None else np.random.default_rng()
    loc = normalize_localization(localization)
    bdepths = list(bdepths) if bdepths is not None else list(DEFAULT_BDEPTHS)
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

    # observations (per date), time-checked and stacked
    batches, times = [], []
    for d in dates:
        batches.append(aggregate_observations(f'{measurements_dir}/{d}.dat', geometry_path,
                                              bdepths, expected_time=float(d), time_tol=TIME_TOL))
        times.append(float(d))
    nobs_per_date = [b['values'].shape[0] for b in batches]
    values = np.vstack([b['values'] for b in batches])
    brefnum = np.vstack([b['brefnum'] for b in batches]).ravel()
    breflev = np.vstack([b['breflev'] for b in batches]).ravel()
    y = np.log(values).ravel()

    validate_observation_coords(brefnum, breflev, bnlev)      # before building the layout
    layout = build_layout(bnlev, brefnum, breflev, nobs_per_date)
    obs_idx = observation_indices(brefnum, breflev, layout)
    anl0 = layout.anl_ind_start
    Iind, Jind, Tind = layout.Iind, layout.Jind, layout.Tind

    # ensemble background (log) + per-member restart metadata (analysis date), with time checks
    Xf = np.zeros((layout.n_state, M), dtype=float)
    metas = [None] * M
    for i, member in enumerate(members):
        per_date = {}
        for k, d in enumerate(dates):
            arr, header, footer, sed = read_state(
                str(ctx.date_dir(d) / member / 'lastconc.xls'), bnlev, expected_time=float(d), time_tol=TIME_TOL)
            per_date[k] = arr
            if k == len(dates) - 1:
                metas[i] = (header, footer, sed)
        for L in range(layout.n_state):
            Xf[L, i] = per_date[int(Tind[L])][int(Iind[L]), int(Jind[L])]
    Xf = np.log(Xf)

    # covariance -> localization -> analysis
    B, _ = localize_covariance(estimate_covariance(Xf), Iind, Jind, centers, depb,
                               loc['exclude'], loc['horizontal_m'], loc['vertical_m'],
                               Tind=Tind, obstimes=times, RcorT=loc['temporal_years'])
    R = sigma ** 2 * np.eye(values.shape[0])
    sol = analysis_update(B, obs_idx, _perturbed_obs(y, M, sigma, perturb, rng), R, Xf)

    result = AnalysisResult(
        target=target, member_ids=list(members), obs_indices=obs_idx, observations=y, log_states=sol,
        box_id=(Iind + 1).astype(int), layer_id=(Jind + 1).astype(int),
        time=np.array([times[t] for t in Tind], float),
        obs_box_id=brefnum.astype(int), obs_layer_id=breflev.astype(int),
        obs_time=np.concatenate([np.full(n, t) for n, t in zip(nobs_per_date, times)]),
        diagnostics={'Nobs': int(values.shape[0]), 'Nstate': int(layout.n_state), 'Nens': M,
                     'anl_ind_start': int(anl0), 'nobs_per_date': nobs_per_date, 'perturb': bool(perturb)},
    )
    payload = RestartPayload(bnlev=bnlev, write_index=layout.lind_extract, anl_ind_start=anl0,
                             members=list(members), metas=metas, batches=batches, dates=dates)
    return result, payload


def write_restarts(result: AnalysisResult, payload: RestartPayload, ctx: RunContext) -> list:
    """Write each member's analysed Initconc.dat into the templates directory."""
    sol, anl0 = result.log_states, payload.anl_ind_start
    written = []
    for i, member in enumerate(payload.members):
        header, footer, sed = payload.metas[i]
        conc = np.exp(sol[anl0:, i])
        if not np.all(np.isfinite(conc) & (conc > 0.0)):
            bad = int(np.where(~(np.isfinite(conc) & (conc > 0.0)))[0][0])
            raise FloatingPointError(
                f'member {member}: non-finite/non-positive analysed concentration at state row {bad} '
                f'(value {conc[bad]}) — refusing to write restart')
        ofname = ctx.template_dir(member) / 'Initconc.dat'
        write_poseidon(str(ofname), conc, payload.write_index, payload.bnlev, header, footer, sed)
        written.append(ofname)
    return written


def write_outputs(result: AnalysisResult, payload: RestartPayload, ctx: RunContext) -> None:
    """Write out_<date>.xlsx: obs sheet(s), one sheet per member, and Ens (geometric mean)."""
    for d, batch in zip(payload.dates, payload.batches):
        xlswrite(str(ctx.output_xlsx(d)), np.concatenate((batch['values'], batch['brefnum']), axis=1), 'obs')
    obs_idx, anl0 = result.obs_indices, payload.anl_ind_start
    sl = slice(anl0, anl0 + payload.batches[-1]['values'].shape[0])
    for i, member in enumerate(payload.members):
        xlswrite(str(ctx.output_xlsx(result.target)), np.exp(result.log_states[obs_idx, i])[sl].reshape(-1, 1), member)
    xlswrite(str(ctx.output_xlsx(result.target)), result.geometric_mean[obs_idx][sl].reshape(-1, 1), 'Ens')


def write_forecast_diagnostics(config, ctx: RunContext, date, bnlev=None):
    """Compare the ensemble forecast at `date` against observations (no assimilation).

    Writes out_<date>.xlsx with an obs sheet, one sim column per member, and the
    arithmetic ensemble mean. Shared by the `diagnose` command and the cycle.
    """
    if bnlev is None:
        bnlev = box_levels()
    batch = aggregate_observations(str(config.measurements_dir / f'{date}.dat'), str(config.geometry),
                                   config.bdepths, expected_time=float(date))
    obs, brn, brl = batch['values'], batch['brefnum'], batch['breflev']
    members = find_members(ctx.date_dir(date))
    No = obs.shape[0]
    ens = np.zeros((No, 1))
    xlswrite(str(ctx.output_xlsx(date)), np.concatenate((obs, brn), axis=1), 'obs')
    for m in members:
        arr, _, _, _ = read_state(str(ctx.date_dir(date) / m / 'lastconc.xls'), bnlev, expected_time=float(date))
        sim = np.array([[arr[int(brn[L, 0]) - 1, int(brl[L, 0]) - 1]] for L in range(No)])
        ens += sim
        xlswrite(str(ctx.output_xlsx(date)), sim, m)
    xlswrite(str(ctx.output_xlsx(date)), ens / np.single(len(members)), 'arithmetic_mean')
    return obs, members
