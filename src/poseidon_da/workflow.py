"""Orchestration: assemble validated inputs, run the shared analysis, write outputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import analysis_update
from .covariance import estimate_covariance, localize_covariance
from .domain import AnalysisResult, AnalysisWindow, StateLayout
from .geometry import box_levels, layer_depths_from_bdepths, load_geometry
from .observations import aggregate_observations
from .physical import to_concentration
from .poseidon_io import read_state, write_poseidon, write_sheets
from .runner import RunContext, find_members
from .state import build_layout, observation_indices, validate_observation_coords

TIME_TOL = 1e-6


@dataclass
class WriteData:
    """File-format data the writers need beyond the AnalysisResult + StateLayout.

    Member order and all index mappings come from result.member_ids / layout; this only carries
    what is genuinely extra: the POSEIDON metadata per member and the raw observation batches.
    """

    bnlev: np.ndarray
    layout: StateLayout
    metas: list                # per member (result.member_ids order): (header, footer, sed)
    batches: list              # per observation date: aggregate_observations() dict
    dates: list


def _perturbed_obs(y, M, sigma, perturb, rng):
    """Observation ensemble Y = y (broadcast), optionally with Gaussian perturbations."""
    Y = np.tile(np.asarray(y, float).reshape(-1, 1), (1, M))
    if perturb:
        Y = Y + sigma * rng.standard_normal(Y.shape)
    return Y


def analyze(window: AnalysisWindow, ctx: RunContext, config, *, perturb=True, rng=None):
    """Run one analysis (a single-date window is the filter). Returns (AnalysisResult, WriteData).

    Reads its parameters from the already-finalized `config`; `rng` is required when perturb is True.
    """
    loc = config.localization
    depb = layer_depths_from_bdepths(config.bdepths)
    sigma = float(np.log(config.gsd))
    dates = list(window.observation_times)
    target = window.target_time
    members = find_members(ctx.date_dir(target))
    if not members:
        raise FileNotFoundError(f'No ensemble members for date {target}')
    M = len(members)
    bnlev = box_levels()
    geometry = load_geometry(config.geometry)           # read + validate the workbook once

    # observations (per date), time-checked and stacked into 1-D (p,) vectors
    times = [float(d) for d in dates]
    batches = [aggregate_observations(f'{config.measurements_dir}/{d}.dat', geometry,
                                      config.bdepths, expected_time=t, time_tol=TIME_TOL)
               for d, t in zip(dates, times)]
    nobs_per_date = [b['values'].shape[0] for b in batches]
    values = np.concatenate([b['values'] for b in batches])
    brefnum = np.concatenate([b['brefnum'] for b in batches])
    breflev = np.concatenate([b['breflev'] for b in batches])
    y = np.log(values)

    validate_observation_coords(brefnum, breflev, bnlev)      # before building the layout
    layout = build_layout(bnlev, brefnum, breflev, nobs_per_date)
    obs_idx = observation_indices(brefnum, breflev, layout)
    Iind, Jind, Tind = layout.Iind, layout.Jind, layout.Tind

    # ensemble background (log) + per-member restart metadata (analysis date), with time checks
    Xf = np.zeros((layout.n_state, M), dtype=float)
    metas = [None] * M
    for i, member in enumerate(members):
        states = []
        for d, t in zip(dates, times):
            arr, header, footer, sed = read_state(
                str(ctx.date_dir(d) / member / 'lastconc.xls'), bnlev, expected_time=t, time_tol=TIME_TOL)
            states.append(arr)
        metas[i] = (header, footer, sed)                 # loop leaves header/footer/sed on the last date
        Xf[:, i] = np.stack(states)[Tind, Iind, Jind]    # gather (date, box, level) per state row
    Xf = np.log(Xf)

    # covariance -> localization -> analysis
    B = localize_covariance(estimate_covariance(Xf), Iind, Jind, geometry.centers, depb,
                            loc['exclude'], loc['horizontal_m'], loc['vertical_m'],
                            Tind=Tind, obstimes=times, RcorT=loc['temporal_years'])
    R = sigma ** 2 * np.eye(values.shape[0])
    sol = analysis_update(B, obs_idx, _perturbed_obs(y, M, sigma, perturb, rng), R, Xf)

    result = AnalysisResult(
        target=target, member_ids=list(members), obs_indices=obs_idx, observations=y, log_states=sol,
        box_id=(Iind + 1).astype(int), layer_id=(Jind + 1).astype(int),
        time=np.asarray(times, float)[Tind],
        obs_box_id=brefnum.astype(int), obs_layer_id=breflev.astype(int),
        obs_time=np.repeat(times, nobs_per_date),
        diagnostics={'Nobs': int(values.shape[0]), 'Nstate': int(layout.n_state), 'Nens': M,
                     'anl_ind_start': int(layout.anl_ind_start), 'nobs_per_date': nobs_per_date,
                     'perturb': bool(perturb)},
    )
    return result, WriteData(bnlev=bnlev, layout=layout, metas=metas, batches=batches, dates=dates)


def write_restarts(result: AnalysisResult, data: WriteData, ctx: RunContext) -> list:
    """Write each member's analysed Initconc.dat into the templates directory."""
    conc = to_concentration(result.log_states[data.layout.target_state_slice], result.member_ids)
    written = []
    for i, member in enumerate(result.member_ids):
        header, footer, sed = data.metas[i]
        ofname = ctx.template_dir(member) / 'Initconc.dat'
        write_poseidon(str(ofname), conc[:, i], data.layout.lind_extract, data.bnlev, header, footer, sed)
        written.append(ofname)
    return written


def write_outputs(result: AnalysisResult, data: WriteData, ctx: RunContext) -> None:
    """Write out_<date>.xlsx: obs sheet(s), and for the target date one sheet per member + Ens."""
    obs_idx, sl = result.obs_indices, data.layout.target_obs_slice   # explicit observation-space slice
    conc_at_obs = result.concentrations[obs_idx]                     # guarded exp once for the whole grid
    for d, batch in zip(data.dates, data.batches):
        sheets = {'obs': np.column_stack((batch['values'], batch['brefnum']))}
        if d == result.target:
            for i, member in enumerate(result.member_ids):
                sheets[member] = conc_at_obs[sl, i]
            sheets['Ens'] = result.geometric_mean[obs_idx][sl]
        write_sheets(str(ctx.output_xlsx(d)), sheets)


def write_forecast_diagnostics(config, ctx: RunContext, date, bnlev=None):
    """Forecast-vs-observation comparison at `date` (no assimilation): write out_<date>.xlsx
    with an obs sheet, a sim column per member and the arithmetic mean. Used by diagnose + cycle."""
    if bnlev is None:
        bnlev = box_levels()
    geometry = load_geometry(config.geometry)
    batch = aggregate_observations(str(config.measurements_dir / f'{date}.dat'), geometry,
                                   config.bdepths, expected_time=float(date))
    obs, brn, brl = batch['values'], batch['brefnum'].astype(int), batch['breflev'].astype(int)
    members = find_members(ctx.date_dir(date))
    ens = np.zeros(obs.shape[0])
    sheets = {'obs': np.column_stack((obs, batch['brefnum']))}
    for m in members:
        arr, _, _, _ = read_state(str(ctx.date_dir(date) / m / 'lastconc.xls'), bnlev, expected_time=float(date))
        sim = arr[brn - 1, brl - 1]
        ens += sim
        sheets[m] = sim
    sheets['arithmetic_mean'] = ens / np.single(len(members))
    write_sheets(str(ctx.output_xlsx(date)), sheets)
    return obs, members
