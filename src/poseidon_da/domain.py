"""Typed contracts passed between workflow steps. Plain dataclasses, no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import physical


@dataclass(frozen=True)
class StateLayout:
    """Maps physical (box, level, date) to the linear state vector and back.

    Lind[box, level, date] holds 1-based linear indices (0 = non-existent). Iind/Jind
    are 0-based box/level per state component; Tind is the 0-based date index.
    lind_extract maps the analysis-date grid to a 1-based index for writing a restart;
    anl_ind_start is the 0-based offset of the analysis-date block (0 for a single date).
    nobs_per_date keeps the observation counts so the state-space and observation-space
    index ranges are derived explicitly, not from a shared offset.
    """

    Lind: np.ndarray
    Iind: np.ndarray
    Jind: np.ndarray
    Tind: np.ndarray
    lind_extract: np.ndarray
    n_state: int
    anl_ind_start: int = 0
    nobs_per_date: tuple = ()

    @property
    def n_obs(self) -> int:
        return int(sum(self.nobs_per_date))

    @property
    def target_obs_slice(self) -> slice:
        """Rows of the stacked observation arrays belonging to the analysis (last) date."""
        prior = int(sum(self.nobs_per_date[:-1]))
        return slice(prior, self.n_obs)

    @property
    def target_state_slice(self) -> slice:
        """Rows of the state vector holding the full grid on the analysis (last) date."""
        return slice(self.anl_ind_start, self.n_state)


@dataclass(frozen=True)
class AnalysisWindow:
    """What to assimilate: the target (restart) date and all observation dates."""

    target_time: str
    observation_times: tuple

    def __post_init__(self):
        if self.target_time != self.observation_times[-1]:
            raise ValueError(
                'AnalysisWindow: target_time must be the last observation time '
                f'(got target={self.target_time!r}, times={self.observation_times!r})')


@dataclass
class AnalysisResult:
    """Structured result of one analysis step (no file side effects).

    Stores the analysis log-states plus explicit physical coordinates for each state
    row (box_id, layer_id, time) and each observation. Derived quantities
    (concentrations, geometric/arithmetic means) are computed on access.
    """

    target: str
    member_ids: list
    obs_indices: np.ndarray
    observations: np.ndarray        # (p,) assimilated values, log-space
    log_states: np.ndarray          # (n, M) per-member analysis, log-space
    box_id: np.ndarray              # (n,) 1-based box numbers
    layer_id: np.ndarray            # (n,) 1-based levels
    time: np.ndarray                # (n,) physical time of each state row
    obs_box_id: np.ndarray          # (p,)
    obs_layer_id: np.ndarray        # (p,)
    obs_time: np.ndarray            # (p,)
    diagnostics: dict = field(default_factory=dict)

    @property
    def concentrations(self) -> np.ndarray:
        return physical.to_concentration(self.log_states, self.member_ids)

    @property
    def geometric_mean(self) -> np.ndarray:
        return physical.geometric_mean(self.log_states)

    @property
    def arithmetic_mean(self) -> np.ndarray:
        return physical.arithmetic_mean(self.log_states)


@dataclass
class RunResult:
    """Outcome of a workflow run: per-step status, output paths and a manifest."""

    status: str
    steps: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
