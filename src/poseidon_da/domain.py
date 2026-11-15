"""Typed contracts passed between workflow steps. Plain dataclasses, no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class StateLayout:
    """Maps physical (box, level[, date]) to the linear state vector and back.

    Lind holds 1-based linear indices (0 = non-existent component). Iind/Jind are
    0-based box/level indices per state component; Tind is the 0-based date index
    (window layouts only). lind_extract maps the analysis-date grid to a 1-based
    index used when writing a restart file; anl_ind_start is the 0-based offset of
    the analysis-date block in the full state (0 for a single date).
    """

    Lind: np.ndarray
    Iind: np.ndarray
    Jind: np.ndarray
    n_state: int
    Tind: np.ndarray | None = None
    lind_extract: np.ndarray | None = None
    anl_ind_start: int = 0


@dataclass
class ObservationBatch:
    """Aggregated observations for one or more dates (box-averaged, pre-log)."""

    values: np.ndarray        # (p, 1) raw concentration aggregates
    brefnum: np.ndarray       # (p, 1) 1-based box ids
    breflev: np.ndarray       # (p, 1) 1-based levels
    nobs_per_date: list       # observations per date, in date order
    box_vs_allobs: np.ndarray
    sigc: np.ndarray          # (p, 1) half-range diagnostic (not used for R)


@dataclass
class ForecastEnsemble:
    """Background ensemble of log-states with member identity and provenance."""

    log_states: np.ndarray    # (n_state, M)
    member_ids: list
    times: list               # physical time(s) of the forecast
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisWindow:
    """What to assimilate: the target (restart) date and all observation dates."""

    target_time: str
    observation_times: tuple

    @property
    def is_filter(self) -> bool:
        return len(self.observation_times) == 1

    def __post_init__(self):
        if self.target_time != self.observation_times[-1]:
            raise ValueError(
                'AnalysisWindow: target_time must be the last observation time '
                f'(got target={self.target_time!r}, times={self.observation_times!r})'
            )


@dataclass
class AnalysisResult:
    """Structured result of one analysis step (no file side effects)."""

    target: str
    member_ids: list
    obs_indices: np.ndarray
    observations: np.ndarray      # (p,) assimilated values, log-space
    log_states: np.ndarray        # (n, M) per-member analysis, log-space
    concentrations: np.ndarray    # (n, M) = exp(log_states)
    geometric_mean: np.ndarray    # (n,) = exp(mean(log_states, axis=1))
    diagnostics: dict = field(default_factory=dict)


@dataclass
class RunResult:
    """Outcome of a workflow run: per-step status, output paths and a manifest."""

    status: str
    steps: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
