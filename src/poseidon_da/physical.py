"""The single log-space -> physical-space conversion, with finite guards and stable statistics."""
from __future__ import annotations

import numpy as np


def to_concentration(log_states, member_ids=None) -> np.ndarray:
    """exp(log_states) with a finiteness guard (one place for every writer/exporter).

    exp is always >= 0, so only non-finite (overflow) values are rejected; a near-zero
    concentration (underflow of a very negative log-state) is valid. Raises FloatingPointError
    naming the offending state row (and member id/label for a 2D array) on overflow.
    """
    x = np.asarray(log_states, dtype=float)
    conc = np.exp(x)
    bad = ~np.isfinite(conc)
    if np.any(bad):
        idx = np.argwhere(bad)[0]
        if x.ndim == 2:
            row, col = int(idx[0]), int(idx[1])
            who = member_ids[col] if member_ids is not None else col
            raise FloatingPointError(
                f'non-representable concentration at state row {row}, member {who}: {conc[row, col]}')
        row = int(idx[0])
        raise FloatingPointError(f'non-representable concentration at state row {row}: {conc[row]}')
    return conc


def _finite(values, what) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        row = int(np.argwhere(~np.isfinite(values))[0][0])
        raise FloatingPointError(f'non-representable {what} at state row {row}: {values[row]}')
    return values


def geometric_mean(log_states) -> np.ndarray:
    """Per-row geometric mean across members, exp(mean of log-states), guarded for finiteness."""
    return _finite(np.exp(np.mean(np.asarray(log_states, dtype=float), axis=1)), 'geometric mean')


def arithmetic_mean(log_states) -> np.ndarray:
    """Per-row arithmetic mean of concentrations across members, overflow-safe and guarded.

    mean(exp(x)) == exp(m) * mean(exp(x - m)) with m = row max, so a row of large log-values
    no longer overflows an intermediate sum before the division.
    """
    x = np.asarray(log_states, dtype=float)
    m = x.max(axis=1, keepdims=True)
    return _finite(np.exp(m).ravel() * np.mean(np.exp(x - m), axis=1), 'arithmetic mean')
