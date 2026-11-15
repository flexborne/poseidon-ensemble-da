"""Shared test fixtures. The package is imported as installed (pip install -e .)."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REF = Path(__file__).resolve().parent / 'reference'
EXAMPLE_CASE = ROOT / 'examples' / 'black-sea-cs137.toml'

# The large experiment archive is optional; data-dependent tests skip without it.
DATA_ROOT = ROOT / 'DAruns' / '01.RunKFS'
HAS_DATA = (DATA_ROOT / '1986.75').is_dir() and (DATA_ROOT / '1988.5').is_dir()


@pytest.fixture
def isolated_ctx(tmp_path):
    from poseidon_da.runner import RunContext
    return RunContext.isolated(tmp_path, runs_root=DATA_ROOT)


def a_lastconc():
    """Path to one real lastconc.xls, or None if the archive is absent."""
    if not HAS_DATA:
        return None
    for p in sorted((DATA_ROOT / '1986.75').glob('*hydrol*dep*')):
        f = p / 'lastconc.xls'
        if f.exists():
            return f
    return None
