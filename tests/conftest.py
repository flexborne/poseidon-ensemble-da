"""Shared test fixtures. The package is imported as installed (pip install -e .)."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REF = Path(__file__).resolve().parent / 'reference'
EXAMPLE_CASE = ROOT / 'examples' / 'black-sea-cs137.toml'
DATA_ROOT = ROOT / 'DAruns' / '01.RunKFS'


def _first_lastconc():
    """One real lastconc.xls, or None. Keys HAS_DATA on actual data (lastconc is
    gitignored), so data-dependent tests skip cleanly on a fresh clone / in CI."""
    for date in ('1986.75', '1987.5', '1988.5'):
        dd = DATA_ROOT / date
        if dd.is_dir():
            for p in sorted(dd.glob('*hydrol*dep*')):
                f = p / 'lastconc.xls'
                if f.exists():
                    return f
    return None


_LASTCONC = _first_lastconc()
HAS_DATA = _LASTCONC is not None

requires_data = pytest.mark.skipif(not HAS_DATA, reason='experiment archive not present')


def a_lastconc():
    return _LASTCONC


@pytest.fixture
def isolated_ctx(tmp_path):
    from poseidon_da.runner import RunContext
    return RunContext.isolated(tmp_path, runs_root=DATA_ROOT)
