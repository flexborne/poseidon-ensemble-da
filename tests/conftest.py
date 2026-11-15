"""Shared test fixtures: put src/ on the path and locate optional run data."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
REF = Path(__file__).resolve().parent / 'reference'
sys.path.insert(0, str(SRC))

# The large experiment archive is optional; data-dependent tests skip without it.
DATA_ROOT = ROOT / 'DAruns' / '01.RunKFS'
HAS_DATA = (DATA_ROOT / '1986.75').is_dir() and (DATA_ROOT / '1988.5').is_dir()


@pytest.fixture
def isolated_ctx(tmp_path):
    from run_context import RunContext
    return RunContext.isolated(tmp_path)


def a_lastconc():
    """Path to one real lastconc.xls, or None if the archive is absent."""
    if not HAS_DATA:
        return None
    for p in sorted((DATA_ROOT / '1986.75').glob('*hydrol*dep*')):
        f = p / 'lastconc.xls'
        if f.exists():
            return f
    return None
