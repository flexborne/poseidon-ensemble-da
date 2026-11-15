"""Input-contract tests: malformed state files and runner preconditions must fail loudly."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import HAS_DATA, a_lastconc

requires_data = pytest.mark.skipif(not HAS_DATA, reason='experiment archive not present')


def _data_lines(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        return fh.readlines()


def _is_box_row(line, nbox=50):
    toks = line.split()
    if not toks:
        return False
    try:
        return 1 <= int(float(toks[0])) <= nbox
    except ValueError:
        return False


@requires_data
def test_read_poseidon_accepts_valid_file():
    from read_poseidon import read_poseidon
    arr, header, footer, sed = read_poseidon(str(a_lastconc()))
    assert np.all(np.isfinite(arr))


@requires_data
def test_read_poseidon_rejects_missing_box(tmp_path):
    from read_poseidon import read_poseidon
    lines = _data_lines(a_lastconc())
    out = [ln for ln in lines if not _is_box_row(ln)] + \
          [ln for ln in lines if _is_box_row(ln)][1:]  # drop the first box row
    bad = tmp_path / 'lastconc.xls'
    bad.write_text(''.join(out))
    with pytest.raises(ValueError, match='missing box'):
        read_poseidon(str(bad))


@requires_data
def test_read_poseidon_rejects_duplicate_box(tmp_path):
    from read_poseidon import read_poseidon
    lines = _data_lines(a_lastconc())
    first_box = next(ln for ln in lines if _is_box_row(ln))
    bad = tmp_path / 'lastconc.xls'
    bad.write_text(''.join(lines) + first_box)   # append a duplicate box row
    with pytest.raises(ValueError, match='duplicate'):
        read_poseidon(str(bad))


@requires_data
def test_read_poseidon_rejects_nonpositive_concentration(tmp_path):
    from read_poseidon import read_poseidon
    lines = _data_lines(a_lastconc())
    bad_lines = []
    patched = False
    for ln in lines:
        if not patched and _is_box_row(ln):
            toks = ln.split()
            toks[1] = '-1.0'                      # make an active concentration negative
            ln = ' '.join(toks) + '\n'
            patched = True
        bad_lines.append(ln)
    bad = tmp_path / 'lastconc.xls'
    bad.write_text(''.join(bad_lines))
    with pytest.raises(ValueError, match='non-positive'):
        read_poseidon(str(bad))


def test_run_poseidon_all_rejects_missing_dir(tmp_path):
    from poseidon_runtime import run_poseidon_all
    with pytest.raises(FileNotFoundError):
        run_poseidon_all(tmp_path / 'does-not-exist')


def test_run_poseidon_all_rejects_empty_member_set(tmp_path):
    from poseidon_runtime import run_poseidon_all
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError):
        run_poseidon_all(tmp_path / 'empty')
