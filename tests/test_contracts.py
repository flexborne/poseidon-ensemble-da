"""Input-contract tests: malformed state files and runner preconditions must fail loudly."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from conftest import HAS_DATA, a_lastconc

from poseidon_da.geometry import box_levels
from poseidon_da.poseidon_io import read_poseidon
from poseidon_da.runner import run_poseidon_all

requires_data = pytest.mark.skipif(not HAS_DATA, reason='experiment archive not present')
BNLEV = box_levels()


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
    arr, header, footer, sed = read_poseidon(str(a_lastconc()), BNLEV)
    assert np.all(np.isfinite(arr))


@requires_data
def test_read_poseidon_rejects_missing_box(tmp_path):
    lines = _data_lines(a_lastconc())
    out = [ln for ln in lines if not _is_box_row(ln)] + \
          [ln for ln in lines if _is_box_row(ln)][1:]
    bad = tmp_path / 'lastconc.xls'
    bad.write_text(''.join(out))
    with pytest.raises(ValueError, match='missing box'):
        read_poseidon(str(bad), BNLEV)


@requires_data
def test_read_poseidon_rejects_duplicate_box(tmp_path):
    lines = _data_lines(a_lastconc())
    first_box = next(ln for ln in lines if _is_box_row(ln))
    bad = tmp_path / 'lastconc.xls'
    bad.write_text(''.join(lines) + first_box)
    with pytest.raises(ValueError, match='duplicate'):
        read_poseidon(str(bad), BNLEV)


@requires_data
def test_read_poseidon_rejects_nonpositive_concentration(tmp_path):
    lines = _data_lines(a_lastconc())
    bad_lines, patched = [], False
    for ln in lines:
        if not patched and _is_box_row(ln):
            toks = ln.split()
            toks[1] = '-1.0'
            ln = ' '.join(toks) + '\n'
            patched = True
        bad_lines.append(ln)
    bad = tmp_path / 'lastconc.xls'
    bad.write_text(''.join(bad_lines))
    with pytest.raises(ValueError, match='non-positive'):
        read_poseidon(str(bad), BNLEV)


@requires_data
@pytest.mark.parametrize('bad_first_token', ['-1', '1.9'])
def test_read_poseidon_rejects_invalid_box_id(tmp_path, bad_first_token):
    lines = _data_lines(a_lastconc())
    bad = tmp_path / 'lastconc.xls'
    bad.write_text(''.join(lines) + f'{bad_first_token} 999 888 777\n')
    with pytest.raises(ValueError, match='invalid box id'):
        read_poseidon(str(bad), BNLEV)


@requires_data
def test_aggregate_rejects_mislabelled_time(tmp_path):
    from poseidon_da.observations import aggregate_observations
    from poseidon_da.paths import box_surfareas_with_centers_path, MEASUREMENTS_DIR
    src = (MEASUREMENTS_DIR / '1987.5.dat').read_text().splitlines()
    patched = [src[0]] + ['\t'.join(['1999'] + ln.split('\t')[1:]) for ln in src[1:] if ln.strip()]
    dat = tmp_path / '1987.5.dat'
    dat.write_text('\n'.join(patched) + '\n')
    with pytest.raises(ValueError, match='time column'):
        aggregate_observations(str(dat), str(box_surfareas_with_centers_path()),
                               [25, 100, 500, 2000], expected_time=1987.5)


@requires_data
def test_aggregate_rejects_nonpositive_raw(tmp_path):
    from poseidon_da.observations import aggregate_observations
    from poseidon_da.paths import box_surfareas_with_centers_path, MEASUREMENTS_DIR
    src = (MEASUREMENTS_DIR / '1987.5.dat').read_text().splitlines()
    rows = [ln for ln in src[1:] if ln.strip()]
    cols = rows[0].split('\t')
    cols[3] = '-1'
    rows[0] = '\t'.join(cols)
    dat = tmp_path / '1987.5.dat'
    dat.write_text('\n'.join([src[0]] + rows) + '\n')
    with pytest.raises(ValueError, match='non-positive'):
        aggregate_observations(str(dat), str(box_surfareas_with_centers_path()),
                               [25, 100, 500, 2000], expected_time=1987.5)


def test_run_poseidon_all_does_not_accept_stale_output(tmp_path):
    # A stale lastconc.xls must not be mistaken for a fresh forecast: the runner deletes
    # it first, so a no-op "model" leaves no output and the run fails.
    member = tmp_path / '01.hydrol1dep1'
    member.mkdir()
    (member / 'lastconc.xls').write_text('stale data, not a model output\n')
    dummy_exe = tmp_path / 'fake.exe'
    dummy_exe.write_text('')
    with pytest.raises(RuntimeError, match='no lastconc'):
        run_poseidon_all(tmp_path, exe_path=dummy_exe, launcher=[sys.executable, '-c', 'pass'])
    assert not (member / 'lastconc.xls').exists()   # stale output was removed


def test_run_executable_under_pty(tmp_path):
    # the pty path (for headless console apps / wine) runs and surfaces the exit code
    from poseidon_da.runner import run_executable_in_dir
    exe = tmp_path / 'fake.exe'
    exe.write_text('')
    run_executable_in_dir(tmp_path, exe_path=exe, launcher=[sys.executable, '-c', 'pass'], use_pty=True)
    with pytest.raises(RuntimeError):
        run_executable_in_dir(tmp_path, exe_path=exe,
                              launcher=[sys.executable, '-c', 'import sys; sys.exit(3)'], use_pty=True)


def test_run_poseidon_all_rejects_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_poseidon_all(tmp_path / 'does-not-exist')


def test_run_poseidon_all_rejects_empty_member_set(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError):
        run_poseidon_all(tmp_path / 'empty')
