"""Tests for the second operational audit (output safety, config independence, publication,
PTY plumbing, stricter validation)."""
from __future__ import annotations

import sys

import pytest

from conftest import EXAMPLE_CASE

from poseidon_da import api
from poseidon_da.config import load_config


# ---- 2.1: an output directory may never overlap the dataset inputs ------------

def _fake_dataset(tmp_path):
    runs = tmp_path / 'runs'
    (runs / '1.0' / '01.hydrol1dep1').mkdir(parents=True)
    (runs / '1.0' / '01.hydrol1dep1' / 'lastconc.xls').write_text('precious\n')
    config = load_config(EXAMPLE_CASE)
    config.runs_root = runs
    return config, runs


def test_run_refuses_output_equal_to_dataset(tmp_path):
    config, runs = _fake_dataset(tmp_path)
    with pytest.raises(ValueError, match='overlaps a protected input'):
        api.run(config, runs, force=True)
    assert (runs / '1.0' / '01.hydrol1dep1' / 'lastconc.xls').read_text() == 'precious\n'


def test_run_refuses_output_inside_dataset(tmp_path):
    config, runs = _fake_dataset(tmp_path)
    with pytest.raises(ValueError, match='overlaps a protected input'):
        api.run(config, runs / '1.0' / 'results', force=True)


def test_run_refuses_output_containing_dataset(tmp_path):
    config, runs = _fake_dataset(tmp_path)
    with pytest.raises(ValueError, match='overlaps a protected input'):
        api.run(config, runs.parent, force=True)


def test_run_cycle_refuses_output_equal_to_dataset(tmp_path):
    from poseidon_da.cycle import run_cycle
    config, runs = _fake_dataset(tmp_path)
    (runs / '2.0' / '01.hydrol1dep1').mkdir(parents=True)
    config.measurements_dir = tmp_path
    with pytest.raises(ValueError, match='overlaps a protected input'):
        run_cycle(config, runs, mode='noda', dates=['1.0', '2.0'],
                  model_runner=lambda *a: None, force=True)


# ---- 2.2: independently loaded configs must not share mutable localization ----

def test_configs_do_not_share_localization_exclude():
    first = load_config(EXAMPLE_CASE)
    second = load_config(EXAMPLE_CASE)
    assert first.localization is not second.localization
    # exclude is an immutable tuple, so it cannot be mutated in place at all
    assert isinstance(first.localization['exclude'], tuple)
    assert isinstance(second.localization['exclude'], tuple)


def test_default_localization_exclude_is_immutable():
    from poseidon_da.config import normalize_localization
    a = normalize_localization()
    b = normalize_localization()
    assert a is not b and a['exclude'] == b['exclude']
    with pytest.raises(AttributeError):
        a['exclude'].append(4)        # tuple has no append; cannot corrupt the shared default


# ---- 2.4: the PTY runner releases its file descriptor ------------------------

def _open_fd_count():
    import os
    try:
        return len(os.listdir(f'/dev/fd'))
    except OSError:
        return len(os.listdir('/proc/self/fd'))


@pytest.mark.skipif(sys.platform == 'win32', reason='pty is POSIX-only')
def test_run_in_pty_does_not_leak_fds(tmp_path):
    from poseidon_da.runner import run_executable_in_dir
    exe = tmp_path / 'fake.exe'
    exe.write_text('')
    before = _open_fd_count()
    for _ in range(8):
        run_executable_in_dir(tmp_path, exe_path=exe,
                              launcher=[sys.executable, '-c', 'pass'], use_pty=True)
    after = _open_fd_count()
    assert after <= before + 1       # no monotonic growth across repeated runs
