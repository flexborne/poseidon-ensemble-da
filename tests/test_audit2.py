"""Tests for the second operational audit (output safety, config independence, publication,
PTY plumbing, stricter validation)."""
from __future__ import annotations

import sys

import pytest

from conftest import requires_data, EXAMPLE_CASE

from poseidon_da import api
from poseidon_da.config import load_config, validate
from poseidon_da.cycle import run_cycle



# ---- 2.1: an output directory may never overlap the dataset inputs ------------

def _fake_dataset(tmp_path):
    runs = tmp_path / 'runs'
    (runs / '1.0' / '01.hydrol1dep1').mkdir(parents=True)
    (runs / '1.0' / '01.hydrol1dep1' / 'lastconc.xls').write_text('precious\n')
    config = load_config(EXAMPLE_CASE)
    config.runs_root = runs
    return config, runs


@pytest.mark.parametrize('pick_output', [
    lambda runs: runs,                       # equal to the dataset
    lambda runs: runs / '1.0' / 'results',   # inside the dataset
    lambda runs: runs.parent,                # containing the dataset
])
def test_run_refuses_output_overlapping_dataset(tmp_path, pick_output):
    config, runs = _fake_dataset(tmp_path)
    with pytest.raises(ValueError, match='overlaps a protected input'):
        api.run(config, pick_output(runs), force=True)
    assert (runs / '1.0' / '01.hydrol1dep1' / 'lastconc.xls').read_text() == 'precious\n'


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
    assert first.localization is not second.localization                 # independent dicts
    assert isinstance(first.localization['exclude'], tuple)              # TOML list coerced to an immutable tuple
    assert isinstance(second.localization['exclude'], tuple)


# ---- 2.3: analyze publishes atomically, never a mix of two runs ---------------

@requires_data
def test_write_analysis_refuses_dirty_dir_without_force(tmp_path):
    config = load_config(EXAMPLE_CASE)
    out = tmp_path / 'a'
    api.write_analysis(config, out, perturb=False, seed=11)
    with pytest.raises(FileExistsError):
        api.write_analysis(config, out, perturb=False, seed=22)


@requires_data
def test_write_analysis_force_leaves_consistent_run(tmp_path):
    import json
    config = load_config(EXAMPLE_CASE)
    out = tmp_path / 'a'
    api.write_analysis(config, out, perturb=True, seed=11)
    api.write_analysis(config, out, perturb=True, seed=22, force=True)
    man = json.loads((out / 'manifest.json').read_text())
    assert man['perturbation']['seed'] == 22      # manifest and npz come from the same (second) run
    assert (out / 'analysis.npz').exists()


# ---- 2.4: the PTY runner releases its file descriptor ------------------------

def _open_fd_count():
    import os
    try:
        return len(os.listdir(f'/dev/fd'))
    except OSError:
        return len(os.listdir('/proc/self/fd'))


@requires_data
def test_unseeded_perturbation_is_recorded_and_reproducible(tmp_path):
    import json
    import numpy as np
    out = tmp_path / 'u'
    api.write_analysis(load_config(EXAMPLE_CASE), out, perturb=True)   # no seed -> one is materialized
    man = json.loads((out / 'manifest.json').read_text())
    seed = man['perturbation']['seed']
    assert isinstance(seed, int)
    archived = np.load(out / 'analysis.npz')['log_states']
    replay = api.analyze(load_config(EXAMPLE_CASE), perturb=True, seed=seed).log_states
    assert np.allclose(archived, replay)       # the recorded seed reproduces the perturbed analysis


@requires_data
def test_analyze_accepts_partial_localization():
    # setting a partial localization dict via the Python API must be completed, not KeyError
    import numpy as np
    config = load_config(EXAMPLE_CASE)
    config.localization = {'horizontal_m': 123000.0}
    assert validate(config) == []
    r = api.analyze(config, perturb=False)
    assert np.all(np.isfinite(r.log_states))


@requires_data
def test_diagnose_publishes_atomically(tmp_path):
    config = load_config(EXAMPLE_CASE)
    out = tmp_path / 'd'
    res = api.diagnose(config, out)
    assert res.status == 'ok' and (out / 'manifest.json').exists()
    with pytest.raises(FileExistsError):
        api.diagnose(config, out)


# ---- 2.4: the PTY runner releases its file descriptor ------------------------


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


# ---- 3: validation reports contract violations before computing -----------------

def test_load_config_rejects_unknown_key(tmp_path):
    case = tmp_path / 'case.toml'
    case.write_text('[case]\nschema_version = 1\ncase_id = "x"\n[analysis]\ngds = 2.0\n')
    with pytest.raises(ValueError, match='analysis.gds'):
        load_config(case)


def test_load_config_rejects_non_dict_table(tmp_path):
    # a table supplied as a scalar must be reported, not crash later with AttributeError
    case = tmp_path / 'case.toml'
    case.write_text('dataset = 5\n[case]\nschema_version = 1\ncase_id = "x"\n')   # top-level key first
    with pytest.raises(ValueError, match=r'\[dataset\] must be a table'):
        load_config(case)


def test_validate_rejects_negative_first_bdepth():
    config = load_config(EXAMPLE_CASE)
    config.bdepths = [-25, 100, 500, 2000]
    assert any('positive depth boundary' in p for p in validate(config))


def test_load_geometry_rejects_missing_columns(tmp_path):
    from openpyxl import Workbook
    from poseidon_da.geometry import load_geometry
    path = tmp_path / 'geom.xlsx'
    wb = Workbook(); ws = wb.active; ws.title = 'Sheet1'
    ws.append([1, 2, 3])                      # far fewer than the required columns
    wb.save(path)
    with pytest.raises(ValueError, match='columns'):
        load_geometry(path)


def _fake_cycle_dataset(tmp_path):
    config, runs = _fake_dataset(tmp_path)
    (runs / '2.0' / '01.hydrol1dep1').mkdir(parents=True)
    meas = tmp_path / 'meas'
    meas.mkdir()                                   # exists but empty -> diagnose skipped
    config.measurements_dir = meas
    return config, runs


def test_run_cycle_rejects_nonfinite_date(tmp_path):
    config, runs = _fake_cycle_dataset(tmp_path)
    with pytest.raises(ValueError, match='finite physical time'):
        run_cycle(config, tmp_path / 'out', mode='noda', dates=['1.0', 'NaN'],
                  model_runner=lambda *a: None)


def test_run_cycle_rejects_unordered_dates(tmp_path):
    config, runs = _fake_cycle_dataset(tmp_path)
    with pytest.raises(ValueError, match='strictly increasing'):
        run_cycle(config, tmp_path / 'out', mode='noda', dates=['2.0', '1.0'],
                  model_runner=lambda *a: None)


def test_run_cycle_applies_parameter_validation(tmp_path):
    # assimilation-parameter validation runs when the plan analyses (filter/smoother)
    config, runs = _fake_cycle_dataset(tmp_path)
    config.gsd = 1.0                                   # invalid: must be > 1
    with pytest.raises(ValueError, match='gsd'):
        run_cycle(config, tmp_path / 'out', mode='filter', dates=['1.0', '2.0'],
                  model_runner=lambda *a: None)


def test_noda_cycle_runs_without_measurements(tmp_path):
    # NODA has no analyze/needed assimilation inputs: a missing measurements dir must not block it
    config, runs = _fake_dataset(tmp_path)
    (runs / '2.0' / '01.hydrol1dep1').mkdir(parents=True)
    config.measurements_dir = tmp_path / 'nonexistent'
    res = run_cycle(config, tmp_path / 'out', mode='noda', dates=['1.0', '2.0'],
                    model_runner=lambda *a: None, perturb=False)
    assert res.status == 'ok'
    assert all(s['step'] != 'analyze' for s in res.steps)


def test_run_cycle_requires_window_states_present(tmp_path):
    # smoother first window needs future states; date 2.0 has a member dir but no lastconc
    config, runs = _fake_cycle_dataset(tmp_path)
    (runs / '3.0' / '01.hydrol1dep1').mkdir(parents=True)
    config.cycle_window_years = 1.0
    with pytest.raises(FileNotFoundError, match='member 01.hydrol1dep1'):
        run_cycle(config, tmp_path / 'out', mode='smoother', dates=['1.0', '2.0', '3.0'],
                  model_runner=lambda *a: None)


# ---- 4.5: log->physical conversion is stable and guarded ----------------------

def test_arithmetic_mean_is_overflow_safe():
    import numpy as np
    from poseidon_da.physical import arithmetic_mean
    log_states = np.full((1, 24), 709.0)          # naive mean(exp(x)) overflows to inf
    mean = arithmetic_mean(log_states)
    assert np.all(np.isfinite(mean)) and mean[0] > 0


def test_to_concentration_reports_member(tmp_path):
    import numpy as np
    from poseidon_da.physical import to_concentration
    log_states = np.zeros((2, 3))
    log_states[1, 2] = 1e9                         # overflows exp -> inf
    with pytest.raises(FloatingPointError, match='member m3'):
        to_concentration(log_states, member_ids=['m1', 'm2', 'm3'])


def test_to_concentration_allows_underflow_to_zero():
    import numpy as np
    from poseidon_da.physical import to_concentration
    conc = to_concentration(np.array([-10000.0, 0.0]))   # first underflows exp to 0.0
    assert conc[0] == 0.0 and np.all(np.isfinite(conc))


# ---- misc robustness fixes ----------------------------------------------------

def test_staged_publish_rejects_file_destination(tmp_path):
    from poseidon_da.publish import staged_publish
    dest = tmp_path / 'afile'
    dest.write_text('x')
    with pytest.raises(NotADirectoryError):
        with staged_publish(dest):
            pass


def test_staged_publish_restores_backup_on_rename_failure(tmp_path, monkeypatch):
    # exercise the actual crash-during-publish recovery: the previous run must survive intact
    import os
    from pathlib import Path
    from poseidon_da import publish
    dest = tmp_path / 'out'
    dest.mkdir()
    (dest / 'orig.txt').write_text('original')
    real_rename = os.rename

    def flaky(src, dst):
        if Path(src).name.startswith('.out.staging'):      # fail only the staging -> dest rename
            raise OSError('simulated rename failure')
        return real_rename(src, dst)

    monkeypatch.setattr(publish.os, 'rename', flaky)
    with pytest.raises(OSError, match='simulated'):
        with publish.staged_publish(dest, force=True) as staging:
            (staging / 'new.txt').write_text('new')
    assert (dest / 'orig.txt').read_text() == 'original'   # backup restored
    assert not (dest / 'new.txt').exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith('.out.')] == []  # no leftovers


def test_cli_validate_bad_case_returns_1(tmp_path):
    from poseidon_da import cli
    case = tmp_path / 'case.toml'
    case.write_text('[case]\nschema_version = 1\ncase_id = "x"\n[analysis]\ngsd = 0.5\n')
    assert cli.main(['validate', str(case)]) == 1   # invalid case -> non-zero exit, no traceback


def test_validate_diagnose_ignores_gsd_and_localization():
    # diagnose runs no Kalman update, so it must not require a valid gsd/localization
    from poseidon_da.config import validate_diagnose
    config = load_config(EXAMPLE_CASE)
    config.gsd = 1.0                                   # invalid for analysis, irrelevant to diagnose
    config.localization = {'horizontal_m': -1.0}       # invalid, irrelevant to diagnose
    problems = validate_diagnose(config, config.target_time)
    assert not any('gsd' in p or 'localization' in p for p in problems)


def test_run_executable_requires_exe_path(tmp_path):
    from poseidon_da.runner import run_executable_in_dir
    with pytest.raises(ValueError, match='no model executable'):
        run_executable_in_dir(tmp_path, exe_path=None)
