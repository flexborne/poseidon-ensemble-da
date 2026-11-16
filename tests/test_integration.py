"""Integration tests on a SYNTHETIC dataset built in a temp dir.

Unlike the data-gated tests, these need no experiment archive, so they exercise the analysis,
the atomic publish and the full cycle orchestration in CI. The dataset is a tiny but complete
POSEIDON setup: a 50-box geometry workbook, measurement files, and a runs tree whose
lastconc.xls files are generated with the package's own writer (so read_poseidon accepts them).
"""
from __future__ import annotations

import numpy as np
import pytest
from openpyxl import Workbook

from poseidon_da import api
from poseidon_da.config import load_config
from poseidon_da.geometry import box_levels
from poseidon_da.poseidon_io import read_poseidon, write_poseidon
from poseidon_da.runner import find_members
from poseidon_da.state import build_layout

DATES = ['2000.0', '2000.5']
MEMBERS = ['01.hydrol1dep1', '02.hydrol1dep2']
OBS_BOXES = [10, 20, 30]
BNLEV = box_levels()


def _write_geometry(path):
    wb = Workbook(); ws = wb.active; ws.title = 'Sheet1'
    for b in range(1, len(BNLEV)):                       # one row per box; boxes laid on a lon strip
        row = [b, 0, 0, 0, 0.0, 1.0, float(b - 1), float(b), int(BNLEV[b]), 0.5, float(b) - 0.5]
        ws.append(row)
    wb.save(path)


def _write_measurements(path, time):
    lines = ['time\tlon\tlat\tval']
    for b in OBS_BOXES:                                  # a point squarely inside box b (level 1)
        lines.append(f'{time}\t{b - 0.5}\t0.5\t{1.0 + 0.01 * b}')
    path.write_text('\n'.join(lines) + '\n')


def _write_lastconc(path, time, fill):
    layout = build_layout(BNLEV)                         # single-date full grid
    vect = np.full(layout.n_state, fill, dtype=float)
    header = [{'tline': f'# header {k}'} for k in range(6)]
    footer = [{'tline': f'   {time:.11f}     '}]
    sed = np.zeros((len(BNLEV) - 1, 2))
    write_poseidon(str(path), vect, layout.lind_extract, BNLEV, header, footer, sed)


@pytest.fixture(scope='module')
def mini_case(tmp_path_factory):
    base = tmp_path_factory.mktemp('mini')
    (base / 'config').mkdir(); (base / 'meas').mkdir()
    _write_geometry(base / 'config' / 'geom.xlsx')
    for d in DATES:
        _write_measurements(base / 'meas' / f'{d}.dat', float(d))
        for i, m in enumerate(MEMBERS):
            member = base / 'runs' / d / m
            member.mkdir(parents=True)
            _write_lastconc(member / 'lastconc.xls', float(d), fill=1.0 + 0.1 * i)
    toml = base / 'case.toml'
    toml.write_text(
        '[case]\nschema_version = 1\ncase_id = "mini"\n'
        '[dataset]\n'
        f'measurements_dir = "{base / "meas"}"\n'
        f'geometry = "{base / "config" / "geom.xlsx"}"\n'
        f'runs_root = "{base / "runs"}"\n'
        '[analysis]\n'
        'target_time = "2000.0"\nobservation_times = ["2000.0"]\ngsd = 1.3\n'
        'bdepths = [25, 100, 500, 2000]\n'
        '[localization]\nexclude = []\n'
        '[cycle]\nmode = "filter"\ndates = ["2000.0", "2000.5"]\nwindow_years = 0.25\n')
    return load_config(toml)


def _mock_model(date_dir, expected_time, _bnlev):
    """Stand in for POSEIDON: turn each member's Initconc.dat into a lastconc.xls at the forecast time."""
    for m in find_members(date_dir):
        arr, header, footer, sed = read_poseidon(str(date_dir / m / 'Initconc.dat'), BNLEV)
        layout = build_layout(BNLEV)
        vect = np.array([arr[int(layout.Iind[L]), int(layout.Jind[L])] for L in range(layout.n_state)])
        footer = [it for it in footer if len(it['tline'].split()) != 1]
        footer.append({'tline': f'   {expected_time:.11f}     '})
        write_poseidon(str(date_dir / m / 'lastconc.xls'), vect, layout.lind_extract, BNLEV, header, footer, sed)


def test_mini_case_validates(mini_case):
    from poseidon_da.config import validate
    assert validate(mini_case) == []


def test_run_writes_restarts_outputs_manifest(mini_case, tmp_path):
    import json
    out = tmp_path / 'run'
    res = api.run(mini_case, out, perturb=False)
    assert res.status == 'ok'
    restarts = sorted((out / 'templates').glob('*hydrol*dep*/Initconc.dat'))
    assert len(restarts) == len(MEMBERS)
    arr, _, _, _ = read_poseidon(str(restarts[0]), BNLEV)   # restart content is a valid state
    assert np.all(np.isfinite(arr))
    man = json.loads((out / 'manifest.json').read_text())
    assert man['operation'] == 'run' and man['target_time'] == '2000.0'
    assert man['inputs']['geometry']['sha256']


def test_run_cycle_filter_with_mock_model(mini_case, tmp_path):
    res = api.run_cycle(mini_case, tmp_path / 'cyc', mode='filter', model_runner=_mock_model, perturb=False)
    assert res.status == 'ok'
    kinds = [(s['step'], s['date']) for s in res.steps]
    assert kinds == [('analyze', '2000.0'), ('forecast', '2000.5'), ('diagnose', '2000.5')]
    # the analyze step records a hash of the states it actually used (not only the archive)
    analyze = next(s for s in res.steps if s['step'] == 'analyze')
    assert set(analyze['inputs_sha256']) == {'2000.0'} and analyze['inputs_sha256']['2000.0']


def test_cycle_atomic_on_model_failure(mini_case, tmp_path):
    def boom(*a):
        raise RuntimeError('model blew up')
    out = tmp_path / 'cyc'
    with pytest.raises(RuntimeError, match='model blew up'):
        api.run_cycle(mini_case, out, mode='filter', model_runner=boom, perturb=False)
    assert not out.exists()                                  # nothing published on failure
