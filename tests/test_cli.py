"""Config / API / CLI tests on the bundled example case."""
from __future__ import annotations

import pytest

from conftest import EXAMPLE_CASE, HAS_DATA

from poseidon_da import api
from poseidon_da.config import load_config, validate
from poseidon_da.domain import AnalysisWindow

requires_data = pytest.mark.skipif(not HAS_DATA, reason='experiment archive not present')


def test_analysis_window_requires_target_last():
    AnalysisWindow('1988.5', ('1988.625', '1988.5'))          # ok
    with pytest.raises(ValueError):
        AnalysisWindow('1988.625', ('1988.625', '1988.5'))    # target not last


@requires_data
def test_example_case_validates():
    config = load_config(EXAMPLE_CASE)
    assert validate(config) == []


@requires_data
def test_api_analyze_runs(tmp_path):
    config = load_config(EXAMPLE_CASE)
    result = api.analyze(config, perturb=False, output_dir=tmp_path)
    assert result.target == '1988.5'
    assert result.log_states.shape[1] == 24


@requires_data
def test_api_run_writes_outputs_and_manifest(tmp_path):
    import json
    config = load_config(EXAMPLE_CASE)
    out = tmp_path / 'run'
    res = api.run(config, out, perturb=False)
    assert res.status == 'ok'
    assert (out / 'manifest.json').exists()
    assert (out / 'analysis.npz').exists()
    restarts = list((out / 'templates').glob('*hydrol*dep*/Initconc.dat'))
    assert len(restarts) == 24
    # manifest records perturbation mode, effective localization and input checksums
    man = json.loads((out / 'manifest.json').read_text())
    assert man['perturbation']['enabled'] is False
    assert man['localization_used']['horizontal_m'] == 100000.0
    assert man['inputs']['geometry']['sha256']


@requires_data
def test_api_run_refuses_dirty_output_dir(tmp_path):
    config = load_config(EXAMPLE_CASE)
    (tmp_path / 'leftover.txt').write_text('x')
    with pytest.raises(FileExistsError):
        api.run(config, tmp_path, perturb=False)


@requires_data
def test_result_has_physical_coordinates(tmp_path):
    config = load_config(EXAMPLE_CASE)
    r = api.analyze(config, perturb=False, output_dir=tmp_path)
    n = r.log_states.shape[0]
    assert r.box_id.shape == (n,) and r.layer_id.shape == (n,) and r.time.shape == (n,)
    assert r.obs_box_id.shape == r.observations.shape
    assert r.arithmetic_mean.shape == (n,) and r.geometric_mean.shape == (n,)


@requires_data
@pytest.mark.parametrize('bad', [
    {'gsd': 1.0},
    {'gsd': float('inf')},
    {'observation_times': ['1988.5', '1988.5'], 'target_time': '1988.5'},
    {'schema_version': 999},
])
def test_api_rejects_invalid_config(bad):
    config = load_config(EXAMPLE_CASE)
    for k, v in bad.items():
        setattr(config, k, v)
    with pytest.raises(ValueError):
        api.analyze(config, perturb=False)


@requires_data
def test_api_analyze_accepts_numpy_generator(tmp_path):
    import numpy as np
    config = load_config(EXAMPLE_CASE)
    # must not raise AttributeError on a modern Generator
    r = api.analyze(config, perturb=True, rng=np.random.default_rng(0), output_dir=tmp_path)
    assert np.all(np.isfinite(r.log_states))


def test_layer_depths_from_bdepths_default():
    from poseidon_da.geometry import layer_depths_from_bdepths
    assert layer_depths_from_bdepths([25, 100, 500, 2000]) == [12.5, 62.5, 300.0, 1250.0]


def test_cli_version_subprocess(tmp_path):
    # the installed console entry point must run from an unrelated working directory
    import subprocess
    import sys
    out = subprocess.run([sys.executable, '-m', 'poseidon_da', '--version'],
                         cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0 and 'poseidon-da' in out.stdout


@requires_data
def test_cli_validate_subprocess(tmp_path):
    import subprocess
    import sys
    out = subprocess.run([sys.executable, '-m', 'poseidon_da', 'validate', str(EXAMPLE_CASE)],
                         cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0 and 'OK' in out.stdout
