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
    config = load_config(EXAMPLE_CASE)
    res = api.run(config, tmp_path, perturb=False)
    assert res.status == 'ok'
    assert (tmp_path / 'manifest.json').exists()
    assert (tmp_path / 'analysis.npz').exists()
    # restarts written for every member
    restarts = list((tmp_path / 'templates').glob('*hydrol*dep*/Initconc.dat'))
    assert len(restarts) == 24
