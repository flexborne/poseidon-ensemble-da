"""Sequential-cycle tests: plan determinism and a mock-model end-to-end run."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import EXAMPLE_CASE, HAS_DATA

from poseidon_da.config import load_config
from poseidon_da.cycle import plan_cycle, run_cycle
from poseidon_da.geometry import box_levels
from poseidon_da.poseidon_io import read_poseidon, write_poseidon
from poseidon_da.runner import find_members
from poseidon_da.state import build_layout

requires_data = pytest.mark.skipif(not HAS_DATA, reason='experiment archive not present')


# ---- plan_cycle (pure, no data) -----------------------------------------------

def _measurements_with(tmp_path, dates):
    for d in dates:
        (tmp_path / f'{d}.dat').write_text('year\n')
    return tmp_path


def test_plan_cycle_filter_order(tmp_path):
    _measurements_with(tmp_path, ['1.0', '2.0', '3.0'])
    kinds = [(s.kind, s.date) for s in plan_cycle('filter', ['1.0', '2.0', '3.0'], tmp_path)]
    assert kinds == [('analyze', '1.0'), ('forecast', '2.0'), ('diagnose', '2.0'),
                     ('analyze', '2.0'), ('forecast', '3.0'), ('diagnose', '3.0')]


def test_plan_cycle_noda_order(tmp_path):
    _measurements_with(tmp_path, ['1.0', '2.0'])
    kinds = [(s.kind, s.date) for s in plan_cycle('noda', ['1.0', '2.0'], tmp_path)]
    assert kinds == [('forecast', '2.0'), ('diagnose', '2.0')]


def test_plan_cycle_smoother_is_gated(tmp_path):
    with pytest.raises(NotImplementedError, match='smoother'):
        plan_cycle('smoother', ['1.0', '2.0'], tmp_path)


def test_plan_cycle_skips_diagnose_without_measurements(tmp_path):
    _measurements_with(tmp_path, ['1.0'])          # only the first date has a measurement file
    steps = plan_cycle('filter', ['1.0', '2.0'], tmp_path)
    diag = [s for s in steps if s.kind == 'diagnose'][0]
    assert diag.provenance.get('skipped') == 'no measurement file'


# ---- end-to-end with a mock model (no POSEIDON needed) ------------------------

def _mock_model_runner():
    """Simulate POSEIDON: turn each member's Initconc.dat into a lastconc.xls stamped
    with the forecast's physical time."""
    bnlev = box_levels()
    layout = build_layout(bnlev)

    def runner(date_dir, expected_time, _bnlev):
        for m in find_members(date_dir):
            init = date_dir / m / 'Initconc.dat'
            arr, header, footer, sed = read_poseidon(str(init), bnlev)
            vect = np.zeros(layout.n_state)
            for L in range(layout.n_state):
                vect[layout.lind_extract[int(layout.Iind[L]), int(layout.Jind[L])] - 1] = \
                    arr[int(layout.Iind[L]), int(layout.Jind[L])]
            newfooter = [it for it in footer if len(it['tline'].split()) != 1]
            newfooter.append({'tline': f'   {expected_time:.11f}     '})
            write_poseidon(str(date_dir / m / 'lastconc.xls'), vect, layout.lind_extract,
                           bnlev, header, newfooter, sed)
    return runner


@requires_data
def test_run_cycle_filter_with_mock_model(tmp_path):
    config = load_config(EXAMPLE_CASE)
    res = run_cycle(config, tmp_path, mode='filter', dates=['1986.75', '1987.5'],
                    model_runner=_mock_model_runner(), perturb=False)
    assert res.status == 'ok'
    kinds = [(s['step'], s['date']) for s in res.steps]
    assert kinds == [('analyze', '1986.75'), ('forecast', '1987.5'), ('diagnose', '1987.5')]
    # forecast provenance records the restart source
    fc = [s for s in res.steps if s['step'] == 'forecast'][0]
    assert fc['source'] == '1986.75'
    # the mock produced a fresh lastconc for every member of the forecast date
    produced = list((tmp_path / 'work' / '1987.5').glob('*hydrol*dep*/lastconc.xls'))
    assert len(produced) == 24
    assert (tmp_path / 'manifest.json').exists()


@requires_data
def test_run_cycle_noda_with_mock_model(tmp_path):
    config = load_config(EXAMPLE_CASE)
    res = run_cycle(config, tmp_path, mode='noda', dates=['1986.75', '1987.5'],
                    model_runner=_mock_model_runner(), perturb=False)
    kinds = [(s['step'], s['date']) for s in res.steps]
    assert kinds == [('forecast', '1987.5'), ('diagnose', '1987.5')]


@requires_data
def test_run_cycle_does_not_mutate_dataset(tmp_path):
    from poseidon_da.paths import RUNS_ROOT
    before = (RUNS_ROOT / '1987.5' / '06.hydrol2dep2' / 'lastconc.xls').read_bytes()
    config = load_config(EXAMPLE_CASE)
    run_cycle(config, tmp_path, mode='noda', dates=['1986.75', '1987.5'],
              model_runner=_mock_model_runner(), perturb=False)
    after = (RUNS_ROOT / '1987.5' / '06.hydrol2dep2' / 'lastconc.xls').read_bytes()
    assert before == after        # the cycle works on a copy, never the dataset
