"""Tests reproducing the re-audit findings (R1–R7)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_data, EXAMPLE_CASE, a_lastconc

from poseidon_da import api
from poseidon_da.config import load_config, resolve_perturbation, validate
from poseidon_da.cycle import run_cycle
from poseidon_da.geometry import box_levels
from poseidon_da.poseidon_io import read_poseidon, read_state

BNLEV = box_levels()


# ---- R7: seed and rng are mutually exclusive ----------------------------------

def test_resolve_perturbation():
    with pytest.raises(ValueError, match='either seed or rng'):
        resolve_perturbation(seed=1, rng=np.random.default_rng(2), perturb=True)
    rng, meta = resolve_perturbation(seed=3, rng=None, perturb=True)
    assert rng is not None and meta['seed'] == 3
    rng, meta = resolve_perturbation(seed=None, rng=None, perturb=True)    # fresh seed materialized
    assert rng is not None and isinstance(meta['seed'], int)              # ...and recorded for reproducibility
    rng, meta = resolve_perturbation(seed=None, rng=None, perturb=False)
    assert rng is None and meta['enabled'] is False


# ---- R5a: geometry with a duplicate box id is rejected ------------------------

def test_load_geometry_rejects_duplicate_box_id(tmp_path):
    from openpyxl import Workbook
    from poseidon_da.geometry import load_geometry
    path = tmp_path / 'geom.xlsx'
    wb = Workbook(); ws = wb.active; ws.title = 'Sheet1'
    for row in ([4] + [0] * 10, [5] + [0] * 10, [5] + [0] * 10):   # box 5 duplicated
        ws.append(row)
    wb.save(path)
    with pytest.raises(ValueError, match='duplicate'):
        load_geometry(path)


# ---- R3: an unequal member set across dates fails before any run --------------

def test_run_cycle_requires_equal_member_set(tmp_path):
    runs = tmp_path / 'runs'
    (runs / '1.0' / '01.hydrol1dep1').mkdir(parents=True)
    (runs / '2.0' / '01.hydrol1dep1').mkdir(parents=True)
    (runs / '2.0' / '02.hydrol1dep2').mkdir(parents=True)   # extra member on date 2
    config = load_config(EXAMPLE_CASE)
    config.runs_root = runs
    (tmp_path / 'meas').mkdir()                    # exists but empty -> diagnose skipped
    config.measurements_dir = tmp_path / 'meas'
    called = []
    with pytest.raises(ValueError, match='member set'):
        run_cycle(config, tmp_path / 'out', mode='noda', dates=['1.0', '2.0'],
                  model_runner=lambda *a: called.append(a))
    assert called == []                          # model never ran


# ---- R2: an unsafe date cannot write outside the output directory -------------

def test_run_cycle_rejects_unsafe_date(tmp_path):
    config = load_config(EXAMPLE_CASE)
    sentinel = tmp_path / 'victim'
    sentinel.write_text('precious')
    with pytest.raises(ValueError, match='safe path component|escapes|not numeric'):
        run_cycle(config, tmp_path / 'out', mode='noda', dates=['1986.75', '../victim'],
                  model_runner=lambda *a: None)
    assert sentinel.read_text() == 'precious'           # untouched
    assert not (tmp_path / 'out').exists()              # nothing published


# ---- R1: a failed (force) run leaves the previous complete output intact ------

@requires_data
def test_run_is_atomic_on_failure(tmp_path, monkeypatch):
    config = load_config(EXAMPLE_CASE)
    out = tmp_path / 'run'
    api.run(config, out, perturb=False)
    original = (out / 'manifest.json').read_bytes()

    def boom(*a, **k):
        raise RuntimeError('disk full during export')
    monkeypatch.setattr('poseidon_da.api.export_result', boom)
    with pytest.raises(RuntimeError):
        api.run(config, out, perturb=False, force=True)
    assert (out / 'manifest.json').read_bytes() == original   # previous run preserved, not mixed


# ---- R4: the physical-time contract is enforced on reads ----------------------

def _rewrite_footer_time(src, dst, new_time):
    """Copy a state file, replacing (new_time=None removes) its trailing time stamp."""
    lines = Path(src).read_text().splitlines()
    out, replaced = [], False
    for ln in reversed(lines):
        toks = ln.split()
        if not replaced and len(toks) == 1:
            try:
                float(toks[0])
            except ValueError:
                out.append(ln); continue
            replaced = True
            if new_time is not None:
                out.append(f'   {new_time:.11f}     ')
            continue
        out.append(ln)
    Path(dst).write_text('\n'.join(reversed(out)) + '\n')


@requires_data
def test_read_state_rejects_mismatched_time(tmp_path):
    bad = tmp_path / 'lastconc.xls'
    _rewrite_footer_time(a_lastconc(), bad, 1987.5)
    with pytest.raises(ValueError, match='does not match'):
        read_state(str(bad), BNLEV, expected_time=1988.5)


@requires_data
def test_read_state_rejects_missing_time(tmp_path):
    bad = tmp_path / 'lastconc.xls'
    _rewrite_footer_time(a_lastconc(), bad, None)
    with pytest.raises(ValueError, match='missing or non-finite'):
        read_state(str(bad), BNLEV, expected_time=1986.75)


# ---- R6: non-finite physical values are rejected ------------------------------

@requires_data
def test_read_poseidon_rejects_nan_sediment(tmp_path):
    lines = Path(a_lastconc()).read_text().splitlines()
    out = []
    patched = False
    for ln in lines:
        toks = ln.split()
        if not patched and toks and toks[0].isdigit() and 1 <= int(toks[0]) <= 50:
            toks[-1] = 'NAN'                      # corrupt a sediment column
            ln = '   ' + '   '.join(toks)
            patched = True
        out.append(ln)
    bad = tmp_path / 'lastconc.xls'
    bad.write_text('\n'.join(out) + '\n')
    with pytest.raises(ValueError, match='non-finite sediment'):
        read_poseidon(str(bad), BNLEV)


# ---- R5b: malformed bdepths are rejected by validate --------------------------

@pytest.mark.parametrize('bd', [[], [25, 100], [25, 100, 50, 2000]])
def test_validate_rejects_bad_bdepths(bd):
    config = load_config(EXAMPLE_CASE)
    config.bdepths = bd
    assert any('bdepths' in p for p in validate(config))
