"""Export an AnalysisResult and write a run manifest (machine- and human-readable)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .domain import AnalysisResult


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def export_result(result: AnalysisResult, output_dir) -> dict:
    """Write NPZ arrays (with physical coordinates), a JSON summary and an XLSX report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    npz = out / 'analysis.npz'
    np.savez(
        npz,
        log_states=result.log_states,
        concentrations=result.concentrations,
        geometric_mean=result.geometric_mean,
        arithmetic_mean=result.arithmetic_mean,
        box_id=result.box_id, layer_id=result.layer_id, time=result.time,
        obs_indices=result.obs_indices, observations=result.observations,
        obs_box_id=result.obs_box_id, obs_layer_id=result.obs_layer_id, obs_time=result.obs_time,
        member_ids=np.array(result.member_ids),
    )

    summary = {
        'target': result.target,
        'member_ids': list(result.member_ids),
        'diagnostics': result.diagnostics,
        'n_state': int(result.log_states.shape[0]),
        'n_members': int(result.log_states.shape[1]),
        'axes': {'state_rows': ['box_id', 'layer_id', 'time'],
                 'means': ['geometric_mean', 'arithmetic_mean']},
    }
    (out / 'result.json').write_text(json.dumps(summary, indent=2))
    return {'npz': npz, 'json': out / 'result.json'}


def build_manifest(config, result: AnalysisResult, package_version: str, steps,
                   *, localization_used, perturbation) -> dict:
    """Manifest recording versions, normalized config, input checksums and per-step status.

    Hashes the actually-read inputs: geometry, the measurement file of each date, and every
    ensemble state (lastconc.xls) that contributed to the background, so the provenance is
    sufficient to identify the exact inputs of the run.
    """
    import scipy

    def _h(path):
        try:
            return _checksum(Path(path))
        except OSError:
            return None

    inputs = {'geometry': {'path': str(config.geometry), 'sha256': _h(config.geometry)},
              'measurements': {}, 'ensemble': {}}
    for d in config.observation_times:
        inputs['measurements'][d] = _h(config.measurements_dir / f'{d}.dat')
        inputs['ensemble'][d] = {m: _h(config.runs_root / d / m / 'lastconc.xls')
                                 for m in result.member_ids}

    return {
        'package_version': package_version,
        'schema_version': config.schema_version,
        'case_id': config.case_id,
        'target_time': result.target,
        'observation_times': list(config.observation_times),
        'gsd': config.gsd,
        'bdepths': config.bdepths,
        'localization_used': localization_used,
        'perturbation': perturbation,
        'member_order': list(result.member_ids),
        'library_versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
        'diagnostics': result.diagnostics,
        'inputs': inputs,
        'steps': steps,
    }


def write_manifest(manifest: dict, output_dir) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
