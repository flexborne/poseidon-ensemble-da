"""Export an AnalysisResult and write a run manifest (machine- and human-readable)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .domain import AnalysisResult
from .poseidon_io import xlswrite


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def export_result(result: AnalysisResult, output_dir) -> dict:
    """Write NPZ arrays, a JSON summary and an XLSX report. Returns written paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    npz = out / 'analysis.npz'
    np.savez(
        npz,
        log_states=result.log_states,
        concentrations=result.concentrations,
        geometric_mean=result.geometric_mean,
        obs_indices=result.obs_indices,
        observations=result.observations,
        member_ids=np.array(result.member_ids),
    )

    summary = {
        'target': result.target,
        'member_ids': list(result.member_ids),
        'diagnostics': result.diagnostics,
        'n_state': int(result.log_states.shape[0]),
        'n_members': int(result.log_states.shape[1]),
    }
    (out / 'result.json').write_text(json.dumps(summary, indent=2))

    report = out / 'report.xlsx'
    xlswrite(str(report), result.geometric_mean.reshape(-1, 1), 'geometric_mean')
    xlswrite(str(report), result.observations.reshape(-1, 1), 'observations')

    return {'npz': npz, 'json': out / 'result.json', 'xlsx': report}


def build_manifest(config, result: AnalysisResult, package_version: str, steps) -> dict:
    """Assemble a manifest recording versions, config, inputs and per-step status."""
    inputs = {}
    try:
        inputs['geometry_sha256'] = _checksum(config.geometry)
    except OSError:
        pass
    return {
        'package_version': package_version,
        'schema_version': config.schema_version,
        'case_id': config.case_id,
        'target_time': result.target,
        'observation_times': list(config.observation_times),
        'gsd': config.gsd,
        'localization': config.localization,
        'member_order': list(result.member_ids),
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
