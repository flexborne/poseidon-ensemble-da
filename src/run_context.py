"""Explicit run context: every step reads/writes through the same directories.

Fixes the split where the driver scripts resolved dates against the repo root
while the analysis functions used DAruns/01.RunKFS. A single RunContext is
passed to all steps; tests and concurrent runs pass an isolated one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import paths


@dataclass(frozen=True)
class RunContext:
    runs_root: Path       # contains the per-date working directories
    templates_dir: Path   # where updated Initconc.dat files are written
    outputs_dir: Path     # where out_*.xlsx / obs_pos.xlsx / obs2pos.mat go

    @classmethod
    def default(cls) -> "RunContext":
        return cls(
            runs_root=paths.RUNS_ROOT,
            templates_dir=paths.TEMPLATES_DIR,
            outputs_dir=paths.OUTPUTS_DIR,
        )

    @classmethod
    def isolated(cls, workdir, runs_root=None) -> "RunContext":
        """A context whose writable outputs live under `workdir` (for tests/parallel runs)."""
        workdir = Path(workdir)
        return cls(
            runs_root=Path(runs_root) if runs_root is not None else paths.RUNS_ROOT,
            templates_dir=workdir / 'templates',
            outputs_dir=workdir / 'outputs',
        )

    def date_dir(self, strdate: str) -> Path:
        return self.runs_root / strdate

    def template_dir(self, name: str) -> Path:
        return self.templates_dir / name

    def _outputs(self) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        return self.outputs_dir

    def output_xlsx(self, strdate: str) -> Path:
        return self._outputs() / f'out_{strdate}.xlsx'

    def obs_pos_xlsx(self) -> Path:
        return self._outputs() / 'obs_pos.xlsx'

    def obs2pos_mat(self) -> Path:
        return self._outputs() / 'obs2pos.mat'
