# POSEIDON — Ensemble Data Assimilation (Python)

> 🇺🇦 Українською: [README.uk.md](README.uk.md)

An installable Python package (`poseidon_da`) for **data assimilation** with the **POSEIDON**
radionuclide transport model. It implements an **ensemble Kalman filter / smoother**: it
updates the POSEIDON state vector from observations, computes the analysis in
**log-concentration space**, writes the updated state back into `Initconc.dat` restart files,
and lets the model continue the forecast to the next analysis date.

The package exposes a stable **CLI** and **Python API**; the covariance, indexing and solver
details live inside the package. A university user supplies a dataset, a case config and an
output directory — nothing else.

---

## Contents

- [Install](#install)
- [Quick start (CLI)](#quick-start-cli)
- [Python API](#python-api)
- [Case configuration](#case-configuration)
- [Method overview](#method-overview)
- [check](#check)
- [Repository layout](#repository-layout)
- [Package modules](#package-modules)
- [Development & tests](#development--tests)

---

## Install

Requires Python ≥ 3.11.

```bash
pip install .            # or: pip install -e .   (editable, for development)
```

This installs the `poseidon_da` package and the `poseidon-da` command. The package is
importable and runnable from any directory, independent of the source checkout.

---

## Quick start (CLI)

```bash
poseidon-da validate examples/black-sea-cs137.toml
poseidon-da analyze  examples/black-sea-cs137.toml -o results/window-01
poseidon-da run      examples/black-sea-cs137.toml -o results/run-01
poseidon-da diagnose examples/black-sea-cs137.toml -o results/check-01
```

- **validate** — check the config and inputs; report problems, non-zero exit if invalid.
- **analyze** — compute the analysis; write `analysis.npz`, `result.json`, `report.xlsx`.
- **run** — analyze + write restart `Initconc.dat` files + `out_<date>.xlsx` + a `manifest.json`.
- **diagnose** — compare the ensemble forecast against observations (no assimilation).

Add `--no-perturb` for a deterministic run (observation perturbations set to zero).
The forecast step itself requires `Poseidon4.exe` (run via `wine` on Linux/macOS); the
analysis commands do not.

---

## Python API

```python
from poseidon_da import load_config, analyze, run

config = load_config("examples/black-sea-cs137.toml")

result = analyze(config, perturb=False)      # structured result, no files written
print(result.target, result.log_states.shape, result.geometric_mean.shape)

run_result = run(config, "results/run-01")   # writes restarts, outputs, manifest
print(run_result.status, run_result.outputs)
```

`analyze` returns an `AnalysisResult` (log-states, concentrations, geometric mean, observation
indices, diagnostics). Writing files is always an explicit step (`run`, or the exporter).

---

## Case configuration

A case is a TOML file; paths are resolved relative to the file's directory.

```toml
[case]
schema_version = 1
case_id = "black-sea-cs137"

[dataset]
measurements_dir = "../measurements.exp2"
geometry         = "../config/Box_surfareas_with_centers.xlsx"
runs_root        = "../DAruns/01.RunKFS"

[analysis]
target_time       = "1988.5"
observation_times = ["1988.625", "1988.5"]   # single entry → filter; many → smoother
gsd               = 1.3
bdepths           = [25, 100, 500, 2000]

[localization]
horizontal_m  = 100000.0
vertical_m    = 100.0
temporal_years = 1.0
exclude       = [1, 2, 3, 49, 50]

[model]
executable = "../bin/Poseidon4.exe"
launcher   = ["wine"]
```

`target_time` must be the last of `observation_times`. See [examples/black-sea-cs137.toml](examples/black-sea-cs137.toml).

---

## Method overview

**Domain.** The sea (the Black Sea) is partitioned into ~50 horizontal boxes with up to 4
vertical levels. Box geometry (lat/lon bounds, levels, centres of mass) lives in `config/*.xlsx`.

**Ensemble.** 24 POSEIDON runs (folders `NN.hydrolXdepY`) with different hydrology/deposition
scenarios; the ensemble spread estimates the forecast-error covariance.

**Log space.** The analysis is performed on the natural logarithms of concentrations
(log-normal quantities), which keeps updated concentrations positive.

**Analysis step.** For an analysis window:
1. read and box-average observations (`<date>.dat`);
2. read the ensemble states (`lastconc.xls`) and build the state layout + observation indices;
3. estimate the ensemble covariance **C** and **localize** it by horizontal/vertical (and, for a
   smoother, temporal) distance; certain boxes are decoupled;
4. solve the Kalman update **in observation space**,
   `Xa = Xf + B Hᵀ (H B Hᵀ + R)⁻¹ (Y − H Xf)`, per ensemble member with perturbed observations
   (stochastic EnKF);
5. write restart files and `out_<date>.xlsx`.

A single-date window is the **filter**; multiple dates form the **smoother** (analysis built for
the last/target date).

---

## check

converted from the model function-by-function. The only randomness is the **observation
perturbations**; because the model and NumPy RNGs differ, check was done with the
**perturbations zeroed in both programs**, which makes both fully deterministic — and they
produced **identical results**, confirming the computational core.

The observation-space solver replaces the earlier explicit `B⁻¹` + LDL + per-member least-squares. It is
the exact least-squares minimizer, so it reproduces the previous result within least-squares's convergence
tolerance (~1e-6), checked against golden arrays in `tests/reference/`.

> In production runs perturbations are left random; the result is reproducible within Python
> (fixed seed) but not bit-for-bit identical to the model.

---

## Repository layout

```text
poseidon-ensemble-da/
  pyproject.toml              # package metadata, deps, console script
  src/poseidon_da/            # the installable package
  examples/                   # example case.toml
  tests/                      # pytest suite + reference/ golden arrays
  tools/                      # make_reference.py
  config/                     # box geometry (dataset)
  measurements.exp2/          # observation files (dataset)
  DAruns/01.RunKFS/<date>/    # POSEIDON working directories (dataset/archive)
  bin/Poseidon4.exe           # model executable
```

Code is installable and read-only; the dataset (geometry, measurements, scenario runs) is kept
separate and referenced by the case config. Each run writes only to its own output directory.

---

## Package modules

| Module | Responsibility |
|---|---|
| `poseidon_da/api.py` | Public API: `load_config`, `analyze`, `run`. |
| `poseidon_da/cli.py`, `__main__.py` | CLI: `validate`, `analyze`, `run`, `diagnose`. |
| `poseidon_da/config.py` | Load + validate a case TOML into a `Config`. |
| `poseidon_da/domain.py` | Typed contracts (`StateLayout`, `ObservationBatch`, `ForecastEnsemble`, `AnalysisWindow`, `AnalysisResult`, `RunResult`). |
| `poseidon_da/workflow.py` | Assemble inputs, run the shared analysis, write restarts/outputs. |
| `poseidon_da/analysis.py` | `analysis_update` — observation-space Kalman update (Cholesky). |
| `poseidon_da/covariance.py` | Ensemble covariance + distance localization (H/V/temporal). |
| `poseidon_da/state.py` | State layout (single/window) + validated `observation_indices`. |
| `poseidon_da/observations.py` | Geolocation + box-aggregation of raw measurements. |
| `poseidon_da/geometry.py` | Box levels, localization depths, box centres. |
| `poseidon_da/poseidon_io.py` | POSEIDON text format read/write + Excel I/O (the only openpyxl user). |
| `poseidon_da/runner.py` | `RunContext`, executable launcher, file plumbing, run checks. |
| `poseidon_da/results.py` | Export (NPZ/JSON/XLSX) + manifest. |
| `poseidon_da/paths.py` | Default repo-relative data locations (overridable by a case). |

---

## Development & tests

```bash
pip install -e ".[test]"
pytest -q
```

Tests cover: regression vs golden arrays, analysis-core algebra (closed-form Gaussian update,
full = reduced window, decoupled coordinate unchanged), input contracts (bad box/level,
malformed `lastconc.xls`, runner preconditions), Excel I/O, and the config/API/CLI on the
example case. Data-dependent tests skip automatically when the archive is absent; all runs write
only to an isolated `RunContext`, never into `DAruns/`.

Regenerate the golden reference (deterministic): `python tools/make_reference.py`.
