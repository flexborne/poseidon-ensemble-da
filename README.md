# POSEIDON — Ensemble Data Assimilation (Python)

A Python workflow for **data assimilation** with the **POSEIDON** radionuclide transport model. The code is a implementation of the original the model scripts (`01.RunKFS/model/*.m`) to Python.

The workflow implements an **ensemble Kalman filter / smoother**: it updates the POSEIDON state vector from observations, computes the analysis in **log-concentration space**, writes the updated state back into `Initconc.dat` files, and lets the model continue the forecast to the next analysis date.

---

## Contents

- [Method overview](#method-overview)
- [check of the the model implementation](#check-of-the-model-implementation)
- [Repository layout](#repository-layout)
- [Directory meaning](#directory-meaning)
- [Source files](#source-files)
- [Input data formats](#input-data-formats)
- [Outputs](#outputs)
- [How to run](#how-to-run)
- [Dependencies](#dependencies)

---

## Method overview

**Domain.** The sea (the Black Sea) is partitioned into ~50 horizontal boxes (computational cells) with up to 4 vertical levels. Box geometry (lat/lon bounds, number of levels, centres of mass) is stored in `config/*.xlsx`.

**Ensemble.** The "ensemble" is a set of 24 POSEIDON model runs (folders `NN.hydrolXdepY`) that differ in hydrology and deposition scenarios. The ensemble spread is used to estimate the forecast-error covariance.

**Log space.** The analysis is performed on the **natural logarithms of concentrations**. Radionuclide concentrations are log-normally distributed, so working in log space makes the regression problem more convenient and prevents physically impossible (negative) updated concentrations.

**Analysis step (for a single date):**

1. Read observations from `<date>.dat`.
2. Map each observation to its box and vertical level (point geolocation).
3. Read ensemble member states from POSEIDON outputs (`lastconc.xls`) and build the state vector and the observation operator **H**.
4. Estimate the forecast-error covariance matrix **C** from the ensemble.
5. Apply **localization**: damp distant correlations by horizontal and vertical distance (and, in the smoother, by time difference). Certain boxes (Mediterranean, Marmara, Azov Seas, etc.) are made independent.
6. Cast the analysis as the quadratic (Kalman) problem — minimize `(x − xf)ᵀ B⁻¹ (x − xf) + (H x − y)ᵀ R⁻¹ (H x − y)` — and solve it **in observation space**, `Xa = Xf + B Hᵀ (H B Hᵀ + R)⁻¹ (Y − H Xf)`, for every ensemble member using **perturbed observations** (stochastic EnKF).
7. Write the result back into `Initconc.dat` restart files and summarize it into `out_<date>.xlsx`.

**Filter vs smoother.**
- `main_LN` — **filter**: only observations at the analysis date are used.
- `main_KSF_LN` — **smoother**: observations from several dates are assimilated jointly (with an added temporal localization), and the analysis is built for the last date in the set.

**Outer cycle** (`run_LN_KSF3`, `run_NODA`, `runall_LN`): for each date, run the analysis, copy the restart files forward, run POSEIDON to forecast to the next date, compare against observations (`main_obsproconly`), and repeat over the 1986–2007 date sequence.

---

## check of the the model implementation

The code was converted from the model to Python function by function (each `.m` file has a `.py` counterpart). 

**check method.** The only source of randomness in the algorithm is the **observation perturbations**: when forming `obs_noise`, each observation gets `(1/invO2)·randn` added to it. Because the random-number generators of the model (`randn`) and NumPy (`np.random.randn`) produce **different** sequences even from the same seed, a direct bit-for-bit comparison with randomness enabled is impossible.

Therefore, for check the **observation perturbations were initialized to zero in both programs** (the model and Python). With the random component zeroed out, both implementations become fully deterministic: all ensemble members receive the same (unperturbed) observations and differ only in their background state. Under these conditions the **the model and Python versions produced identical final results**, which confirms the correctness of the converted computational core (observation geolocation, construction of `H`, covariance estimation and localization, LDL factorization, and the least-squares solve).

> **Note.** In production (non-check) runs the observation perturbations are left random. The Python version is then reproducible **within Python** (a fixed `np.random.seed` → the same result), but does not match the model bit-for-bit because the random-number generators differ.

In addition, the deterministic front end was checked separately: the output of `obs_vs_poseidon` for date `1986.75` matches the historical the model file `out_1986.75.xlsx` (sheet `obs`) **exactly** (to 0.0).

---

## Repository layout

```text
poseidon-ensemble-da/
  bin/
    Poseidon4.exe               # POSEIDON model executable

  config/
    Box_surfareas.xlsx          # box geometry (base)
    Box_surfareas_with_centers.xlsx  # geometry + box centres of mass (for localization)

  measurements.exp2/
    <date>.dat                  # observation files

  src/
    *.py                        # all Python code

  DAruns/
    01.RunKFS/
      templates/                # hydrol folders where updated Initconc.dat is written
      outputs/                  # auxiliary outputs (out_*.xlsx, obs2pos.mat, obs_pos.xlsx)
      1986.75/  1987.5/  ...    # working POSEIDON run directories per date
```

---

## Directory meaning

| Directory | Purpose |
|---|---|
| `bin/` | The `Poseidon4.exe` model executable. |
| `config/` | Static box metadata: lat/lon bounds, vertical structure, centres of mass. Used for observation mapping and localization. |
| `measurements.exp2/` | Observation files `<date>.dat`. |
| `src/` | Python code (all workflow modules). |
| `DAruns/01.RunKFS/templates/` | Template hydrol folders where updated `Initconc.dat` files are written. |
| `DAruns/01.RunKFS/<date>/` | Working POSEIDON run directories for each date. |
| `DAruns/01.RunKFS/outputs/` | Auxiliary Python outputs: `out_<date>.xlsx`, `obs2pos.mat`, `obs_pos.xlsx`. |

---

## Source files

### Entry points (main scripts)

| File | What it does |
|---|---|
| `main_LN.py` | **Single-date analysis** (filter). Full EnKF step: observations → covariance → localization → least-squares solve → write `Initconc.dat`. |
| `main_KSF_LN.py` | **Multi-date smoother analysis**. Jointly assimilates observations from several dates with temporal localization; analysis for the last date. |
| `main_obsproconly.py` | **Observation processing and diagnostics only** (no analysis): compares the ensemble forecast against observations and writes `out_<date>.xlsx`. |
| `runall_LN.py` | **Full sequential** analysis–forecast cycle over all dates (filter-only variant, `main_LN`). |
| `run_LN_KSF3.py` | Full cycle **combining filter and smoother** (`main_KSF_LN` for some dates) with a ±3-month observation window. |
| `run_NODA.py` | Control cycle **without data assimilation** (No DA): forecast and comparison against observations only. |

### Computational core

| File | What it does |
|---|---|
| `obs_vs_poseidon.py` | Reads `<date>.dat`, maps observations to boxes/levels, averages observations within a box, returns the observation vector and box references. |
| `locate_point3D.py` | Determines the box and level a point (lon, lat, depth) falls into. |
| `box_levs.py` | Returns the number of vertical levels in each box (hard-coded configuration). |
| `IJ_Lind.py` | Builds the mapping between the linear state index `L` and the pair (box `i`, level `j`) — for a single date. |
| `IJT_Lind.py` | Same, but with a **time** dimension (box, level, date) — for the smoother. |
| `init_Hmatr.py` | Builds the observation operator **H** (observation → state component) for a single date. |
| `init_Hmatr_T.py` | Builds **H** with the observation time index — for the smoother. |
| `Box_Neighbors.py` | Box-adjacency matrix (hard-coded topology). *Currently unused by the workflow.* |
| `Dist_geocords.py` | Distance between two geographic points (in metres). |
| `regularize_Covmatr.py` | Simple covariance regularization: makes excluded boxes independent. |
| `regularize_Covmatr_analit.py` | Covariance localization by **horizontal and vertical** distance (analytic damping function). |
| `Tregularize_Covmatr_analit.py` | Localization by **horizontal, vertical and temporal** distance — for the smoother. |
| `read_poseidon.py` | Reads and **validates** a POSEIDON output file (`lastconc.xls`): concentrations per box/level, header, footer, sediments. Rejects missing/duplicated box rows and non-positive active concentrations. |
| `write_poseidon.py` | Writes a file in POSEIDON format (`Initconc.dat`) for model restart. |
| `analysis.py` | **Pure analysis core** (no I/O): `observation_indices` (validated gather, rejects non-existent box/level) and `analysis_update` (observation-space Kalman update via a single Cholesky factorization), plus the `AnalysisResult` record. |

### Infrastructure

| File | What it does |
|---|---|
| `paths.py` | Centralized paths to project directories and files. |
| `run_context.py` | `RunContext` — the single set of input/output directories passed to every step; `RunContext.isolated(workdir)` gives tests and concurrent runs their own output tree. |
| `model_compat.py` | Compatibility layer: Excel read/write (named sheets), running the `.exe`, file copying, labelled printing. |
| `poseidon_runtime.py` | Runs `Poseidon4.exe` in hydrol folders (with existence/output checks) and copies restart files between dates. |

---

## Input data formats

**`measurements.exp2/<date>.dat`** — a text file of observations. The first line is a header; the remaining lines have 4 numeric columns, of which the workflow uses: column 2 — longitude, column 3 — latitude, column 4 — measured value. Depth is fixed in the current experiment (surface measurements).

**`config/Box_surfareas_with_centers.xlsx`** — the `Sheet1` sheet contains (with no header row) numeric per-box data: box number, depth, volume, area, lat/lon bounds, number of levels, and the **box centres of mass** (columns 10–11) needed for localization. A separate `header` sheet holds only the text column labels.

---

## Outputs

**In `DAruns/01.RunKFS/templates/`:**
- updated `Initconc.dat` (model restart files).

**In `DAruns/01.RunKFS/outputs/`:**
- `out_<date>.xlsx` — analysis summaries (sheets `obs`, one per hydrol folder, and `Ens`);
- `obs2pos.mat` — intermediate observation-mapping data;
- `obs_pos.xlsx` — observation diagnostics (sheets `obs`, `sigc`, `bnum`, `box_vs_allobs`).

**In each hydrol folder after POSEIDON runs:**
- `lastconc.xls`, `concsed.txt`, `concwat_*.txt`, `concbiol_*.txt`, `mat.out`.

---

## How to run

Install dependencies:

```bash
pip install -r src/requirements.txt
```

Single-date analysis (filter):

```bash
python src/main_LN.py 1986.75
```

Multi-date smoother analysis:

```bash
python src/main_KSF_LN.py 1988.625 1988.5
```

Full sequential cycle:

```bash
python src/runall_LN.py
```

Observation diagnostics only:

```bash
python src/main_obsproconly.py 1987.5
```

> The **forecast step** requires `Poseidon4.exe`. On Linux/macOS it runs via `wine` (which must be installed). The analysis modules (`main_LN`, `main_KSF_LN`, `main_obsproconly`) work without the `.exe`.

The analysis functions accept an explicit `run_ctx` and a `perturb` flag:

```python
from run_context import RunContext
from main_LN import main_LN

ctx = RunContext.isolated("results/run-01")      # outputs under results/run-01, reads real data
result = main_LN("1986.75", run_ctx=ctx, perturb=False)   # deterministic run
print(result.log_states.shape, result.geometric_mean.shape)
```

---

## Development, tests, and reference case

The analysis is solved in **observation space** —
`Xa = Xf + B Hᵀ (H B Hᵀ + R)⁻¹ (Y − H Xf)` (`analysis.analysis_update`) — computed with
one Cholesky factorization. This replaces the earlier explicit `B⁻¹` + LDL + per-member
least-squares. It is the exact least-squares minimizer, so it reproduces the previous result to
within least-squares's convergence tolerance (~1e-6), checked against golden arrays.

**Reference case (golden arrays).** `tests/reference/*.npz` holds the per-member analysis
log-states for the filter (`1986.75`) and the smoother (`[1988.625, 1988.5]`), captured
from the previous solver with perturbations zeroed. Regenerate with:

```bash
python tools/make_reference.py      # then: git checkout -- DAruns
```

**Tests.** Regression (vs golden), analysis-core algebra (closed-form Gaussian update,
full = reduced window, decoupled coordinate unchanged), input contracts (bad box/level,
malformed `lastconc.xls`, runner preconditions) and Excel I/O:

```bash
pip install pytest
python -m pytest tests/ -q
```

Tests that need the experiment archive skip automatically when it is absent; all runs
write only to an isolated `RunContext`, never into `DAruns/`.

---

## Dependencies

- `numpy`
- `scipy`
- `openpyxl`
- `pytest` (tests only)
