# POSEIDON Data Assimilation Workflow

Python workflow for ensemble data assimilation with the POSEIDON radionuclide transport model.

The workflow uses an **ensemble Kalman smoother / filter data assimilation method** to update POSEIDON state vectors from observations. The analyzed state is computed in **log-concentration space**, written back as updated `Initconc.dat`, and then used to continue the forecast to the next analysis date.

## What the code does

For each analysis date, the workflow:

1. reads observations from `<date>.dat`
2. maps observations to POSEIDON computational boxes and vertical levels
3. reads ensemble forecast states from POSEIDON outputs (`lastconc.xls`)
4. estimates the forecast-error covariance matrix from the ensemble
5. applies covariance localization
6. solves the analysis problem for each ensemble member
7. writes updated `Initconc.dat`
8. runs POSEIDON to produce the next forecast step

In the general method, observations from several times can be assimilated jointly in a smoother-type formulation. When only observations at the analysis time are used, the method reduces to a filter-type formulation.

## Method summary

The model state is the vector of radionuclide concentrations in POSEIDON computational cells. The analysis is performed for the **natural logarithms of concentrations**, which makes the regression problem more convenient and avoids nonphysical negative updated concentrations.

Observations are linked to model cells through a correspondence matrix `H`. Observation error and representativeness error are combined into a single observation-error covariance matrix. Forecast-error covariance is estimated from the ensemble of POSEIDON runs.

Because ensemble-estimated covariance matrices are noisy, localization is applied. In this method, localization depends on:
- horizontal distance
- vertical distance
- and, in the temporal version, time difference

To reduce the size of the minimization problem, the method excludes state components that do not contain observations and do not belong to the target analysis time, without changing the solution for the reinitialization state.

The final minimization is written in regression form after covariance inversion and LDL decomposition.

## Repository layout

```text
repo/
  bin/
    Poseidon4.exe

  config/
    Box_surfareas.xlsx
    Box_surfareas_with_centers.xlsx

  measurements.exp2/
    *.dat

  src/
    *.py

  DAruns/
    01.RunKFS/
      templates/
      outputs/
      1986.75/
      1987.5/
      ...
```

## Directory meaning

### `bin/`
Contains the POSEIDON executable:
- `Poseidon4.exe`

### `config/`
Contains static box metadata used by the Python workflow:
- `Box_surfareas.xlsx`
- `Box_surfareas_with_centers.xlsx`

These files define box geometry, vertical structure, and box centers used for observation mapping and localization.

### `measurements.exp2/`
Contains observation files:
- `<date>.dat`

### `src/`
Contains Python source code.

### `DAruns/01.RunKFS/templates/`
Contains hydrol template folders where updated `Initconc.dat` files are written.

### `DAruns/01.RunKFS/<date>/`
Contains working POSEIDON run directories for each date.

### `DAruns/01.RunKFS/outputs/`
Contains auxiliary outputs of the Python workflow:
- `out_<date>.xlsx`
- `obs2pos.mat`
- `obs_pos.xls`

## Main scripts

- `main_LN.py` — single-date analysis
- `main_KSF_LN.py` — multi-date smoother-style analysis
- `main_obsproconly.py` — observation processing and diagnostics only
- `runall_LN.py` — full sequential cycle over all configured dates
- `poseidon_runtime.py` — runs `Poseidon4.exe` in hydrol folders

## Inputs

The workflow expects:
- observation files in `measurements.exp2/`
- box metadata in `config/`
- POSEIDON run folders in `DAruns/01.RunKFS/<date>/`
- `Poseidon4.exe` in `bin/`

## Outputs

In `DAruns/01.RunKFS/templates/`:
- updated `Initconc.dat`

In `DAruns/01.RunKFS/outputs/`:
- `out_<date>.xlsx`
- `obs2pos.mat`
- `obs_pos.xls`

In each date hydrol folder after POSEIDON runs:
- `lastconc.xls`
- `concsed.txt`
- `concwat_*.txt`
- `concbiol_*.txt`
- `mat.out`

## How to run

Single-date analysis:
```bash
python src/main_LN.py 1986.75
```

Full sequential cycle:
```bash
python src/runall_LN.py
```

Multi-date smoother analysis:
```bash
python src/main_KSF_LN.py
```

Observation-only diagnostics:
```bash
python src/main_obsproconly.py
```

## Dependencies

- `numpy`
- `scipy`
- `openpyxl`
