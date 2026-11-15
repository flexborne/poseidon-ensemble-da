# Frozen reference arrays

These `.npz` files are the **frozen regression baseline**. They must not be overwritten
by routine regeneration — a future bug could otherwise be "blessed" into the expected result.

- `filter_1986.75.npz` — per-member analysis log-states, filter, date `1986.75`.
- `smoother_1988.625_1988.5.npz` — per-member analysis log-states, smoother window `[1988.625, 1988.5]`.

## Origin

- Captured from the **previous stacked-least-squares solver** with observation perturbations set to zero
  (deterministic), before the solver was replaced.
- First committed at repository revision `b58060a` (reference-case commit `9b23ad6`).
- Inputs: the bundled dataset (`config/Box_surfareas_with_centers.xlsx`, `measurements.exp2/*.dat`,
  `DAruns/01.RunKFS/<date>/*/lastconc.xls`).
- Parameters: `gsd=1.3`, localization `horizontal_m=100000, vertical_m=100, temporal_years=1`,
  `exclude=[1,2,3,49,50]`, `bdepths=[25,100,500,2000]`.

## check against the current solver

The current observation-space solver reproduces these arrays within least-squares's convergence
tolerance (~1e-6), checked by `tests/test_regression.py`.

## checked against the original the model (performed)

Using `tools/model_dump.m` (original the model functions, perturbations zeroed) +
`tools/compare_model_dump.py` on identical inputs (the model tree at
`DAwork/DAruns.Sasha/01.RunKFS`), the model R2026b vs the Python implementation:

| quantity | filter 1986.75 | smoother [1988.625,1988.5] |
|---|---|---|
| y   | 3.7e-8  | 4.6e-8  |
| Xf  | 5.3e-15 | 5.3e-15 |
| B   | 4.4e-16 | 8.3e-17 |
| sol | 1.4e-6  | 2.9e-7  |

y/Xf/B match to machine precision; `sol` matches to the model's iterative-least-squares tolerance
(the Python solver is the exact minimizer of the same problem).

## Regeneration (does NOT overwrite this baseline)

`python tools/make_reference.py` writes **candidate** arrays to `tests/reference/candidate/`
and prints the max deviation from the frozen baseline. Promoting a candidate to the baseline is
a deliberate manual step (and should be justified in the commit message).
