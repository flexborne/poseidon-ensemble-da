# Frozen reference arrays

These `.npz` files are the **frozen regression baseline** for the solver. They must not be
overwritten by routine regeneration — a future change could otherwise be silently "blessed"
into the expected result.

- `filter_1986.75.npz` — per-member analysis log-states, filter, date `1986.75`.
- `smoother_1988.625_1988.5.npz` — per-member analysis log-states, smoother window `[1988.625, 1988.5]`.

## Parameters

Produced deterministically (observation perturbations off) with:
`gsd=1.3`; localization `horizontal_m=100000, vertical_m=100, temporal_years=1`;
`exclude=[1,2,3,49,50]`; `bdepths=[25,100,500,2000]`.

## Regeneration (does NOT overwrite this baseline)

`python tools/make_reference.py` writes **candidate** arrays to `tests/reference/candidate/`
and prints the max deviation from the frozen baseline. Promoting a candidate to the baseline is
a deliberate manual step, justified in the commit message.
