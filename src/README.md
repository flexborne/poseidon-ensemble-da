# Python port of the MATLAB DA workflow

This directory contains Python replacements for all `.m` files in `01.RunKFS/matlab`, plus the legacy `1986.75/main.m` port in `../1986.75/main.py`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Main entrypoints:

- `python main_LN.py 1986.75`
- `python main_obsproconly.py 1987.5`
- `python main_KSF_LN.py 1988.25 1988.5 1988.625`
- `python runall_LN.py`
- `python run_LN_KSF3.py`
- `python run_NODA.py`

Forecast steps still depend on `Poseidon4.exe`. On Linux/macOS use `wine` or set `POSEIDON_RUNNER`.
