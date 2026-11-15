"""Command-line interface: validate / analyze / run / diagnose."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__, api
from .config import load_config, validate


def _default_output(config, sub):
    return Path.cwd() / 'results' / f'{config.case_id}-{sub}'


def _cmd_validate(args):
    config = load_config(args.case)
    problems = validate(config)
    if problems:
        print(f'INVALID: {config.case_id} ({len(problems)} problem(s))')
        for p in problems:
            print(f'  - {p}')
        return 1
    print(f'OK: {config.case_id} is runnable '
          f'(target {config.target_time}, {len(config.observation_times)} observation date(s))')
    return 0


def _cmd_analyze(args):
    config = load_config(args.case)
    out = Path(args.output) if args.output else _default_output(config, 'analysis')
    result = api.analyze(config, perturb=not args.no_perturb)
    from .results import export_result
    paths = export_result(result, out)
    print(f'analyze {config.case_id}: target={result.target} '
          f'state={result.log_states.shape[0]} members={result.log_states.shape[1]}')
    print(f'  wrote {paths["npz"]}')
    return 0


def _cmd_run(args):
    config = load_config(args.case)
    problems = validate(config)
    if problems:
        print('cannot run, case is invalid:')
        for p in problems:
            print(f'  - {p}')
        return 1
    out = Path(args.output) if args.output else _default_output(config, 'run')
    result = api.run(config, out, perturb=not args.no_perturb)
    print(f'run {config.case_id}: {result.status}')
    for step in result.steps:
        print(f'  {step["step"]}: {step["status"]}')
    print(f'  outputs: {out}')
    return 0


def _cmd_diagnose(args):
    config = load_config(args.case)
    out = Path(args.output) if args.output else _default_output(config, 'diagnose')
    out.mkdir(parents=True, exist_ok=True)

    from .geometry import box_levels
    from .observations import aggregate_observations
    from .poseidon_io import read_poseidon, xlswrite
    from .runner import find_members

    batch = aggregate_observations(str(config.measurements_dir / f'{config.target_time}.dat'),
                                   str(config.geometry), config.bdepths)
    obs, brefnum, breflev = batch['values'], batch['brefnum'], batch['breflev']
    bnlev = box_levels()
    members = find_members(config.runs_root / config.target_time)
    No = obs.shape[0]
    ens = np.zeros((No, 1))
    report = out / 'diagnose.xlsx'
    xlswrite(str(report), np.concatenate((obs, brefnum), axis=1), 'obs')
    for m in members:
        arr, _, _, _ = read_poseidon(str(config.runs_root / config.target_time / m / 'lastconc.xls'), bnlev)
        sim = np.array([[arr[int(brefnum[L, 0]) - 1, int(breflev[L, 0]) - 1]] for L in range(No)])
        ens += sim
        xlswrite(str(report), sim, m)
    ens /= np.single(len(members))
    xlswrite(str(report), ens, 'Ens')
    print(f'diagnose {config.case_id}: {No} observations, {len(members)} members -> {report}')
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog='poseidon-da', description='POSEIDON ensemble data assimilation')
    p.add_argument('--version', action='version', version=f'poseidon-da {__version__}')
    sub = p.add_subparsers(dest='command', required=True)

    for name, func, needs_output in [
        ('validate', _cmd_validate, False),
        ('analyze', _cmd_analyze, True),
        ('run', _cmd_run, True),
        ('diagnose', _cmd_diagnose, True),
    ]:
        sp = sub.add_parser(name, help=f'{name} a case')
        sp.add_argument('case', help='path to case.toml')
        if needs_output:
            sp.add_argument('--output', '-o', default=None, help='output directory')
            sp.add_argument('--no-perturb', action='store_true',
                            help='disable observation perturbations (deterministic)')
        sp.set_defaults(func=func)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
