"""Command-line interface: validate / analyze / run / diagnose."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, api
from .config import load_config, validate


def _default_output(config, sub):
    return Path.cwd() / 'results' / f'{config.case_id}-{sub}'


def _report_problems(config):
    problems = validate(config)
    if problems:
        print(f'INVALID: {config.case_id} ({len(problems)} problem(s))')
        for p in problems:
            print(f'  - {p}')
    return problems


def _cmd_validate(args):
    config = load_config(args.case)
    if _report_problems(config):
        return 1
    print(f'OK: {config.case_id} is runnable '
          f'(target {config.target_time}, {len(config.observation_times)} observation date(s))')
    return 0


def _cmd_analyze(args):
    config = load_config(args.case)
    if _report_problems(config):
        return 1
    out = Path(args.output) if args.output else _default_output(config, 'analysis')
    result = api.write_analysis(config, out, perturb=not args.no_perturb, seed=args.seed, force=args.force)
    print(f'analyze {config.case_id}: {result.status}')
    print(f'  outputs: {out}')
    return 0


def _cmd_run(args):
    config = load_config(args.case)
    if _report_problems(config):
        return 1
    out = Path(args.output) if args.output else _default_output(config, 'run')
    result = api.run(config, out, perturb=not args.no_perturb, seed=args.seed, force=args.force)
    print(f'run {config.case_id}: {result.status}')
    for step in result.steps:
        print(f'  {step["step"]}: {step["status"]}')
    print(f'  outputs: {out}')
    return 0


def _cmd_diagnose(args):
    config = load_config(args.case)
    if _report_problems(config):
        return 1
    out = Path(args.output) if args.output else _default_output(config, 'diagnose')
    result = api.diagnose(config, out, force=args.force)
    s = result.steps[0]
    print(f'diagnose {config.case_id}: {result.status} ({s["n_obs"]} observations, {s["n_members"]} members)')
    print(f'  outputs: {out}')
    return 0


def _cmd_run_cycle(args):
    config = load_config(args.case)
    out = Path(args.output) if args.output else _default_output(config, 'cycle')
    result = api.run_cycle(config, out, mode=args.mode, dates=args.dates or None,
                           perturb=not args.no_perturb, seed=args.seed, force=args.force)
    print(f'run-cycle {config.case_id}: {result.status} (mode={result.manifest["mode"]})')
    for step in result.steps:
        tag = step.get('source', '')
        print(f'  {step["step"]:<9} {step["date"]:<10} {step["status"]}' + (f'  <- {tag}' if tag else ''))
    print(f'  outputs: {out}')
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
            sp.add_argument('--seed', type=int, default=None, help='RNG seed for perturbations')
        if name in ('analyze', 'run', 'diagnose'):
            sp.add_argument('--force', action='store_true', help='overwrite a non-empty output directory')
        sp.set_defaults(func=func)

    sc = sub.add_parser('run-cycle', help='run a sequential NODA/filter assimilation-forecast cycle')
    sc.add_argument('case', help='path to case.toml')
    sc.add_argument('--output', '-o', default=None, help='output directory')
    sc.add_argument('--mode', choices=['noda', 'filter', 'smoother'], default=None,
                    help='cycle mode (default from [cycle] in the config)')
    sc.add_argument('--dates', nargs='+', default=None, help='override the date sequence')
    sc.add_argument('--no-perturb', action='store_true', help='deterministic (no perturbations)')
    sc.add_argument('--seed', type=int, default=None, help='RNG seed')
    sc.add_argument('--force', action='store_true', help='overwrite a non-empty output directory')
    sc.set_defaults(func=_cmd_run_cycle)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
