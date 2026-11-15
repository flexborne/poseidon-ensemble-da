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
    result = api.analyze(config, perturb=not args.no_perturb, seed=args.seed)
    from .results import export_result
    paths = export_result(result, out)
    print(f'analyze {config.case_id}: target={result.target} '
          f'state={result.log_states.shape[0]} members={result.log_states.shape[1]}')
    print(f'  wrote {paths["npz"]}')
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
    from .runner import RunContext
    from .workflow import write_forecast_diagnostics
    config = load_config(args.case)
    if _report_problems(config):
        return 1
    out = Path(args.output) if args.output else _default_output(config, 'diagnose')
    ctx = RunContext.isolated(out, runs_root=config.runs_root)
    obs, members = write_forecast_diagnostics(config, ctx, config.target_time)
    print(f'diagnose {config.case_id}: {obs.shape[0]} observations, {len(members)} members -> {out}')
    return 0


def _cmd_run_cycle(args):
    from . import cycle
    config = load_config(args.case)
    out = Path(args.output) if args.output else _default_output(config, 'cycle')
    result = cycle.run_cycle(config, out, mode=args.mode, dates=args.dates or None,
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
        if name == 'run':
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
