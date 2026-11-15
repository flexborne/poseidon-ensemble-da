"""Command-line interface: validate / analyze / run / diagnose / run-cycle.

Thin wrapper: each command loads the case and calls one API operation, which owns validation
and publication; a ValueError (invalid case, unsafe output) is printed and returns exit 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, api
from .config import load_config, validate


def _default_output(config, sub):
    return Path.cwd() / 'results' / f'{config.case_id}-{sub}'


def _cmd_validate(args):
    config = load_config(args.case)
    problems = validate(config)
    if config.geometry.exists():                       # the only place the command reads geometry content
        from .geometry import load_geometry
        try:
            load_geometry(config.geometry)
        except Exception as exc:
            problems.append(str(exc))
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
    result = api.write_analysis(config, out, perturb=not args.no_perturb, seed=args.seed, force=args.force)
    print(f'analyze {config.case_id}: {result.status}')
    print(f'  outputs: {out}')
    return 0


def _cmd_run(args):
    config = load_config(args.case)
    out = Path(args.output) if args.output else _default_output(config, 'run')
    result = api.run(config, out, perturb=not args.no_perturb, seed=args.seed, force=args.force)
    print(f'run {config.case_id}: {result.status}')
    for step in result.steps:
        print(f'  {step["step"]}: {step["status"]}')
    print(f'  outputs: {out}')
    return 0


def _cmd_diagnose(args):
    config = load_config(args.case)
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


def _add_output(sp, perturb=False):
    sp.add_argument('--output', '-o', default=None, help='output directory')
    sp.add_argument('--force', action='store_true', help='overwrite a non-empty output directory')
    if perturb:
        sp.add_argument('--no-perturb', action='store_true', help='deterministic (no perturbations)')
        sp.add_argument('--seed', type=int, default=None, help='RNG seed for perturbations')


def build_parser():
    p = argparse.ArgumentParser(prog='poseidon-da', description='POSEIDON ensemble data assimilation')
    p.add_argument('--version', action='version', version=f'poseidon-da {__version__}')
    sub = p.add_subparsers(dest='command', required=True)

    sv = sub.add_parser('validate', help='check a case and its inputs')
    sv.add_argument('case', help='path to case.toml')
    sv.set_defaults(func=_cmd_validate)

    sa = sub.add_parser('analyze', help='compute and publish the analysis (NPZ/JSON + manifest)')
    sa.add_argument('case', help='path to case.toml'); _add_output(sa, perturb=True)
    sa.set_defaults(func=_cmd_analyze)

    sr = sub.add_parser('run', help='analyze + restarts + outputs + manifest')
    sr.add_argument('case', help='path to case.toml'); _add_output(sr, perturb=True)
    sr.set_defaults(func=_cmd_run)

    sd = sub.add_parser('diagnose', help='compare the forecast against observations')
    sd.add_argument('case', help='path to case.toml'); _add_output(sd)
    sd.set_defaults(func=_cmd_diagnose)

    sc = sub.add_parser('run-cycle', help='sequential NODA/filter/smoother assimilation-forecast cycle')
    sc.add_argument('case', help='path to case.toml'); _add_output(sc, perturb=True)
    sc.add_argument('--mode', choices=['noda', 'filter', 'smoother'], default=None,
                    help='cycle mode (default from [cycle] in the config)')
    sc.add_argument('--dates', nargs='+', default=None, help='override the date sequence')
    sc.set_defaults(func=_cmd_run_cycle)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
