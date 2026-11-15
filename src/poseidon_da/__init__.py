"""POSEIDON ensemble data assimilation — installable package.

Public API:
    from poseidon_da import load_config, analyze, run
"""
from __future__ import annotations

__version__ = '0.1.0'

from .api import analyze, diagnose, load_config, run, run_cycle, write_analysis  # noqa: E402
from .domain import AnalysisResult, AnalysisWindow, RunResult  # noqa: E402

__all__ = ['load_config', 'analyze', 'write_analysis', 'run', 'diagnose', 'run_cycle',
           'AnalysisResult', 'AnalysisWindow', 'RunResult', '__version__']
