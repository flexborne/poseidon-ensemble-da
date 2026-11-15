"""Transactional output publication: build in a sibling staging dir, publish atomically.

A run is assembled in a staging directory on the SAME filesystem as the destination and
published with a single directory rename, backing up any existing destination so a failure
mid-publish leaves either the previous complete output or nothing — never a mixed set.
"""
from __future__ import annotations

import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path


def _within(a: Path, b: Path) -> bool:
    """True if `a` is `b` or lives inside `b` (both already resolved)."""
    try:
        a.relative_to(b)
        return True
    except ValueError:
        return False


def assert_safe_dest(dest, protected) -> None:
    """Refuse an output directory that overlaps a protected input path.

    Publication replaces `dest` wholesale (rename + backup), so a destination that equals,
    contains, or sits inside the dataset / measurements / geometry would destroy or
    contaminate the inputs. This is the single check both `run` and `run_cycle` apply before
    writing anything.
    """
    dest = Path(dest).resolve()
    for raw in protected:
        p = Path(raw).resolve()
        if dest == p or _within(dest, p) or _within(p, dest):
            raise ValueError(
                f'output directory {dest} overlaps a protected input ({p}); '
                f'choose an output path outside the dataset')


@contextmanager
def staged_publish(dest, force: bool = False):
    """Yield a staging dir; on clean exit publish it to `dest` atomically.

    Refuses a non-empty `dest` unless force=True. On any exception the staging dir is
    removed and `dest` is left untouched.
    """
    dest = Path(dest)
    if dest.exists() and not dest.is_dir():
        raise NotADirectoryError(f'output path is not a directory: {dest}')
    if dest.is_dir() and any(dest.iterdir()) and not force:
        raise FileExistsError(f'output directory is not empty: {dest} (use force=True to overwrite)')
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(f'.{dest.name}.staging-{uuid.uuid4().hex[:8]}')
    staging.mkdir()
    try:
        yield staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    backup = None
    if dest.exists():
        backup = dest.with_name(f'.{dest.name}.bak-{uuid.uuid4().hex[:8]}')
        os.rename(dest, backup)
    try:
        os.rename(staging, dest)                 # atomic on a single filesystem
    except BaseException:
        if backup is not None:
            os.rename(backup, dest)              # restore the previous complete output
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
