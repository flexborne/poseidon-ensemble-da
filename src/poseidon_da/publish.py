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


@contextmanager
def staged_publish(dest, force: bool = False):
    """Yield a staging dir; on clean exit publish it to `dest` atomically.

    Refuses a non-empty `dest` unless force=True. On any exception the staging dir is
    removed and `dest` is left untouched.
    """
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()) and not force:
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
