"""Filesystem sealing helpers.

Kept in their own module so the classification runner can import the lock
without importing the module that reads the benchmark. A process running as the
same user cannot traverse a directory with mode 000 on POSIX systems, which
blocks accidental reads for the duration of a run.

Honest limits: this is a barrier, not a sandbox. It does not stop root, and it
does not stop a deliberately malicious process that waits for the unlock. See
SPEC.md section 3 for the layered argument.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def _sealed_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "sealed"


def lock_sealed(out_dir: str | Path) -> bool:
    path = _sealed_dir(out_dir)
    if not path.exists():
        return False
    try:
        os.chmod(path, 0o000)
        return True
    except OSError:
        return False


def unlock_sealed(out_dir: str | Path) -> bool:
    path = _sealed_dir(out_dir)
    if not path.exists():
        return False
    try:
        os.chmod(path, stat.S_IRWXU)
        return True
    except OSError:
        return False


def sealed_is_locked(out_dir: str | Path) -> bool:
    path = _sealed_dir(out_dir)
    if not path.exists():
        return False
    return (path.stat().st_mode & stat.S_IRWXU) == 0