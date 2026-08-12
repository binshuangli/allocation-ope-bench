"""Git utilities for stamping results with a reproducible hash."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_hash(short: bool = True) -> str:
    """Return current HEAD commit hash, or 'unknown' if not in a git repo.

    Runs git anchored to this package's own directory (not the process cwd) so the
    recorded provenance is correct even when an experiment is launched from a
    different working directory.
    """
    repo_dir = Path(__file__).resolve().parent
    try:
        cmd = ["git", "rev-parse", "--short" if short else "", "HEAD"]
        cmd = [c for c in cmd if c]  # drop empty string when not short
        return (
            subprocess.check_output(cmd, stderr=subprocess.DEVNULL, cwd=repo_dir).decode().strip()
        )
    except Exception:
        return "unknown"
