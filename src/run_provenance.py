"""
Provenance record for a pipeline run: which code, which environment, which data.

A run is only reproducible if its outputs can be tied to exact code and inputs. The
manifest written here records the git commit and any uncommitted modifications (so a run
made from a modified working tree says so), the interpreter and package versions, the CPU
architecture (LSTM training is not reproducible across architectures), the run settings,
and SHA-256 hashes of the input data and any saved checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "cvxpy", "cvxpylayers", "diffcp"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    # only trailing newlines: porcelain status lines start with a significant space (' M file')
    return out.stdout.rstrip("\n") if out.returncode == 0 else None


def git_state(root: Path) -> dict:
    """Commit, branch and the list of modified or untracked-but-not-ignored files. The
    list is the point: an empty list means the commit is exactly the code that ran."""
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "uncommitted_files": None, "note": "not a git checkout"}
    status = _git(root, "status", "--porcelain") or ""
    return {
        "commit": commit,
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_files": sorted(line[3:] for line in status.splitlines() if line.strip()),
    }


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }


def write_run_manifest(path: Path, root: Path, settings: dict, hashed_files: dict[str, Path], git: dict | None = None) -> dict:
    """Write the manifest as JSON and return it. `hashed_files` maps a label to a path; labels
    whose file is missing are recorded as null rather than raising. Pass `git` captured at the
    START of the run: taken afterwards, the run's own output files would show as uncommitted."""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git if git is not None else git_state(root),
        "environment": environment(),
        "settings": settings,
        "sha256": {label: (file_sha256(p) if Path(p).exists() else None) for label, p in hashed_files.items()},
    }
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
