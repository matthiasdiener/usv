"""Result serialization and loading."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Any

__all__ = ["save_results", "load_results"]


def _get_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def save_results(
    all_results: dict[str, Any],
    *,
    label: str | None = None,
    results_dir: str | None = None,
) -> str:
    """Write raw per-call samples to ``<results_dir>/<hash>[-<label>].json``.

    Returns the path written.
    """
    commit = _get_commit_hash()
    results_dir = results_dir or os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    suffix = ""
    if label:
        suffix = "-" + re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    path = os.path.join(results_dir, f"{commit[:8]}{suffix}.json")

    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    else:
        data = {
            "commit_hash": commit,
            "timestamp": int(time.time()),
            "results": {},
        }
    data["results"].update(all_results)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_results(path: str) -> dict[str, Any]:
    """Load a result JSON produced by :func:`save_results`."""
    with open(path) as f:
        return json.load(f)
