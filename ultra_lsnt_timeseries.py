"""Compatibility wrapper for legacy script imports.

The public repository keeps the implementation under ``src/models`` but many
experiment scripts import ``ultra_lsnt_timeseries`` from the repository root.
"""

import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.models.ultra_lsnt_timeseries import *  # noqa: F401,F403
