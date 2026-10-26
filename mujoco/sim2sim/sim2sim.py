#!/usr/bin/env python
"""Entry point for the modular sim2sim evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``python path/to/sim2sim.py`` work from any current directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dogsim.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
