#!/usr/bin/env python3
"""Compatibility wrapper for priorauth.apps.agent.setup_rag."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __name__ == "__main__":
    runpy.run_module("priorauth.apps.agent.setup_rag", run_name="__main__")
