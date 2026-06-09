#!/usr/bin/env python3
"""Compatibility wrapper for the smoke consistency verifier suite."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from consistency_verifier.verifier import main


if __name__ == "__main__":
    raise SystemExit(main(["run", "--suite", "smoke", *sys.argv[1:]]))
