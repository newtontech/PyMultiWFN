"""Compatibility wrapper for the manifest-driven consistency verifier."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from consistency_verifier.verifier import main


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] in {"smoke", "pr", "full"}:
        args = ["run", "--suite", args[0], *args[1:]]
    elif not args or args[0].startswith("-"):
        args = ["run", *args]
    raise SystemExit(main(args))
