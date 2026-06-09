"""Command-line entry point for ``python -m consistency_verifier``."""

from .verifier import main


if __name__ == "__main__":
    raise SystemExit(main())
