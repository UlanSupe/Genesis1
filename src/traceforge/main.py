"""Console entry point."""
from __future__ import annotations

import sys

from .cli import run


def main(argv=None) -> int:
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
