import sys
from typing import Optional

from .. import cli as core_cli


def main(argv: Optional[list[str]] = None) -> None:
    """Thin wrapper around the legacy edge CLI.

    Example:
        python -m project.src.edge.cli_edge baseline_glm ...
    """
    if argv is None:
        argv = sys.argv[1:]
    core_cli._legacy_main(argv)


if __name__ == "__main__":
    main()

