"""Validate the registry and keep its documentation current.

`swce-registry` validates the registry and renders docs/benchmarks.md, exiting with status one
when it rewrote the page. `swce-registry --check` writes nothing and only reports whether the page
is current, for continuous integration.
"""

import argparse
import sys

from swce.registry.docs import DOCS_PATH, render, write_docs
from swce.registry.store import load_registry


def main(argv: list[str] | None = None) -> int:
    """Validate the registry and render or check the benchmark page.

    Args:
        argv: Command line arguments without the program name, or None to read `sys.argv`.

    Returns:
        Exit status, one when the page was rewritten or, under `--check`, when it is stale, and
        zero otherwise.
    """
    parser = argparse.ArgumentParser(prog="swce-registry", description=__doc__)
    parser.add_argument("--check", action="store_true", help="report staleness without writing")
    args = parser.parse_args(argv)
    registry = load_registry()
    if args.check:
        current = DOCS_PATH.exists() and DOCS_PATH.read_text(encoding="utf-8") == render(registry)
        if not current:
            print("docs/benchmarks.md is stale, run swce-registry", file=sys.stderr)
        return int(not current)
    written = write_docs(registry)
    if written:
        print("docs/benchmarks.md regenerated")
    return int(written)


if __name__ == "__main__":
    sys.exit(main())
