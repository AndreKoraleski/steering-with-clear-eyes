"""Single entry point of the command line tools.

`swce evaluate`, `swce diagnose`, `swce data` and `swce registry` forward to the commands
`swce-evaluate`, `swce-diagnose`, `swce-data` and `swce-registry`. `swce results` rewrites the
Markdown summary of a leaderboard from its CSV file, for instance after merging rows from another
machine.
"""

import argparse
import sys
from pathlib import Path

from swce.data.__main__ import main as data_main
from swce.diagnostics.__main__ import main as diagnose_main
from swce.evaluation.__main__ import main as evaluate_main
from swce.evaluation.leaderboard import Leaderboard
from swce.registry.__main__ import main as registry_main

COMMANDS = {
    "evaluate": evaluate_main,
    "diagnose": diagnose_main,
    "data": data_main,
    "registry": registry_main,
}
"""Commands that forward their arguments to another entry point."""


def main(argv: list[str] | None = None) -> int:
    """Dispatch a command.

    Args:
        argv: Command line arguments without the program name, or None to read `sys.argv`.

    Returns:
        Exit status of the command.
    """
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in COMMANDS:
        return COMMANDS[arguments[0]](arguments[1:])
    parser = argparse.ArgumentParser(prog="swce", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        commands.add_parser(name, help=f"same as swce-{name}")
    results = commands.add_parser("results", help="rewrite the summary of a leaderboard")
    results.add_argument("--results", type=Path, default=Path("results"), help="results directory")
    args = parser.parse_args(arguments)
    Leaderboard(args.results / "leaderboard.csv").refresh()
    return 0


if __name__ == "__main__":
    sys.exit(main())
