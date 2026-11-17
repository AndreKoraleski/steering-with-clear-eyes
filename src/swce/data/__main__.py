"""Inspect, download and verify the benchmark files.

`swce-data benchmarks` lists every benchmark with the local state of its files, `swce-data tasks`
lists the tasks, `swce-data fetch` downloads pinned files ahead of an evaluation and `swce-data
verify` checks that every host still serves what the registry pins.
"""

import argparse
import sys
from pathlib import Path

from swce.data.download import fetch, matches
from swce.data.paths import default_data_root
from swce.data.verify import RemoteState, remote_state
from swce.registry import Registry, load_registry


def _benchmarks(registry: Registry, root: Path) -> int:
    """Print every benchmark with the local state of its pinned files."""
    for card in registry.benchmarks:
        print(f"{card.key}: {card.name}. {card.licence}")
        for file in card.files:
            path = root / file.path
            state = "missing" if not path.exists() else "ok" if matches(path, file) else "differs"
            print(f"  {state:8} {file.path} ({file.size} bytes)")
    return 0


def _tasks(registry: Registry) -> int:
    """Print every task with its kind and primary metric."""
    for card in registry.benchmarks:
        for task in card.tasks:
            print(f"{task.key}: {task.kind.value}, {task.primary}. {task.description}")
    return 0


def _fetch(registry: Registry, root: Path, keys: list[str]) -> int:
    """Download the pinned files of the named benchmarks, or of all of them."""
    cards = [registry.benchmark(key) for key in keys] if keys else list(registry.benchmarks)
    for card in cards:
        for file in card.files:
            print(fetch(file, root))
    return 0


def _verify(registry: Registry) -> int:
    """Print the remote state of every pinned file and fail unless all are unchanged."""
    failed = False
    for card in registry.benchmarks:
        for file in card.files:
            state = remote_state(file)
            failed |= state is not RemoteState.OK
            print(f"{state.value:11} {file.path}")
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    """Run one of the data commands.

    Args:
        argv: Command line arguments without the program name, or None to read `sys.argv`.

    Returns:
        Exit status, one when `verify` found a changed or unreachable file, and zero otherwise.
    """
    parser = argparse.ArgumentParser(prog="swce-data", description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="data root, default SWCE_DATA")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("benchmarks", help="list benchmarks and the local state of their files")
    commands.add_parser("tasks", help="list tasks")
    fetching = commands.add_parser("fetch", help="download pinned files")
    fetching.add_argument("keys", nargs="*", help="benchmark keys, default all")
    commands.add_parser("verify", help="check that the hosts serve the pinned files")
    args = parser.parse_args(argv)
    registry = load_registry()
    root = args.data or default_data_root()
    if args.command == "benchmarks":
        return _benchmarks(registry, root)
    if args.command == "tasks":
        return _tasks(registry)
    if args.command == "fetch":
        return _fetch(registry, root, args.keys)
    return _verify(registry)


if __name__ == "__main__":
    sys.exit(main())
