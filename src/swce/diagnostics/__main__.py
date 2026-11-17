"""Read the internals of a model under steering methods on the items of a task.

`swce-diagnose --subject MODULE:FACTORY METHOD ... --task KEY` applies the logit lens and the
residual shift to every method and to the unmodified model, block by block. The readings of
every item are written under `diagnostics` in the results directory, one file per method
setting, so that jobs on several machines can be copied into one folder. `--summarize-only`
needs no model and summarizes every reading found there. No answer is generated and nothing is
scored.
"""

import argparse
import os
import sys
from pathlib import Path

from swce.data.paths import DATA_ENV
from swce.diagnostics.run import diagnose, summarize, summarize_results, write_readings
from swce.evaluation.__main__ import load_factory, load_steerers
from swce.logging import configure_logging
from swce.models.subject import Subject


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostics from the command line.

    Args:
        argv: Command line arguments without the program name, or None to read `sys.argv`.

    Returns:
        Exit status, zero on success.
    """
    parser = argparse.ArgumentParser(prog="swce-diagnose", description=__doc__)
    parser.add_argument("methods", nargs="*", help="references of method factories")
    parser.add_argument("--subject", help="reference of the subject factory")
    parser.add_argument(
        "--summarize-only", action="store_true", help="summarize the readings already written"
    )
    parser.add_argument("--task", default="pope-adversarial", help="key of the task to read")
    parser.add_argument("--limit", type=int, default=50, help="images to read, 0 for all")
    parser.add_argument("--results", type=Path, default=Path("results"), help="results directory")
    parser.add_argument("--data", type=Path, default=None, help="data root, default SWCE_DATA")
    args = parser.parse_args(argv)
    configure_logging()
    if args.data is not None:
        os.environ[DATA_ENV] = str(args.data)
    if args.summarize_only:
        for path in summarize_results(args.results):
            print(path)
        return 0
    if args.subject is None:
        parser.error("--subject is required unless --summarize-only is given")
    subject = load_factory(args.subject)()
    if not isinstance(subject, Subject):
        raise TypeError(f"{args.subject} built {type(subject).__name__}, not a Subject")
    steerers = load_steerers(args.methods)
    readings = diagnose(subject, steerers, args.task, limit=args.limit or None, data=args.data)
    write_readings(readings, args.results, args.task)
    for path in summarize_results(args.results):
        print(path)
    print(summarize(readings).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
