"""Evaluate steering methods on a vision language model.

`swce-evaluate --subject MODULE:FACTORY METHOD ...` runs every method on the selected tasks. A
reference is `module:factory` or `path/to/file.py:factory`, and the factory takes no argument.
The factory of a subject returns a `Subject`, and that of a method a `Steerer` or a list of them,
one per published setting. Without methods, the unmodified model alone is evaluated.

An evaluation has three stages, which run together by default and apart on request, so that
several machines can share the work. `--prepare-only` runs the recipes of the methods, such as
fitting directions or calibrating a threshold. `--generate-only` writes the answers of the
model. `--score-only` needs no model and scores every answer found under the results
directory, once the folders written elsewhere are copied into it.
"""

import argparse
import importlib
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from swce.baselines.reference import Unmodified
from swce.data.paths import DATA_ENV
from swce.evaluation.analysis import MARGIN
from swce.evaluation.runner import (
    EvaluationConfig,
    dry_run,
    dumps,
    evaluate,
    generate,
    prepare,
    score_results,
)
from swce.logging import configure_logging
from swce.metrics.bootstrap import RESAMPLES
from swce.models.subject import Subject
from swce.steering.steerer import Steerer


def load_factory(reference: str) -> Callable[[], Any]:
    """Factory named by a reference.

    Args:
        reference: `module:factory`, or `path/to/file.py:factory` for a file outside any package.

    Returns:
        The callable the reference names.

    Raises:
        ValueError: If the reference has no factory name or its file cannot be imported.
        AttributeError: If the module defines no such factory.
    """
    source, separator, name = reference.rpartition(":")
    if not separator or not source:
        raise ValueError(f"{reference!r} is not of the form module:factory")
    if source.endswith(".py"):
        path = Path(source)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(source)
    factory: Callable[[], Any] = getattr(module, name)
    return factory


def load_steerers(references: list[str]) -> list[Steerer]:
    """Method settings built by the factories of the references.

    Args:
        references: References of method factories. Empty selects the unmodified model.

    Returns:
        Every method setting, in the order given.

    Raises:
        TypeError: If a factory returns something other than steerers.
    """
    if not references:
        return [Unmodified()]
    steerers: list[Steerer] = []
    for reference in references:
        built = load_factory(reference)()
        for steerer in built if isinstance(built, (list, tuple)) else [built]:
            if not isinstance(steerer, Steerer):
                raise TypeError(f"{reference} built {type(steerer).__name__}, not a Steerer")
            steerers.append(steerer)
    return steerers


def main(argv: list[str] | None = None) -> int:
    """Run an evaluation from the command line.

    Args:
        argv: Command line arguments without the program name, or None to read `sys.argv`.

    Returns:
        Exit status, zero on success.
    """
    parser = argparse.ArgumentParser(prog="swce-evaluate", description=__doc__)
    parser.add_argument("methods", nargs="*", help="references of method factories")
    parser.add_argument("--subject", help="reference of the subject factory")
    parser.add_argument("--tasks", nargs="*", default=[], help="patterns of task keys")
    parser.add_argument("--limit", type=int, default=None, help="images per task, for smoke tests")
    parser.add_argument("--bootstrap", type=int, default=RESAMPLES, help="resamples, 0 to skip")
    parser.add_argument("--seed", type=int, default=0, help="seed of the resampling")
    parser.add_argument(
        "--margin", type=float, default=MARGIN, help="non inferiority margin of the capability"
    )
    parser.add_argument("--results", type=Path, default=Path("results"), help="results directory")
    parser.add_argument("--data", type=Path, default=None, help="data root, default SWCE_DATA")
    parser.add_argument("--resume", action="store_true", help="reuse records already written")
    parser.add_argument("--lens", action="store_true", help="record the logit lens with answers")
    parser.add_argument("--dry-run", action="store_true", help="list the tasks and stop")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--prepare-only", action="store_true", help="run the recipes of the methods and stop"
    )
    stage.add_argument(
        "--generate-only", action="store_true", help="write the answers without scoring them"
    )
    stage.add_argument(
        "--score-only", action="store_true", help="score the answers under the results directory"
    )
    args = parser.parse_args(argv)
    configure_logging()
    if args.data is not None:
        # The steps of the methods read the data root from the environment.
        os.environ[DATA_ENV] = str(args.data)
    config = EvaluationConfig(
        tasks=tuple(args.tasks),
        limit=args.limit,
        bootstrap=args.bootstrap,
        seed=args.seed,
        results=args.results,
        data=args.data,
        resume=args.resume,
        lens=args.lens,
    )
    if args.dry_run:
        print("\n".join(dry_run(config)))
        return 0
    if args.score_only:
        rows = score_results(
            config.root,
            data=config.data,
            bootstrap=config.bootstrap,
            seed=config.seed,
            margin=args.margin,
        )
        print(dumps(rows))
        return 0
    if args.subject is None:
        parser.error("--subject is required unless --dry-run or --score-only is given")
    subject = load_factory(args.subject)()
    if not isinstance(subject, Subject):
        raise TypeError(f"{args.subject} built {type(subject).__name__}, not a Subject")
    steerers = load_steerers(args.methods)
    if args.prepare_only:
        for steerer in prepare(subject, steerers):
            print(f"{steerer.name} {steerer.setting}: ready")
    elif args.generate_only:
        for folder in generate(subject, steerers, config):
            print(folder)
    else:
        print(dumps(evaluate(subject, steerers, config)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
