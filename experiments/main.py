"""Run the main experiment on one model.

The main experiment evaluates VTI and REVIS, each with its random control, next to the unmodified
model, on POPE adversarial with its RePOPE labels, on CHAIR and on MME. The logit lens is recorded
with every answer, the residual shift of every method is measured on a few images, and everything
is scored. A published setting is used on the model it is published for and the defaults of the
official code on the others.

```bash
uv run python experiments/main.py --model llava_15 --limit 16   # smoke test
uv run python experiments/main.py --model llava_15
```

The run continues where an earlier one stopped. Its three jobs are independent, so they can run
side by side on one results folder and be scored once at the end.

```bash
uv run python experiments/main.py --model llava_15 --jobs baseline --no-score
uv run python experiments/main.py --model llava_15 --jobs vti --no-score
uv run python experiments/main.py --model llava_15 --jobs revis --no-score
uv run python experiments/main.py --model llava_15 --score-only
```
"""

import argparse
import os
import sys
from pathlib import Path

from swce.baselines import Unmodified
from swce.data.paths import DATA_ENV
from swce.diagnostics import diagnose, summarize_results, write_readings
from swce.evaluation.__main__ import load_factory, load_steerers
from swce.evaluation.runner import EvaluationConfig, dumps, generate, score_results
from swce.logging import configure_logging
from swce.models.subject import Subject
from swce.steering import Steerer

METHODS = Path(__file__).parents[1] / "methods"
"""Folder of the method files."""
TASKS = ("pope-adversarial", "repope-adversarial", "chair", "mme")
"""Tasks of the experiment."""
SHIFT_TASK = "pope-adversarial"
"""Task on whose first images the residual shift is measured."""
JOBS = {
    "baseline": (),
    "vti": ("vti.py:build", "vti.py:control"),
    "revis": ("revis.py:build", "revis.py:control"),
}
"""Factories of every job, in the method folder. The baseline is the unmodified model."""


def main(argv: list[str] | None = None) -> int:
    """Run the jobs of the main experiment and score them.

    Args:
        argv: Command line arguments without the program name, or None to read `sys.argv`.

    Returns:
        Exit status, zero on success.
    """
    parser = argparse.ArgumentParser(prog="experiments/main.py", description=__doc__)
    parser.add_argument(
        "--model",
        default="llava_15",
        help="factory of `swce.models`, such as llava_15, qwen3_vl_8b, llava_onevision_15_8b or "
        "llava_15_rehearsal, or any `module:factory`",
    )
    parser.add_argument("--jobs", nargs="*", choices=list(JOBS), default=list(JOBS))
    parser.add_argument("--limit", type=int, default=None, help="images per task, for smoke tests")
    parser.add_argument(
        "--shift-images", type=int, default=50, help="images of the residual shift, 0 to skip it"
    )
    parser.add_argument("--results", type=Path, default=Path("results") / "main")
    parser.add_argument("--data", type=Path, default=None, help="data root, default SWCE_DATA")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--no-score", action="store_true", help="leave the scoring to a later call")
    stage.add_argument("--score-only", action="store_true", help="score what was already answered")
    args = parser.parse_args(argv)
    configure_logging()
    if args.data is not None:
        os.environ[DATA_ENV] = str(args.data)
    config = EvaluationConfig(
        tasks=TASKS, limit=args.limit, results=args.results, data=args.data, resume=True, lens=True
    )
    if not args.score_only:
        reference = args.model if ":" in args.model else f"swce.models:{args.model}"
        subject = load_factory(reference)()
        if not isinstance(subject, Subject):
            raise TypeError(f"{reference} built {type(subject).__name__}, not a Subject")
        factories = [f"{METHODS / name}" for job in args.jobs for name in JOBS[job]]
        methods: list[Steerer] = load_steerers(factories) if factories else []
        baseline: list[Steerer] = [Unmodified()] if "baseline" in args.jobs else []
        generate(subject, [*baseline, *methods], config)
        images = min(args.shift_images, args.limit or args.shift_images)
        if methods and images:
            readings = diagnose(subject, methods, SHIFT_TASK, limit=images, data=args.data)
            write_readings(readings, config.root, SHIFT_TASK)
    if not args.no_score:
        rows = score_results(config.root, data=args.data, bootstrap=config.bootstrap)
        summarize_results(config.root)
        print(dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
