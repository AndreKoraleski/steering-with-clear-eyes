"""Read the internals of a model under steering methods on the items of a task.

`swce-diagnose --subject MODULE:FACTORY METHOD ... --task KEY` applies the logit lens and the
residual shift to every method and to the unmodified model, block by block. The readings of
every item are written under `diagnostics` in the results directory, one file per method
setting, so that jobs on several machines can be copied into one folder. `--summarize-only`
needs no model and summarizes every reading found there. `--probe` fits linear probes of the
expected answer on every block instead, cross-validated over images, and writes them to
`diagnostics/MODEL/probe-TASK.csv`.

Three readings localize where a wrong answer of POPE comes from, on the unmodified model.
`--prior` asks every question without its image and sets the answers against COCO co-occurrence,
`--encoder` probes the vision encoder for the object of every question, and `--knockout` blocks
the attention between the answer, the text and the image in windows of layers. They write
`prior-TASK.csv`, `encoder-TASK.csv` and `knockout-TASK.csv`, with summaries, under
`diagnostics/MODEL`. No answer is generated and nothing is scored.
"""

import argparse
import os
import sys
from pathlib import Path

from swce.data.paths import DATA_ENV
from swce.diagnostics.probe import FOLDS, probe_task, write_probes
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
    parser.add_argument(
        "--probe", action="store_true", help="fit linear probes instead of the lens and shift"
    )
    parser.add_argument("--folds", type=int, default=FOLDS, help="folds of the probes over images")
    parser.add_argument(
        "--prior", action="store_true", help="answer every question without its image"
    )
    parser.add_argument("--encoder", action="store_true", help="probe the vision encoder")
    parser.add_argument("--knockout", action="store_true", help="block attention routes by window")
    parser.add_argument(
        "--knockout-items", type=int, default=200, help="wrong yeses and correct answers to block"
    )
    parser.add_argument(
        "--encoder-train", type=int, default=5000, help="COCO train2014 images the probes fit on"
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
    if args.prior or args.encoder or args.knockout:
        return localize(subject, args)
    steerers = load_steerers(args.methods)
    if args.probe:
        probes = probe_task(
            subject,
            steerers,
            args.task,
            limit=args.limit or None,
            folds=args.folds,
            data=args.data,
        )
        for path in write_probes(probes, args.results):
            print(path)
        print(probes.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return 0
    readings = diagnose(subject, steerers, args.task, limit=args.limit or None, data=args.data)
    write_readings(readings, args.results, args.task)
    for path in summarize_results(args.results):
        print(path)
    print(summarize(readings).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return 0


def localize(subject: Subject, args: argparse.Namespace) -> int:
    """Run the readings that localize a wrong answer, on the unmodified model."""
    from swce.diagnostics.coco import categories, image_objects
    from swce.diagnostics.encoder import encoder_task
    from swce.diagnostics.knockout import knockout_task, summarize_knockout
    from swce.diagnostics.prior import margins, prior_task, write_frames
    from swce.models.hf import HuggingFaceSubject
    from swce.sae.harvest import descriptions, fetch_all, open_image

    if not isinstance(subject, HuggingFaceSubject):
        raise TypeError(f"{subject.name} is not a HuggingFaceSubject")
    folder = args.results / "diagnostics" / subject.name
    limit = args.limit or None
    frames = []
    answers = None
    if args.prior:
        answers, summary = prior_task(subject, args.task, limit=limit, data=args.data)
        frames += [(f"prior-{args.task}", answers), (f"prior-{args.task}-summary", summary)]
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if args.knockout:
        if answers is None:
            answers = margins(subject, args.task, limit=limit, data=args.data)
        knocked = knockout_task(
            subject, args.task, answers, per_kind=args.knockout_items, data=args.data
        )
        summary = summarize_knockout(knocked)
        frames += [(f"knockout-{args.task}", knocked), (f"knockout-{args.task}-summary", summary)]
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if args.encoder:
        train = image_objects("train")
        images = list(dict.fromkeys(example.image for example in descriptions()))
        images = images[: args.encoder_train]
        fetch_all(images)
        training = [
            (lambda name=name: open_image(name), train.get(name, frozenset())) for name in images
        ]
        probed = encoder_task(
            subject, args.task, training, categories(), limit=limit, data=args.data
        )
        frames.append((f"encoder-{args.task}", probed))
        print(probed.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    for path in write_frames(frames, folder):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
