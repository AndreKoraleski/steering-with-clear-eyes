"""Diagnostics of steering methods over the items of a task."""

import fnmatch
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from swce.baselines.reference import Unmodified
from swce.diagnostics.lens import logit_lens, observe, residual_shift
from swce.evaluation.runner import prepare
from swce.metrics.yes_no import auroc
from swce.models.subject import Subject
from swce.registry import Registry
from swce.steering.steerer import Steerer
from swce.tasks.task import Task

KEYS = ["model", "method", "setting", "layer"]
"""Columns a summary is grouped by."""


def diagnose(
    subject: Subject,
    steerers: Sequence[Steerer],
    task: str,
    *,
    limit: int | None = None,
    data: Path | None = None,
    registry: Registry | None = None,
) -> pd.DataFrame:
    """Logit lens and residual shift of every method on every item of a task.

    The unmodified model is always read, since every shift is measured against it. The recipes
    of the methods run first, and a method that is not meant for the task is left out.

    Args:
        subject: Model to observe.
        steerers: Method settings to diagnose.
        task: Key of the task whose items are read.
        limit: Number of leading images to read, or None for the whole task.
        data: Data root, or None for the default one.
        registry: Registry defining the task, or None for the committed one.

    Returns:
        One row per method setting, item and decoder block, with the columns of `logit_lens` and
        of `residual_shift`, the `fingerprint` of the setting, the `item` and its `label`.
    """
    bound = Task(task, data, registry)
    chosen = [
        steerer
        for steerer in prepare(subject, steerers)
        if not isinstance(steerer, Unmodified)
        and (not steerer.tasks or any(fnmatch.fnmatch(task, p) for p in steerer.tasks))
    ]
    frames = []
    for item in tqdm(bound.items(limit), desc=f"diagnose {task}"):
        image = bound.image(item)
        unmodified = observe(subject, image, item.prompt)
        for steerer in (Unmodified(), *chosen):
            under = None if isinstance(steerer, Unmodified) else steerer
            states = unmodified if under is None else observe(subject, image, item.prompt, under)
            frame = logit_lens(subject, states).merge(residual_shift(states, unmodified))
            frames.append(
                frame.assign(
                    model=subject.name,
                    method=steerer.name,
                    setting=steerer.setting,
                    fingerprint=steerer.fingerprint,
                    item=item.id,
                    label=item.label,
                )
            )
    return pd.concat(frames, ignore_index=True)


def write_readings(readings: pd.DataFrame, root: Path, task: str) -> list[Path]:
    """Write the readings of a task, one file per model and method setting.

    Args:
        readings: Rows returned by `diagnose`.
        root: Results folder.
        task: Key of the task that was read.

    Returns:
        The files written under `diagnostics`, which jobs on several machines can copy into one
        results folder without overwriting one another.
    """
    written = []
    for (model, method, mark), group in readings.groupby(["model", "method", "fingerprint"]):
        folder = root / "diagnostics" / str(model) / f"{method}-{mark}"
        folder.mkdir(parents=True, exist_ok=True)
        group.to_csv(folder / f"{task}.csv", index=False, lineterminator="\n")
        written.append(folder / f"{task}.csv")
    return written


def summarize_results(root: Path) -> list[Path]:
    """Summarize every reading found under a results folder.

    Args:
        root: Results folder holding `diagnostics`.

    Returns:
        The summaries written, one per model and task, next to the folders of the settings.
    """
    found: dict[tuple[Path, str], list[pd.DataFrame]] = {}
    for path in sorted(root.glob("diagnostics/*/*/*.csv")):
        frame = pd.read_csv(path, dtype={"fingerprint": str, "item": str})
        found.setdefault((path.parent.parent, path.stem), []).append(frame)
    written = []
    for (folder, task), frames in found.items():
        target = folder / f"{task}-summary.csv"
        summarize(pd.concat(frames, ignore_index=True)).to_csv(
            target, index=False, lineterminator="\n"
        )
        written.append(target)
    return written


def summarize(readings: pd.DataFrame) -> pd.DataFrame:
    """Readings of a task averaged over its items, per method setting and decoder block.

    Args:
        readings: Rows returned by `diagnose`.

    Returns:
        One row per method setting and block with `auroc`, how well the margin of the logit lens
        separates the items whose answer is yes from those whose answer is no, the mean margin
        of each kind of item as `margin_yes` and `margin_no`, and the mean `cosine`, `norm_ratio`
        and `distance` of the residual shift. The area is NaN for a task without yes or no
        labels.
    """
    rows = []
    for key, group in readings.groupby(KEYS, sort=False):
        labelled = group[group.label.isin(["yes", "no"])]
        positive = (labelled.label == "yes").to_numpy()
        margins = labelled.margin.to_numpy(dtype=np.float64)
        rows.append(
            {
                **dict(zip(KEYS, key, strict=True)),
                "auroc": auroc(margins, positive) if positive.size else float("nan"),
                "margin_yes": float(margins[positive].mean()) if positive.any() else float("nan"),
                "margin_no": float(margins[~positive].mean())
                if (~positive).any()
                else float("nan"),
                "cosine": float(group.cosine.mean()),
                "norm_ratio": float(group.norm_ratio.mean()),
                "distance": float(group.distance.mean()),
            }
        )
    return pd.DataFrame(rows)
