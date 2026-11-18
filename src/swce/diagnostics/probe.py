"""Linear probes of the residual stream for the answer a yes or no question expects.

A probe is a logistic regression fitted, block by block, on the residual stream at the position
that predicts the first token of the answer, to tell the questions whose expected answer is yes
from those whose answer is no. Unlike the logit lens, it does not read a block through the head
of the model, so it also finds what an early block holds in a form the head cannot read.

Every probe is evaluated by cross-validation over images, so that no image is seen both in
fitting and in evaluation. Under a steering method, three readings are reported per block.

- The probe fitted and evaluated on the states under the method, which says whether the method
  keeps, removes or adds what the stream holds about the answer.
- The probe of the unmodified model evaluated on the states under the method, which says whether
  the method moves the stream out of the region the unmodified probe reads.
- The mean change of the score of that probe, which says whether the method moves the stream
  along the direction that separates the answers, toward yes or toward no.

The regularization is the default of scikit-learn, on standardized states, and is not tuned.
"""

import fnmatch
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from swce.baselines.reference import Unmodified
from swce.diagnostics.lens import observe
from swce.evaluation.runner import prepare
from swce.metrics.yes_no import auroc
from swce.models.subject import Subject
from swce.registry import Registry
from swce.steering.steerer import Steerer
from swce.tasks.task import Task

FOLDS = 5
"""Folds of the cross-validation over images."""
ITERATIONS = 1000
"""Iterations the solver may take before it stops."""


def collect(
    subject: Subject,
    steerers: Sequence[Steerer],
    task: str,
    *,
    limit: int | None = None,
    data: Path | None = None,
    registry: Registry | None = None,
) -> tuple[list[tuple[Steerer, NDArray[np.float32]]], NDArray[np.bool_], NDArray[np.str_]]:
    """Residual stream at the last prompt position of every labelled item, under every method.

    The recipes of the methods run first, and a method that is not meant for the task is left
    out. The unmodified model is always read.

    Args:
        subject: Model to observe.
        steerers: Method settings to read.
        task: Key of a task whose items expect yes or no.
        limit: Number of leading images to read, or None for the whole task.
        data: Data root, or None for the default one.
        registry: Registry defining the task, or None for the committed one.

    Returns:
        Every setting with its states, of shape `(items, layers, width)`, the unmodified model
        first, whether each item expects yes, and the image of each item.
    """
    bound = Task(task, data, registry)
    chosen = [
        steerer
        for steerer in prepare(subject, steerers)
        if not isinstance(steerer, Unmodified)
        and (not steerer.tasks or any(fnmatch.fnmatch(task, p) for p in steerer.tasks))
    ]
    settings: list[Steerer] = [Unmodified(), *chosen]
    states: list[list[NDArray[np.float32]]] = [[] for _ in settings]
    positive, groups = [], []
    for item in tqdm(bound.items(limit), desc=f"probe {task}"):
        if item.label not in ("yes", "no"):
            continue
        image = bound.image(item)
        for setting, found in zip(settings, states, strict=True):
            under = None if isinstance(setting, Unmodified) else setting
            read = observe(subject, image, item.prompt, under)
            found.append(read.float().numpy().astype(np.float32))
        positive.append(item.label == "yes")
        groups.append(item.group)
    stacked = [(setting, np.stack(found)) for setting, found in zip(settings, states, strict=True)]
    return stacked, np.array(positive, dtype=bool), np.array(groups)


def classifier() -> Pipeline:
    """Standardized logistic regression at the defaults of scikit-learn."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=ITERATIONS))


def probe(
    unmodified: NDArray[np.float32],
    steered: NDArray[np.float32],
    positive: NDArray[np.bool_],
    groups: NDArray[np.str_],
    *,
    folds: int = FOLDS,
) -> pd.DataFrame:
    """Cross-validated probes of one setting against those of the unmodified model.

    Args:
        unmodified: States of the unmodified model, of shape `(items, layers, width)`.
        steered: States of the same items under a method, of the same shape. Passing the
            unmodified states again gives the readings of the unmodified model.
        positive: Whether each item expects yes.
        groups: Image of each item. The items of one image always fall in the same fold.
        folds: Number of folds.

    Returns:
        One row per block with its index `layer`, `auroc`, the area under the ROC curve of the
        probe fitted on the steered states, `transfer_auroc`, that of the probe of the unmodified
        model on the steered states, and `score_shift`, the mean change of the score of that
        probe from the unmodified states to the steered ones, in standard deviations of its
        score on the unmodified states. A positive shift moves the stream toward yes.
    """
    splits = list(GroupKFold(n_splits=folds).split(unmodified[:, 0], positive, groups))
    rows = []
    same = steered is unmodified
    for layer in range(unmodified.shape[1]):
        own = np.zeros(len(positive))
        transfer = np.zeros(len(positive))
        base = np.zeros(len(positive))
        for train, test in splits:
            fitted = classifier().fit(unmodified[train, layer], positive[train])
            base[test] = fitted.decision_function(unmodified[test, layer])
            transfer[test] = fitted.decision_function(steered[test, layer])
            if same:
                own[test] = base[test]
            else:
                refit = classifier().fit(steered[train, layer], positive[train])
                own[test] = refit.decision_function(steered[test, layer])
        spread = float(base.std())
        rows.append(
            {
                "layer": layer,
                "auroc": auroc(own.astype(np.float64), positive),
                "transfer_auroc": auroc(transfer.astype(np.float64), positive),
                "score_shift": float((transfer - base).mean() / spread) if spread else 0.0,
            }
        )
    return pd.DataFrame(rows)


def probe_task(
    subject: Subject,
    steerers: Sequence[Steerer],
    task: str,
    *,
    limit: int | None = None,
    folds: int = FOLDS,
    data: Path | None = None,
    registry: Registry | None = None,
) -> pd.DataFrame:
    """Probes of the unmodified model and of every method on the items of a task.

    Args:
        subject: Model to observe.
        steerers: Method settings to read.
        task: Key of a task whose items expect yes or no.
        limit: Number of leading images to read, or None for the whole task.
        folds: Number of folds of the cross-validation over images.
        data: Data root, or None for the default one.
        registry: Registry defining the task, or None for the committed one.

    Returns:
        The rows of `probe` for every setting, with its `model`, `method`, `setting`,
        `fingerprint`, the `task` and the number of `items`.
    """
    states, positive, groups = collect(
        subject, steerers, task, limit=limit, data=data, registry=registry
    )
    unmodified = states[0][1]
    frames = []
    for setting, steered in states:
        frame = probe(unmodified, steered, positive, groups, folds=folds)
        frames.append(
            frame.assign(
                model=subject.name,
                method=setting.name,
                setting=setting.setting,
                fingerprint=setting.fingerprint,
                task=task,
                items=len(positive),
            )
        )
    columns = ["model", "method", "setting", "fingerprint", "task", "layer"]
    combined = pd.concat(frames, ignore_index=True)
    return combined[[*columns, "auroc", "transfer_auroc", "score_shift", "items"]]


def write_probes(probes: pd.DataFrame, root: Path) -> list[Path]:
    """Write the probes of every model and task under the diagnostics of a results folder.

    Args:
        probes: Rows returned by `probe_task`.
        root: Results folder.

    Returns:
        The files written, one per model and task, named `probe-TASK.csv`.
    """
    written = []
    for (model, task), group in probes.groupby(["model", "task"]):
        folder = root / "diagnostics" / str(model)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"probe-{task}.csv"
        group.to_csv(target, index=False, lineterminator="\n")
        written.append(target)
    return written
