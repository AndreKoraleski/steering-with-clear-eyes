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
The probes of every block are fitted together with PyTorch, on the GPU when there is one, and
`tests/diagnostics/test_probe.py` checks them against scikit-learn.
"""

import fnmatch
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold
from torch import Tensor, nn
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


def default_device() -> torch.device:
    """The first GPU when there is one, else the processor."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def released(subject: Subject) -> Iterator[None]:
    """Hold the modules of a subject on the processor, and put them back where they were after.

    The probes of one fold are fitted in double precision on the states of every block, which on
    a 24 GB GPU do not fit beside an 8B model. Once every state is read the model is not needed,
    so it waits on the processor while the probes are fitted, and the probes stay on the GPU.

    Args:
        subject: Model whose module attributes are moved. A module whose parameters lie on
            several devices is left where it is, since one move could not put it back.
    """
    moved: list[tuple[nn.Module, torch.device]] = []
    for module in vars(subject).values():
        if not isinstance(module, nn.Module):
            continue
        devices = {tensor.device for tensor in [*module.parameters(), *module.buffers()]}
        if len(devices) != 1:
            continue
        (device,) = devices
        if device.type != "cpu":
            module.to("cpu")
            moved.append((module, device))
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        yield
    finally:
        for module, device in moved:
            module.to(device)


class Probes:
    """Standardized logistic regressions of every block, fitted together.

    The objective is that of `LogisticRegression` of scikit-learn at its default `C` of one, an
    L2 penalty of one half the squared norm of the weights and the summed log loss, with an
    unpenalized intercept, on states standardized as `StandardScaler` does. The blocks do not
    interact, so their sum is minimized at once by L-BFGS, which is what makes it fast on a GPU.

    Attributes:
        mean: Mean of the fitted states, of shape `(layers, width)`.
        scale: Standard deviation of the fitted states, one where it is zero.
        weight: Weights of every block, of shape `(layers, width)`.
        bias: Intercept of every block, of shape `(layers,)`.
    """

    def __init__(self, states: Tensor, positive: Tensor, *, iterations: int = ITERATIONS) -> None:
        """Fit the probes.

        Args:
            states: States of shape `(items, layers, width)`, in double precision.
            positive: Whether each item expects yes, of shape `(items,)`.
            iterations: Iterations the solver may take before it stops.
        """
        self.mean = states.mean(dim=0)
        scale = states.std(dim=0, unbiased=False)
        self.scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        standard = (states - self.mean) / self.scale
        sign = positive.to(states.dtype) * 2 - 1
        layers, width = states.shape[1:]
        self.weight = torch.zeros(layers, width, dtype=states.dtype, device=states.device)
        self.bias = torch.zeros(layers, dtype=states.dtype, device=states.device)
        self.weight.requires_grad_(True)
        self.bias.requires_grad_(True)
        solver = torch.optim.LBFGS(
            [self.weight, self.bias],
            max_iter=iterations,
            history_size=20,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        count = states.shape[0]

        def objective() -> Tensor:
            solver.zero_grad()
            logits = torch.einsum("nld,ld->nl", standard, self.weight) + self.bias
            # Divided by the number of items, which leaves the minimum where it is.
            loss = (F.softplus(-sign[:, None] * logits).sum() + 0.5 * (self.weight**2).sum()) / (
                count
            )
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        solver.step(objective)  # type: ignore[no-untyped-call]
        self.weight = self.weight.detach()
        self.bias = self.bias.detach()

    def decision(self, states: Tensor) -> Tensor:
        """Score of yes of every item at every block, of shape `(items, layers)`."""
        standard = (states - self.mean) / self.scale
        return torch.einsum("nld,ld->nl", standard, self.weight) + self.bias


def scores(
    fitted_on: NDArray[np.float32],
    applied_to: Sequence[NDArray[np.float32]],
    positive: NDArray[np.bool_],
    splits: Sequence[tuple[NDArray[np.intp], NDArray[np.intp]]],
    device: torch.device,
) -> list[NDArray[np.float64]]:
    """Out-of-fold scores of probes fitted on one set of states and applied to others.

    Args:
        fitted_on: States the probes are fitted on, of shape `(items, layers, width)`.
        applied_to: States of the same items the probes of each fold score on its held-out items.
        positive: Whether each item expects yes.
        splits: Indices of the fitted and the held-out items of every fold.
        device: Device the probes are fitted on.

    Returns:
        For every array of `applied_to`, the score of every item at every block, of shape
        `(items, layers)`.
    """

    def tensor(array: NDArray[np.float32]) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(device, torch.float64)

    found = [np.zeros(fitted_on.shape[:2]) for _ in applied_to]
    for train, test in splits:
        target = torch.from_numpy(positive[train]).to(device)
        fitted = Probes(tensor(fitted_on[train]), target)
        for result, states in zip(found, applied_to, strict=True):
            result[test] = fitted.decision(tensor(states[test])).cpu().numpy()
    return found


def probe_settings(
    unmodified: NDArray[np.float32],
    steered: Sequence[NDArray[np.float32]],
    positive: NDArray[np.bool_],
    groups: NDArray[np.str_],
    *,
    folds: int = FOLDS,
    device: torch.device | None = None,
) -> list[pd.DataFrame]:
    """Cross-validated probes of the unmodified model and of every steered setting.

    The probes of the unmodified model are fitted once per fold and applied to every setting.

    Args:
        unmodified: States of the unmodified model, of shape `(items, layers, width)`.
        steered: States of the same items under every method, of the same shape.
        positive: Whether each item expects yes.
        groups: Image of each item. The items of one image always fall in the same fold.
        folds: Number of folds.
        device: Device the probes are fitted on, or None for the first GPU when there is one.

    Returns:
        The readings of `probe` for the unmodified model, then for every steered setting.
    """
    device = device or default_device()
    splits = list(GroupKFold(n_splits=folds).split(unmodified[:, 0], positive, groups))
    base, *transfers = scores(unmodified, [unmodified, *steered], positive, splits, device)
    frames = [readings(base, base, base, positive)]
    for states, transfer in zip(steered, transfers, strict=True):
        (own,) = scores(states, [states], positive, splits, device)
        frames.append(readings(own, transfer, base, positive))
    return frames


def readings(
    own: NDArray[np.float64],
    transfer: NDArray[np.float64],
    base: NDArray[np.float64],
    positive: NDArray[np.bool_],
) -> pd.DataFrame:
    """Readings of one setting from its out-of-fold scores, block by block."""
    rows = []
    for layer in range(own.shape[1]):
        spread = float(base[:, layer].std())
        shift = (transfer[:, layer] - base[:, layer]).mean()
        rows.append(
            {
                "layer": layer,
                "auroc": auroc(own[:, layer], positive),
                "transfer_auroc": auroc(transfer[:, layer], positive),
                "score_shift": float(shift / spread) if spread else 0.0,
            }
        )
    return pd.DataFrame(rows)


def probe(
    unmodified: NDArray[np.float32],
    steered: NDArray[np.float32],
    positive: NDArray[np.bool_],
    groups: NDArray[np.str_],
    *,
    folds: int = FOLDS,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Cross-validated probes of one setting against those of the unmodified model.

    Args:
        unmodified: States of the unmodified model, of shape `(items, layers, width)`.
        steered: States of the same items under a method, of the same shape. Passing the
            unmodified states again gives the readings of the unmodified model.
        positive: Whether each item expects yes.
        groups: Image of each item. The items of one image always fall in the same fold.
        folds: Number of folds.
        device: Device the probes are fitted on, or None for the first GPU when there is one.

    Returns:
        One row per block with its index `layer`, `auroc`, the area under the ROC curve of the
        probe fitted on the steered states, `transfer_auroc`, that of the probe of the unmodified
        model on the steered states, and `score_shift`, the mean change of the score of that
        probe from the unmodified states to the steered ones, in standard deviations of its
        score on the unmodified states. A positive shift moves the stream toward yes.
    """
    others = [] if steered is unmodified else [steered]
    frames = probe_settings(unmodified, others, positive, groups, folds=folds, device=device)
    return frames[-1]


def probe_task(
    subject: Subject,
    steerers: Sequence[Steerer],
    task: str,
    *,
    limit: int | None = None,
    folds: int = FOLDS,
    data: Path | None = None,
    registry: Registry | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Probes of the unmodified model and of every method on the items of a task.

    Once every state is read, the modules of the subject wait on the processor while the probes
    are fitted, and are put back after, so that the probes have the GPU to themselves.

    Args:
        subject: Model to observe.
        steerers: Method settings to read.
        task: Key of a task whose items expect yes or no.
        limit: Number of leading images to read, or None for the whole task.
        folds: Number of folds of the cross-validation over images.
        data: Data root, or None for the default one.
        registry: Registry defining the task, or None for the committed one.
        device: Device the probes are fitted on, or None for the first GPU when there is one.

    Returns:
        The rows of `probe` for every setting, with its `model`, `method`, `setting`,
        `fingerprint`, the `task` and the number of `items`.
    """
    states, positive, groups = collect(
        subject, steerers, task, limit=limit, data=data, registry=registry
    )
    unmodified = states[0][1]
    steered = [read for _, read in states[1:]]
    with released(subject):
        found = probe_settings(unmodified, steered, positive, groups, folds=folds, device=device)
    frames = []
    for (setting, _), frame in zip(states, found, strict=True):
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
