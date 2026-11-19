"""Linear probes of the vision encoder for the objects POPE asks about.

A probe of the residual stream at the position of the answer reads one question at a time. The
features of the vision encoder do not know which object is asked about, so they are probed for
every COCO category at once instead: one logistic regression per category, on the mean of the
image tokens after every encoder block and after the projector. The probes are fitted on COCO
train2014 images labelled by their annotations, and each POPE question is scored by the probe of
its object on its own image, which no probe has seen.

The area under the ROC curve of those scores over the POPE questions is then comparable with that
of the probes of the residual stream. If the encoder already separates present from absent
objects as well as the language model does, a wrong answer does not start in the encoder.

The regularization is the default of scikit-learn on standardized features, as for the probes of
the stream, and is not tuned.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor, nn
from tqdm import tqdm

from swce.diagnostics.coco import object_of
from swce.diagnostics.probe import ITERATIONS, default_device
from swce.metrics.yes_no import auroc
from swce.models.hf import HuggingFaceSubject
from swce.registry import Registry
from swce.tasks.task import Task


class _StopError(Exception):
    """Raised by a hook once every feature is read, so the language model never runs."""


@torch.inference_mode()
def image_features(subject: HuggingFaceSubject, image: Image.Image) -> tuple[Tensor, Tensor | None]:
    """Mean of the image tokens after every encoder block, and after the projector.

    Args:
        subject: Model whose vision encoder reads the image.
        image: Image in RGB.

    Returns:
        The features of the encoder, of shape `(blocks, width)`, and those of the projector, of
        shape `(width,)`, or None when the subject names no projector. Both are in single
        precision on the processor.
    """
    found: dict[int, Tensor] = {}
    projected: list[Tensor] = []
    blocks = subject.encoder_layers()
    projector = subject.projector()

    def block_hook(index: int) -> Any:
        def read(_module: nn.Module, _inputs: Any, output: Any) -> None:
            state = output[0] if isinstance(output, tuple) else output
            found[index] = state.reshape(-1, state.shape[-1]).float().mean(dim=0).cpu()
            if projector is None and index == len(blocks) - 1:
                raise _StopError

        return read

    def projector_hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
        state = output[0] if isinstance(output, tuple) else output
        projected.append(state.reshape(-1, state.shape[-1]).float().mean(dim=0).cpu())
        raise _StopError

    handles = [block.register_forward_hook(block_hook(i)) for i, block in enumerate(blocks)]
    if projector is not None:
        handles.append(projector.register_forward_hook(projector_hook))
    try:
        subject.model(**subject.inputs(image, subject.template("")))
    except _StopError:
        pass
    finally:
        for handle in handles:
            handle.remove()
    encoder = torch.stack([found[index] for index in range(len(blocks))])
    return encoder, (projected[0] if projected else None)


class MultiProbes:
    """Standardized logistic regressions of every category on every block, fitted together.

    The objective of each is that of `LogisticRegression` of scikit-learn at its default `C` of
    one, as in `swce.diagnostics.probe.Probes`, so they could be fitted one by one with the same
    result.

    Attributes:
        mean: Mean of the fitted features, `(blocks, width)`.
        scale: Their standard deviation, one where it is zero.
        weight: Weights, `(blocks, categories, width)`.
        bias: Intercepts, `(blocks, categories)`.
    """

    def __init__(self, features: Tensor, targets: Tensor, *, iterations: int = ITERATIONS) -> None:
        """Fit the probes.

        Args:
            features: Features of shape `(items, blocks, width)`, in double precision.
            targets: Whether each item holds each category, `(items, categories)`.
            iterations: Iterations the solver may take before it stops.
        """
        self.mean = features.mean(dim=0)
        scale = features.std(dim=0, unbiased=False)
        self.scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        standard = (features - self.mean) / self.scale
        sign = targets.to(features.dtype) * 2 - 1
        blocks, width = features.shape[1:]
        count = targets.shape[1]
        self.weight = torch.zeros(
            blocks, count, width, dtype=features.dtype, device=features.device
        )
        self.bias = torch.zeros(blocks, count, dtype=features.dtype, device=features.device)
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
        items = features.shape[0]

        def objective() -> Tensor:
            solver.zero_grad()
            logits = torch.einsum("nld,lcd->nlc", standard, self.weight) + self.bias
            loss = (
                F.softplus(-sign[:, None] * logits).sum() + 0.5 * (self.weight**2).sum()
            ) / items
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        solver.step(objective)  # type: ignore[no-untyped-call]
        self.weight = self.weight.detach()
        self.bias = self.bias.detach()

    def decision(self, features: Tensor) -> Tensor:
        """Score of every category on every block, `(items, blocks, categories)`."""
        standard = (features - self.mean) / self.scale
        return torch.einsum("nld,lcd->nlc", standard, self.weight) + self.bias


def presence(names: Sequence[str], objects: Sequence[frozenset[str]]) -> NDArray[np.bool_]:
    """Whether each image holds each category, `(images, categories)`."""
    return np.array([[name in found for name in names] for found in objects], dtype=bool)


def read_images(
    subject: HuggingFaceSubject, images: Sequence[Image.Image] | Sequence[Any], desc: str
) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """Features of the encoder and of the projector for every image, stacked."""
    encoder, projector = [], []
    for image in tqdm(images, desc=desc, mininterval=30):
        found, projected = image_features(subject, image() if callable(image) else image)
        encoder.append(found.numpy())
        if projected is not None:
            projector.append(projected.numpy())
    return np.stack(encoder), (np.stack(projector) if projector else None)


def probe_encoder(
    train: Mapping[str, NDArray[np.float32]],
    train_objects: Sequence[frozenset[str]],
    test: Mapping[str, NDArray[np.float32]],
    test_index: NDArray[np.intp],
    questions: Sequence[tuple[str, bool]],
    names: Sequence[str],
    *,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Probes fitted on one set of images and read on the questions about another.

    Args:
        train: Features of the training images by source, `encoder` of shape
            `(images, blocks, width)` and optionally `projector` of shape `(images, 1, width)`.
        train_objects: Objects of every training image.
        test: Features of the images the questions are about, by the same sources.
        test_index: Row of `test` holding the image of every question.
        questions: Object and expected answer, True for yes, of every question.
        names: Categories the probes are fitted for.
        device: Device the probes are fitted on, or None for the first GPU when there is one.

    Returns:
        One row per source and block with the `auroc` over the questions and the number of
        `questions` whose object is one of the categories.
    """
    device = device or default_device()
    targets = torch.from_numpy(presence(names, train_objects)).to(device)
    column = {name: index for index, name in enumerate(names)}
    known = [
        (row, column[name], label) for row, (name, label) in enumerate(questions) if name in column
    ]
    rows = []
    for source, features in train.items():
        fitted = MultiProbes(torch.from_numpy(features).to(device, torch.float64), targets)
        decided = fitted.decision(torch.from_numpy(test[source]).to(device, torch.float64))
        scores = decided.cpu().numpy()
        for block in range(features.shape[1]):
            picked = np.array([scores[test_index[row], block, cat] for row, cat, _ in known])
            labels = np.array([label for _, _, label in known])
            rows.append(
                {
                    "source": source,
                    "block": block,
                    "auroc": auroc(picked, labels),
                    "questions": len(known),
                }
            )
    return pd.DataFrame(rows)


def encoder_task(
    subject: HuggingFaceSubject,
    task: str,
    training: Sequence[tuple[Any, frozenset[str]]],
    names: Sequence[str],
    *,
    limit: int | None = None,
    data: Path | None = None,
    registry: Registry | None = None,
) -> pd.DataFrame:
    """Probes of the vision encoder of a subject read on the questions of a POPE task.

    Args:
        subject: Model whose vision encoder is read.
        task: Key of a POPE task.
        training: Every training image, or a function returning it, with its objects.
        names: Categories the probes are fitted for.
        limit: Number of leading images of the task to read, or None for all of them.
        data: Data root, or None for the default one.
        registry: Registry defining the task, or None for the committed one.

    Returns:
        The rows of `probe_encoder`, with the `model` and the `task`.
    """
    encoder, projector = read_images(subject, [image for image, _ in training], "encoder train")
    train = {"encoder": encoder}
    if projector is not None:
        train["projector"] = projector[:, None]
    bound = Task(task, data, registry)
    items = bound.items(limit)
    groups = list(dict.fromkeys(item.group for item in items))
    first = {group: next(item for item in items if item.group == group) for group in groups}
    test_encoder, test_projector = read_images(
        subject, [lambda g=group: bound.image(first[g]) for group in groups], f"encoder {task}"
    )
    test = {"encoder": test_encoder}
    if test_projector is not None:
        test["projector"] = test_projector[:, None]
    index = {group: row for row, group in enumerate(groups)}
    test_index = np.array([index[item.group] for item in items])
    questions = [(object_of(item.prompt), item.label == "yes") for item in items]
    frame = probe_encoder(
        train, [objects for _, objects in training], test, test_index, questions, names
    )
    return frame.assign(model=subject.name, task=task)
