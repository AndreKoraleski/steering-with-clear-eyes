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

from collections.abc import Callable, Mapping, Sequence
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

from swce.diagnostics.attention import Grid, ResizedGrid
from swce.diagnostics.coco import object_of
from swce.diagnostics.probe import ITERATIONS, default_device, released
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
    return _image_features(
        subject,
        subject.inputs(image, subject.template("")),
        lambda state, source: state.mean(dim=0),
    )


@torch.inference_mode()
def _image_features(
    subject: HuggingFaceSubject,
    inputs: Mapping[str, Tensor],
    pool: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor | None]:
    """Reduce tokens inside the hooks, retaining only the requested features."""
    found: dict[int, Tensor] = {}
    projected: list[Tensor] = []
    blocks = subject.encoder_layers()
    projector = subject.projector()

    def block_hook(index: int) -> Any:
        def read(_module: nn.Module, _inputs: Any, output: Any) -> None:
            state = output[0] if isinstance(output, tuple) else output
            found[index] = pool(state.reshape(-1, state.shape[-1]).float(), "encoder").cpu()
            if projector is None and index == len(blocks) - 1:
                raise _StopError

        return read

    def projector_hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
        state = output[0] if isinstance(output, tuple) else output
        projected.append(pool(state.reshape(-1, state.shape[-1]).float(), "projector").cpu())
        raise _StopError

    handles = [block.register_forward_hook(block_hook(i)) for i, block in enumerate(blocks)]
    if projector is not None:
        handles.append(projector.register_forward_hook(projector_hook))
    try:
        subject.model(**inputs)
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
    with released(subject):
        frame = probe_encoder(
            train, [objects for _, objects in training], test, test_index, questions, names
        )
    return frame.assign(model=subject.name, task=task)


def region_cells(
    grid: Grid | ResizedGrid,
    width: int,
    height: int,
    boxes: Sequence[tuple[str, list[float]]],
) -> list[tuple[str, float, list[int]]]:
    """Category, original area and cells of boxes at least half visible in the crop."""
    kept = []
    for name, box in boxes:
        cells = grid.cells(width, height, box)
        if grid.visible(width, height, box) >= 0.5 and cells:
            kept.append((name, box[2] * box[3], cells))
    return kept


def _merged_region_indices(
    cells: Sequence[Sequence[int]], merge: int, offset: int
) -> list[list[int]]:
    """Expand merged cells into contiguous patch groups, after any class token."""
    return [
        [index * merge**2 + patch + offset for index in cell for patch in range(merge**2)]
        for cell in cells
    ]


def _resized_region_offset(
    subject: HuggingFaceSubject, inputs: Mapping[str, Tensor], grid: ResizedGrid
) -> int:
    """Validate an unpermuted merged vision tower and locate its first patch token."""
    shape = inputs.get("image_grid_thw")
    if shape is None or shape.shape != (1, 3) or shape.tolist()[0] != [1, grid.rows, grid.columns]:
        raise ValueError("Encoder regions require one still image with matching image_grid_thw")
    if (
        min(grid.rows, grid.columns, grid.merge) <= 0
        or grid.rows % grid.merge
        or grid.columns % grid.merge
    ):
        raise ValueError("Encoder regions require a positive grid divisible by spatial_merge_size")
    projector = subject.projector()
    blocks = list(subject.encoder_layers())
    for tower in subject.model.modules():
        if projector is None or getattr(tower, "merger", None) is not projector:
            continue
        if (
            list(getattr(tower, "blocks", [])) != blocks
            or getattr(tower, "spatial_merge_size", None) != grid.merge
            or getattr(tower, "patch_size", None) != grid.patch
            or not hasattr(tower, "patch_embed")
            or getattr(tower, "window_size", None) is not None
        ):
            break
        if isinstance(getattr(tower, "class_embedding", None), Tensor):
            if hasattr(tower, "class_pos_emb") and hasattr(tower, "pre_layernorm"):
                return 1
            break
        if hasattr(tower, "deepstack_merger_list") and hasattr(tower, "pos_embed"):
            return 0
        break
    raise ValueError("Cannot map encoder regions: unsupported merged vision tower layout")


def region_features(
    subject: HuggingFaceSubject,
    inputs: Mapping[str, Tensor],
    cells: Sequence[Sequence[int]],
    grid: Grid | ResizedGrid,
) -> tuple[Tensor, Tensor | None]:
    """Pool patch tokens inside each hook, returning `(regions, blocks, width)`.

    LLaVA-1.5 CLIP blocks include a class token at zero: grid cell i reads token i+1.
    Its projector has only patch tokens and reads cell i directly. Unlike image_features,
    region pooling excludes the class token. Qwen3-VL-8B and LLaVA-OneVision-1.5-8B use
    ResizedGrid cells in row-major merged-grid order. Cell i reads the contiguous encoder
    patches i*m*m through (i+1)*m*m-1, with an offset of one for RICE's leading class token,
    and projector token i. Only one still image (grid_t == 1) is supported. Unknown layouts
    and mismatched token counts raise ValueError rather than guessing offsets.

    Args:
        subject: Model whose encoder is read.
        inputs: Processed inputs for one image.
        cells: Grid cell indices of each region, in row-major order, merged for ResizedGrid.
        grid: Geometry used to map the boxes to cells.

    Returns:
        Encoder features `(regions, blocks, width)` and projector features `(regions, width)`,
        or None when no projector is named, in single precision on the processor.
    """
    model: Any = subject.model
    if isinstance(grid, Grid) and model.config.model_type == "llava":
        count, merge, offset = grid.side**2, 1, 1
    elif isinstance(grid, ResizedGrid):
        offset = _resized_region_offset(subject, inputs, grid)
        merge = grid.merge
        count = grid.rows * grid.columns // merge**2
    else:
        raise ValueError("Cannot map encoder regions: unsupported vision tower layout")
    if not cells or any(not cell or min(cell) < 0 or max(cell) >= count for cell in cells):
        raise ValueError("Region cells must be nonempty and inside the patch grid")
    encoder_cells = _merged_region_indices(cells, merge, offset)

    def pool(state: Tensor, source: str) -> Tensor:
        expected = count * merge**2 + offset if source == "encoder" else count
        if state.shape[0] != expected:
            raise ValueError(
                f"Cannot map {source} tokens: expected {expected}, got {state.shape[0]}"
            )
        indices = encoder_cells if source == "encoder" else cells
        return torch.stack(
            [state[torch.as_tensor(cell, device=state.device)].mean(dim=0) for cell in indices]
        )

    encoder, projector = _image_features(subject, inputs, pool)
    return encoder.transpose(0, 1), projector


def read_regions(
    subject: HuggingFaceSubject,
    images: Sequence[tuple[Any, Sequence[tuple[str, list[float]]]]],
    desc: str,
    *,
    per_category: int | None = None,
) -> tuple[dict[str, NDArray[np.float32]], list[tuple[int, str, float]]]:
    """Read images once and retain pooled regions and their image, category and area.

    Args:
        subject: Model whose encoder is read.
        images: Images or loaders with their category and box annotations.
        desc: Progress description.
        per_category: Maximum regions per category, or None to keep all evaluation regions.
            Select the first eligible boxes in image and annotation order deterministically,
            before pooling or stacking features.

    Returns:
        Features by source and the image index, category and area of each retained region.
    """
    if per_category is not None and per_category < 1:
        raise ValueError("per_category must be positive")
    counts: dict[str, int] = {}
    encoders, projectors = [], []
    regions: list[tuple[int, str, float]] = []
    for index, (image, boxes) in enumerate(tqdm(images, desc=desc, mininterval=30)):
        image = image() if callable(image) else image
        inputs = subject.inputs(image, subject.template(""))
        grid = Grid.of(subject) if subject.fixed_grid else ResizedGrid.of(subject, inputs)
        kept = region_cells(grid, image.width, image.height, boxes)
        if per_category is not None:
            selected = []
            for name, area, cells in kept:
                if counts.get(name, 0) < per_category:
                    selected.append((name, area, cells))
                    counts[name] = counts.get(name, 0) + 1
            kept = selected
        if not kept:
            continue
        encoder, projector = region_features(subject, inputs, [cells for _, _, cells in kept], grid)
        encoders.append(encoder.numpy())
        if projector is not None:
            projectors.append(projector.numpy()[:, None])
        regions.extend((index, name, area) for name, area, _ in kept)
    if not encoders:
        raise ValueError(f"No kept boxes in {desc}")
    features = {"encoder": np.concatenate(encoders)}
    if projectors:
        features["projector"] = np.concatenate(projectors)
    return features, regions


def probe_regions(
    train: Mapping[str, NDArray[np.float32]],
    train_names: Sequence[str],
    test: Mapping[str, NDArray[np.float32]],
    test_names: Sequence[str],
    questions: Sequence[tuple[int, str, bool]],
    names: Sequence[str],
    *,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Fit one-vs-rest region probes and read recognition and missed present objects.

    Questions contain the largest kept region's row, category and whether the margin is
    positive. Region AUROC averages categories with at least ten positive and ten negative
    boxes. Missed AUROC treats an answered yes as positive. Counts and means refer to questions;
    categories and boxes report recognition support. Undefined readings are NaN.

    Args:
        train: Training features by source, each shaped `(regions, blocks, width)`.
        train_names: Category of each training region.
        test: Evaluation features by the same sources.
        test_names: Category of each evaluation region.
        questions: Region row, category and model answer of each present question.
        names: Categories of the one-vs-rest probes.
        device: Device for fitting, or None for the first GPU when available.

    Returns:
        One row per source and block with both AUROCs and their support.
    """
    device = device or default_device()
    targets = torch.from_numpy(presence(names, [frozenset([name]) for name in train_names])).to(
        device
    )
    labels = presence(names, [frozenset([name]) for name in test_names])
    column = {name: index for index, name in enumerate(names)}
    known = [(row, column[name], yes) for row, name, yes in questions if name in column]
    yes = np.array([answer for _, _, answer in known], dtype=bool)
    eligible = [
        i for i in range(len(names)) if labels[:, i].sum() >= 10 and (~labels[:, i]).sum() >= 10
    ]
    rows = []
    for source, features in train.items():
        fitted = MultiProbes(torch.from_numpy(features).to(device, torch.float64), targets)
        scores = (
            fitted.decision(torch.from_numpy(test[source]).to(device, torch.float64)).cpu().numpy()
        )
        for block in range(features.shape[1]):
            recognition = [auroc(scores[:, block, cat], labels[:, cat]) for cat in eligible]
            picked = np.array([scores[row, block, cat] for row, cat, _ in known])
            rows.append(
                {
                    "source": source,
                    "block": block,
                    "region_auroc": float(np.mean(recognition)) if recognition else float("nan"),
                    "categories": len(eligible),
                    "boxes": len(test_names),
                    "missed_auroc": auroc(picked, yes),
                    "correct_yes": int(yes.sum()),
                    "missed": int((~yes).sum()),
                    "correct_yes_score": float(picked[yes].mean()) if yes.any() else float("nan"),
                    "missed_score": float(picked[~yes].mean()) if (~yes).any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def encoder_regions_task(
    subject: HuggingFaceSubject,
    task: str,
    training: Sequence[tuple[Any, Sequence[tuple[str, list[float]]]]],
    names: Sequence[str],
    answers: pd.DataFrame,
    boxes: Mapping[str, tuple[int, int, list[tuple[str, list[float]]]]],
    *,
    per_category: int = 300,
    limit: int | None = None,
    data: Path | None = None,
    registry: Registry | None = None,
) -> pd.DataFrame:
    """Read region probes on POPE and the largest kept box of each present question.

    Training contains images and COCO boxes. Answers are rows of prior.margins, joined by id;
    only positive margins count as yes. Each evaluation image contributes its boxes once,
    regardless of how many questions ask about it. Largest refers to original box area.

    The keyword per_category caps training regions per category at 300 by default, selecting
    the first eligible boxes in image and annotation order. Evaluation regions are uncapped.
    The model waits on the processor during fitting and scoring and is restored afterwards.
    """
    train, train_regions = read_regions(
        subject, training, "encoder regions train", per_category=per_category
    )
    bound = Task(task, data, registry)
    items = bound.items(limit)
    groups = list(dict.fromkeys(item.group for item in items))
    first = {item.group: item for item in items}
    test, regions = read_regions(
        subject,
        [
            (lambda g=group: bound.image(first[g]), boxes.get(group, (1, 1, []))[2])
            for group in groups
        ],
        f"encoder regions {task}",
    )
    largest: dict[tuple[str, str], int] = {}
    for row, (index, name, area) in enumerate(regions):
        key = (groups[index], name)
        if key not in largest or area > regions[largest[key]][2]:
            largest[key] = row
    margins = dict(zip(answers["id"].astype(str), answers["margin"], strict=True))
    questions = []
    for item in items:
        name = object_of(item.prompt)
        key = (item.group, name)
        if item.label == "yes" and key in largest and item.id in margins:
            questions.append((largest[key], name, margins[item.id] > 0))
    with released(subject):
        frame = probe_regions(
            train,
            [name for _, name, _ in train_regions],
            test,
            [name for _, name, _ in regions],
            questions,
            names,
        )
    return frame.assign(model=subject.name, task=task)
