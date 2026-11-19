"""How much of the answer of a model to POPE its language prior alone predicts.

Every question is asked twice, with its image and with the image span removed, as VISTA builds its
prompt without the image. The margin of yes over no at the first answer token is read both
times, from the logits of the model. Without the image the margin is the language prior: what the
question alone makes the model expect. Each question is also set against how often its object
co-occurs, in COCO train2014, with the objects actually in its image, since the adversarial split
of POPE draws its absent objects by co-occurrence.

A wrong yes is a question whose object is absent and whose margin with the image is positive. The
readings ask whether the prior, or the co-occurrence, tells the wrong yeses from the correct
noes.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import Tensor
from tqdm import tqdm

from swce.diagnostics.coco import Cooccurrence, cooccurrence, image_objects, object_of
from swce.diagnostics.lens import NO_WORDS, YES_WORDS
from swce.metrics.yes_no import auroc
from swce.models.hf import HuggingFaceSubject
from swce.registry import Registry
from swce.tasks.task import Task


def margin(subject: HuggingFaceSubject, inputs: Mapping[str, Tensor]) -> float:
    """Log probability of yes minus that of no at the next position, from the logits.

    Each word sums its spellings, as the probabilities of every answer do.
    """
    with torch.inference_mode():
        logits = subject.model(**inputs).logits[0, -1].float()
    yes, no = subject.first_tokens(YES_WORDS), subject.first_tokens(NO_WORDS)
    return float(logits[yes].logsumexp(0) - logits[no].logsumexp(0))


def margins(
    subject: HuggingFaceSubject,
    task: str,
    *,
    limit: int | None = None,
    data: Path | None = None,
    registry: Registry | None = None,
) -> pd.DataFrame:
    """The margin of every question of a task, with and without its image.

    Args:
        subject: Model to read.
        task: Key of a POPE task.
        limit: Number of leading images to read, or None for the whole task.
        data: Data root, or None for the default one.
        registry: Registry defining the task, or None for the committed one.

    Returns:
        One row per question, with its `id`, `group`, `object`, `label`, `margin` with the image
        and `blind` margin without it.
    """
    bound = Task(task, data, registry)
    rows = []
    for item in tqdm(bound.items(limit), desc=f"prior {task}", mininterval=30):
        text = subject.template(item.prompt)
        rows.append(
            {
                "id": item.id,
                "group": item.group,
                "object": object_of(item.prompt),
                "label": item.label,
                "margin": margin(subject, subject.inputs(bound.image(item), text)),
                "blind": margin(subject, subject.text_inputs(text)),
            }
        )
    return pd.DataFrame(rows)


def with_cooccurrence(
    frame: pd.DataFrame, present: Mapping[str, frozenset[str]], counts: Cooccurrence
) -> pd.DataFrame:
    """The margins with the objects of every image and the co-occurrence of every object.

    Args:
        frame: Rows of `margins`.
        present: Objects of every image of the task, by file name.
        counts: Co-occurrence of the categories in the images the prior is set against.

    Returns:
        The rows with `present`, the number of objects of the image, and `cooccurrence`, the
        largest share of the images holding one of them that also hold the object asked about.
    """
    objects = [present.get(group, frozenset()) for group in frame["group"]]
    return frame.assign(
        present=[len(found) for found in objects],
        cooccurrence=[
            counts.strongest(target, found)
            for target, found in zip(frame["object"], objects, strict=True)
        ],
    )


def summarize_prior(frame: pd.DataFrame) -> pd.DataFrame:
    """Readings of the prior on the questions whose object is absent.

    Returns:
        One row per reading, with its `value`. The AUROCs tell the wrong yeses from the correct
        noes, a wrong yes being positive.
    """
    absent = frame[frame["label"] == "no"]
    wrong = (absent["margin"] > 0).to_numpy()
    rows: list[dict[str, Any]] = [
        {"reading": "questions", "value": len(frame)},
        {"reading": "absent", "value": len(absent)},
        {"reading": "wrong_yes", "value": int(wrong.sum())},
        {"reading": "yes_ratio_with_image", "value": float((frame["margin"] > 0).mean())},
        {"reading": "yes_ratio_without_image", "value": float((frame["blind"] > 0).mean())},
        {
            "reading": "auroc_answer_without_image",
            "value": auroc(frame["blind"].to_numpy(), (frame["label"] == "yes").to_numpy()),
        },
    ]
    if 0 < wrong.sum() < len(wrong):
        rows += [
            {
                "reading": "auroc_wrong_yes_by_prior",
                "value": auroc(absent["blind"].to_numpy(), wrong),
            },
            {
                "reading": "auroc_wrong_yes_by_cooccurrence",
                "value": auroc(absent["cooccurrence"].to_numpy(), wrong),
            },
            {
                "reading": "wrong_yes_with_positive_prior",
                "value": float((absent["blind"].to_numpy()[wrong] > 0).mean()),
            },
            {
                "reading": "correct_no_with_positive_prior",
                "value": float((absent["blind"].to_numpy()[~wrong] > 0).mean()),
            },
        ]
    rho = spearmanr(absent["blind"], absent["cooccurrence"]).statistic
    rows.append({"reading": "spearman_prior_cooccurrence", "value": float(np.nan_to_num(rho))})
    return pd.DataFrame(rows)


def prior_task(
    subject: HuggingFaceSubject,
    task: str,
    *,
    limit: int | None = None,
    data: Path | None = None,
    registry: Registry | None = None,
    present: Mapping[str, frozenset[str]] | None = None,
    counts: Cooccurrence | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The margins of a POPE task with and without its images, and their readings.

    Args:
        subject: Model to read.
        task: Key of a POPE task.
        limit: Number of leading images to read, or None for the whole task.
        data: Data root, or None for the default one.
        registry: Registry defining the task, or None for the committed one.
        present: Objects of the images of the task, or None for the COCO val2014 annotations.
        counts: Co-occurrence the prior is set against, or None for COCO train2014.

    Returns:
        The rows of `with_cooccurrence` and the readings of `summarize_prior`.
    """
    frame = margins(subject, task, limit=limit, data=data, registry=registry)
    present = present if present is not None else image_objects("val")
    counts = counts if counts is not None else cooccurrence(image_objects("train"))
    frame = with_cooccurrence(frame, present, counts)
    return frame, summarize_prior(frame)


def write_frames(frames: Sequence[tuple[str, pd.DataFrame]], folder: Path) -> list[Path]:
    """Write named frames as CSV files in a folder, returning their paths."""
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in frames:
        target = folder / f"{name}.csv"
        frame.to_csv(target, index=False, lineterminator="\n")
        written.append(target)
    return written
