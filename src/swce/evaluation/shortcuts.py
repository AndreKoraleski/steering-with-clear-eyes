"""How much of a gain the unmodified model reaches by a trivial change of its own answers.

Two shortcuts lower a hallucination score without any better use of the image. On yes or no
questions, a model that says yes more or less often moves its accuracy and its F1 (Yin et al.,
2025). On descriptions, a shorter caption mentions fewer objects and so fewer absent ones
(Fazli et al., 2026). Both are traced here on the answers of the unmodified model alone, and a
method is compared with the point of the trace that matches its own yes ratio or its own recall.
What remains is the part of its gain the shortcut does not explain.

Nothing here runs a model. The traces need the probabilities of yes and no, or the captions,
that an evaluation saves with every answer.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from swce.metrics.chair import (
    CaptionCounts,
    Lemmatizer,
    caption_counts,
    chair_scores,
    mentions_of,
    nltk_lemmas,
)
from swce.metrics.yes_no import binary_scores, pope_answer, yes_margin
from swce.tasks.schema import Record

STEP = 5
"""Number of words between two truncations of a caption."""


def bias_curve(margins: NDArray[np.float64], labels: NDArray[np.bool_]) -> pd.DataFrame:
    """Scores of a model at every threshold on its preference for yes.

    Args:
        margins: Score of yes for each question, larger meaning more likely yes.
        labels: Whether each expected answer is yes.

    Returns:
        One row per number of questions answered yes, from none to all, taking the questions in
        decreasing order of margin, with the `yes_ratio`, the `accuracy` and the `f1` reached.
    """
    order = np.argsort(-margins, kind="stable")
    positives = np.concatenate([[0], np.cumsum(labels[order])]).astype(np.float64)
    answered = np.arange(labels.size + 1, dtype=np.float64)
    total = float(labels.sum())
    negatives_kept = (labels.size - total) - (answered - positives)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(answered + total > 0, 2 * positives / (answered + total), np.nan)
    return pd.DataFrame(
        {
            "yes_ratio": answered / labels.size,
            "accuracy": (positives + negatives_kept) / labels.size,
            "f1": f1,
        }
    )


def truncation_curve(
    captions: Sequence[str],
    present: Sequence[frozenset[str]],
    lemmatize: Lemmatizer = nltk_lemmas,
    step: int = STEP,
) -> pd.DataFrame:
    """CHAIR scores of captions cut after a growing number of words.

    Args:
        captions: Captions of the unmodified model.
        present: Objects of the image of each caption.
        lemmatize: Function turning a caption into lemmas, applied once per caption.
        step: Number of words between two truncations.

    Returns:
        One row per truncation, from `step` words to the longest caption, with the `words` kept
        and the `chair_s`, `chair_i`, `recall` and `length` of the truncated captions.
    """
    lemmas = [lemmatize(caption) for caption in captions]
    longest = max((len(found) for found in lemmas), default=0)
    rows = []
    for words in range(step, longest + step, step):
        counts: list[CaptionCounts] = [
            caption_counts(mentions_of(found[:words]), objects)
            for found, objects in zip(lemmas, present, strict=True)
        ]
        rows.append({"words": words, **chair_scores(counts)})
    return pd.DataFrame(rows)


def matched(curve: pd.DataFrame, on: str, at: float, column: str) -> float:
    """Value a trace reaches where one of its columns equals a given value.

    Args:
        curve: Trace of the unmodified model.
        on: Column to match, such as the yes ratio or the recall.
        at: Value of that column for the method.
        column: Column to read at the match.

    Returns:
        The value interpolated between the two neighbouring points of the trace, or the value of
        its nearest end when the match lies outside the trace. NaN for an empty trace.
    """
    kept = curve[[on, column]].dropna().sort_values(on, kind="stable")
    if kept.empty or np.isnan(at):
        return float("nan")
    return float(np.interp(at, kept[on].to_numpy(), kept[column].to_numpy()))


def pope_shortcut(records: Sequence[Record], base: Sequence[Record]) -> list[dict[str, Any]]:
    """Scores of a method next to those the unmodified model reaches at the same yes ratio.

    Args:
        records: Answers of the method to yes or no questions.
        base: Answers of the unmodified model to the same questions.

    Returns:
        One row for the accuracy and one for the F1, with the `value` of the method, the value
        `unmodified`, the `shortcut` the unmodified model reaches by a threshold giving the yes
        ratio of the method, and `beyond`, what the method adds to that. Empty when the
        unmodified answers carry no probability of yes and no.
    """
    margins = yes_margin([record.yes for record in base], [record.no for record in base])
    if not np.all(np.isfinite(margins)):
        return []
    labels = np.array([record.label == "yes" for record in base], dtype=np.bool_)
    curve = bias_curve(margins, labels)

    def scores(answers: Sequence[Record]) -> dict[str, float]:
        said = np.array([pope_answer(record.text) for record in answers], dtype=np.bool_)
        wanted = np.array([record.label == "yes" for record in answers], dtype=np.bool_)
        return binary_scores(said, wanted)

    steered, unmodified = scores(records), scores(base)
    rows: list[dict[str, Any]] = []
    for metric in ("accuracy", "f1"):
        shortcut = matched(curve, "yes_ratio", steered["yes_ratio"], metric)
        rows.append(
            {
                "metric": metric,
                "value": steered[metric],
                "unmodified": unmodified[metric],
                "matched_on": "yes_ratio",
                "at": steered["yes_ratio"],
                "shortcut": shortcut,
                "beyond": steered[metric] - shortcut,
            }
        )
    return rows


def chair_shortcut(
    records: Sequence[Record],
    base: Sequence[Record],
    present: dict[int, frozenset[str]],
    lemmatize: Lemmatizer = nltk_lemmas,
) -> list[dict[str, Any]]:
    """Scores of a method next to those the unmodified captions reach when cut to its recall.

    Args:
        records: Captions of the method.
        base: Captions of the unmodified model for the same images.
        present: Objects of every image, by image identifier.
        lemmatize: Function turning a caption into lemmas.

    Returns:
        One row for CHAIR_S and one for CHAIR_I, with the `value` of the method, the value
        `unmodified`, the `shortcut` the unmodified captions reach once truncated to the recall
        of the method, and `beyond`, how far the method is below that.
    """
    objects = [present[int(record.id)] for record in base]
    curve = truncation_curve([record.text for record in base], objects, lemmatize)

    def scores(captions: Sequence[Record]) -> dict[str, float]:
        return chair_scores(
            [
                caption_counts(mentions_of(lemmatize(record.text)), present[int(record.id)])
                for record in captions
            ]
        )

    steered, unmodified = scores(records), scores(base)
    rows: list[dict[str, Any]] = []
    for metric in ("chair_s", "chair_i"):
        shortcut = matched(curve, "recall", steered["recall"], metric)
        rows.append(
            {
                "metric": metric,
                "value": steered[metric],
                "unmodified": unmodified[metric],
                "matched_on": "recall",
                "at": steered["recall"],
                "shortcut": shortcut,
                "beyond": shortcut - steered[metric],
            }
        )
    return rows
