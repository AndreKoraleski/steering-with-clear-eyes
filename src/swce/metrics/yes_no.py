"""Metrics of closed yes or no questions.

The answer parser and the first five metrics are those of the official POPE script. The others
are computed from the same answers and never replace them. Sensitivity and criterion follow
signal detection theory (Macmillan and Creelman, 2005) and separate how well a model tells
present from absent objects from how readily it says yes.
"""

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, rankdata

NEGATIVE_WORDS = ("No", "not", "no")
"""Words that make the official POPE script read an answer as no."""


def pope_answer(text: str) -> bool:
    """Whether the official POPE script reads an answer as yes.

    The script keeps the first sentence, removes commas, splits on spaces and reads the answer as
    no when one word is `No`, `not` or `no`. Every other answer is read as yes, including an empty
    or unrelated one.

    Args:
        text: Generated answer.

    Returns:
        True for yes and False for no.
    """
    if text.find(".") != -1:
        text = text.split(".")[0]
    words = text.replace(",", "").split(" ")
    return not any(word in words for word in NEGATIVE_WORDS)


def _ratio(numerator: float, denominator: float) -> float:
    """Quotient, or NaN when the denominator is zero."""
    return numerator / denominator if denominator else math.nan


def binary_scores(predictions: NDArray[np.bool_], labels: NDArray[np.bool_]) -> dict[str, float]:
    """Metrics of yes or no predictions, with yes as the positive class.

    Args:
        predictions: Whether each answer is yes.
        labels: Whether each expected answer is yes.

    Returns:
        Accuracy, precision, recall, F1 and yes ratio as the official POPE script computes them,
        followed by specificity, balanced accuracy, the Matthews correlation coefficient, and the
        sensitivity `d_prime` and `criterion` of signal detection theory. A metric whose
        denominator is zero is NaN.
    """
    tp = float(np.sum(predictions & labels))
    fp = float(np.sum(predictions & ~labels))
    tn = float(np.sum(~predictions & ~labels))
    fn = float(np.sum(~predictions & labels))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    # The rates are corrected as Hautus (1995) proposes, so that perfect rates stay finite.
    hit = (tp + 0.5) / (tp + fn + 1.0)
    alarm = (fp + 0.5) / (fp + tn + 1.0)
    return {
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "yes_ratio": _ratio(tp + fp, tp + tn + fp + fn),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": _ratio(tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
        "d_prime": float(norm.ppf(hit) - norm.ppf(alarm)),
        "criterion": float(-(norm.ppf(hit) + norm.ppf(alarm)) / 2),
    }


def auroc(scores: NDArray[np.float64], labels: NDArray[np.bool_]) -> float:
    """Area under the ROC curve of a score for yes.

    It is the probability that a question whose answer is yes receives a larger score than one
    whose answer is no, with ties counted as one half, and does not depend on any threshold.

    Args:
        scores: Score of yes for each question, larger meaning more likely yes.
        labels: Whether each expected answer is yes.

    Returns:
        The area, or NaN when one of the two classes is absent.
    """
    positives = int(np.sum(labels))
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(scores)
    return float((np.sum(ranks[labels]) - positives * (positives + 1) / 2) / positives / negatives)


def yes_margin(yes: Sequence[float | None], no: Sequence[float | None]) -> NDArray[np.float64]:
    """Score of yes from the probabilities of the tokens yes and no.

    Args:
        yes: Probability of yes for each question, or None where unknown.
        no: Probability of no for each question, or None where unknown.

    Returns:
        The difference of the log probabilities, NaN where either is unknown.
    """
    floor = 1e-12
    return np.array(
        [
            math.nan
            if p is None or q is None
            else math.log(max(p, floor)) - math.log(max(q, floor))
            for p, q in zip(yes, no, strict=True)
        ]
    )
