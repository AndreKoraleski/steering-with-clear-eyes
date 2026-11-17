"""Confidence intervals and paired tests over the items of a task.

Questions about one image are not independent, so the bootstrap resamples images and keeps the
questions of an image together (Miller, 2024). Two conditions answering the same items are
compared on the same resamples, which removes the variance the items share. Intervals are the
percentile intervals of Efron and Tibshirani (1993). The evaluation computes every metric of a
scorer on each resample, so this module only draws the resamples and reads the intervals.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest

RESAMPLES = 1000
"""Default number of bootstrap resamples."""
LEVEL = 0.95
"""Default coverage of the confidence interval."""


def cluster_resamples(
    groups: Sequence[str], resamples: int = RESAMPLES, seed: int = 0
) -> list[NDArray[np.intp]]:
    """Resamples of the items of a task, drawn by whole groups.

    Args:
        groups: Group of each item, usually its image.
        resamples: Number of resamples.
        seed: Seed of the resampling.

    Returns:
        For every resample, the indices of the items of as many groups as the task has, drawn
        with replacement.
    """
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    blocks = [np.array(indices, dtype=np.intp) for indices in members.values()]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), (resamples, len(blocks)))
    return [np.concatenate([blocks[block] for block in draw]) for draw in draws]


def percentile_interval(values: Sequence[float], level: float = LEVEL) -> tuple[float, float]:
    """Percentile interval of the values of a metric over resamples.

    Args:
        values: Value of the metric on every resample.
        level: Coverage of the interval.

    Returns:
        The lower and upper bounds over the finite values, since a metric can be undefined on a
        resample, or NaN bounds when no value is finite.
    """
    finite = np.array([value for value in values if np.isfinite(value)])
    if finite.size == 0:
        return float("nan"), float("nan")
    tail = (1 - level) / 2
    low, high = np.quantile(finite, [tail, 1 - tail])
    return float(low), float(high)


def mcnemar(first: NDArray[np.bool_], second: NDArray[np.bool_]) -> float:
    """Exact McNemar test of two conditions on the same items.

    Args:
        first: Whether the first condition answers each item correctly.
        second: Whether the second condition does.

    Returns:
        The two sided p value of the hypothesis that both conditions are equally often the only
        correct one, which is one when they never disagree. The test ignores the grouping of
        items by image.
    """
    only_first = int(np.sum(first & ~second))
    only_second = int(np.sum(~first & second))
    if only_first + only_second == 0:
        return 1.0
    return float(binomtest(only_first, only_first + only_second, 0.5).pvalue)


def flips(first: NDArray[np.bool_], second: NDArray[np.bool_]) -> dict[str, int]:
    """Items whose correctness differs between two conditions.

    Args:
        first: Whether the reference condition answers each item correctly.
        second: Whether the compared condition does.

    Returns:
        `fixed`, the items only the compared condition answers correctly, and `broken`, the items
        only the reference answers correctly. A net change of zero can hide many flips.
    """
    return {"fixed": int(np.sum(~first & second)), "broken": int(np.sum(first & ~second))}
