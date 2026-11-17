import numpy as np
import pytest

from swce.metrics.bootstrap import cluster_resamples, flips, mcnemar, percentile_interval

GROUPS = [f"image{index // 2}" for index in range(40)]


def test_resamples_draw_whole_groups_and_depend_on_the_seed_only() -> None:
    draws = cluster_resamples(GROUPS, resamples=20, seed=3)
    assert len(draws) == 20
    for draw in draws:
        assert draw.size == 40
        assert all(draw[position] + 1 == draw[position + 1] for position in range(0, 40, 2))
    again = cluster_resamples(GROUPS, resamples=20, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip(draws, again, strict=True))
    other = cluster_resamples(GROUPS, resamples=20, seed=4)
    assert not all(np.array_equal(a, b) for a, b in zip(draws, other, strict=True))


def test_percentile_interval_ignores_undefined_resamples() -> None:
    values = [*np.linspace(0.0, 1.0, 101).tolist(), float("nan")]
    low, high = percentile_interval(values, level=0.9)
    assert low == pytest.approx(0.05)
    assert high == pytest.approx(0.95)
    assert np.isnan(percentile_interval([float("nan")])).all()


def test_mcnemar_and_flips() -> None:
    first = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.bool_)
    second = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool_)
    assert mcnemar(first, first) == 1.0
    assert mcnemar(first, second) == pytest.approx(2 * 0.5**6)
    assert flips(first, second) == {"fixed": 6, "broken": 0}
    assert flips(second, first) == {"fixed": 0, "broken": 6}
