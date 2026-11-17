import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from swce.evaluation.analysis import holm, trade_offs
from swce.evaluation.leaderboard import COLUMNS
from swce.evaluation.shortcuts import (
    bias_curve,
    chair_shortcut,
    matched,
    pope_shortcut,
    truncation_curve,
)
from swce.registry import load_registry
from swce.tasks.schema import Record


def words(caption: str) -> list[str]:
    return caption.lower().replace(".", "").split()


def row(
    method: str,
    task: str,
    metric: str,
    primary: bool,
    value: float,
    delta: float,
    low: float,
    high: float,
    p: float = float("nan"),
) -> dict[str, object]:
    return {
        "model": "m",
        "method": method,
        "setting": "default",
        "fingerprint": method,
        "fitted_on": "nothing",
        "task": task,
        "task_fingerprint": "t",
        "metric": metric,
        "primary": primary,
        "value": value,
        "low": value,
        "high": value,
        "delta": delta,
        "delta_low": low,
        "delta_high": high,
        "p_mcnemar": p,
        "fixed": 0,
        "broken": 0,
        "items": 10,
    }


def test_holm_steps_down_and_keeps_untested_comparisons() -> None:
    adjusted = holm([0.01, 0.04, 0.03, float("nan")])
    assert adjusted[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert math.isnan(adjusted[3])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm([]) == []


def test_trade_offs_sign_the_gain_and_test_the_capability() -> None:
    rows = pd.DataFrame(
        [
            row("unmodified", "pope-adversarial", "f1", True, 0.80, float("nan"), 0, 0),
            row("a", "pope-adversarial", "f1", True, 0.82, 0.02, 0.01, 0.03, 0.01),
            row("a", "chair", "chair_s", True, 0.40, -0.05, -0.08, -0.02, 0.04),
            row("a", "mme", "other", False, 995.0, -5.0, -9.0, -1.0),
            row("a", "mme", "total", True, 1500.0, 10.0, 0.0, 20.0),
            row("a-random", "pope-adversarial", "f1", True, 0.815, 0.015, 0.0, 0.03, 0.2),
            row("b", "pope-adversarial", "f1", True, 0.81, 0.01, -0.01, 0.03, 0.03),
            row("b", "mme", "other", False, 970.0, -30.0, -45.0, -15.0),
        ],
        columns=COLUMNS,
    )
    found = trade_offs(rows, load_registry(), margin=0.02)
    table: dict[tuple[str, str], dict[str, Any]] = {
        (str(entry["method"]), str(entry["task"])): {str(k): v for k, v in entry.items()}
        for entry in found.to_dict("records")
    }
    assert len(table) == 4
    pope = table[("a", "pope-adversarial")]
    assert pope["gain"] == pytest.approx(0.02)
    assert pope["capability"] == pytest.approx(-5.0)
    assert pope["capability_relative"] == pytest.approx(-0.005)
    assert pope["non_inferior"]
    assert pope["control_gain"] == pytest.approx(0.015)
    assert pope["beyond_control"] == pytest.approx(0.005)
    assert pope["dominated_by"] == ""
    chair = table[("a", "chair")]
    assert (chair["gain"], chair["gain_low"], chair["gain_high"]) == pytest.approx(
        (0.05, 0.02, 0.08)
    )
    worse = table[("b", "pope-adversarial")]
    assert not worse["non_inferior"]
    assert worse["dominated_by"] == "a default"
    control = table[("a-random", "pope-adversarial")]
    assert math.isnan(control["capability"])
    assert control["non_inferior"] is None
    assert found.p_holm.tolist() == pytest.approx(holm(found.p_mcnemar.tolist()))
    assert trade_offs(rows.iloc[:1], load_registry()).empty


def test_bias_curve_follows_the_threshold() -> None:
    curve = bias_curve(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 0, 1, 0], dtype=np.bool_))
    assert curve.yes_ratio.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert curve.accuracy.tolist() == [0.5, 0.75, 0.5, 0.75, 0.5]
    assert curve.f1.iloc[1] == pytest.approx(2 / 3)
    assert matched(curve, "yes_ratio", 0.375, "accuracy") == pytest.approx(0.625)
    assert matched(curve, "yes_ratio", 2.0, "accuracy") == pytest.approx(0.5)
    assert math.isnan(matched(curve.iloc[:0], "yes_ratio", 0.5, "accuracy"))


def test_pope_shortcut_credits_only_what_a_threshold_cannot_reach() -> None:
    labels = ["yes", "no", "yes", "no"]
    base = [
        Record(id=str(i), group=str(i), label=label, text="No", tokens=1, yes=0.4, no=0.6)
        for i, label in enumerate(labels)
    ]
    always_yes = [record.model_copy(update={"text": "Yes"}) for record in base]
    perfect = [record.model_copy(update={"text": (record.label or "").title()}) for record in base]
    accuracy, _f1 = pope_shortcut(always_yes, base)
    assert (accuracy["value"], accuracy["unmodified"]) == (0.5, 0.5)
    assert accuracy["beyond"] == pytest.approx(0.0)
    accuracy, _f1 = pope_shortcut(perfect, base)
    assert accuracy["at"] == pytest.approx(0.5)
    assert accuracy["beyond"] == pytest.approx(0.5)
    blind = [record.model_copy(update={"yes": None}) for record in base]
    assert pope_shortcut(perfect, blind) == []


def test_truncation_lowers_chair_together_with_recall() -> None:
    captions = ["a dog on a couch near a cat and a car", "a person with a dog and a bird"]
    present = [frozenset({"dog", "couch"}), frozenset({"person", "dog"})]
    curve = truncation_curve(captions, present, words, step=3)
    assert curve.words.tolist() == [3, 6, 9, 12]
    assert curve.recall.is_monotonic_increasing
    assert curve.chair_s.iloc[0] == 0.0
    assert curve.chair_s.iloc[-1] == 1.0
    base = [
        Record(id=str(i), group=str(i), text=caption, tokens=1)
        for i, caption in enumerate(captions)
    ]
    careful = [
        Record(id="0", group="0", text="a dog on a couch", tokens=1),
        Record(id="1", group="1", text="a person with a dog", tokens=1),
    ]
    chair_s, chair_i = chair_shortcut(careful, base, dict(enumerate(present)), words)
    assert (chair_s["value"], chair_s["unmodified"]) == (0.0, 1.0)
    assert chair_s["at"] == pytest.approx(1.0)
    assert chair_s["beyond"] == pytest.approx(1.0)
    assert chair_i["beyond"] > 0
