import math

import numpy as np
import pytest

from swce.metrics.mme import mme_answer, mme_scores
from swce.metrics.yes_no import auroc, binary_scores, pope_answer, yes_margin


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", True),
        ("No", False),
        ("no", False),
        ("No, there is no dog.", False),
        ("Yes, there is a dog. It is not small.", True),
        ("There is not a dog in the image.", False),
        ("NO", True),
        ("", True),
        ("I cannot tell", True),
        ("Yes. No.", True),
    ],
)
def test_pope_answer_follows_the_official_script(text: str, expected: bool) -> None:
    assert pope_answer(text) is expected


def test_binary_scores_match_hand_computed_values() -> None:
    predictions = np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=np.bool_)
    labels = np.array([1, 1, 0, 0, 0, 1, 1, 1], dtype=np.bool_)
    scores = binary_scores(predictions, labels)
    assert scores["accuracy"] == pytest.approx(5 / 8)
    assert scores["precision"] == pytest.approx(3 / 4)
    assert scores["recall"] == pytest.approx(3 / 5)
    assert scores["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert scores["yes_ratio"] == pytest.approx(0.5)
    assert scores["specificity"] == pytest.approx(2 / 3)
    assert scores["balanced_accuracy"] == pytest.approx((0.6 + 2 / 3) / 2)
    assert scores["mcc"] == pytest.approx((3 * 2 - 1 * 2) / math.sqrt(4 * 5 * 3 * 4))


def test_bias_moves_the_criterion_and_not_the_sensitivity() -> None:
    labels = np.array([1] * 50 + [0] * 50, dtype=np.bool_)
    balanced = np.array([1] * 40 + [0] * 10 + [1] * 10 + [0] * 40, dtype=np.bool_)
    always_yes = np.ones(100, dtype=np.bool_)
    assert binary_scores(balanced, labels)["criterion"] == pytest.approx(0.0, abs=1e-9)
    assert binary_scores(balanced, labels)["d_prime"] > 1.0
    assert binary_scores(always_yes, labels)["d_prime"] == pytest.approx(0.0, abs=1e-9)
    assert binary_scores(always_yes, labels)["criterion"] < -1.0
    assert math.isnan(binary_scores(~always_yes, labels)["precision"])


def test_auroc_counts_ties_as_one_half() -> None:
    labels = np.array([1, 1, 0, 0], dtype=np.bool_)
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == pytest.approx(1.0)
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(0.0)
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == pytest.approx(0.5)
    assert math.isnan(auroc(np.array([0.5, 0.5]), np.array([1, 1], dtype=np.bool_)))


def test_yes_margin_is_a_log_ratio() -> None:
    margins = yes_margin([0.8, None, 0.0], [0.2, 0.5, 1.0])
    assert margins[0] == pytest.approx(math.log(4))
    assert math.isnan(margins[1])
    assert margins[2] < -20


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", "yes"),
        ("no", "no"),
        ("Yes, it is.", "yes"),
        ("No.", "no"),
        ("The answer is yes", "other"),
        ("Not sure", "no"),
        ("", "other"),
    ],
)
def test_mme_answer_follows_the_official_script(text: str, expected: str) -> None:
    assert mme_answer(text) == expected


def test_mme_scores_add_accuracy_and_paired_accuracy() -> None:
    categories = ["existence"] * 4 + ["code_reasoning"] * 2
    correct = [True, True, True, False, False, False]
    scores = mme_scores(categories, correct)
    assert scores["existence"] == pytest.approx(75 + 50)
    assert scores["code_reasoning"] == 0.0
    assert scores["perception"] == pytest.approx(125)
    assert scores["cognition"] == 0.0
    assert scores["total"] == pytest.approx(125)
    assert scores["hallucination"] == pytest.approx(125)
    assert scores["other"] == 0.0
    assert scores["accuracy"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="odd number"):
        mme_scores(["existence"], [True])
