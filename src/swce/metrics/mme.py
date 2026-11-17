"""Scores of MME as its official script computes them.

A subtask scores the percentage of correct answers plus the percentage of images whose two
questions are both answered correctly, 200 at most. Perception adds ten subtasks, cognition four
and the total all fourteen, 2800 at most.
"""

from collections import defaultdict
from collections.abc import Sequence

PERCEPTION = (
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
)
"""Perception subtasks, in the order of the official script."""
COGNITION = (
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
)
"""Cognition subtasks, in the order of the official script."""
HALLUCINATION = ("existence", "count", "position", "color")
"""Subtasks that hallucination papers report as the hallucination subset."""


def mme_answer(text: str) -> str:
    """Answer as the official MME script reads it.

    The script lowers the answer and accepts it when it is exactly `yes` or `no`. Otherwise it
    looks for `yes`, then for `no`, in the first four characters.

    Args:
        text: Generated answer.

    Returns:
        `yes`, `no` or `other`.
    """
    answer = text.lower()
    if answer in ("yes", "no"):
        return answer
    prefix = answer[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def mme_scores(categories: Sequence[str], correct: Sequence[bool]) -> dict[str, float]:
    """Scores of MME from the correctness of every answer.

    As in the official script, the two questions of an image are consecutive within their
    subtask. Resamples that repeat an image therefore count it once per repetition.

    Args:
        categories: Subtask of each question.
        correct: Whether each answer equals the expected one. An answer read as `other` is wrong.

    Returns:
        The score of every subtask present under its name, `perception`, `cognition`, `total`,
        the `hallucination` subset, `other` as the total without that subset, and `accuracy` over
        all questions. A group score adds the subtasks present.

    Raises:
        ValueError: If a subtask has an odd number of questions.
    """
    answers: dict[str, list[bool]] = defaultdict(list)
    for category, right in zip(categories, correct, strict=True):
        answers[category].append(right)
    scores: dict[str, float] = {}
    for category, rights in answers.items():
        if len(rights) % 2:
            raise ValueError(f"subtask {category} has an odd number of questions")
        pairs = [rights[index : index + 2] for index in range(0, len(rights), 2)]
        accuracy = sum(rights) / len(rights)
        accuracy_plus = sum(all(pair) for pair in pairs) / len(pairs)
        scores[category] = 100 * accuracy + 100 * accuracy_plus
    subtasks = dict(scores)
    scores["perception"] = sum(subtasks.get(name, 0.0) for name in PERCEPTION)
    scores["cognition"] = sum(subtasks.get(name, 0.0) for name in COGNITION)
    scores["total"] = scores["perception"] + scores["cognition"]
    scores["hallucination"] = sum(subtasks.get(name, 0.0) for name in HALLUCINATION)
    scores["other"] = scores["total"] - scores["hallucination"]
    scores["accuracy"] = sum(correct) / len(correct) if correct else float("nan")
    return scores
