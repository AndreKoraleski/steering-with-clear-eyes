"""Scorers turning the records of a task into its metrics.

A scorer first parses every record once, which is where the cost lies, and then aggregates any
selection of the parsed records. The bootstrap aggregates resamples of the same parsed records, so
an interval never parses an answer twice.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from swce.metrics.chair import CaptionCounts, caption_counts, chair_scores, ground_truth, mentions
from swce.metrics.mme import mme_answer, mme_scores
from swce.metrics.yes_no import auroc, binary_scores, pope_answer, yes_margin
from swce.tasks.schema import Record
from swce.tasks.task import Task


class Scorer(ABC):
    """Procedure scoring the records of one kind of task.

    Attributes:
        name: Name the task cards refer to.
    """

    name: ClassVar[str]

    @abstractmethod
    def parse(self, records: Sequence[Record], task: Task) -> Sequence[Any]:
        """Read every record once.

        Args:
            records: Records of the task, in item order.
            task: Task the records answer.

        Returns:
            One parsed value per record.
        """

    @abstractmethod
    def aggregate(self, parsed: Sequence[Any]) -> dict[str, float]:
        """Metrics of a selection of parsed records.

        Args:
            parsed: Parsed values, possibly resampled.

        Returns:
            Every metric of the task by name.
        """

    @abstractmethod
    def correct(self, parsed: Sequence[Any]) -> list[bool] | None:
        """Whether each parsed record is a correct answer.

        Args:
            parsed: Parsed values in item order.

        Returns:
            One flag per record, or None when the task has no notion of a correct answer.
        """


class PopeScorer(Scorer):
    """Official POPE scoring, with the measures of signal detection added."""

    name = "pope"

    def parse(self, records: Sequence[Record], task: Task) -> Sequence[Any]:
        """Read every answer with the official parser.

        Args:
            records: Records of the task.
            task: Task the records answer, unused.

        Returns:
            For every record, the answer and the label as booleans and the score of yes.
        """
        margins = yes_margin([record.yes for record in records], [record.no for record in records])
        return [
            (pope_answer(record.text), record.label == "yes", float(margin))
            for record, margin in zip(records, margins, strict=True)
        ]

    def aggregate(self, parsed: Sequence[Any]) -> dict[str, float]:
        """Metrics of yes or no answers.

        Args:
            parsed: Values returned by `parse`.

        Returns:
            The metrics of `binary_scores`, and `auroc` when the model exposes the probabilities
            of yes and no for every item.
        """
        predictions = np.array([row[0] for row in parsed], dtype=np.bool_)
        labels = np.array([row[1] for row in parsed], dtype=np.bool_)
        margins = np.array([row[2] for row in parsed], dtype=np.float64)
        scores = binary_scores(predictions, labels)
        scores["auroc"] = auroc(margins, labels) if np.all(np.isfinite(margins)) else float("nan")
        return scores

    def correct(self, parsed: Sequence[Any]) -> list[bool] | None:
        """Whether each answer equals its label.

        Args:
            parsed: Values returned by `parse`.

        Returns:
            One flag per record.
        """
        return [row[0] == row[1] for row in parsed]


class MmeScorer(Scorer):
    """Official MME scoring."""

    name = "mme"

    def parse(self, records: Sequence[Record], task: Task) -> Sequence[Any]:
        """Read every answer with the official parser.

        Args:
            records: Records of the task.
            task: Task the records answer, unused.

        Returns:
            For every record, its subtask and whether the answer is correct.
        """
        return [
            (record.category or "", mme_answer(record.text) == record.label) for record in records
        ]

    def aggregate(self, parsed: Sequence[Any]) -> dict[str, float]:
        """Scores of MME.

        Args:
            parsed: Values returned by `parse`.

        Returns:
            The scores of `mme_scores`.
        """
        return mme_scores([row[0] for row in parsed], [row[1] for row in parsed])

    def correct(self, parsed: Sequence[Any]) -> list[bool] | None:
        """Whether each answer equals its label.

        Args:
            parsed: Values returned by `parse`.

        Returns:
            One flag per record.
        """
        return [bool(row[1]) for row in parsed]


class ChairScorer(Scorer):
    """CHAIR as the script distributed with OPERA computes it."""

    name = "chair"

    def parse(self, records: Sequence[Record], task: Task) -> Sequence[Any]:
        """Count the objects every caption mentions against those of its image.

        Args:
            records: Records of the task.
            task: Task the records answer, whose second file is the COCO annotation archive.

        Returns:
            The counts of every caption.
        """
        annotations = task.files()[1]
        present = ground_truth(annotations, {int(record.id) for record in records})
        return [
            caption_counts(mentions(record.text), present[int(record.id)]) for record in records
        ]

    def aggregate(self, parsed: Sequence[Any]) -> dict[str, float]:
        """Corpus metrics of CHAIR.

        Args:
            parsed: Values returned by `parse`.

        Returns:
            The metrics of `chair_scores`.
        """
        counts: list[CaptionCounts] = list(parsed)
        return chair_scores(counts)

    def correct(self, parsed: Sequence[Any]) -> list[bool] | None:
        """Whether each caption is free of hallucinated objects.

        Args:
            parsed: Values returned by `parse`.

        Returns:
            One flag per caption.
        """
        return [row.hallucinated == 0 for row in parsed]


SCORERS: dict[str, Scorer] = {
    scorer.name: scorer for scorer in (PopeScorer(), MmeScorer(), ChairScorer())
}
"""Scorers by the name the task cards give them."""
