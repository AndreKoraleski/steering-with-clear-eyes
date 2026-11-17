"""Interface of the steering methods.

A steerer is a black box that modifies a subject for the duration of a context. Its settings are
pydantic fields and enter its fingerprint, so that two runs of one method at different published
settings are kept apart on the leaderboard.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from swce.models.subject import Subject
from swce.utils.hashing import canonical_sha256


class InapplicableError(RuntimeError):
    """Raised when a method cannot be instantiated on a subject as its paper describes it.

    The case is a result of its own and is reported, never worked around by changing the method.
    """


class Steerer(BaseModel, ABC):
    """Steering method at one setting.

    Attributes:
        name: Unique name of the method, which keys the leaderboard.
        description: One sentence naming the method and its source.
        setting: Name of the published setting the fields reproduce, for instance the benchmark
            the authors chose it for, or `default`.
        fitted_on: Keys of the benchmarks and corpora whose data built the method, an empty tuple
            when it uses none, or None when undeclared, which the leaderboard marks.
        published_for: Names of the subjects the setting is published for. An evaluation skips
            the setting on any other subject, since a value its paper does not give is never
            assumed. Empty means that the setting does not depend on the subject.
        tasks: Patterns of the task keys the setting is meant for, in the syntax of `fnmatch`.
            A paper that publishes one setting per benchmark is evaluated on that benchmark
            and on the capability benchmarks only. Empty means every task.
        fallback: Whether the setting holds the default arguments of the official code and not
            values a paper publishes. An evaluation uses it only on a subject for which the
            method has no published setting.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: ClassVar[str]
    description: ClassVar[str]

    setting: str = "default"
    fitted_on: tuple[str, ...] | None = None
    published_for: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    fallback: bool = False

    @property
    def fingerprint(self) -> str:
        """Hash of the method name and of every field."""
        fields = self.model_dump(mode="json")
        return canonical_sha256({"name": self.name, "fields": fields})[:16]

    def prepare(self, subject: Subject) -> None:
        """Run the steps the recipe of the method takes before inference.

        An evaluation calls it once before steering a subject. A method whose recipe fits
        directions or calibrates a threshold does so here when their outcome is missing, as its
        official code does, so that nothing is left to run by hand.

        Args:
            subject: Model about to be steered.

        Raises:
            InapplicableError: If the recipe cannot be followed on the subject.
        """

    @abstractmethod
    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Steer a subject for the duration of a context.

        Args:
            subject: Model to steer.

        Returns:
            A context inside which every generation of the subject is steered, and after which
            the subject is unmodified.
        """
