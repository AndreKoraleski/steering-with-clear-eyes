"""Schema of the benchmark registry.

A benchmark card documents where a benchmark comes from, under which terms it may be used and
which files it consists of. Every file is pinned by its size and digest, so that an evaluation
either reads the exact bytes the registry names or fails. Tasks are the units an evaluation runs,
and each reads pinned files of its own benchmark and of the benchmarks it requires.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class TaskKind(StrEnum):
    """Form of the answers a task expects.

    Attributes:
        YES_NO: One closed question per item, answered yes or no.
        CAPTION: One free description per image.
    """

    YES_NO = "yes_no"
    CAPTION = "caption"


class PinnedFile(BaseModel):
    """File of a benchmark, pinned by its size and digest.

    Attributes:
        path: Location under the data root, with forward slashes.
        url: Address the file is downloaded from, on the host of its authors.
        size: Size in bytes.
        digest: Hexadecimal digest of the content.
        algorithm: Hash function of `digest`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    url: str
    size: int
    digest: str
    algorithm: Literal["sha256", "md5"] = "sha256"


class TaskCard(BaseModel):
    """Task defined on a benchmark.

    Attributes:
        key: Unique name of the task, which keys the leaderboard.
        kind: Form of the answers.
        description: One sentence on what the task measures.
        loader: Name of the loader building its items.
        scorer: Name of the scorer turning answers into metrics.
        primary: Metric that summarizes the task.
        higher_is_better: Whether a larger primary metric is better.
        max_new_tokens: Largest number of tokens generated per item.
        files: Paths of the pinned files the loader reads, of this benchmark or a required one.
        requires: Keys of other benchmarks whose files or images the task reads.
        options: Settings passed to the loader.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    kind: TaskKind
    description: str
    loader: str
    scorer: str
    primary: str
    higher_is_better: bool = True
    max_new_tokens: int
    files: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    options: dict[str, str] = {}


class BenchmarkCard(BaseModel):
    """Provenance, terms and content of one benchmark.

    Attributes:
        key: Unique name of the benchmark.
        name: Display name.
        reference: Publication that introduced it.
        source: Page or repository of its authors.
        revision: Commit or revision the files are pinned at, empty when the host has none.
        licence: Licence of the files, or a statement that none is declared.
        terms: What the licence means for use and redistribution.
        image_url: Template of the address of one image, with the placeholder `{name}`, for
            benchmarks whose images are fetched one by one, or None.
        image_dir: Directory under the data root holding those images.
        files: Pinned files.
        tasks: Tasks defined on the benchmark.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    name: str
    reference: str
    source: str
    revision: str = ""
    licence: str
    terms: str
    image_url: str | None = None
    image_dir: str = ""
    files: tuple[PinnedFile, ...] = ()
    tasks: tuple[TaskCard, ...] = ()


class Registry(BaseModel):
    """Every benchmark the project evaluates on.

    Attributes:
        version: Version of the registry, bumped whenever a pin or a task changes.
        benchmarks: Cards of the benchmarks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    benchmarks: tuple[BenchmarkCard, ...]

    @model_validator(mode="after")
    def _consistent(self) -> "Registry":
        """Check that keys are unique and that every reference resolves."""
        benchmarks = [card.key for card in self.benchmarks]
        tasks = [task.key for card in self.benchmarks for task in card.tasks]
        paths = [file.path for card in self.benchmarks for file in card.files]
        for kind, names in (("benchmark", benchmarks), ("task", tasks), ("file", paths)):
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(f"repeated {kind} keys: {repeated}")
        for card in self.benchmarks:
            for task in card.tasks:
                missing = sorted(set(task.requires) - set(benchmarks))
                if missing:
                    raise ValueError(f"task {task.key} requires unknown benchmarks: {missing}")
                missing = sorted(set(task.files) - set(paths))
                if missing:
                    raise ValueError(f"task {task.key} reads unpinned files: {missing}")
        return self

    def benchmark(self, key: str) -> BenchmarkCard:
        """Card of a benchmark.

        Args:
            key: Key of the benchmark.

        Returns:
            The card with that key.

        Raises:
            KeyError: If no benchmark has that key.
        """
        for card in self.benchmarks:
            if card.key == key:
                return card
        raise KeyError(f"unknown benchmark {key!r}")

    def task(self, key: str) -> tuple[BenchmarkCard, TaskCard]:
        """Task and the benchmark that defines it.

        Args:
            key: Key of the task.

        Returns:
            The card of the benchmark and the card of the task.

        Raises:
            KeyError: If no task has that key.
        """
        for card in self.benchmarks:
            for task in card.tasks:
                if task.key == key:
                    return card, task
        raise KeyError(f"unknown task {key!r}")

    def task_keys(self) -> tuple[str, ...]:
        """Keys of every task, in registry order.

        Returns:
            The task keys.
        """
        return tuple(task.key for card in self.benchmarks for task in card.tasks)

    def file(self, path: str) -> PinnedFile:
        """Pinned file at a path under the data root.

        Args:
            path: Path of the file as the registry spells it.

        Returns:
            The pinned file.

        Raises:
            KeyError: If no file is pinned at that path.
        """
        for card in self.benchmarks:
            for file in card.files:
                if file.path == path:
                    return file
        raise KeyError(f"no file pinned at {path!r}")
