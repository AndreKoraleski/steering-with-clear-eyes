"""Tasks ready to be evaluated, with their items and images."""

import io
import zipfile
from pathlib import Path

from PIL import Image

from swce.data.download import fetch, fetch_image
from swce.data.paths import default_data_root
from swce.registry import BenchmarkCard, Registry, TaskCard, load_registry
from swce.tasks.loaders import LOADERS
from swce.tasks.schema import Item
from swce.utils.hashing import canonical_sha256


class Task:
    """Task of the registry bound to a data root.

    Attributes:
        benchmark: Card of the benchmark defining the task.
        card: Card of the task.
        root: Data root holding the local copies.
    """

    def __init__(self, key: str, root: Path | None = None, registry: Registry | None = None):
        self._registry = registry or load_registry()
        self.benchmark, self.card = self._registry.task(key)
        self.root = root or default_data_root()
        self._archives: dict[str, zipfile.ZipFile] = {}

    @property
    def key(self) -> str:
        """Key of the task."""
        return self.card.key

    @property
    def fingerprint(self) -> str:
        """Hash of everything that defines the task.

        It covers the task card and the pins of the files it reads. Results obtained under
        different fingerprints are never compared.
        """
        pins = [self._registry.file(path).model_dump() for path in self.card.files]
        return canonical_sha256({"card": self.card.model_dump(mode="json"), "pins": pins})[:16]

    def files(self) -> list[Path]:
        """Local copies of the pinned files of the task, downloaded when missing.

        Returns:
            The verified local paths, in the order of the task card.

        Raises:
            swce.data.download.PinError: If a file differs from its pin.
        """
        return [fetch(self._registry.file(path), self.root) for path in self.card.files]

    def items(self, limit: int | None = None) -> list[Item]:
        """Items of the task.

        Args:
            limit: Number of leading images to keep with all their items, for smoke tests, or
                None for the whole task.

        Returns:
            The items in the order of the benchmark files.
        """
        items = LOADERS[self.card.loader](self.files(), self.card.files, self.card.options)
        if limit is None:
            return items
        groups = list(dict.fromkeys(item.group for item in items))[:limit]
        return [item for item in items if item.group in set(groups)]

    def image(self, item: Item) -> Image.Image:
        """Image of an item, fetched from its host or read from its archive.

        Args:
            item: Item of this task.

        Returns:
            The image converted to RGB.

        Raises:
            ValueError: If the item names neither a host nor an archive.
        """
        if item.archive is not None:
            if item.archive not in self._archives:
                self._archives[item.archive] = zipfile.ZipFile(self.root / item.archive)
            content = self._archives[item.archive].read(item.image)
            return Image.open(io.BytesIO(content)).convert("RGB")
        if item.host is None:
            raise ValueError(f"item {item.id} names neither a host nor an archive")
        host: BenchmarkCard = self._registry.benchmark(item.host)
        with Image.open(fetch_image(host, item.image, self.root)) as image:
            return image.convert("RGB")


def describe(card: TaskCard) -> str:
    """One line on how a task is run, for dry runs.

    Args:
        card: Card of the task.

    Returns:
        The key, kind, token budget and files of the task.
    """
    files = ", ".join(card.files) or "no pinned file"
    return f"{card.key}: {card.kind.value}, {card.max_new_tokens} new tokens, reads {files}"
