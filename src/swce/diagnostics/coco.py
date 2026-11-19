"""Objects of COCO 2014 images, read from their annotations, and how often they co-occur.

POPE asks whether one of the 80 COCO categories is in an image, and draws the absent objects of
its adversarial split from those that co-occur most with the present ones. Reading the same
annotations lets a diagnostic set every question against the objects of its image and against
how often its object appears with them elsewhere.
"""

import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import combinations
from pathlib import Path

from swce.data.download import fetch
from swce.registry import load_registry

QUESTION = re.compile(r"^Is there an? (?P<object>.+?) in the image\?$")
"""Form of every POPE question."""


def object_of(question: str) -> str:
    """Object a POPE question asks about.

    Raises:
        ValueError: If the question does not have the form of POPE.
    """
    found = QUESTION.match(question.strip())
    if found is None:
        raise ValueError(f"not a POPE question: {question!r}")
    return found.group("object")


def annotations_path() -> Path:
    """Local copy of the archive of the COCO 2014 annotations, downloaded when missing."""
    card = load_registry().benchmark("coco-2014")
    (file,) = [file for file in card.files if file.path.endswith("annotations_trainval2014.zip")]
    return fetch(file)


def image_objects(split: str, archive: Path | None = None) -> dict[str, frozenset[str]]:
    """Categories of the instances of every image of a split, by file name.

    Args:
        split: `train` or `val`.
        archive: Archive of the annotations, or None for the pinned one.

    Returns:
        For every image with at least one instance, the names of its categories.
    """
    path = archive or annotations_path()
    with zipfile.ZipFile(path) as opened:
        data = json.loads(opened.read(f"annotations/instances_{split}2014.json"))
    names = {category["id"]: category["name"] for category in data["categories"]}
    files = {image["id"]: image["file_name"] for image in data["images"]}
    found: dict[str, set[str]] = {}
    for annotation in data["annotations"]:
        found.setdefault(files[annotation["image_id"]], set()).add(names[annotation["category_id"]])
    return {name: frozenset(objects) for name, objects in found.items()}


def categories(archive: Path | None = None) -> list[str]:
    """Names of the 80 COCO categories, in the order of their identifiers."""
    path = archive or annotations_path()
    with zipfile.ZipFile(path) as opened:
        data = json.loads(opened.read("annotations/instances_val2014.json"))
    return [category["name"] for category in sorted(data["categories"], key=lambda c: c["id"])]


class Cooccurrence:
    """How often COCO categories appear together in a set of images.

    Attributes:
        images: Number of images each category appears in.
        pairs: Number of images each unordered pair of categories appears in together.
    """

    def __init__(self, objects: Iterable[frozenset[str]]) -> None:
        """Count the categories and pairs of every image."""
        self.images: Counter[str] = Counter()
        self.pairs: Counter[frozenset[str]] = Counter()
        for found in objects:
            self.images.update(found)
            self.pairs.update(frozenset(pair) for pair in combinations(sorted(found), 2))

    def given(self, target: str, present: str) -> float:
        """Share of the images holding `present` that also hold `target`."""
        count = self.images[present]
        return self.pairs[frozenset((target, present))] / count if count else 0.0

    def strongest(self, target: str, present: Iterable[str]) -> float:
        """Largest share of `given` over the objects present, zero when none is."""
        return max((self.given(target, other) for other in present if other != target), default=0.0)


def cooccurrence(objects: Mapping[str, frozenset[str]]) -> Cooccurrence:
    """Co-occurrence of the categories over the images of a mapping."""
    return Cooccurrence(objects.values())
