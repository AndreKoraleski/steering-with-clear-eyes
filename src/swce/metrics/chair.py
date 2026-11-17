"""CHAIR, the rate of objects a caption mentions that the image does not contain.

This is a port of the script distributed with OPERA (Huang et al., 2024, MIT licence), itself the
Python 3 version by Maxlinn of the original script of Rohrbach et al. (2018). The synonym list,
the handling of two word objects and the special cases are theirs, unchanged. The objects of an
image are the union of the categories of its instance annotations and of the objects its reference
captions mention.

The original script reads the training and validation annotations together. Images are scored
against their own annotations only, so reading those of the evaluated images gives the same
scores.
"""

import json
import zipfile
from collections.abc import Callable, Collection, Iterable, Sequence
from functools import cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

SYNONYMS_PATH = Path(__file__).parent / "data" / "synonyms.txt"
"""Synonyms of the 80 COCO categories, one category per line, as distributed with the script."""
DOUBLE_WORDS = (
    "motor bike",
    "motor cycle",
    "air plane",
    "traffic light",
    "street light",
    "traffic signal",
    "stop light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "suit case",
    "sports ball",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "wine glass",
    "hot dog",
    "cell phone",
    "mobile phone",
    "teddy bear",
    "hair drier",
    "potted plant",
    "bow tie",
    "laptop computer",
    "stove top oven",
    "home plate",
    "train track",
)
"""Pairs of words the script treats as one object."""
ANIMAL_WORDS = (
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "animal",
    "cub",
)
"""Animals whose qualifiers baby and adult would otherwise count as a person."""
VEHICLE_WORDS = ("jet", "train")
"""Vehicles whose qualifier passenger would otherwise count as a person."""
ANNOTATION_MEMBERS = ("annotations/instances_val2014.json", "annotations/captions_val2014.json")
"""Members of the COCO annotation archive holding the validation instances and captions."""

Lemmatizer = Callable[[str], list[str]]
"""Function turning a caption into its lemmas, in order."""


class Vocabulary(BaseModel):
    """Words the metric recognizes as COCO objects.

    Attributes:
        nodes: Category of every recognized word, including the categories themselves.
        double_words: Replacement of every pair of words treated as one.
    """

    model_config = ConfigDict(frozen=True)

    nodes: dict[str, str]
    double_words: dict[str, str]


@cache
def vocabulary(path: Path = SYNONYMS_PATH) -> Vocabulary:
    """Vocabulary built from the synonym list as the script builds it.

    Args:
        path: File holding the synonyms.

    Returns:
        The vocabulary, cached per path.
    """
    nodes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        # The script splits on a comma and one space, which keeps the second space of the few
        # entries written with two. Those entries therefore never match, here as there.
        synonyms = line.strip().split(", ")
        for synonym in synonyms:
            nodes[synonym] = synonyms[0]
    double_words = {word: word for word in DOUBLE_WORDS}
    for animal in ANIMAL_WORDS:
        double_words[f"baby {animal}"] = animal
        double_words[f"adult {animal}"] = animal
    for vehicle in VEHICLE_WORDS:
        double_words[f"passenger {vehicle}"] = vehicle
    double_words["bow tie"] = "tie"
    double_words["toilet seat"] = "toilet"
    double_words["wine glas"] = "wine glass"
    return Vocabulary(nodes=nodes, double_words=double_words)


def nltk_lemmas(caption: str) -> list[str]:
    """Lemmas of a caption as the script computes them with NLTK.

    The caption is lowered, tokenized and tagged, and every token is lemmatized with the WordNet
    part of speech of its tag, noun when the tag maps to none.

    Args:
        caption: Caption to process.

    Returns:
        The lemmas, one per token.

    Raises:
        LookupError: If the NLTK resources `punkt_tab`, `averaged_perceptron_tagger_eng` and
            `wordnet` are not installed.
    """
    import nltk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    parts = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
    lemmatizer = WordNetLemmatizer()
    tagged = nltk.pos_tag(nltk.word_tokenize(caption.lower()))
    return [
        lemmatizer.lemmatize(word, pos=parts.get(tag[:1], wordnet.NOUN)) for word, tag in tagged
    ]


class Mentions(BaseModel):
    """Objects a caption mentions.

    Attributes:
        nodes: Category of every mention of a COCO object, in order, with repetitions.
        words: Number of words of the caption once pairs are merged.
    """

    model_config = ConfigDict(frozen=True)

    nodes: tuple[str, ...]
    words: int


def mentions(caption: str, lemmatize: Lemmatizer = nltk_lemmas) -> Mentions:
    """COCO objects mentioned by a caption.

    Args:
        caption: Caption to read.
        lemmatize: Function turning the caption into lemmas.

    Returns:
        The mentions and the length of the caption.
    """
    return mentions_of(lemmatize(caption))


def mentions_of(lemmas: Sequence[str]) -> Mentions:
    """COCO objects mentioned by the lemmas of a caption.

    Args:
        lemmas: Lemmas of the caption, or of its beginning.

    Returns:
        The mentions and the number of words once pairs are merged.
    """
    known = vocabulary()
    words: list[str] = []
    index = 0
    while index < len(lemmas):
        pair = " ".join(lemmas[index : index + 2])
        if pair in known.double_words:
            words.append(known.double_words[pair])
            index += 2
        else:
            words.append(lemmas[index])
            index += 1
    length = len(words)
    # A toilet seat is not a chair, so the seat is dropped whenever a toilet is mentioned. The
    # script measures the length before dropping it.
    if "toilet" in words and "seat" in words:
        words = [word for word in words if word != "seat"]
    nodes = tuple(known.nodes[word] for word in words if word in known.nodes)
    return Mentions(nodes=nodes, words=length)


def ground_truth(
    annotations: Path, images: Collection[int], lemmatize: Lemmatizer = nltk_lemmas
) -> dict[int, frozenset[str]]:
    """Objects of images according to their COCO annotations.

    Args:
        annotations: Archive of the COCO 2014 annotations.
        images: Identifiers of the validation images to read.
        lemmatize: Function turning a reference caption into lemmas.

    Returns:
        For every requested image, the categories of its instances together with the objects its
        reference captions mention.
    """
    known = vocabulary()
    wanted = set(images)
    objects: dict[int, set[str]] = {image: set() for image in wanted}
    with zipfile.ZipFile(annotations) as archive:
        instances = json.loads(archive.read(ANNOTATION_MEMBERS[0]))
        names = {category["id"]: category["name"] for category in instances["categories"]}
        for annotation in instances["annotations"]:
            if annotation["image_id"] in wanted:
                objects[annotation["image_id"]].add(known.nodes[names[annotation["category_id"]]])
        del instances
        captions = json.loads(archive.read(ANNOTATION_MEMBERS[1]))
        for annotation in captions["annotations"]:
            if annotation["image_id"] in wanted:
                found = mentions(annotation["caption"], lemmatize)
                objects[annotation["image_id"]].update(found.nodes)
    return {image: frozenset(found) for image, found in objects.items()}


class CaptionCounts(BaseModel):
    """Counts of one caption that the corpus metrics add up.

    Attributes:
        mentioned: Number of mentions of COCO objects.
        hallucinated: Number of those mentions whose object is absent from the image.
        present: Number of objects of the image.
        recalled: Number of objects of the image the caption mentions.
        words: Number of words of the caption.
    """

    model_config = ConfigDict(frozen=True)

    mentioned: int
    hallucinated: int
    present: int
    recalled: int
    words: int


def caption_counts(found: Mentions, present: frozenset[str]) -> CaptionCounts:
    """Counts of one caption against the objects of its image.

    Args:
        found: Objects the caption mentions.
        present: Objects of the image.

    Returns:
        The counts of the caption.
    """
    return CaptionCounts(
        mentioned=len(found.nodes),
        hallucinated=sum(node not in present for node in found.nodes),
        present=len(present),
        recalled=len(set(found.nodes) & present),
        words=found.words,
    )


def chair_scores(counts: Sequence[CaptionCounts] | Iterable[CaptionCounts]) -> dict[str, float]:
    """Corpus metrics of CHAIR.

    Args:
        counts: Counts of every caption.

    Returns:
        `chair_s`, the fraction of captions with at least one hallucinated object, `chair_i`, the
        fraction of object mentions that are hallucinated, `recall`, the fraction of the objects
        of the images that are mentioned, and `length`, the mean number of words. The script
        prints the length divided by a hundred. A ratio whose denominator is zero is NaN.
    """
    rows = list(counts)
    mentioned = sum(row.mentioned for row in rows)
    present = sum(row.present for row in rows)
    nan = float("nan")
    return {
        "chair_s": sum(row.hallucinated > 0 for row in rows) / len(rows) if rows else nan,
        "chair_i": sum(row.hallucinated for row in rows) / mentioned if mentioned else nan,
        "recall": sum(row.recalled for row in rows) / present if present else nan,
        "length": sum(row.words for row in rows) / len(rows) if rows else nan,
    }
