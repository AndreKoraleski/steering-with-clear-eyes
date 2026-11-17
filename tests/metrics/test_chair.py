import json
import math
import zipfile
from pathlib import Path

import pytest

from swce.metrics.chair import (
    ANNOTATION_MEMBERS,
    Mentions,
    caption_counts,
    chair_scores,
    ground_truth,
    mentions,
    vocabulary,
)


def words(caption: str) -> list[str]:
    """Lemmatizer standing in for NLTK, which needs downloaded resources."""
    return caption.lower().replace(".", "").replace(",", "").split()


def test_vocabulary_maps_synonyms_to_the_80_categories() -> None:
    known = vocabulary()
    assert len(set(known.nodes.values())) == 80
    assert known.nodes["puppy"] == "dog"
    assert known.nodes["dog"] == "dog"
    assert known.nodes["sofa"] == "couch"
    assert known.double_words["baby bird"] == "bird"
    assert known.double_words["passenger train"] == "train"
    assert known.double_words["bow tie"] == "tie"


def test_mentions_merge_pairs_and_keep_repetitions() -> None:
    found = mentions("A puppy and a dog near a fire hydrant and a hot dog", words)
    assert found.nodes == ("dog", "dog", "fire hydrant", "hot dog")
    assert found.words == 11


def test_qualifiers_do_not_count_as_a_person() -> None:
    assert mentions("a baby elephant", words).nodes == ("elephant",)
    assert mentions("a baby", words).nodes == ("person",)
    assert mentions("a passenger jet", words).nodes == ("airplane",)


def test_a_toilet_seat_is_not_a_chair() -> None:
    found = mentions("the seat of the toilet", words)
    assert found.nodes == ("toilet",)
    assert found.words == 5
    assert mentions("a seat", words).nodes == ("chair",)


def test_counts_and_corpus_scores() -> None:
    first = caption_counts(Mentions(nodes=("dog", "dog", "cat"), words=10), frozenset({"dog"}))
    second = caption_counts(Mentions(nodes=("car",), words=6), frozenset({"car", "bus", "person"}))
    assert (first.mentioned, first.hallucinated, first.present, first.recalled) == (3, 1, 1, 1)
    scores = chair_scores([first, second])
    assert scores["chair_s"] == pytest.approx(0.5)
    assert scores["chair_i"] == pytest.approx(1 / 4)
    assert scores["recall"] == pytest.approx(2 / 4)
    assert scores["length"] == pytest.approx(8.0)
    assert all(math.isnan(value) for value in chair_scores([]).values())


def test_ground_truth_unites_instances_and_reference_captions(tmp_path: Path) -> None:
    instances = {
        "categories": [{"id": 1, "name": "person"}, {"id": 18, "name": "dog"}],
        "annotations": [
            {"image_id": 7, "category_id": 18},
            {"image_id": 8, "category_id": 1},
            {"image_id": 9, "category_id": 1},
        ],
    }
    captions = {
        "annotations": [
            {"image_id": 7, "caption": "A puppy on a sofa."},
            {"image_id": 9, "caption": "A man."},
        ]
    }
    archive = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr(ANNOTATION_MEMBERS[0], json.dumps(instances))
        file.writestr(ANNOTATION_MEMBERS[1], json.dumps(captions))
    truth = ground_truth(archive, {7, 8}, words)
    assert truth == {7: frozenset({"dog", "couch"}), 8: frozenset({"person"})}
