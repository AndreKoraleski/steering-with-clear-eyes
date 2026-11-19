"""Loaders turning the pinned files of a benchmark into items.

Every loader reads the files exactly as their authors publish them. Prompts are the questions as
written, without any suffix, and item order is the order of the files. The one exception is a
task registered as a variant, such as `pope-adversarial-yn`, whose options name the instruction
it appends to every question. A variant has a key and a fingerprint of its own, so its rows never
mix with those of the benchmark as written.
"""

import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath

from swce.tasks.schema import Item

COCO = "coco-2014"
"""Key of the benchmark serving the COCO images."""
MME_SUBTASKS = (
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
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
)
"""Subtasks of MME in the order of its official scoring script."""
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
"""Suffixes of image members of an archive, compared in lower case."""

Loader = Callable[[Sequence[Path], Sequence[str], Mapping[str, str]], list[Item]]
"""Function of the local files of a task, their registry paths and the task options."""


def _json_lines(path: Path) -> list[dict[str, object]]:
    """Objects of a file holding one JSON object per line."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pope(files: Sequence[Path], paths: Sequence[str], options: Mapping[str, str]) -> list[Item]:
    """Items of a POPE question file.

    RePOPE publishes its corrected labels in the same format and keeps the question identifiers,
    so it is read by this loader too.

    Args:
        files: The question file, alone.
        paths: Registry paths of the files, unused.
        options: Task options, whose `suffix`, when given, is appended to every question after a
            space.

    Returns:
        One item per question, in file order.
    """
    suffix = f" {options['suffix']}" if options.get("suffix") else ""
    return [
        Item(
            id=str(row["question_id"]),
            group=str(row["image"]),
            prompt=str(row["text"]) + suffix,
            image=str(row["image"]),
            host=COCO,
            label=str(row["label"]).lower(),
        )
        for row in _json_lines(files[0])
    ]


def chair(files: Sequence[Path], paths: Sequence[str], options: Mapping[str, str]) -> list[Item]:
    """Items of the CHAIR image list.

    The list is the log of a published run, of which only the image identifiers are read.

    Args:
        files: The log, followed by files the scorer reads, which are ignored here.
        paths: Registry paths of the files, unused.
        options: Task options, whose `prompt` is asked for every image.

    Returns:
        One item per image, in file order.
    """
    return [
        Item(
            id=str(row["image_id"]),
            group=str(row["image_id"]),
            prompt=options["prompt"],
            image=f"COCO_val2014_{int(str(row['image_id'])):012d}.jpg",
            host=COCO,
        )
        for row in _json_lines(files[0])
    ]


def _mme_image(names: frozenset[str], questions: PurePosixPath) -> str:
    """Archive member holding the image a question file is about."""
    folders = [questions.parent]
    if questions.parent.name == "questions_answers_YN":
        folders = [questions.parent.parent / "images"]
    for folder in folders:
        for suffix in IMAGE_SUFFIXES:
            member = str(folder / (questions.stem + suffix))
            if member in names:
                return member
    raise ValueError(f"no image for {questions}")


def mme(files: Sequence[Path], paths: Sequence[str], options: Mapping[str, str]) -> list[Item]:
    """Items of the MME release archive, read without unpacking it.

    Every image comes with a text file of two lines, each a question and its answer separated by
    a tabulation. Images stay in the archive and are read from it during the evaluation.

    Args:
        files: The release archive, alone.
        paths: Registry paths of the files, the first of which locates the archive for the items.
        options: Task options, unused.

    Returns:
        Two items per image, grouped by subtask in the order of the official scoring script and
        sorted by file name within a subtask.

    Raises:
        ValueError: If a question file has no image or a subtask is missing.
    """
    items: list[Item] = []
    with zipfile.ZipFile(files[0]) as archive:
        names = frozenset(archive.namelist())
        texts = sorted(PurePosixPath(name) for name in names if name.endswith(".txt"))
        for subtask in MME_SUBTASKS:
            members = [text for text in texts if subtask in text.parts]
            if not members:
                raise ValueError(f"subtask {subtask} is missing from {files[0]}")
            for member in members:
                image = _mme_image(names, member)
                lines = archive.read(str(member)).decode("utf-8").splitlines()
                for index, line in enumerate(line for line in lines if line.strip()):
                    question, answer = line.split("\t")
                    items.append(
                        Item(
                            id=f"{subtask}/{member.stem}/{index}",
                            group=f"{subtask}/{member.stem}",
                            prompt=question,
                            image=image,
                            archive=paths[0],
                            label=answer.strip().lower(),
                            category=subtask,
                        )
                    )
    return items


LOADERS: dict[str, Loader] = {"pope": pope, "chair": chair, "mme": mme}
"""Loaders by the name the task cards give them."""
