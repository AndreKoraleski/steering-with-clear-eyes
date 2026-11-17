import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from support import tiny_registry

from swce.registry import load_registry
from swce.tasks.loaders import MME_SUBTASKS, chair, mme
from swce.tasks.task import Task, describe


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def _mme_archive(path: Path, subtasks: tuple[str, ...] = MME_SUBTASKS) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for subtask in subtasks:
            root = f"MME/{subtask}"
            lines = f"Is it {subtask}? Please answer yes or no.\tYes\nIs it not?\tNo"
            if subtask == "artwork":
                archive.writestr(f"{root}/images/1.jpg", _png())
                archive.writestr(f"{root}/questions_answers_YN/1.txt", lines)
            else:
                archive.writestr(f"{root}/1.png", _png())
                archive.writestr(f"{root}/1.txt", lines)
    return path


def test_pope_task_reads_questions_and_images(tmp_path: Path) -> None:
    task = Task("pope-tiny", tmp_path / "data", tiny_registry(tmp_path / "host"))
    items = task.items()
    assert [item.id for item in items] == ["1", "2", "3", "4", "5", "6"]
    assert items[0].prompt == "Is there a dog in the image?"
    assert items[0].label == "yes"
    assert items[0].group == "a.jpg"
    assert task.image(items[0]).size == (4, 4)
    assert [item.group for item in task.items(limit=2)] == ["a.jpg", "a.jpg", "b.jpg", "b.jpg"]


def test_fingerprint_follows_the_task_definition(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    task = Task("pope-tiny", tmp_path / "data", registry)
    tiny = registry.benchmark("tiny")
    card = tiny.tasks[0].model_copy(update={"max_new_tokens": 16})
    changed = registry.model_copy(
        update={"benchmarks": (registry.benchmarks[0], tiny.model_copy(update={"tasks": (card,)}))}
    )
    assert task.fingerprint == Task("pope-tiny", tmp_path / "data", registry).fingerprint
    assert task.fingerprint != Task("pope-tiny", tmp_path / "data", changed).fingerprint


def test_chair_items_ask_the_prompt_of_the_task(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"image_id": 42, "caption": "ignored"}) + "\n", encoding="utf-8")
    (item,) = chair([log], ["chair/log.jsonl"], {"prompt": "Describe."})
    assert item.id == "42"
    assert item.image == "COCO_val2014_000000000042.jpg"
    assert item.prompt == "Describe."
    assert item.label is None


def test_mme_items_follow_both_layouts_and_the_official_order(tmp_path: Path) -> None:
    archive = _mme_archive(tmp_path / "mme.zip")
    items = mme([archive], ["mme/mme.zip"], {})
    assert len(items) == 2 * len(MME_SUBTASKS)
    assert list(dict.fromkeys(item.category for item in items)) == list(MME_SUBTASKS)
    artwork = [item for item in items if item.category == "artwork"]
    assert artwork[0].image == "MME/artwork/images/1.jpg"
    assert artwork[0].archive == "mme/mme.zip"
    assert [item.label for item in artwork] == ["yes", "no"]
    assert artwork[0].group == artwork[1].group
    assert artwork[0].prompt.endswith("Please answer yes or no.")


def test_mme_reports_a_missing_subtask(tmp_path: Path) -> None:
    archive = _mme_archive(tmp_path / "mme.zip", MME_SUBTASKS[:-1])
    with pytest.raises(ValueError, match="code_reasoning is missing"):
        mme([archive], ["mme/mme.zip"], {})


def test_describe_names_the_budget_and_files() -> None:
    line = describe(load_registry().task("chair")[1])
    assert "512 new tokens" in line
    assert "annotations_trainval2014.zip" in line
