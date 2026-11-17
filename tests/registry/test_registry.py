from pathlib import Path

import pytest
from pydantic import ValidationError
from support import tiny_registry

from swce.registry import Registry, load_registry
from swce.registry.__main__ import main
from swce.registry.docs import DOCS_PATH, render, write_docs


def test_committed_registry_is_valid_and_documented() -> None:
    registry = load_registry()
    assert "pope-adversarial" in registry.task_keys()
    assert "chair" in registry.task_keys()
    assert "mme" in registry.task_keys()
    assert DOCS_PATH.read_text(encoding="utf-8") == render(registry)
    assert main(["--check"]) == 0


def test_every_pin_is_a_full_digest() -> None:
    lengths = {"sha256": 64, "md5": 32}
    for card in load_registry().benchmarks:
        for file in card.files:
            assert len(file.digest) == lengths[file.algorithm]
            assert file.size > 0


def test_lookups(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path)
    card, task = registry.task("pope-tiny")
    assert card.key == "tiny"
    assert registry.file(task.files[0]).path == "tiny/questions.json"
    assert registry.benchmark("coco-2014").image_url is not None
    with pytest.raises(KeyError):
        registry.task("missing")
    with pytest.raises(KeyError):
        registry.file("missing")
    with pytest.raises(KeyError):
        registry.benchmark("missing")


def test_repeated_keys_and_dangling_references_are_rejected(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path)
    tiny = registry.benchmark("tiny")
    with pytest.raises(ValidationError, match="repeated benchmark"):
        Registry(version=1, benchmarks=(tiny, tiny))
    with pytest.raises(ValidationError, match="requires unknown"):
        Registry(version=1, benchmarks=(tiny,))
    task = tiny.tasks[0].model_copy(update={"files": ("tiny/other.json",)})
    broken = tiny.model_copy(update={"tasks": (task,)})
    with pytest.raises(ValidationError, match="unpinned"):
        Registry(version=1, benchmarks=(registry.benchmark("coco-2014"), broken))


def test_write_docs_only_when_stale(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    page = tmp_path / "docs" / "benchmarks.md"
    assert write_docs(registry, page)
    assert not write_docs(registry, page)
    assert "pope-tiny" in page.read_text(encoding="utf-8")
