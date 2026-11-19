import json
from pathlib import Path

import pandas as pd
import pytest

from swce.diagnostics.coco import Cooccurrence
from swce.diagnostics.items import (
    change_readings,
    first_token_readings,
    load_answers,
    object_readings,
    relabel_readings,
    subtask_readings,
)
from swce.tasks.schema import Record


def write(root: Path, method: str, task: str, records: list[Record]) -> None:
    folder = root / "per_item" / "toy" / f"{method}-0123"
    folder.mkdir(parents=True, exist_ok=True)
    setting = {"model": "toy", "method": method, "setting": "default", "fingerprint": "0123"}
    (folder / "setting.json").write_text(json.dumps(setting), encoding="utf-8")
    lines = "".join(record.model_dump_json() + "\n" for record in records)
    (folder / f"{task}-abcd.jsonl").write_text(lines, encoding="utf-8")


def record(identifier: str, label: str, text: str, yes: float, group: str = "a.jpg") -> Record:
    return Record(id=identifier, group=group, label=label, text=text, tokens=1, yes=yes)


@pytest.fixture
def answers(tmp_path: Path) -> pd.DataFrame:
    base = [
        record("1", "yes", "Yes.", 0.9),
        record("2", "no", "Yes, there is.", 0.8, "b.jpg"),
        record("3", "no", "No.", 0.1, "b.jpg"),
    ]
    steered = [
        record("1", "yes", "No.", 0.4),
        record("2", "no", "No.", 0.3, "b.jpg"),
        record("3", "no", "No, there is not.", 0.1, "b.jpg"),
    ]
    write(tmp_path, "unmodified", "pope-adversarial", base)
    write(tmp_path, "vti", "pope-adversarial", steered)
    relabelled = [
        record("1", "yes", "Yes.", 0.9),
        record("2", "yes", "Yes, there is.", 0.8, "b.jpg"),
    ]
    write(tmp_path, "unmodified", "repope-adversarial", relabelled)
    return load_answers(tmp_path)


def test_answers_are_parsed_as_their_scorers_parse_them(answers: pd.DataFrame) -> None:
    chosen = answers[(answers.method == "vti") & (answers.task == "pope-adversarial")]
    assert chosen["answer"].tolist() == ["no", "no", "no"]
    assert chosen["correct"].tolist() == [False, True, True]


def test_changes_count_fixes_and_breaks_by_expected_answer(answers: pd.DataFrame) -> None:
    changes = change_readings(answers).set_index("expected")
    assert changes.loc["yes", "broken"] == 1 and changes.loc["yes", "fixed"] == 0
    assert changes.loc["no", "fixed"] == 1 and changes.loc["no", "broken"] == 0


def test_relabel_splits_removed_and_relabelled_questions(answers: pd.DataFrame) -> None:
    (row,) = relabel_readings(answers).to_dict("records")
    assert row["removed"] == 1 and row["relabelled"] == 1
    assert row["errors"] == 1 and row["errors_relabelled"] == 1
    assert row["f1_repope"] == pytest.approx(1.0)


def test_first_token_separates_moved_answers_from_later_text(answers: pd.DataFrame) -> None:
    (row,) = first_token_readings(answers).to_dict("records")
    assert row["first_token_moved"] == pytest.approx(2 / 3)
    assert row["later_text_only"] == pytest.approx(1 / 3)


def test_objects_set_wrong_yeses_against_cooccurrence(answers: pd.DataFrame) -> None:
    questions = {"2": "Is there a fork in the image?", "3": "Is there a dog in the image?"}
    present = {"b.jpg": frozenset({"dining table"})}
    counts = Cooccurrence([frozenset({"fork", "dining table"}), frozenset({"dining table"})])
    table, correlation = object_readings(answers, questions, present, counts)
    rates = dict(zip(table["object"], table["wrong_yes"], strict=True))
    assert rates == {"fork": 1.0, "dog": 0.0}
    assert correlation["spearman_question"].iloc[0] == pytest.approx(1.0)


def test_subtasks_keep_the_object_subtasks_of_mme() -> None:
    board = pd.DataFrame(
        {
            "model": ["toy"] * 3,
            "method": ["vti"] * 3,
            "setting": ["pope"] * 3,
            "task": ["mme", "mme", "chair"],
            "metric": ["existence", "OCR", "chair_s"],
            "value": [190.0, 120.0, 0.3],
            "delta": [-5.0, 0.0, -0.1],
        }
    )
    assert subtask_readings(board)["subtask"].tolist() == ["existence"]
