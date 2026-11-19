"""The yes or no benchmarks read item by item, from the answers a run saved.

No model runs. The answers under `per_item` of a results folder are parsed as their scorers parse
them, and five readings are taken.

- `objects`: for every absent object of POPE adversarial, how often the unmodified model says yes
  to it, against how often it co-occurs with the objects of the image in COCO train2014.
- `changes`: the questions every method turns from wrong to right and from right to wrong,
  against the unmodified model, split by whether the object is present.
- `relabel`: how RePOPE moves the score of the unmodified model, split into the questions it
  removes and the questions it relabels.
- `subtasks`: the MME subtasks that ask about objects, read from the leaderboard.
- `first_token`: how often a method changes the probability of the first answer token, and how
  often it changes the text while keeping its first word, which a yes or no score never sees.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from swce.diagnostics.coco import Cooccurrence, object_of
from swce.evaluation.runner import SETTING_FILE, load_records
from swce.metrics.mme import mme_answer
from swce.metrics.yes_no import binary_scores, pope_answer

OBJECT_SUBTASKS = ("existence", "count", "position", "color")
"""Subtasks of MME that ask about objects."""
UNMODIFIED = "unmodified"
"""Name of the unmodified model among the methods."""


def load_answers(root: Path) -> pd.DataFrame:
    """Every answer saved under a results folder, parsed as yes or no where the task asks it.

    Returns:
        One row per model, method setting, task and item, with the `text`, the `yes` probability
        of the first token, the parsed `answer` and whether it is `correct`.
    """
    rows: list[dict[str, Any]] = []
    for setting in sorted(root.glob(f"per_item/*/*/{SETTING_FILE}")):
        described = json.loads(setting.read_text(encoding="utf-8"))
        for path in sorted(setting.parent.glob("*.jsonl")):
            task = path.stem.rpartition("-")[0]
            if not task.startswith(("pope", "repope", "mme")):
                continue
            for record in load_records(path):
                if task.startswith("mme"):
                    answer = mme_answer(record.text)
                else:
                    answer = "yes" if pope_answer(record.text) else "no"
                rows.append(
                    {
                        "model": described["model"],
                        "method": described["method"],
                        "setting": described["setting"],
                        "task": task,
                        "id": record.id,
                        "group": record.group,
                        "label": record.label,
                        "category": record.category,
                        "text": record.text,
                        "yes": record.yes,
                        "answer": answer,
                        "correct": answer == record.label,
                    }
                )
    return pd.DataFrame(rows)


def object_readings(
    answers: pd.DataFrame,
    questions: Mapping[str, str],
    present: Mapping[str, frozenset[str]],
    counts: Cooccurrence,
    task: str = "pope-adversarial",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wrong yeses of the unmodified model on every absent object, against co-occurrence.

    Args:
        answers: Rows of `load_answers`.
        questions: Text of every question of the task, by identifier.
        present: Objects of every image, by file name.
        counts: Co-occurrence of the categories in COCO train2014.
        task: Key of the task read.

    Returns:
        One row per model and absent object with its questions, its rate of wrong yeses and its
        mean co-occurrence, and one row per model with the Spearman correlation of the two.
    """
    chosen = answers[
        (answers["method"] == UNMODIFIED) & (answers["task"] == task) & (answers["label"] == "no")
    ]
    chosen = chosen.assign(
        object=[object_of(questions[identifier]) for identifier in chosen["id"]],
        cooccurrence=[
            counts.strongest(object_of(questions[identifier]), present.get(group, frozenset()))
            for identifier, group in zip(chosen["id"], chosen["group"], strict=True)
        ],
        wrong=chosen["answer"] == "yes",
    )
    table = chosen.groupby(["model", "object"], as_index=False).agg(
        questions=("id", "count"),
        wrong_yes=("wrong", "mean"),
        cooccurrence=("cooccurrence", "mean"),
    )
    correlation = []
    for model, frame in chosen.groupby("model"):
        rho = spearmanr(frame["cooccurrence"], frame["wrong"]).statistic
        by_object = table[table["model"] == model]
        rho_objects = spearmanr(by_object["cooccurrence"], by_object["wrong_yes"]).statistic
        correlation.append(
            {
                "model": model,
                "questions": len(frame),
                "spearman_question": float(np.nan_to_num(rho)),
                "spearman_object": float(np.nan_to_num(rho_objects)),
            }
        )
    return table.sort_values(["model", "wrong_yes"], ascending=[True, False]), pd.DataFrame(
        correlation
    )


def change_readings(answers: pd.DataFrame) -> pd.DataFrame:
    """Questions every method fixes and breaks against the unmodified model, by expected answer."""
    rows = []
    for (model, task), frame in answers[answers["task"].str.startswith(("pope", "repope"))].groupby(
        ["model", "task"]
    ):
        base = frame[frame["method"] == UNMODIFIED].set_index("id")
        if base.empty:
            continue
        for (method, setting), steered in frame[frame["method"] != UNMODIFIED].groupby(
            ["method", "setting"]
        ):
            joined = steered.merge(
                base[["correct"]], left_on="id", right_index=True, suffixes=("", "_base")
            )
            for label, part in joined.groupby("label"):
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "method": method,
                        "setting": setting,
                        "expected": label,
                        "questions": len(part),
                        "fixed": int((part["correct"] & ~part["correct_base"]).sum()),
                        "broken": int((~part["correct"] & part["correct_base"]).sum()),
                    }
                )
    return pd.DataFrame(rows)


def _f1(frame: pd.DataFrame, label: str) -> float:
    predictions = (frame["answer"] == "yes").to_numpy()
    labels = (frame[label] == "yes").to_numpy()
    return float(binary_scores(predictions, labels)["f1"])


def relabel_readings(answers: pd.DataFrame) -> pd.DataFrame:
    """How RePOPE moves the F1 of every method, split into removed and relabelled questions."""
    rows = []
    for (model, method, setting), frame in answers.groupby(["model", "method", "setting"]):
        pope = frame[frame["task"] == "pope-adversarial"].set_index("id")
        repope = frame[frame["task"] == "repope-adversarial"].set_index("id")
        if pope.empty or repope.empty:
            continue
        kept = pope.loc[pope.index.intersection(repope.index)].assign(
            relabel=repope["label"].reindex(pope.index.intersection(repope.index))
        )
        changed = kept["label"] != kept["relabel"]
        wrong = kept["answer"] != kept["label"]
        rows.append(
            {
                "model": model,
                "method": method,
                "setting": setting,
                "pope_questions": len(pope),
                "removed": len(pope) - len(kept),
                "relabelled": int(changed.sum()),
                "errors": int((pope["answer"] != pope["label"]).sum()),
                "errors_relabelled": int((wrong & changed).sum()),
                "f1_pope": _f1(pope, "label"),
                "f1_kept_pope_labels": _f1(kept, "label"),
                "f1_repope": _f1(kept, "relabel"),
            }
        )
    return pd.DataFrame(rows)


def subtask_readings(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Scores of the object subtasks of MME, with their change against the unmodified model."""
    chosen = leaderboard[
        (leaderboard["task"] == "mme") & leaderboard["metric"].isin(OBJECT_SUBTASKS)
    ]
    return chosen[["model", "method", "setting", "metric", "value", "delta"]].rename(
        columns={"metric": "subtask"}
    )


def _first_word(texts: pd.Series) -> pd.Series:
    """First word of every text, lower case and without punctuation."""
    return texts.str.split().str[0].fillna("").str.strip(".,!?;:").str.lower()


def first_token_readings(answers: pd.DataFrame, tolerance: float = 1e-3) -> pd.DataFrame:
    """How often a method changes the first answer token, and how often only later text."""
    rows = []
    for (model, task), frame in answers.groupby(["model", "task"]):
        base = frame[frame["method"] == UNMODIFIED].set_index("id")
        if base.empty:
            continue
        for (method, setting), steered in frame[frame["method"] != UNMODIFIED].groupby(
            ["method", "setting"]
        ):
            joined = steered.merge(
                base[["yes", "text", "answer"]],
                left_on="id",
                right_index=True,
                suffixes=("", "_base"),
            )
            moved = (joined["yes"] - joined["yes_base"]).abs() > tolerance
            first_word = _first_word(joined["text"]) == _first_word(joined["text_base"])
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "method": method,
                    "setting": setting,
                    "questions": len(joined),
                    "first_token_moved": float(moved.mean()),
                    "answer_changed": float((joined["answer"] != joined["answer_base"]).mean()),
                    "text_changed": float((joined["text"] != joined["text_base"]).mean()),
                    "later_text_only": float(
                        ((joined["text"] != joined["text_base"]) & first_word & ~moved).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)
