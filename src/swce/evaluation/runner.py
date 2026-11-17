"""Evaluation of steering methods on a subject, in two independent stages.

Generation runs the model. Every method answers the items of every task through the same subject,
and its answers are written verbatim, one file per model, method setting and task. Scoring runs no
model. It reads the answers found under a results folder, scores them with the scorer of their
task, and gives each metric a bootstrap interval over images and, when the unmodified model
answered the same items, its paired difference from it.

The two stages share nothing but the files, so answers generated on several machines are scored
together once their folders are copied into one results folder.
"""

import fnmatch
import json
import platform
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from swce.baselines.reference import Unmodified
from swce.evaluation.analysis import MARGIN, trade_offs
from swce.evaluation.leaderboard import COLUMNS, Leaderboard
from swce.evaluation.scorers import SCORERS, Scorer
from swce.evaluation.shortcuts import chair_shortcut, pope_shortcut
from swce.logging import get_logger
from swce.metrics.bootstrap import (
    RESAMPLES,
    cluster_resamples,
    flips,
    mcnemar,
    percentile_interval,
)
from swce.metrics.chair import ground_truth
from swce.metrics.yes_no import auroc
from swce.models.subject import Generation, Subject
from swce.registry import Registry, load_registry
from swce.steering.steerer import InapplicableError, Steerer
from swce.tasks.schema import Item, Record
from swce.tasks.task import Task

ANALYSIS_FILE = "analysis.csv"
"""File of the gain of every method next to its capability change."""
SHORTCUTS_FILE = "shortcuts.csv"
"""File comparing every method with the shortcuts of the unmodified model."""
LENS_FILE = "lens.csv"
"""File of the summary of the logit lens readings recorded with the answers."""
SETTING_FILE = "setting.json"
"""File describing the method setting whose answers a folder holds."""

log = get_logger(__name__)


class EvaluationConfig(BaseModel):
    """Settings of an evaluation.

    Attributes:
        tasks: Patterns of the task keys to run, in the syntax of `fnmatch`. Empty runs all.
        limit: Number of leading images per task, for smoke tests, or None for whole tasks. A
            limited run writes under `smoke` and never to the published leaderboard.
        bootstrap: Number of bootstrap resamples, zero to skip the intervals.
        seed: Seed of the resampling.
        results: Directory of the leaderboard and of the per item records.
        data: Data root, or None for the default one.
        resume: Whether to reuse per item records already written for the same model, method
            setting and task definition instead of generating again. An interrupted task
            continues after its last answered item.
        lens: Whether to record the logit lens of every prompt with its answer.
    """

    model_config = ConfigDict(frozen=True)

    tasks: tuple[str, ...] = ()
    limit: int | None = None
    bootstrap: int = RESAMPLES
    seed: int = 0
    results: Path = Path("results")
    data: Path | None = None
    resume: bool = False
    lens: bool = False

    @property
    def root(self) -> Path:
        """Directory the run writes to, apart for limited runs."""
        return self.results / "smoke" if self.limit is not None else self.results


def select_tasks(patterns: Sequence[str], registry: Registry | None = None) -> list[str]:
    """Task keys matching any of the patterns, in registry order.

    Args:
        patterns: Patterns in the syntax of `fnmatch`. Empty selects every task.
        registry: Registry to select from, or None for the committed one.

    Returns:
        The matching keys.

    Raises:
        ValueError: If a pattern matches no task.
    """
    keys = (registry or load_registry()).task_keys()
    for pattern in patterns:
        if not fnmatch.filter(keys, pattern):
            raise ValueError(f"no task matches {pattern!r}")
    return [key for key in keys if not patterns or any(fnmatch.fnmatch(key, p) for p in patterns)]


def setting_folder(root: Path, subject: Subject, steerer: Steerer) -> Path:
    """Folder holding the answers of one model under one method setting.

    Args:
        root: Directory the run writes to.
        subject: Model evaluated.
        steerer: Method setting evaluated.

    Returns:
        The path of the folder.
    """
    return root / "per_item" / subject.name / f"{steerer.name}-{steerer.fingerprint}"


def records_path(root: Path, subject: Subject, steerer: Steerer, task: Task) -> Path:
    """File holding the per item records of one model, method setting and task definition.

    Args:
        root: Directory the run writes to.
        subject: Model evaluated.
        steerer: Method setting evaluated.
        task: Task evaluated.

    Returns:
        The path of the JSON lines file.
    """
    return setting_folder(root, subject, steerer) / f"{task.key}-{task.fingerprint}.jsonl"


def load_records(path: Path) -> list[Record]:
    """Records of a file of answers.

    Args:
        path: File of per item records.

    Returns:
        The records, in file order.
    """
    with path.open(encoding="utf-8") as stream:
        return [Record.model_validate_json(line) for line in stream if line.strip()]


def read_records(path: Path, items: Sequence[Item]) -> list[Record] | None:
    """Records of a file, when they answer exactly the given items.

    Args:
        path: File of per item records, which may be missing.
        items: Items the records must answer, in order.

    Returns:
        The records, or None when the file is missing or answers other items.
    """
    if not path.exists():
        return None
    records = load_records(path)
    if [record.id for record in records] != [item.id for item in items]:
        return None
    return records


def leading_records(path: Path, items: Sequence[Item]) -> list[Record]:
    """Records of an interrupted file that answer the leading items, in order.

    Args:
        path: File of per item records left by an interrupted run, which may be missing.
        items: Items of the task, in order.

    Returns:
        The longest beginning of the file that answers the first items. A line cut short by the
        interruption and everything after it are dropped.
    """
    records: list[Record] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line, item in zip(stream, items, strict=False):
            try:
                record = Record.model_validate_json(line)
            except ValueError:
                break
            if record.id != item.id:
                break
            records.append(record)
    return records


def cache_key(item: Item, task: Task) -> tuple[str, str | None, str, int]:
    """Key under which the generation of an item is shared between tasks.

    Args:
        item: Item of the task.
        task: Task the item belongs to.

    Returns:
        The image, the prompt and the token budget, which decide the generation.
    """
    return (item.image, item.archive, item.prompt, task.card.max_new_tokens)


def answer(
    subject: Subject,
    task: Task,
    items: Sequence[Item],
    cache: dict[tuple[str, str | None, str, int], Generation],
    sink: TextIO | None = None,
) -> list[Record]:
    """Answers of a subject, steered or not, to the items of a task.

    Args:
        subject: Model to query, already inside the context of its steering method.
        task: Task the items belong to.
        items: Items to answer.
        cache: Generations of the same method setting by image, prompt and token budget, shared
            across tasks so that tasks asking the same questions cost one generation.
        sink: Stream every record is written to as soon as it exists, one JSON object per line,
            so that an interrupted run keeps what it answered.

    Returns:
        One record per item, in order.
    """
    records = []
    for item in tqdm(items, desc=task.key, leave=False):
        key = cache_key(item, task)
        if key not in cache:
            cache[key] = subject.generate(task.image(item), item.prompt, key[3])
        generation = cache[key]
        record = Record(
            id=item.id,
            group=item.group,
            label=item.label,
            category=item.category,
            text=generation.text,
            tokens=generation.tokens,
            yes=generation.yes,
            no=generation.no,
            lens_yes=generation.lens_yes,
            lens_no=generation.lens_no,
        )
        if sink is not None:
            sink.write(record.model_dump_json() + "\n")
            sink.flush()
        records.append(record)
    return records


def environment(subject: Subject) -> dict[str, object]:
    """Software, hardware and subject settings a run depends on.

    Args:
        subject: Model of the run.

    Returns:
        The versions of Python, PyTorch and this package, the platform, the accelerator, and the
        description and settings of the subject.
    """
    accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "swce": metadata.version("steering-with-clear-eyes"),
        "accelerator": accelerator,
        "subject": subject.description,
        **subject.settings(),
    }


def applicable(steerers: Sequence[Steerer], subject: Subject) -> list[Steerer]:
    """Settings an evaluation runs on a subject.

    A setting published for other subjects is left out. A fallback setting, which holds the
    defaults of the official code, is kept only when the method has no published setting for the
    subject.

    Args:
        steerers: Method settings to choose from.
        subject: Model to evaluate.

    Returns:
        The settings to run, the unmodified model first.
    """

    def fits(steerer: Steerer) -> bool:
        return not steerer.published_for or subject.name in steerer.published_for

    published = {steerer.name for steerer in steerers if fits(steerer) and not steerer.fallback}
    kept = [
        steerer
        for steerer in steerers
        if fits(steerer) and not (steerer.fallback and steerer.name in published)
    ]
    for steerer in steerers:
        if steerer not in kept:
            log.info(
                "left out",
                model=subject.name,
                method=steerer.name,
                setting=steerer.setting,
                reason="fallback" if fits(steerer) else "published for other models",
            )
    return sorted(kept, key=lambda steerer: not isinstance(steerer, Unmodified))


def prepare(subject: Subject, steerers: Sequence[Steerer]) -> list[Steerer]:
    """Run the steps the recipes of the methods take before inference.

    Args:
        subject: Model the methods will steer.
        steerers: Method settings to choose from.

    Returns:
        The applicable settings whose recipe could be followed on the subject. The others are
        logged.
    """
    ready = []
    for steerer in applicable(steerers, subject):
        try:
            steerer.prepare(subject)
        except InapplicableError as error:
            log.warning(
                "inapplicable",
                model=subject.name,
                method=steerer.name,
                setting=steerer.setting,
                reason=str(error),
            )
        else:
            ready.append(steerer)
    return ready


def generate_setting(
    subject: Subject, steerer: Steerer, tasks: Sequence[Task], config: EvaluationConfig
) -> Path:
    """Write the answers of a subject under one prepared method setting.

    Args:
        subject: Model to steer and query.
        steerer: Method setting, whose recipe already ran.
        tasks: Tasks to answer. Those the setting is not meant for are skipped.
        config: Settings of the evaluation.

    Returns:
        The folder holding the answers and the description of the setting.

    Raises:
        InapplicableError: If the method cannot be instantiated on the subject.
    """
    folder = setting_folder(config.root, subject, steerer)
    cache: dict[tuple[str, str | None, str, int], Generation] = {}
    for task in tasks:
        if steerer.tasks and not any(fnmatch.fnmatch(task.key, p) for p in steerer.tasks):
            continue
        items = task.items(config.limit)
        path = records_path(config.root, subject, steerer, task)
        partial = path.with_name(path.name + ".part")
        done: list[Record] = []
        if config.resume:
            done = read_records(path, items) or leading_records(partial, items)
        for item, record in zip(items, done, strict=False):
            cache[cache_key(item, task)] = Generation(
                **record.model_dump(include=set(Generation.model_fields))
            )
        if path.exists() and len(done) == len(items):
            continue
        folder.mkdir(parents=True, exist_ok=True)
        # Answers are written one by one to a partial file, which takes the name of the complete
        # one only when every item is answered. Scoring never reads a partial file.
        with partial.open("w", encoding="utf-8", newline="\n") as sink:
            sink.writelines(record.model_dump_json() + "\n" for record in done)
            sink.flush()
            with steerer.attach(subject):
                answer(subject, task, items[len(done) :], cache, sink)
        partial.replace(path)
    if folder.exists():
        declared = steerer.fitted_on
        description = {
            "model": subject.name,
            "method": steerer.name,
            "setting": steerer.setting,
            "fingerprint": steerer.fingerprint,
            "fitted_on": "undeclared" if declared is None else ", ".join(declared) or "nothing",
            "fields": steerer.model_dump(mode="json"),
            "environment": environment(subject),
        }
        text = json.dumps(description, indent=2) + "\n"
        (folder / SETTING_FILE).write_text(text, encoding="utf-8", newline="\n")
    return folder


def generate(
    subject: Subject,
    steerers: Sequence[Steerer],
    config: EvaluationConfig | None = None,
    registry: Registry | None = None,
) -> list[Path]:
    """Run the recipes of the methods and write their answers, without scoring them.

    A published setting runs on the subjects it is published for, and the defaults of the
    official code on the others. A method that cannot be instantiated on the subject is logged
    and left out, so its cells stay empty.

    Args:
        subject: Model to steer and query.
        steerers: Method settings to evaluate.
        config: Settings of the evaluation, or None for the defaults.
        registry: Registry defining the tasks, or None for the committed one.

    Returns:
        The folders of answers written, one per method setting.
    """
    config = config or EvaluationConfig()
    registry = registry or load_registry()
    tasks = [Task(key, config.data, registry) for key in select_tasks(config.tasks, registry)]
    folders = []
    recorded = subject.lens
    subject.lens = config.lens
    try:
        for steerer in prepare(subject, steerers):
            try:
                folders.append(generate_setting(subject, steerer, tasks, config))
            except InapplicableError as error:
                log.warning(
                    "inapplicable",
                    model=subject.name,
                    method=steerer.name,
                    setting=steerer.setting,
                    reason=str(error),
                )
    finally:
        subject.lens = recorded
    return [folder for folder in folders if folder.exists()]


def score(
    scorer: Scorer,
    parsed: Sequence[Any],
    reference: Sequence[Any] | None,
    groups: Sequence[str],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Metrics of one method setting on one task, with intervals and paired differences.

    Args:
        scorer: Scorer of the task.
        parsed: Parsed records of the method setting.
        reference: Parsed records of the unmodified model on the same items, or None.
        groups: Image of each item.
        resamples: Number of bootstrap resamples, zero to skip the intervals.
        seed: Seed of the resampling.

    Returns:
        One row per metric with the columns `metric`, `value`, `low`, `high`, `delta`,
        `delta_low`, `delta_high`, `p_mcnemar`, `fixed` and `broken`. The last three compare the
        correctness of single items and are equal on every row.
    """
    values = scorer.aggregate(parsed)
    base = scorer.aggregate(reference) if reference is not None else None
    draws = cluster_resamples(groups, resamples, seed) if resamples else []
    sampled = [scorer.aggregate([parsed[index] for index in draw]) for draw in draws]
    paired = (
        [scorer.aggregate([reference[index] for index in draw]) for draw in draws]
        if reference is not None
        else None
    )
    nan = float("nan")
    comparison: dict[str, float] = {"p_mcnemar": nan, "fixed": nan, "broken": nan}
    right = scorer.correct(parsed)
    before = scorer.correct(reference) if reference is not None else None
    if right is not None and before is not None:
        first, second = np.array(before, dtype=np.bool_), np.array(right, dtype=np.bool_)
        comparison = {"p_mcnemar": mcnemar(first, second), **flips(first, second)}
    rows = []
    for metric, value in values.items():
        low, high = percentile_interval([draw.get(metric, nan) for draw in sampled])
        delta = delta_low = delta_high = nan
        if base is not None and paired is not None:
            delta = value - base.get(metric, nan)
            delta_low, delta_high = percentile_interval(
                [
                    a.get(metric, nan) - b.get(metric, nan)
                    for a, b in zip(sampled, paired, strict=True)
                ]
            )
        rows.append(
            {
                "metric": metric,
                "value": value,
                "low": low,
                "high": high,
                "delta": delta,
                "delta_low": delta_low,
                "delta_high": delta_high,
                **comparison,
            }
        )
    return pd.DataFrame(rows)


def score_folder(
    folder: Path, tasks: dict[str, Task], *, bootstrap: int, seed: int
) -> list[pd.DataFrame]:
    """Leaderboard rows of the answers of one method setting.

    Args:
        folder: Folder of answers, with the description of its setting.
        tasks: Tasks of the registry in use, by key.
        bootstrap: Number of bootstrap resamples, zero to skip the intervals.
        seed: Seed of the resampling.

    Returns:
        The rows of every file of answers whose task definition is the present one. Files of
        other definitions are left out.
    """
    described = json.loads((folder / SETTING_FILE).read_text(encoding="utf-8"))
    frames = []
    for path in sorted(folder.glob("*.jsonl")):
        key, _, mark = path.stem.rpartition("-")
        if key not in tasks or tasks[key].fingerprint != mark:
            continue
        task = tasks[key]
        scorer = SCORERS[task.card.scorer]
        records = load_records(path)
        parsed = scorer.parse(records, task)
        reference = None
        if described["method"] != Unmodified.name:
            identifiers = [record.id for record in records]
            for candidate in sorted(folder.parent.glob(f"{Unmodified.name}-*")):
                other = candidate / path.name
                if other.exists():
                    found = load_records(other)
                    if [record.id for record in found] == identifiers:
                        reference = scorer.parse(found, task)
        groups = [record.group for record in records]
        frame = score(scorer, parsed, reference, groups, bootstrap, seed)
        frame = frame.assign(
            model=described["model"],
            method=described["method"],
            setting=described["setting"],
            fingerprint=described["fingerprint"],
            fitted_on=described["fitted_on"],
            task=key,
            task_fingerprint=mark,
            primary=frame.metric == task.card.primary,
            items=len(records),
        )
        primary = frame[frame.primary]
        log.info(
            "scored",
            model=described["model"],
            method=described["method"],
            setting=described["setting"],
            task=key,
            metric=task.card.primary,
            value=None if primary.empty else float(primary.value.iloc[0]),
        )
        frames.append(frame[COLUMNS])
    return frames


def lens_rows(folder: Path, tasks: dict[str, Task]) -> list[dict[str, Any]]:
    """Summary of the logit lens readings recorded with the answers of one method setting.

    Args:
        folder: Folder of answers, with the description of its setting.
        tasks: Tasks of the registry in use, by key.

    Returns:
        One row per task and decoder block with `auroc`, how well the log ratio of yes to no
        read after the block separates the items whose answer is yes from the others, and the
        mean log ratio of each kind of item as `margin_yes` and `margin_no`. Tasks without yes
        or no labels and answers recorded without the lens give no row.
    """
    described = json.loads((folder / SETTING_FILE).read_text(encoding="utf-8"))
    rows = []
    for path in sorted(folder.glob("*.jsonl")):
        key, _, mark = path.stem.rpartition("-")
        if key not in tasks or tasks[key].fingerprint != mark:
            continue
        records = [
            record
            for record in load_records(path)
            if record.lens_yes is not None and record.label in ("yes", "no")
        ]
        if not records:
            continue
        positive = np.array([record.label == "yes" for record in records], dtype=np.bool_)
        yes = np.array([record.lens_yes for record in records], dtype=np.float64)
        no = np.array([record.lens_no for record in records], dtype=np.float64)
        margins = np.log(np.maximum(yes, 1e-12)) - np.log(np.maximum(no, 1e-12))
        for layer in range(margins.shape[1]):
            column = margins[:, layer]
            rows.append(
                {
                    "model": described["model"],
                    "method": described["method"],
                    "setting": described["setting"],
                    "fingerprint": described["fingerprint"],
                    "task": key,
                    "layer": layer,
                    "auroc": auroc(column, positive),
                    "margin_yes": float(column[positive].mean()) if positive.any() else np.nan,
                    "margin_no": float(column[~positive].mean()) if (~positive).any() else np.nan,
                    "items": len(records),
                }
            )
    return rows


def shortcut_rows(folder: Path, tasks: dict[str, Task]) -> list[dict[str, Any]]:
    """Comparison of one method setting with the shortcuts of the unmodified model.

    Args:
        folder: Folder of answers of the method, with the description of its setting.
        tasks: Tasks of the registry in use, by key.

    Returns:
        The rows of `pope_shortcut` and `chair_shortcut` for every task the unmodified model of
        the same model folder answered on the same items, with the model, the method and the
        task. Empty for the unmodified model itself.
    """
    described = json.loads((folder / SETTING_FILE).read_text(encoding="utf-8"))
    if described["method"] == Unmodified.name:
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.jsonl")):
        key, _, mark = path.stem.rpartition("-")
        if key not in tasks or tasks[key].fingerprint != mark:
            continue
        records = load_records(path)
        base = None
        for candidate in sorted(folder.parent.glob(f"{Unmodified.name}-*")):
            other = candidate / path.name
            if other.exists():
                found = load_records(other)
                if [record.id for record in found] == [record.id for record in records]:
                    base = found
        if base is None:
            continue
        scorer = tasks[key].card.scorer
        compared: list[dict[str, Any]] = []
        if scorer == "pope":
            compared = pope_shortcut(records, base)
        elif scorer == "chair":
            present = ground_truth(tasks[key].files()[1], {int(record.id) for record in base})
            compared = chair_shortcut(records, base, present)
        rows += [
            {
                "model": described["model"],
                "method": described["method"],
                "setting": described["setting"],
                "fingerprint": described["fingerprint"],
                "task": key,
                **row,
            }
            for row in compared
        ]
    return rows


def merge_rows(target: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    """Add rows to a CSV file, replacing earlier rows with the same keys.

    Args:
        target: File to write.
        rows: Rows to add. Nothing is written when there are none.
        keys: Columns that identify a row.
    """
    if not rows:
        return
    found = pd.DataFrame(rows)
    if target.exists():
        found = pd.concat([pd.read_csv(target, dtype={"fingerprint": str}), found])
    found = found.drop_duplicates(keys, keep="last").sort_values(keys, kind="stable")
    found.to_csv(target, index=False, lineterminator="\n")


def score_results(
    root: Path,
    registry: Registry | None = None,
    *,
    data: Path | None = None,
    bootstrap: int = RESAMPLES,
    seed: int = 0,
    margin: float = MARGIN,
    folders: Sequence[Path] | None = None,
) -> pd.DataFrame:
    """Score answers found under a results folder and publish them on its leaderboard.

    No model runs. Every method is compared with the answers of the unmodified model of the same
    model folder, wherever those were generated.

    Args:
        root: Results folder holding `per_item`.
        registry: Registry defining the tasks, or None for the committed one.
        data: Data root holding the files the scorers read, or None for the default one.
        bootstrap: Number of bootstrap resamples, zero to skip the intervals.
        seed: Seed of the resampling.
        margin: Non inferiority margin of the capability change, as a fraction of the score
            of the unmodified model.
        folders: Folders of answers to score, or None for every one under the results folder.

    Returns:
        The rows added to the leaderboard. Three more files are written next to it. `lens.csv`
        summarizes the logit lens readings recorded with the answers, `shortcuts.csv` compares
        every method with what the unmodified model reaches by answering yes as often or by
        describing as little, and `analysis.csv` sets the gain of every method next to its
        capability change.
    """
    registry = registry or load_registry()
    tasks = {key: Task(key, data, registry) for key in registry.task_keys()}
    if folders is None:
        folders = sorted(path.parent for path in root.glob(f"per_item/*/*/{SETTING_FILE}"))
    frames: list[pd.DataFrame] = []
    lens: list[dict[str, Any]] = []
    shortcuts: list[dict[str, Any]] = []
    for folder in folders:
        frames += score_folder(folder, tasks, bootstrap=bootstrap, seed=seed)
        lens += lens_rows(folder, tasks)
        shortcuts += shortcut_rows(folder, tasks)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    Leaderboard(root / "leaderboard.csv", registry).add(rows)
    setting = ["model", "method", "fingerprint", "task"]
    merge_rows(root / LENS_FILE, lens, [*setting, "layer"])
    merge_rows(root / SHORTCUTS_FILE, shortcuts, [*setting, "metric"])
    board = Leaderboard(root / "leaderboard.csv", registry)
    table = trade_offs(board.current(), registry, margin)
    if not table.empty:
        table.to_csv(root / ANALYSIS_FILE, index=False, lineterminator="\n")
    return rows


def evaluate(
    subject: Subject,
    steerers: Sequence[Steerer],
    config: EvaluationConfig | None = None,
    registry: Registry | None = None,
) -> pd.DataFrame:
    """Generate the answers of steering methods on a subject, then score and publish them.

    The unmodified model is evaluated first whenever it is among the methods, so that the others
    can be compared with it. A method is otherwise compared with the answers of the unmodified
    model left in the same results folder by an earlier run.

    Args:
        subject: Model to steer and query.
        steerers: Method settings to evaluate.
        config: Settings of the evaluation, or None for the defaults.
        registry: Registry defining the tasks, or None for the committed one.

    Returns:
        The rows added to the leaderboard.
    """
    config = config or EvaluationConfig()
    folders = generate(subject, steerers, config, registry)
    return score_results(
        config.root,
        registry,
        data=config.data,
        bootstrap=config.bootstrap,
        seed=config.seed,
        folders=folders,
    )


def dry_run(config: EvaluationConfig, registry: Registry | None = None) -> list[str]:
    """Lines describing what an evaluation would run, without downloading or generating.

    Args:
        config: Settings of the evaluation.
        registry: Registry defining the tasks, or None for the committed one.

    Returns:
        One line per selected task with its kind, token budget and files.
    """
    from swce.tasks.task import describe

    registry = registry or load_registry()
    return [describe(registry.task(key)[1]) for key in select_tasks(config.tasks, registry)]


def dumps(rows: pd.DataFrame) -> str:
    """Primary metrics of an evaluation as JSON, for the end of a command.

    Args:
        rows: Rows returned by `evaluate` or `score_results`.

    Returns:
        A JSON list of the primary rows with their model, method, task, metric and value.
    """
    primary = rows[rows.primary.astype(bool)] if not rows.empty else rows
    fields = ["model", "method", "setting", "task", "metric", "value", "low", "high", "delta"]
    return json.dumps(json.loads(primary[fields].to_json(orient="records")), indent=2)
