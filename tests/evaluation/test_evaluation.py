import json
import shutil
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest
from support import ToySubject, tiny_registry

from swce.baselines import RandomDirection, Unmodified
from swce.cli import main as cli_main
from swce.evaluation.__main__ import load_factory, load_steerers, main
from swce.evaluation.leaderboard import COLUMNS, Leaderboard
from swce.evaluation.runner import (
    EvaluationConfig,
    applicable,
    dry_run,
    evaluate,
    generate,
    prepare,
    score_results,
    select_tasks,
)
from swce.models.subject import Generation, Subject
from swce.steering import InapplicableError, Steerer


def config(tmp_path: Path, **settings: object) -> EvaluationConfig:
    values = {"results": tmp_path / "results", "data": tmp_path / "data", "bootstrap": 50}
    return EvaluationConfig.model_validate({**values, **settings})


def test_evaluate_publishes_metrics_intervals_and_paired_differences(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    control = RandomDirection(norms={1: 5.0}, width=subject.width)
    rows = evaluate(subject, [control, Unmodified()], config(tmp_path), registry)
    assert list(rows.columns) == COLUMNS
    assert set(rows.method) == {"unmodified", "random-direction"}
    assert rows.groupby("method").primary.sum().tolist() == [1, 1]
    base = rows[(rows.method == "unmodified") & (rows.metric == "accuracy")].iloc[0]
    steered = rows[(rows.method == "random-direction") & (rows.metric == "accuracy")].iloc[0]
    assert base["items"] == 6
    assert base.low <= base.value <= base.high
    assert pd.isna(base.delta)
    assert steered.delta == pytest.approx(steered.value - base.value)
    assert steered.fixed + steered.broken >= 0
    assert steered.fitted_on == "nothing"
    board = tmp_path / "results" / "leaderboard.csv"
    assert len(Leaderboard(board, registry).rows()) == len(rows)
    summary = board.with_suffix(".md").read_text(encoding="utf-8")
    assert "## toy" in summary
    assert "random-direction" in summary
    records = list((tmp_path / "results" / "per_item" / "toy").glob("*/*.jsonl"))
    assert len(records) == 2


def test_limited_runs_stay_out_of_the_published_leaderboard(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    rows = evaluate(ToySubject(), [Unmodified()], config(tmp_path, limit=1), registry)
    assert set(rows["items"]) == {2}
    assert (tmp_path / "results" / "smoke" / "leaderboard.csv").exists()
    assert not (tmp_path / "results" / "leaderboard.csv").exists()


def test_resume_reuses_records_and_reruns_replace_rows(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    first = evaluate(subject, [Unmodified()], config(tmp_path), registry)
    calls = subject.calls
    evaluate(subject, [Unmodified()], config(tmp_path, resume=True), registry)
    assert subject.calls == calls
    evaluate(subject, [Unmodified()], config(tmp_path), registry)
    assert subject.calls == 2 * calls
    board = Leaderboard(tmp_path / "results" / "leaderboard.csv", registry)
    assert len(board.rows()) == len(first)


def test_rows_of_a_changed_task_definition_leave_the_summary(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    evaluate(ToySubject(), [Unmodified()], config(tmp_path), registry)
    tiny = registry.benchmark("tiny")
    card = tiny.tasks[0].model_copy(update={"max_new_tokens": 16})
    changed = registry.model_copy(
        update={"benchmarks": (registry.benchmarks[0], tiny.model_copy(update={"tasks": (card,)}))}
    )
    board = Leaderboard(tmp_path / "results" / "leaderboard.csv", changed)
    assert not board.rows().empty
    assert board.current().empty
    assert "No published result yet." in board.summary()


def test_settings_published_for_another_model_are_skipped(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    elsewhere = RandomDirection(norms={1: 5.0}, width=subject.width, published_for=("other",))
    here = RandomDirection(norms={1: 5.0}, width=subject.width, published_for=("toy",))
    rows = evaluate(subject, [elsewhere], config(tmp_path), registry)
    assert rows.empty
    rows = evaluate(subject, [here], config(tmp_path), registry)
    assert set(rows.method) == {"random-direction"}


class Probe(Steerer):
    """Method that counts its preparations and may refuse the subject."""

    name: ClassVar[str] = "probe"
    description: ClassVar[str] = "Stand in."
    prepared: ClassVar[list[str]] = []

    refuses: bool = False
    fitted_on: tuple[str, ...] | None = ()

    def prepare(self, subject: Subject) -> None:
        self.prepared.append(self.setting)

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        if self.refuses:
            raise InapplicableError("the subject lacks what the method needs")
        return nullcontext()


def test_code_defaults_run_only_where_nothing_is_published(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    default = Probe(setting="code-default", fallback=True)
    elsewhere = Probe(setting="paper", published_for=("other",))
    here = Probe(setting="paper", published_for=("toy",))
    assert applicable([elsewhere, default], subject) == [default]
    assert applicable([here, default], subject) == [here]
    assert applicable([default, Unmodified(), here], subject) == [Unmodified(), here]
    rows = evaluate(subject, [elsewhere, default], config(tmp_path), registry)
    assert set(rows.setting) == {"code-default"}


def test_recipes_run_before_steering_and_inapplicable_methods_leave_no_row(
    tmp_path: Path,
) -> None:
    registry = tiny_registry(tmp_path / "host")
    Probe.prepared.clear()
    steerers = [Probe(setting="works"), Probe(setting="refuses", refuses=True), Unmodified()]
    rows = evaluate(ToySubject(), steerers, config(tmp_path), registry)
    assert Probe.prepared == ["works", "refuses"]
    assert set(rows.setting) == {"default", "works"}


def test_answers_generated_apart_are_scored_together(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    control = RandomDirection(norms={1: 5.0}, width=subject.width)
    first = config(tmp_path).model_copy(update={"results": tmp_path / "first"})
    second = config(tmp_path).model_copy(update={"results": tmp_path / "second"})
    (baseline,) = generate(subject, [Unmodified()], first, registry)
    (steered,) = generate(subject, [control], second, registry)
    assert not (tmp_path / "first" / "leaderboard.csv").exists()
    assert (
        json.loads((steered / "setting.json").read_text(encoding="utf-8"))["setting"] == "default"
    )
    shutil.copytree(
        tmp_path / "second" / "per_item", tmp_path / "first" / "per_item", dirs_exist_ok=True
    )
    rows = score_results(tmp_path / "first", registry, data=tmp_path / "data", bootstrap=50)
    assert set(rows.method) == {"unmodified", "random-direction"}
    accuracy = rows[(rows.method == "random-direction") & (rows.metric == "accuracy")].iloc[0]
    assert not pd.isna(accuracy.delta)
    assert baseline.parent == tmp_path / "first" / "per_item" / "toy"
    alone = score_results(tmp_path / "second", registry, data=tmp_path / "data", bootstrap=0)
    assert pd.isna(alone[alone.metric == "accuracy"].iloc[0].delta)


def test_a_setting_answers_only_the_tasks_it_is_meant_for(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    assert evaluate(subject, [Probe(tasks=("chair",))], config(tmp_path), registry).empty
    rows = evaluate(subject, [Probe(tasks=("pope-*", "mme"))], config(tmp_path), registry)
    assert set(rows.task) == {"pope-tiny"}


def test_prepare_reports_the_settings_whose_recipe_ran(tmp_path: Path) -> None:
    class Unfit(Probe):
        name: ClassVar[str] = "unfit"

        def prepare(self, subject: Subject) -> None:
            raise InapplicableError("no recipe for this subject")

    ready = prepare(ToySubject(), [Probe(), Unfit(), Probe(setting="other", published_for=("x",))])
    assert [(steerer.name, steerer.setting) for steerer in ready] == [("probe", "default")]


def test_the_logit_lens_is_recorded_with_the_answers_and_summarized(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    control = RandomDirection(norms={1: 5.0}, width=subject.width)
    evaluate(subject, [Unmodified()], config(tmp_path), registry)
    assert not (tmp_path / "results" / "lens.csv").exists()
    settings = config(tmp_path).model_copy(update={"lens": True})
    evaluate(subject, [Unmodified(), control], settings, registry)
    assert subject.lens is False
    (path,) = (tmp_path / "results" / "per_item" / "toy").glob("random-direction-*/*.jsonl")
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert len(record["lens_yes"]) == len(record["lens_no"]) == 3
    lens = pd.read_csv(tmp_path / "results" / "lens.csv")
    assert set(lens.method) == {"unmodified", "random-direction"}
    assert lens.groupby("method").layer.apply(list).tolist() == [[0, 1, 2], [0, 1, 2]]
    assert lens.auroc.between(0, 1).all()
    assert set(lens["items"]) == {6}
    first = lens[(lens.layer == 0)].set_index("method")
    assert first.loc["unmodified", "margin_yes"] == pytest.approx(
        first.loc["random-direction", "margin_yes"]
    )
    last = lens[(lens.layer == 2)].set_index("method")
    assert last.loc["unmodified", "margin_yes"] != pytest.approx(
        last.loc["random-direction", "margin_yes"]
    )
    score_results(tmp_path / "results", registry, data=tmp_path / "data", bootstrap=0)
    assert len(pd.read_csv(tmp_path / "results" / "lens.csv")) == len(lens)


def test_an_interrupted_task_continues_after_its_last_answer(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")

    class Fragile(ToySubject):
        fails_at: int | None = 4

        def generate(self, *args: Any, **kwargs: Any) -> Generation:
            if self.fails_at is not None and self.calls == self.fails_at:
                raise RuntimeError("interrupted")
            return super().generate(*args, **kwargs)

    subject = Fragile()
    with pytest.raises(RuntimeError, match="interrupted"):
        generate(subject, [Unmodified()], config(tmp_path), registry)
    folder = next((tmp_path / "results" / "per_item" / "toy").iterdir())
    (partial,) = folder.glob("*.part")
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 4
    assert not list(folder.glob("*.jsonl"))
    assert score_results(tmp_path / "results", registry, data=tmp_path / "data").empty
    subject.fails_at = None
    resumed = config(tmp_path).model_copy(update={"resume": True})
    rows = evaluate(subject, [Unmodified()], resumed, registry)
    assert subject.calls == 6
    assert not list(folder.glob("*.part"))
    assert set(rows["items"]) == {6}
    answers = next(folder.glob("*.jsonl")).read_text(encoding="utf-8")
    fresh = ToySubject()
    other = config(tmp_path).model_copy(update={"results": tmp_path / "fresh"})
    generate(fresh, [Unmodified()], other, registry)
    expected = next((tmp_path / "fresh" / "per_item" / "toy").glob("*/*.jsonl"))
    assert answers == expected.read_text(encoding="utf-8")


def test_the_environment_and_the_analyses_are_written_with_the_results(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    control = RandomDirection(norms={1: 5.0}, width=subject.width)
    evaluate(subject, [Unmodified(), control], config(tmp_path), registry)
    (described,) = (tmp_path / "results" / "per_item" / "toy").glob(
        "random-direction-*/setting.json"
    )
    environment = json.loads(described.read_text(encoding="utf-8"))["environment"]
    assert {"python", "torch", "swce", "platform", "accelerator", "subject"} <= set(environment)
    shortcuts = pd.read_csv(tmp_path / "results" / "shortcuts.csv")
    assert set(shortcuts.metric) == {"accuracy", "f1"}
    assert set(shortcuts.method) == {"random-direction"}
    assert (shortcuts.matched_on == "yes_ratio").all()
    analysis = pd.read_csv(tmp_path / "results" / "analysis.csv")
    assert analysis.method.tolist() == ["random-direction"]
    assert analysis.task.tolist() == ["pope-tiny"]
    assert analysis.gain.iloc[0] == pytest.approx(
        shortcuts[shortcuts.metric == "f1"].value.iloc[0]
        - shortcuts[shortcuts.metric == "f1"].unmodified.iloc[0]
    )


def test_task_selection(tmp_path: Path) -> None:
    assert select_tasks(["pope-*"]) == ["pope-random", "pope-popular", "pope-adversarial"]
    assert "mme" in select_tasks([])
    with pytest.raises(ValueError, match="no task matches"):
        select_tasks(["missing"])
    lines = dry_run(EvaluationConfig(tasks=("chair", "mme")))
    assert [line.split(":")[0] for line in lines] == ["chair", "mme"]


def test_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--tasks", "pope-adversarial", "--dry-run"]) == 0
    assert "pope-adversarial" in capsys.readouterr().out
    assert [steerer.name for steerer in load_steerers([])] == ["unmodified"]
    assert load_factory("swce.baselines:Unmodified")().name == "unmodified"
    with pytest.raises(ValueError, match="module:factory"):
        load_factory("swce.baselines")
    with pytest.raises(TypeError, match="not a Steerer"):
        load_steerers(["support:ToySubject"])
    assert cli_main(["results", "--results", str(tmp_path)]) == 0
    assert "No published result yet." in (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert cli_main(["registry", "--check"]) == 0
    assert main(["--score-only", "--results", str(tmp_path / "empty")]) == 0
    assert "[]" in capsys.readouterr().out
