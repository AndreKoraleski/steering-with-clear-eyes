from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from support import ToySubject, tiny_registry

from swce.baselines import RandomDirection
from swce.diagnostics import probe, probe_task, write_probes
from swce.diagnostics.probe import Probes, released
from swce.models.subject import Subject

# The package exports a function named `probe`, which hides the module of the same name.
probe_module = import_module("swce.diagnostics.probe")


def states_with_signal(
    groups: int = 40, per_group: int = 2, width: int = 8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """States of two blocks, the first pure noise and the second holding the label on axis 0."""
    generator = np.random.default_rng(seed)
    count = groups * per_group
    positive = np.array([index % 2 == 0 for index in range(count)])
    noise = generator.normal(size=(count, 2, width)).astype(np.float32)
    noise[:, 1, 0] += np.where(positive, 2.0, -2.0)
    names = np.repeat([f"image{index}" for index in range(groups)], per_group)
    return noise, positive, names


def test_probes_solve_the_logistic_regression_of_scikit_learn() -> None:
    generator = np.random.default_rng(2)
    count, layers, width = 120, 3, 6
    positive = generator.random(count) < 0.5
    states = generator.normal(size=(count, layers, width)) * np.array([1.0, 3.0, 0.5])[:, None]
    states[:, :, 0] += np.where(positive, 1.0, -1.0)[:, None] * np.array([0.2, 1.0, 2.0])
    fitted = Probes(torch.from_numpy(states), torch.from_numpy(positive))
    ours = fitted.decision(torch.from_numpy(states)).numpy()
    for layer in range(layers):
        reference = make_pipeline(StandardScaler(), LogisticRegression(tol=1e-10, max_iter=10000))
        reference.fit(states[:, layer], positive)
        expected = reference.decision_function(states[:, layer])
        assert ours[:, layer] == pytest.approx(expected, abs=1e-4)


def test_probe_reads_the_block_that_holds_the_answer() -> None:
    states, positive, groups = states_with_signal()
    readings = probe(states, states, positive, groups)
    assert readings.layer.tolist() == [0, 1]
    assert 0.25 < readings.auroc.iloc[0] < 0.75
    assert readings.auroc.iloc[1] > 0.95
    assert readings.transfer_auroc.tolist() == pytest.approx(readings.auroc.tolist())
    assert readings.score_shift.tolist() == pytest.approx([0.0, 0.0])


def test_probe_measures_a_shift_along_the_answer() -> None:
    states, positive, groups = states_with_signal()
    steered = states.copy()
    steered[:, 1, 0] += 3.0
    readings = probe(states, steered, positive, groups)
    assert readings.score_shift.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert readings.score_shift.iloc[1] > 0.5
    assert readings.auroc.iloc[1] > 0.95
    away = states.copy()
    away[:, 1, 0] -= 3.0
    assert probe(states, away, positive, groups).score_shift.iloc[1] < -0.5


def test_probe_never_fits_and_evaluates_on_the_same_image() -> None:
    generator = np.random.default_rng(1)
    groups, per_group, width = 40, 3, 50
    labels = generator.random(groups) < 0.5
    positive = np.repeat(labels, per_group)
    shared = generator.normal(size=(groups, width)).repeat(per_group, axis=0)
    states = (shared + 0.01 * generator.normal(size=shared.shape)).astype(np.float32)[:, None]
    names = np.repeat([f"image{index}" for index in range(groups)], per_group)
    # Each image carries its own label and nothing else, so only a probe that has seen the
    # image could tell its answer.
    assert probe(states, states, positive, names).auroc.iloc[0] < 0.75


def test_probe_task_reads_every_method_and_writes_one_file(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    control = RandomDirection(norms={1: 5.0}, width=subject.width)
    elsewhere = RandomDirection(norms={1: 5.0}, width=subject.width, tasks=("chair",))
    probes = probe_task(
        subject,
        [control, elsewhere],
        "pope-tiny",
        folds=2,
        data=tmp_path / "data",
        registry=registry,
    )
    assert set(probes.method) == {"unmodified", "random-direction"}
    assert len(probes) == 2 * 3
    assert (probes["items"] == 6).all()
    unmodified = probes[probes.method == "unmodified"]
    assert (unmodified.score_shift == 0).all()
    areas = probes[["auroc", "transfer_auroc"]].to_numpy()
    assert ((areas >= 0) & (areas <= 1)).all()
    (written,) = write_probes(probes, tmp_path / "results")
    assert written == tmp_path / "results" / "diagnostics" / "toy" / "probe-pope-tiny.csv"
    assert len(pd.read_csv(written)) == len(probes)


def test_released_holds_the_modules_on_the_processor_and_puts_them_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = ToySubject()
    subject.head.to("meta")
    moves: list[str] = []
    monkeypatch.setattr(subject.head, "to", lambda device: moves.append(str(device)))
    with released(subject):
        assert moves == ["cpu"]
    assert moves == ["cpu", "meta"]
    with pytest.raises(RuntimeError), released(subject):
        raise RuntimeError
    assert moves == ["cpu", "meta", "cpu", "meta"]


def test_probe_task_gives_the_same_probes_whether_the_model_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tiny_registry(tmp_path / "host")
    control = RandomDirection(norms={1: 5.0}, width=ToySubject().width)

    def run() -> pd.DataFrame:
        return probe_task(
            ToySubject(), [control], "pope-tiny", folds=2, data=tmp_path / "data", registry=registry
        )

    calls: list[Subject] = []

    @contextmanager
    def counted(subject: Subject) -> Iterator[None]:
        calls.append(subject)
        with released(subject):
            yield

    monkeypatch.setattr(probe_module, "released", counted)
    with_release = run()
    assert len(calls) == 1
    monkeypatch.setattr(probe_module, "released", lambda subject: nullcontext())
    pd.testing.assert_frame_equal(with_release, run())
