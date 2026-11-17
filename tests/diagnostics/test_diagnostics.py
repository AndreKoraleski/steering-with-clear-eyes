import importlib.util
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from support import ToySubject, tiny_registry
from tiny import inputs, outputs, tiny_llava, tiny_qwen

from swce.baselines import RandomDirection, Unmodified
from swce.diagnostics import (
    diagnose,
    logit_lens,
    observe,
    read,
    residual_shift,
    summarize,
    summarize_results,
    write_readings,
)
from swce.diagnostics.__main__ import main
from swce.models.hf import HuggingFaceSubject

IMAGE = Image.new("RGB", (4, 4), "white")
VISTA = Path(__file__).parents[2] / "methods" / "vista.py"


def test_reading_the_last_state_gives_the_logits_of_the_model() -> None:
    subject = ToySubject()
    states = subject.residuals(IMAGE, "prompt")
    assert torch.allclose(read(subject, states[-1]), subject.logits("prompt")[0, -1])
    assert read(subject, states).shape == (3, 5)


@pytest.mark.parametrize("build", [tiny_llava, tiny_qwen])
def test_reading_applies_the_final_norm_of_real_architectures(
    build: Callable[[], HuggingFaceSubject],
) -> None:
    subject = build()
    output = outputs(subject)
    states = subject.block_outputs(inputs(subject))
    assert states.shape == (4, 16)
    assert torch.allclose(read(subject, states[-1]), output.logits[0, -1], atol=1e-5)
    assert not torch.allclose(read(subject, states[0]), output.logits[0, -1], atol=1e-3)


def test_logit_lens_reports_every_block() -> None:
    subject = ToySubject()
    states = subject.residuals(IMAGE, "prompt")
    lens = logit_lens(subject, states, top=2)
    assert list(lens.columns) == ["layer", "yes", "no", "margin", "entropy", "top"]
    assert lens.layer.tolist() == [0, 1, 2]
    final = torch.softmax(subject.logits("prompt")[0, -1], dim=-1)
    assert lens.yes.iloc[-1] == pytest.approx(final[0].item(), rel=1e-5)
    assert lens.no.iloc[-1] == pytest.approx(final[1].item(), rel=1e-5)
    assert lens.margin.iloc[-1] == pytest.approx(torch.log(final[0] / final[1]).item(), rel=1e-4)
    assert lens.entropy.between(0, torch.log(torch.tensor(5.0)).item() + 1e-6).all()
    assert all(len(tokens.split()) == 2 for tokens in lens.top)


def test_residual_shift_starts_at_the_steered_block() -> None:
    subject = ToySubject()
    unmodified = observe(subject, IMAGE, "prompt")
    same = residual_shift(observe(subject, IMAGE, "prompt", Unmodified()), unmodified)
    assert same.cosine.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert same.distance.tolist() == pytest.approx([0.0, 0.0, 0.0])
    control = RandomDirection(norms={1: 5.0}, width=subject.width)
    shift = residual_shift(observe(subject, IMAGE, "prompt", control), unmodified)
    assert shift.distance.iloc[0] == pytest.approx(0.0)
    assert shift.distance.iloc[1] > 0.1
    assert shift.distance.iloc[2] > 0.0
    assert torch.equal(observe(subject, IMAGE, "prompt"), unmodified)


def test_observing_notifies_methods_that_depend_on_the_input() -> None:
    spec = importlib.util.spec_from_file_location("vista", VISTA)
    assert spec is not None
    assert spec.loader is not None
    vista = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vista)
    subject = ToySubject()
    steerer = vista.Vista(strength=0.5, logits_layers=(0, 1))
    steered = observe(subject, IMAGE, "prompt", steerer)
    assert not torch.allclose(steered, observe(subject, IMAGE, "prompt"))
    assert subject.listeners == []


def test_diagnose_reads_every_item_under_every_method(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    control = RandomDirection(norms={1: 5.0}, width=subject.width)
    elsewhere = RandomDirection(norms={1: 5.0}, width=subject.width, tasks=("chair",))
    readings = diagnose(
        subject, [control, elsewhere], "pope-tiny", data=tmp_path / "data", registry=registry
    )
    assert len(readings) == 6 * 3 * 2
    assert set(readings.method) == {"unmodified", "random-direction"}
    assert (readings[readings.method == "unmodified"].distance == 0).all()
    assert set(readings.label) == {"yes", "no"}
    limited = diagnose(subject, [], "pope-tiny", limit=1, data=tmp_path / "data", registry=registry)
    assert len(limited) == 2 * 3
    summary = summarize(readings)
    assert len(summary) == 2 * 3
    assert summary.auroc.between(0, 1).all()
    steered = summary[(summary.method == "random-direction") & (summary.layer == 2)].iloc[0]
    assert steered.distance > 0
    assert steered.margin_yes == pytest.approx(
        readings[
            (readings.method == "random-direction")
            & (readings.layer == 2)
            & (readings.label == "yes")
        ].margin.mean()
    )


def test_readings_of_separate_jobs_are_summarized_together(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    subject = ToySubject()
    first = RandomDirection(norms={1: 5.0}, width=subject.width)
    second = RandomDirection(norms={2: 3.0}, width=subject.width, setting="other")
    for steerer, folder in ((first, "a"), (second, "b")):
        readings = diagnose(
            subject, [steerer], "pope-tiny", data=tmp_path / "data", registry=registry
        )
        written = write_readings(readings, tmp_path / folder, "pope-tiny")
        assert len(written) == 2
    shutil.copytree(
        tmp_path / "b" / "diagnostics", tmp_path / "a" / "diagnostics", dirs_exist_ok=True
    )
    (summary,) = summarize_results(tmp_path / "a")
    assert summary == tmp_path / "a" / "diagnostics" / "toy" / "pope-tiny-summary.csv"
    merged = pd.read_csv(summary)
    assert sorted(set(merged.setting)) == ["default", "other"]
    assert len(merged) == 3 * 3
    assert main(["--summarize-only", "--results", str(tmp_path / "a")]) == 0
    assert summarize_results(tmp_path / "empty") == []
