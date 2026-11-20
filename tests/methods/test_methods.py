import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from support import ToySubject

from swce.steering import InapplicableError

METHODS = Path(__file__).parents[2] / "methods"
IMAGE = Image.new("RGB", (4, 4), "white")


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, METHODS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vista = load("vista")
vti = load("vti")
revis = load("revis")
nullu = load("nullu")


def test_vista_edit_keeps_the_norm_and_pushes_harder_against_opposition() -> None:
    vector = torch.tensor([1.0, 0.0, 0.0])
    along = torch.tensor([[[2.0, 1.0, 0.0]]])
    against = torch.tensor([[[-2.0, 1.0, 0.0]]])
    edit = vista.steer(vector, 0.5)
    assert edit(along).norm().item() == pytest.approx(along.norm().item())
    assert edit(against).norm().item() == pytest.approx(against.norm().item())
    gain_along = edit(along)[0, 0, 0] / along.norm() - along[0, 0, 0] / along.norm()
    gain_against = edit(against)[0, 0, 0] / against.norm() - against[0, 0, 0] / against.norm()
    assert gain_against > gain_along > 0
    assert edit(along.half()).dtype == torch.float16


def test_vista_vectors_are_the_difference_an_image_makes() -> None:
    torch.manual_seed(0)
    with_image, without_image = torch.randn(3, 8) * 10, torch.randn(3, 8) * 10
    found = vista.steering_vectors(with_image, without_image)
    assert found.shape == (3, 8)
    # The official analysis of one sample adds one vector of unit norm over all layers.
    assert (found - (with_image - without_image)).norm().item() == pytest.approx(1.0, abs=1e-4)


def test_vista_steers_with_the_vector_of_each_input_and_detaches() -> None:
    subject = ToySubject()
    before = subject.hidden("prompt")
    steerer = vista.Vista(strength=0.5, logits_layers=(0, 1))
    with steerer.attach(subject):
        assert torch.equal(subject.hidden("prompt"), before)
        subject.notify(IMAGE, "prompt")
        assert not torch.allclose(subject.hidden("prompt"), before)
    assert subject.listeners == []
    assert torch.equal(subject.hidden("prompt"), before)
    assert steerer.fitted_on == ()


def test_vista_mixes_the_logits_of_intermediate_layers() -> None:
    subject = ToySubject()
    plain = subject.logits("prompt")
    states = subject.states("prompt")
    expected = 0.5 * (subject.head(states[0]) + subject.head(states[1])) / 2 + 0.5 * plain
    with vista.Vista(strength=0.0, logits_alpha=0.5, logits_layers=(0, 1)).attach(subject):
        assert torch.equal(subject.logits("prompt"), plain)
        subject.notify(IMAGE, "prompt")
        assert torch.allclose(subject.logits("prompt"), expected, atol=1e-5)
    assert torch.equal(subject.logits("prompt"), plain)
    chair, pope, default = vista.build()
    assert (chair.setting, chair.strength) == ("chair", 0.17)
    assert chair.tasks == ("chair", "mme")
    assert pope.tasks == ("pope-*", "repope-*", "mme")
    assert default.tasks == ()
    assert (pope.setting, pope.strength) == ("pope", 0.01)
    assert chair.published_for == ("llava-1.5-7b",)
    assert (default.setting, default.strength, default.fallback) == ("code-default", 0.1, True)
    assert default.published_for == ()


def test_vista_reports_logit_layers_a_model_does_not_have() -> None:
    with (
        pytest.raises(InapplicableError, match="blocks 25 to 30 of 3 blocks"),
        vista.Vista(strength=0.1).attach(ToySubject()),
    ):
        pass


def test_the_lens_reads_the_stream_without_triggering_the_logit_mixing() -> None:
    subject = ToySubject()
    subject.lens = True
    plain = subject.generate(IMAGE, "prompt", 1)
    idle = vista.Vista(strength=0.0, logits_alpha=0.5, logits_layers=(0, 1))
    with idle.attach(subject):
        steered = subject.generate(IMAGE, "prompt", 1)
    assert steered.lens_yes is not None
    assert plain.lens_yes is not None
    assert steered.lens_yes == pytest.approx(plain.lens_yes, rel=1e-5)


def test_vista_control_steers_with_the_same_random_vectors_for_every_input() -> None:
    subject = ToySubject()
    controls = vista.control()
    assert [control.name for control in controls] == ["vista-random"] * 3
    assert [control.tasks for control in controls] == [s.tasks for s in vista.build()]
    assert [control.fallback for control in controls] == [False, False, True]
    control = vista.VistaRandom(strength=0.5, logits_layers=(0, 1))
    first = control.vectors_for(subject, IMAGE, "one prompt")
    assert first.shape == (3, 8)
    assert torch.equal(first, control.vectors_for(subject, IMAGE, "another"))
    other = control.model_copy(update={"seed": 1})
    assert not torch.equal(first, other.vectors_for(subject, IMAGE, "one prompt"))
    before = subject.hidden("prompt")
    with control.attach(subject):
        subject.notify(IMAGE, "prompt")
        assert not torch.allclose(subject.hidden("prompt"), before)
    assert torch.equal(subject.hidden("prompt"), before)


def test_vti_control_draws_directions_of_the_fitted_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SWCE_DATA", str(tmp_path))
    (tmp_path / "artifacts").mkdir()
    subject = ToySubject()
    fitted = {"visual": torch.ones(3, 2, 8), "textual": torch.ones(3, 8)}
    torch.save(fitted, vti.directions_path("toy"))
    controls = vti.control()
    assert [control.name for control in controls] == ["vti-random"] * 3
    assert [(c.visual, c.textual) for c in controls] == [(0.2, 0.4), (0.9, 0.9), (0.0, 0.8)]
    control = controls[1]
    assert control.fitted_on == ()
    with pytest.raises(FileNotFoundError, match="prepare vti-random"):
        control.attach(subject)
    control.prepare(subject)
    drawn = torch.load(tmp_path / "artifacts" / "vti_toy_random0.pt", weights_only=True)
    assert {key: found.shape for key, found in drawn.items()} == {
        key: found.shape for key, found in fitted.items()
    }
    assert not torch.allclose(drawn["textual"], fitted["textual"])
    before = subject.hidden("prompt")
    with control.attach(subject):
        assert not torch.allclose(subject.hidden("prompt"), before)
    assert torch.equal(subject.hidden("prompt"), before)


def test_vti_edit_keeps_the_norm_and_moves_a_tenth_of_the_strength() -> None:
    direction = torch.tensor([0.0, 3.0])
    hidden = torch.tensor([[[2.0, 0.0]]])
    moved = vti.shift(direction, 1.0)(hidden)
    assert moved.norm().item() == pytest.approx(2.0)
    assert (moved[0, 0, 1] / moved[0, 0, 0]).item() == pytest.approx(0.1)
    per_token = vti.shift(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), 1.0)(torch.ones(1, 2, 2))
    assert per_token[0, 0, 1] > per_token[0, 0, 0]
    assert per_token[0, 1, 0] > per_token[0, 1, 1]


def test_vti_direction_is_the_mean_plus_the_first_component() -> None:
    torch.manual_seed(0)
    differences = torch.randn(6, 5) * torch.tensor([5.0, 1.0, 0.1, 0.1, 0.1]) + 2.0
    direction = vti.principal_direction(differences)
    component = direction - differences.mean(dim=0)
    assert component.norm().item() == pytest.approx(1.0, rel=1e-5)
    assert abs(component[0].item()) > 0.9
    batched = vti.principal_direction(torch.stack([differences, differences * 2]))
    assert torch.allclose(batched[0], direction, atol=1e-5)
    assert batched.shape == (2, 5)


def test_vti_directions_drop_the_embeddings() -> None:
    torch.manual_seed(0)
    clean, masked = torch.randn(4, 3, 5, 2), torch.randn(4, 3, 5, 2)
    assert vti.visual_directions(clean, masked).shape == (2, 5, 2)
    faithful, hallucinated = torch.randn(4, 3, 2), torch.randn(4, 3, 2)
    assert vti.textual_directions(faithful, hallucinated).shape == (2, 2)


def test_vti_masks_whole_patches_with_the_channel_mean() -> None:
    pixels = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4)
    masked = vti.mask_patches(pixels, torch.tensor([3]), patch=2)
    assert torch.equal(masked[:, :2], pixels[:, :2])
    assert torch.equal(masked[:, 2:, :2], pixels[:, 2:, :2])
    assert torch.all(masked[0, 2:, 2:] == pixels[0].mean())
    assert torch.all(masked[1, 2:, 2:] == pixels[1].mean())


def test_vti_needs_its_directions_and_steers_both_towers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = ToySubject()
    monkeypatch.setenv("SWCE_DATA", str(tmp_path))
    (tmp_path / "artifacts").mkdir()
    chair, pope, default = vti.build()
    assert (chair.visual, chair.textual, pope.visual, pope.textual) == (0.2, 0.4, 0.9, 0.9)
    assert (default.visual, default.textual, default.fallback) == (0.0, 0.8, True)
    with pytest.raises(FileNotFoundError):
        chair.attach(subject)
    torch.save(
        {"visual": torch.ones(3, 2, 8), "textual": torch.ones(3, 8)}, vti.directions_path("toy")
    )
    before = subject.hidden("prompt")
    with pope.attach(subject):
        assert not torch.allclose(subject.hidden("prompt"), before)
    assert torch.equal(subject.hidden("prompt"), before)
    with vti.Vti(visual=0.0, textual=0.0).attach(subject):
        assert torch.equal(subject.hidden("prompt"), before)


def test_vti_runs_its_textual_branch_alone_when_fitted_without_the_visual_one(
    tmp_path: Path,
) -> None:
    subject = ToySubject()
    path = tmp_path / "textual.pt"
    torch.save({"textual": torch.ones(3, 8)}, path)
    before = subject.hidden("prompt")
    with vti.Vti(visual=0.0, textual=0.9, directions=path).attach(subject):
        assert not torch.allclose(subject.hidden("prompt"), before)
    with (
        pytest.raises(InapplicableError, match="visual directions of VTI were not fitted"),
        vti.Vti(visual=0.9, textual=0.9, directions=path).attach(subject),
    ):
        pass


def test_revis_adds_the_vector_only_where_the_risk_is_high() -> None:
    vector = torch.tensor([2.0, 0.0])
    hidden = torch.tensor([[[3.0, 0.0], [-3.0, 0.0], [0.0, 3.0]]])
    edited = revis.gate(vector, strength=1.5, tau=0.2, gamma=1.0)(hidden)
    assert torch.equal(edited[0, 0], hidden[0, 0])
    assert torch.equal(edited[0, 1], hidden[0, 1] + 1.5 * vector)
    assert torch.equal(edited[0, 2], hidden[0, 2])
    assert torch.equal(
        revis.gate(vector, 1.5, -0.5, 1.0)(hidden)[0, 2], hidden[0, 2] + 1.5 * vector
    )


def test_revis_reads_the_vector_of_the_hidden_state_after_its_layer(tmp_path: Path) -> None:
    subject = ToySubject()
    path = tmp_path / "vectors.pt"
    torch.save({index: {"visual": torch.full((8,), float(index))} for index in range(4)}, path)
    before = subject.hidden("prompt")
    with revis.Revis(layer=2, strength=1.0, tau=-2.0, vectors=path).attach(subject):
        assert torch.allclose(subject.hidden("prompt"), before + 3.0)
    assert torch.equal(subject.hidden("prompt"), before)
    assert all(len(file.digest) == 64 for file in (*revis.RELEASED.values(), revis.PAIRS))


def test_revis_leaves_its_layer_and_threshold_to_the_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SWCE_DATA", str(tmp_path))
    (tmp_path / "artifacts").mkdir()
    published, table, default = revis.build()
    assert (published.strength, published.layer, published.tau) == (1.1, None, None)
    assert published.published_for == table.published_for == ("llava-1.5-7b",)
    assert (table.setting, table.strength, table.layer, table.tau) == ("table-11", 1.1, 30, None)
    assert (default.setting, default.strength, default.fallback) == ("code-default", 2.5, True)
    subject = ToySubject()
    vectors = {index: {"visual": torch.full((8,), float(index))} for index in range(4)}
    torch.save(vectors, revis.vectors_path("toy"))
    with pytest.raises(FileNotFoundError, match="calibrate revis for toy"):
        default.attach(subject)
    outcome = json.dumps({"layer": 1, "tau": -2.0})
    revis.calibration_path("toy").write_text(outcome, encoding="utf-8")
    before = subject.states("prompt")[1]
    with default.attach(subject):
        assert torch.allclose(subject.states("prompt")[1], before + 2.5 * 2.0)
    default.prepare(subject)
    fixed = default.model_copy(update={"layer": 2})
    with pytest.raises(FileNotFoundError, match="holds no threshold for block 2"):
        fixed.attach(subject)
    outcome = json.dumps({"layer": 1, "tau": -2.0, "taus": {"1": -2.0, "2": -3.0}})
    revis.calibration_path("toy").write_text(outcome, encoding="utf-8")
    fixed.prepare(subject)
    before = subject.states("prompt")[2]
    with fixed.attach(subject):
        assert torch.allclose(subject.states("prompt")[2], before + 2.5 * 3.0)
    with pytest.raises(FileNotFoundError, match="compute the vectors"):
        revis.load_vectors("qwen3-vl-8b")


def test_revis_control_passes_a_random_vector_of_the_same_norm_through_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SWCE_DATA", str(tmp_path))
    (tmp_path / "artifacts").mkdir()
    subject = ToySubject()
    torch.manual_seed(5)
    vectors = {index: {"visual": torch.randn(8) * (index + 1)} for index in range(4)}
    torch.save(vectors, revis.vectors_path("toy"))
    outcome = json.dumps({"layer": 1, "tau": -2.0})
    revis.calibration_path("toy").write_text(outcome, encoding="utf-8")
    published, table, control = revis.control()
    assert (published.name, published.strength, control.strength) == ("revis-random", 1.1, 2.5)
    assert (table.setting, table.layer) == ("table-11", 30)
    assert control.fallback
    assert control.fitted_on == ()
    with pytest.raises(FileNotFoundError, match="prepare revis-random"):
        control.attach(subject)
    control.prepare(subject)
    drawn = torch.load(tmp_path / "artifacts" / "revis_toy_random0.pt", weights_only=True)
    assert drawn[2]["visual"].norm().item() == pytest.approx(vectors[2]["visual"].norm().item())
    assert not torch.allclose(drawn[2]["visual"], vectors[2]["visual"])
    before = subject.states("prompt")[1]
    with control.attach(subject):
        steered = subject.states("prompt")[1]
    assert torch.allclose(steered, before + 2.5 * drawn[2]["visual"], atol=1e-5)
    assert torch.equal(subject.states("prompt")[1], before)
    other = control.model_copy(update={"seed": 1})
    other.prepare(subject)
    again = torch.load(tmp_path / "artifacts" / "revis_toy_random1.pt", weights_only=True)
    assert not torch.allclose(again[2]["visual"], drawn[2]["visual"])


def test_revis_principal_direction_and_its_special_cases() -> None:
    torch.manual_seed(0)
    samples = torch.randn(20, 4) * torch.tensor([6.0, 0.5, 0.1, 0.1]) + torch.tensor([9.0, 0, 0, 0])
    direction = revis.principal(samples)
    assert direction.norm().item() == pytest.approx(samples.norm(dim=1).mean().item(), rel=1e-5)
    assert abs(F.normalize(direction, dim=0)[0].item()) > 0.95
    assert torch.equal(revis.principal(samples[:1]), samples[0])
    assert torch.equal(revis.principal(torch.ones(5, 4)), torch.zeros(4))


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize(
    "answer", ["A faithful caption.", "A hallucinated caption.", revis.UNKNOWN[0]]
)
def test_revis_qwen_preserves_empty_image_prompt_without_multimodal_tokenization(
    with_image: bool, answer: str
) -> None:
    image = IMAGE if with_image else None
    prompt = "<|vision_start|><|image_pad|><|vision_end|> preserved template"
    encoded = {"input_ids": torch.tensor([[1, 2, 3]])}
    calls: list[str] = []

    def template(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        assert messages == [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": revis.QUESTION if with_image else "Describe this image.",
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["tokenize"] is with_image
        calls.append("template")
        if with_image:
            assert kwargs["return_dict"] is True and kwargs["return_tensors"] == "pt"
            return encoded
        return prompt

    def tokenize(texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        assert not with_image and texts == [prompt]
        assert kwargs == {"return_tensors": "pt", "padding": True}
        calls.append("tokenizer")
        return encoded

    states = torch.arange(24).reshape(2, 3, 4)

    def moved(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert inputs is encoded
        return inputs

    def all_states(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        assert inputs is encoded
        return states

    subject = SimpleNamespace(
        name="qwen3-vl-8b",
        processor=SimpleNamespace(apply_chat_template=template, tokenizer=tokenize),
        moved=moved,
        all_states=all_states,
    )
    assert torch.equal(revis._states(subject, revis.QUESTION, answer, image), states[:, -1])
    assert calls == (["template"] if with_image else ["template", "tokenizer"])


def test_revis_visual_vector_is_free_of_the_language_prior() -> None:
    torch.manual_seed(0)
    pairs = 30
    prior_axis = torch.tensor([1.0, 0.0, 0.0, 0.0])
    unknown = torch.randn(pairs, 4) * 0.01
    prior = unknown + prior_axis * (5 + torch.randn(pairs, 1))
    plain = torch.randn(pairs, 4) * 0.01
    faithful = plain + prior_axis * 4 + torch.tensor([0.0, 1.0, 0.0, 0.0]) * torch.randn(pairs, 1)
    vectors = revis.combine(faithful, faithful - 1.0, plain, prior, unknown)
    assert set(vectors) == {"truth", "language", "visual"}
    visual = F.normalize(vectors["visual"], dim=0)
    assert abs(visual[0].item()) < 0.05
    assert abs(visual[1].item()) > 0.95


def test_revis_selects_the_deepest_layer_that_separates_false_positives() -> None:
    categories = ["TP", "TN", "FP", "FP", "Other"]
    risks = {
        12: [0.5, 0.6, 0.1, 0.2, 9.0],
        11: [0.1, 0.3, 0.6, 0.8, 9.0],
        10: [0.0, 0.1, 0.9, 0.9, 9.0],
    }
    assert revis.select(risks, categories, percentile=50.0, min_layer=10) == (11, 0.2)
    assert revis.select(risks, categories, percentile=100.0, min_layer=10) == (11, 0.3)
    assert revis.select(risks, categories, percentile=50.0, min_layer=12) is None
    assert revis.thresholds(risks, categories, percentile=50.0) == {12: 0.55, 11: 0.2, 10: 0.05}
    assert revis.thresholds(risks, ["TP", "TN", "TN", "TP", "Other"], percentile=50.0) == {}
    assert (
        revis.select(risks, ["TP", "TN", "TN", "TP", "Other"], percentile=50.0, min_layer=0) is None
    )


def test_revis_calibration_observes_every_layer_in_one_pass(tmp_path: Path) -> None:
    subject = ToySubject()
    torch.manual_seed(3)
    vectors = tmp_path / "vectors.pt"
    loaded = {index: {"visual": torch.randn(8)} for index in range(1, 4)}
    torch.save(loaded, vectors)
    rows = [
        {"image": "a.jpg", "text": "Is there a " + "very " * index + "dog?", "label": label}
        for index, label in enumerate(["yes", "no", "no", "yes", "no", "no", "yes", "no"])
    ]
    questions = tmp_path / "questions.jsonl"
    questions.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    IMAGE.save(tmp_path / "a.jpg")
    risks: dict[int, list[float]] = {0: [], 1: [], 2: []}
    categories = []
    outcomes = {("yes", "yes"): "TP", ("no", "yes"): "FP", ("no", "no"): "TN"}
    for row in rows:
        prompt = row["text"] + " Please answer yes or no."
        answer = subject.generate(IMAGE, prompt, 5).text.lower()
        categories.append(outcomes.get((row["label"], answer), "Other"))
        for layer, state in enumerate(subject.states(prompt)):
            unit = F.normalize(loaded[layer + 1]["visual"], dim=-1)
            risks[layer].append(-torch.matmul(F.normalize(state[0, -1], dim=-1), unit).item())
    expected = revis.select(risks, categories, percentile=85.0, min_layer=0)
    output = tmp_path / "calibration.json"

    def run() -> tuple[int, float]:
        found: tuple[int, float] = revis.calibrate(
            subject, tmp_path, vectors=vectors, questions=questions, output=output, min_layer=0
        )
        return found

    if expected is None:
        with pytest.raises(InapplicableError):
            run()
    else:
        assert run() == pytest.approx(expected)
        written = json.loads(output.read_text(encoding="utf-8"))
        assert written["layer"] == expected[0]
        every = revis.thresholds(risks, categories, percentile=85.0)
        # The two passes round differently in single precision.
        expected_taus = {str(layer): tau for layer, tau in every.items()}
        assert written["taus"] == pytest.approx(expected_taus, abs=1e-6)
    assert subject.listeners == []


def test_nullu_removes_the_subspace_from_the_feed_forward_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    basis = torch.linalg.qr(torch.randn(8, 4)).Q
    hidden = torch.randn(1, 3, 8)
    projected = nullu.project_out(basis)(hidden)
    assert torch.allclose(projected @ basis, torch.zeros(1, 3, 4), atol=1e-5)
    assert torch.allclose(nullu.project_out(basis)(projected), projected, atol=1e-5)
    subject = ToySubject()
    monkeypatch.setenv("SWCE_DATA", str(tmp_path))
    (tmp_path / "artifacts").mkdir()
    published, default = nullu.build()
    assert (published.rank, published.first_layer, published.last_layer) == (4, 16, 31)
    assert published.published_for == ("llava-1.5-7b",)
    assert (default.rank, default.last_layer, default.fallback) == (8, 31, True)
    with pytest.raises(FileNotFoundError):
        published.attach(subject)
    torch.save({1: basis, 2: basis}, nullu.subspaces_path("toy"))
    before = subject.hidden("prompt")
    with nullu.Nullu(rank=2, first_layer=1, last_layer=2).attach(subject):
        assert not torch.allclose(subject.hidden("prompt"), before)
    assert torch.equal(subject.hidden("prompt"), before)
    with pytest.raises(InapplicableError, match="blocks 16 to 31 of 3 blocks"):
        published.attach(subject)
    nullu.Nullu(rank=2, first_layer=1, last_layer=2).prepare(subject)


def test_nullu_control_projects_a_random_subspace_without_any_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SWCE_DATA", str(tmp_path))
    subject = ToySubject()
    controls = nullu.control()
    assert [(c.name, c.rank, c.fallback) for c in controls] == [
        ("nullu-random", 4, False),
        ("nullu-random", 8, True),
    ]
    control = nullu.NulluRandom(rank=2, first_layer=1, last_layer=2)
    control.prepare(subject)
    assert not (tmp_path / "artifacts").exists()
    before = subject.hidden("prompt")
    with control.attach(subject):
        steered = subject.hidden("prompt")
    assert not torch.allclose(steered, before)
    with control.attach(subject):
        assert torch.equal(subject.hidden("prompt"), steered)
    with control.model_copy(update={"seed": 1}).attach(subject):
        assert not torch.allclose(subject.hidden("prompt"), steered)
    assert torch.equal(subject.hidden("prompt"), before)
    with pytest.raises(InapplicableError, match="blocks 16 to 31 of 3 blocks"):
        controls[0].attach(subject)


def test_nullu_subspaces_span_the_leading_differences() -> None:
    torch.manual_seed(0)
    axis = F.normalize(torch.randn(8), dim=0)
    hallucinated = torch.randn(12, 3, 8) * 0.01
    truthful = hallucinated + axis * (4 + torch.randn(12, 3, 1))
    found = nullu.subspaces(truthful, hallucinated, kept=5)
    assert set(found) == {0, 1, 2}
    assert found[0].shape == (8, 5)
    assert torch.allclose(found[0].T @ found[0], torch.eye(5), atol=1e-5)
    assert abs(float(found[1][:, 0] @ axis)) > 0.99
