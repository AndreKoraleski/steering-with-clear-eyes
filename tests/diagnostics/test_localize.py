from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tiny import inputs as tiny_inputs
from tiny import tiny_llava
from torch import Tensor

from swce.diagnostics.coco import Cooccurrence, object_of
from swce.diagnostics.encoder import MultiProbes, image_features, probe_encoder
from swce.diagnostics.knockout import (
    blocked_mask,
    knocked_out,
    route_positions,
    summarize_knockout,
    windows,
)
from swce.diagnostics.prior import summarize_prior, with_cooccurrence
from swce.models.hf import HuggingFaceSubject


def test_object_of_reads_every_form_of_pope() -> None:
    assert object_of("Is there a dining table in the image?") == "dining table"
    assert object_of("Is there an apple in the image?") == "apple"
    with pytest.raises(ValueError, match="not a POPE question"):
        object_of("What is in the image?")


def test_cooccurrence_counts_pairs_once_per_image() -> None:
    counts = Cooccurrence(
        [frozenset({"fork", "dining table"}), frozenset({"dining table"}), frozenset({"dog"})]
    )
    assert counts.given("fork", "dining table") == pytest.approx(0.5)
    assert counts.given("dining table", "fork") == pytest.approx(1.0)
    assert counts.strongest("fork", {"dining table", "dog"}) == pytest.approx(0.5)
    assert counts.strongest("fork", set()) == 0.0


def test_prior_readings_tell_wrong_yeses_apart() -> None:
    frame = pd.DataFrame(
        {
            "id": [str(i) for i in range(6)],
            "group": ["a", "a", "b", "b", "c", "c"],
            "object": ["fork", "dog", "fork", "cat", "dog", "fork"],
            "label": ["no", "no", "no", "yes", "no", "yes"],
            "margin": [2.0, -1.0, 1.5, 3.0, -2.0, 1.0],
            "blind": [1.0, -2.0, 0.5, 0.2, -1.5, 0.3],
        }
    )
    present = {"a": frozenset({"dining table"}), "b": frozenset({"cat"}), "c": frozenset()}
    counts = Cooccurrence([frozenset({"fork", "dining table"}), frozenset({"dining table"})])
    filled = with_cooccurrence(frame, present, counts)
    assert filled["cooccurrence"].tolist() == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    readings = dict(zip(*summarize_prior(filled)[["reading", "value"]].to_numpy().T, strict=True))
    assert readings["wrong_yes"] == 2
    assert readings["auroc_wrong_yes_by_prior"] == pytest.approx(1.0)
    assert readings["wrong_yes_with_positive_prior"] == pytest.approx(1.0)


def test_multi_probes_solve_the_logistic_regression_of_every_category() -> None:
    generator = np.random.default_rng(0)
    items, blocks, width, count = 150, 2, 5, 3
    features = generator.normal(size=(items, blocks, width))
    targets = generator.random((items, count)) < 0.4
    features[:, 1, 0] += np.where(targets[:, 0], 1.5, -1.5)
    fitted = MultiProbes(torch.from_numpy(features), torch.from_numpy(targets))
    ours = fitted.decision(torch.from_numpy(features)).numpy()
    for block in range(blocks):
        for category in range(count):
            reference = make_pipeline(
                StandardScaler(), LogisticRegression(tol=1e-10, max_iter=10000)
            )
            reference.fit(features[:, block], targets[:, category])
            expected = reference.decision_function(features[:, block])
            assert ours[:, block, category] == pytest.approx(expected, abs=1e-4)


def test_probe_encoder_finds_the_block_that_holds_the_object() -> None:
    generator = np.random.default_rng(1)
    names = ["dog", "cat"]

    def images(count: int) -> tuple[np.ndarray, list[frozenset[str]]]:
        objects = [frozenset(n for n in names if generator.random() < 0.5) for _ in range(count)]
        features = generator.normal(size=(count, 2, 6)).astype(np.float32)
        features[:, 1, 0] += np.array([2.0 if "dog" in o else -2.0 for o in objects])
        return features, objects

    train, train_objects = images(200)
    test, test_objects = images(40)
    questions = [("dog", "dog" in found) for found in test_objects] + [("unicorn", True)]
    index = np.array([*range(40), 0])
    frame = probe_encoder(
        {"encoder": train},
        train_objects,
        {"encoder": test},
        index,
        questions,
        names,
        device=torch.device("cpu"),
    )
    assert frame["questions"].tolist() == [40, 40]
    assert 0.2 < frame["auroc"].iloc[0] < 0.8
    assert frame["auroc"].iloc[1] > 0.95


def test_blocked_mask_removes_only_the_routes_asked() -> None:
    mask = blocked_mask(
        5, torch.tensor([4]), torch.tensor([1, 2]), boolean=True, dtype=torch.float32, device="cpu"
    )
    expected = torch.ones(5, 5, dtype=torch.bool).tril()
    expected[4, 1] = expected[4, 2] = False
    assert torch.equal(mask[0, 0], expected)
    additive = blocked_mask(
        5, torch.tensor([4]), torch.tensor([1]), boolean=False, dtype=torch.float32, device="cpu"
    )
    assert additive[0, 0, 4, 1] == torch.finfo(torch.float32).min
    assert additive[0, 0, 4, 0] == 0


def test_routes_split_the_image_the_text_and_the_answer() -> None:
    routes = route_positions(torch.tensor([1, 59, 59, 59, 59, 5, 6, 7]), 59)
    queries, keys = routes["answer-image"]
    assert queries.tolist() == [7] and keys.tolist() == [1, 2, 3, 4]
    assert routes["answer-text"][1].tolist() == [5, 6]
    assert routes["text-image"][0].tolist() == [5, 6]
    assert windows(4, 3) == [[0, 1, 2], [1, 2, 3]]


def logits_of(subject: HuggingFaceSubject) -> Tensor:
    with torch.no_grad():
        found: Tensor = subject.model(**tiny_inputs(subject)).logits[0, -1]
    return found


def test_knockout_changes_the_answer_only_inside_its_context() -> None:
    subject = tiny_llava()
    layers = subject.decoder_layers()
    base = logits_of(subject)
    with knocked_out(layers, [0, 1, 2, 3], torch.tensor([6]), torch.arange(0)):
        assert torch.allclose(logits_of(subject), base, atol=1e-5)
    with knocked_out(layers, [0, 1, 2, 3], torch.tensor([6]), torch.arange(1, 6)):
        assert not torch.allclose(logits_of(subject), base, atol=1e-4)
    assert torch.allclose(logits_of(subject), base, atol=1e-6)


def test_summary_of_knockout_counts_flips() -> None:
    frame = pd.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "kind": ["wrong_yes"] * 4,
            "route": ["answer-image"] * 4,
            "start": [0, 1, 0, 1],
            "end": [8, 9, 8, 9],
            "margin": [1.0, 1.0, 2.0, 2.0],
            "knocked": [-0.5, 0.5, 1.0, 3.0],
            "change": [-1.5, -0.5, -1.0, 1.0],
        }
    )
    summary = summarize_knockout(frame)
    assert summary["flipped"].tolist() == [0.5, 0.0]
    assert summary["change"].tolist() == pytest.approx([-1.25, 0.25])


def test_image_features_read_every_encoder_block(monkeypatch: pytest.MonkeyPatch) -> None:
    subject = tiny_llava()
    monkeypatch.setattr(HuggingFaceSubject, "template", lambda self, prompt: "template")
    monkeypatch.setattr(subject, "inputs", lambda image, text: tiny_inputs(subject))
    encoder, projected = image_features(subject, Image.new("RGB", (28, 28)))
    assert encoder.shape == (3, 16)
    assert projected is not None and projected.shape == (16,)


def test_object_of_reads_the_variant_with_its_instruction() -> None:
    assert object_of("Is there a dog in the image? Please answer yes or no.") == "dog"


def test_find_span_finds_the_last_run_of_tokens() -> None:
    from swce.diagnostics.patching import find_span

    assert find_span([5, 7, 8, 1, 7, 8], [7, 8]) == [4, 5]
    assert find_span([5, 7], [9]) == []


def test_patching_every_position_reproduces_the_other_run() -> None:
    from swce.diagnostics.patching import patched
    from swce.sae.harvest import capture

    subject = tiny_llava()
    layers = subject.decoder_layers()
    bad = tiny_inputs(subject)
    good = dict(bad, pixel_values=torch.ones_like(bad["pixel_values"]))
    with torch.no_grad(), capture(layers, [0]) as found:
        target = subject.model(**good).logits[0, -1]
    states = found[0].clone()
    everything = torch.arange(bad["input_ids"].shape[1])
    with torch.no_grad(), patched(layers[0], everything, states):
        restored = subject.model(**bad).logits[0, -1]
    with torch.no_grad():
        untouched = subject.model(**bad).logits[0, -1]
    assert torch.allclose(restored, target, atol=1e-5)
    assert not torch.allclose(untouched, target, atol=1e-4)


def test_pairs_join_a_wrong_yes_to_a_correct_no_on_another_image() -> None:
    from swce.diagnostics.patching import pairs_of

    answers = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "group": ["a", "b", "a", "c"],
            "object": ["dog", "dog", "dog", "cat"],
            "label": ["no", "no", "no", "no"],
            "margin": [1.0, -1.0, -2.0, 3.0],
        }
    )
    assert pairs_of(answers, 5) == [("1", "2")]


def test_grid_places_boxes_on_the_centre_crop() -> None:
    from swce.diagnostics.attention import Grid

    grid = Grid(shortest=336, crop=336, patch=14)
    assert grid.side == 24
    assert grid.visible(672, 336, [0, 0, 100, 100]) == 0.0
    assert grid.cells(672, 336, [0, 0, 100, 100]) == []
    assert grid.cells(672, 336, [168, 0, 14, 14]) == [0]
    assert grid.cells(672, 336, [168 + 14, 14, 28, 1]) == [25, 26]
    assert grid.visible(672, 336, [100, 0, 136, 10]) == pytest.approx(0.5)
    padded = Grid(shortest=336, crop=336, patch=14, pad=True)
    assert padded.visible(672, 336, [0, 0, 100, 100]) == pytest.approx(1.0)


def test_head_value_norms_match_the_product_of_every_head() -> None:
    from swce.diagnostics.attention import head_value_norms

    values = torch.randn(5, 8)
    output = torch.randn(6, 8)
    norms = head_value_norms(values, output, heads=2)
    expected = (values[:, 4:] @ output[:, 4:].T).norm(dim=-1)
    assert torch.allclose(norms[1], expected, atol=1e-5)
    shared = head_value_norms(values[:, :4], output, heads=2)
    assert torch.allclose(shared[1], (values[:, :4] @ output[:, 4:].T).norm(dim=-1), atol=1e-5)


def test_answer_attention_is_a_share_of_every_position() -> None:
    from swce.diagnostics.attention import answer_attention

    subject = tiny_llava()
    model: Any = subject.model
    before = model.config._attn_implementation
    shares = answer_attention(subject, tiny_inputs(subject))
    assert shares.shape == (4, 7)
    assert torch.allclose(shares.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert model.config._attn_implementation == before
