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
    assert pairs_of(answers, 5) == [("1", ["2"])]


def test_grid_places_boxes_on_the_centre_crop() -> None:
    from swce.diagnostics.attention import Grid

    grid = Grid(shortest=336, crop=336, patch=14)
    assert grid.side == 24
    assert grid.visible(672, 336, [0, 0, 100, 100]) == 0.0
    assert grid.cells(672, 336, [0, 0, 100, 100]) == []
    assert grid.cells(672, 336, [168, 0, 14, 14]) == [0]
    assert grid.cells(672, 336, [168 + 14, 14, 28, 1]) == [25, 26]
    assert grid.visible(672, 336, [100, 0, 136, 10]) == pytest.approx(0.5)
    assert grid.visible(640, 427, [0.3, 0.1, 639.4, 426.7]) <= 1.0
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


def test_answer_attention_asks_for_the_weights() -> None:
    from swce.diagnostics.attention import answer_attention

    subject = tiny_llava()
    asked: list[Any] = []
    handle = subject.model.register_forward_pre_hook(
        lambda _module, _args, kwargs: asked.append(kwargs.get("output_attentions")),
        with_kwargs=True,
    )
    try:
        answer_attention(subject, tiny_inputs(subject))
    finally:
        handle.remove()
    assert asked == [True]


def test_resized_grid_places_boxes_on_merged_tokens() -> None:
    from swce.diagnostics.attention import ResizedGrid

    grid = ResizedGrid(rows=4, columns=6, patch=14, merge=2)
    assert grid.cells(168, 112, [0, 0, 50, 50]) == [0]
    assert grid.cells(168, 112, [120, 60, 40, 40]) == [5]
    assert grid.cells(168, 112, [0, 0, 168, 112]) == [0, 1, 2, 3, 4, 5]
    assert grid.visible(168, 112, [0, 0, 10, 10]) == 1.0


def test_resized_grid_reads_the_grid_the_processor_returns() -> None:
    from tiny import tiny_qwen

    from swce.diagnostics.attention import ResizedGrid

    subject = tiny_qwen()
    grid = ResizedGrid.of(subject, tiny_inputs(subject))
    assert (grid.rows, grid.columns, grid.patch, grid.merge) == (2, 2, 14, 2)
    assert grid.cells(28, 28, [0, 0, 28, 28]) == [0]


def test_patching_reproduces_the_other_run_on_a_tiny_qwen() -> None:
    from tiny import tiny_qwen

    from swce.diagnostics.patching import patched
    from swce.sae.harvest import capture

    subject = tiny_qwen()
    layers = subject.decoder_layers()
    bad = tiny_inputs(subject)
    good = dict(bad, pixel_values=torch.ones_like(bad["pixel_values"]))
    # The tiny Qwen3-VL injects its vision features once more after block 0, as DeepStack does,
    # so a patch reproduces the other run from block 1 on.
    with torch.no_grad(), capture(layers, [0, 1]) as found:
        target = subject.model(**good).logits[0, -1]
    everything = torch.arange(bad["input_ids"].shape[1])
    with torch.no_grad(), patched(layers[1], everything, found[1].clone()):
        restored = subject.model(**bad).logits[0, -1]
    with torch.no_grad(), patched(layers[0], everything, found[0].clone()):
        overwritten = subject.model(**bad).logits[0, -1]
    assert torch.allclose(restored, target, atol=1e-5)
    assert not torch.allclose(overwritten, target, atol=1e-5)


def test_patching_tries_partners_until_one_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    from swce.diagnostics import patching

    answers = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "group": ["a", "b", "c"],
            "object": ["dog"] * 3,
            "label": ["no"] * 3,
            "margin": [1.0, -1.0, -2.0],
        }
    )

    class Item:
        def __init__(self, identifier: str) -> None:
            self.id = identifier

    class Bound:
        def __init__(self, *_: object) -> None:
            pass

        def items(self) -> list[Item]:
            return [Item(identifier) for identifier in ("1", "2", "3")]

    calls: list[str] = []

    def fake(subject: object, task: object, wrong: Item, right: Item) -> list[dict[str, object]]:
        calls.append(right.id)
        return [{"wrong": wrong.id, "right": right.id}] if len(calls) == 2 else []

    monkeypatch.setattr(patching, "Task", Bound)
    monkeypatch.setattr(patching, "patch_pair", fake)
    frame = patching.patching_task(object(), "pope-tiny", answers, pairs=1)  # type: ignore[arg-type]
    assert len(calls) == 2
    assert frame["tried"].tolist() == [2]


def test_region_tokens_preserve_the_old_mean_and_exclude_cls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from swce.diagnostics.attention import Grid
    from swce.diagnostics.encoder import _image_features, region_features

    subject = tiny_llava()
    inputs = tiny_inputs(subject)
    monkeypatch.setattr(HuggingFaceSubject, "template", lambda self, prompt: "template")
    monkeypatch.setattr(subject, "inputs", lambda image, text: inputs)
    tokens, projected_tokens = _image_features(subject, inputs, lambda state, source: state)
    before = [dict(block._forward_hooks) for block in subject.encoder_layers()]
    mean, projected_mean = image_features(subject, Image.new("RGB", (28, 28)))
    pooled, projected = region_features(subject, inputs, [[0, 1, 2, 3], [2]], Grid(28, 28, 14))
    assert torch.allclose(tokens.mean(dim=1), mean, atol=1e-6)
    assert torch.allclose(pooled[0], tokens[:, 1:].mean(dim=1), atol=1e-6)
    assert torch.allclose(pooled[1], tokens[:, 3], atol=1e-6)
    assert torch.allclose((pooled[0] * 4 + tokens[:, 0]) / 5, mean, atol=1e-6)
    assert projected is not None and projected_mean is not None and projected_tokens is not None
    assert torch.allclose(projected[0], projected_mean, atol=1e-6)
    assert torch.allclose(projected[1], projected_tokens[2], atol=1e-6)
    with pytest.raises(ValueError, match="Cannot map"):
        region_features(subject, inputs, [[0]], Grid(42, 42, 14))
    assert [dict(block._forward_hooks) for block in subject.encoder_layers()] == before


def test_region_cells_drop_boxes_outside_the_crop() -> None:
    from swce.diagnostics.attention import Grid
    from swce.diagnostics.encoder import region_cells

    kept = region_cells(
        Grid(336, 336, 14),
        672,
        336,
        [
            ("outside", [0, 0, 100, 100]),
            ("edge", [100, 0, 136, 10]),
            ("inside", [168, 0, 14, 14]),
            ("empty", [168, 0, 0, 14]),
            ("under", [99, 0, 136, 10]),
        ],
    )
    assert [name for name, _, _ in kept] == ["edge", "inside"]
    assert kept[1] == ("inside", 196, [0])


def test_region_probes_recognize_separable_categories() -> None:
    from swce.diagnostics.encoder import probe_regions

    rng = np.random.default_rng(8)
    names = ["dog", "cat"] * 40
    features = rng.normal(size=(80, 2, 5)).astype(np.float32)
    features[:, 1, 0] += np.array([5 if name == "dog" else -5 for name in names])
    frame = probe_regions(
        {"encoder": features},
        names,
        {"encoder": features},
        names,
        [(i, name, i % 3 == 0) for i, name in enumerate(names)],
        ["dog", "cat"],
        device=torch.device("cpu"),
    )
    assert frame["region_auroc"].iloc[1] == pytest.approx(1)
    assert frame["categories"].tolist() == [2, 2]
    assert frame["correct_yes"].tolist() == [27, 27]
    assert frame["missed"].tolist() == [53, 53]
    assert not frame.isna().any().any()


def test_encoder_regions_task_reads_boxes_and_margins(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager
    from types import SimpleNamespace

    from swce.diagnostics import encoder as encoder_module
    from swce.diagnostics.attention import Grid
    from swce.diagnostics.encoder import encoder_regions_task
    from swce.diagnostics.probe import released
    from swce.tasks.task import Task

    subject = tiny_llava()
    original = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject.model.to(original)
    original = next(subject.model.parameters()).device
    calls = []

    @contextmanager
    def counted(model: HuggingFaceSubject) -> Iterator[None]:
        calls.append(model)
        with released(model):
            assert {p.device.type for p in model.model.parameters()} == {"cpu"}
            yield

    monkeypatch.setattr(encoder_module, "released", counted)
    monkeypatch.setattr(HuggingFaceSubject, "template", lambda self, prompt: "template")
    monkeypatch.setattr(
        subject,
        "inputs",
        lambda image, text: {
            key: value.to(original) for key, value in tiny_inputs(subject).items()
        },
    )
    monkeypatch.setattr(Grid, "of", classmethod(lambda cls, subject: Grid(28, 28, 14)))
    image = Image.new("RGB", (28, 28))
    boxes = [(name, [0.0, 0.0, 14.0, 14.0]) for name in ["dog", "cat"] * 10]
    boxes += [("dog", [0.0, 0.0, 28.0, 28.0])]
    items = [
        SimpleNamespace(
            id=str(i),
            group="image",
            prompt="Is there a dog in the image?",
            label="yes" if i < 2 else "no",
        )
        for i in range(3)
    ]
    monkeypatch.setattr(Task, "items", lambda self, limit=None: items)
    monkeypatch.setattr(Task, "image", lambda self, item: image)
    answers = pd.DataFrame({"id": ["2", "1", "0"], "margin": [4.0, 0.0, 1.0]})
    frame = encoder_regions_task(
        subject,
        "pope-adversarial",
        [(image, boxes)],
        ["dog", "cat"],
        answers,
        {"image": (28, 28, boxes)},
        per_category=2,
    )
    assert set(frame["source"]) == {"encoder", "projector"}
    assert len(frame) == 4
    assert set(frame["model"]) == {"tiny-llava"}
    assert set(frame["task"]) == {"pope-adversarial"}
    assert frame["correct_yes"].tolist() == [1] * 4
    assert frame["missed"].tolist() == [1] * 4
    assert frame["boxes"].tolist() == [21] * 4
    assert not frame.isna().any().any()

    assert calls == [subject]
    assert all(p.device == original for p in subject.model.parameters())


@pytest.mark.parametrize("rows, columns", [(2, 2), (4, 6), (6, 4)])
def test_qwen_region_tokens_match_every_block(rows: int, columns: int) -> None:
    from tiny import tiny_qwen

    from swce.diagnostics.attention import ResizedGrid
    from swce.diagnostics.encoder import _image_features, region_features

    subject = tiny_qwen()
    inputs = tiny_inputs(subject)
    inputs["image_grid_thw"] = torch.tensor([[1, rows, columns]])
    inputs["pixel_values"] = torch.randn(rows * columns, 3 * 2 * 14 * 14)
    grid = ResizedGrid.of(subject, inputs)
    tokens, projected_tokens = _image_features(subject, inputs, lambda state, source: state)
    cells = grid.cells(columns * 14, rows * 14, [0, 0, columns * 14, rows * 14])
    pooled, projected = region_features(subject, inputs, [cells], grid)
    assert tokens.shape[:2] == (3, rows * columns)
    torch.testing.assert_close(pooled[0], tokens.mean(dim=1))
    assert projected is not None and projected_tokens is not None
    assert projected_tokens.shape[0] == rows * columns // 4
    torch.testing.assert_close(projected[0], projected_tokens.mean(dim=0))


def test_merged_region_indices_follow_processor_order() -> None:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    from swce.diagnostics.encoder import _merged_region_indices

    pixels = torch.arange(24).reshape(1, 1, 4, 6)
    patches, rows, columns = Qwen2VLImageProcessor().patchify(pixels, 1, 2, 1)
    assert (rows, columns) == (4, 6)
    assert _merged_region_indices([[1], [3, 5]], 2, 0) == [
        [4, 5, 6, 7],
        [12, 13, 14, 15, 20, 21, 22, 23],
    ]
    assert patches[0, [4, 5, 6, 7], 0].tolist() == [2, 3, 8, 9]


def test_qwen_one_cell_pools_constructed_patch_tokens() -> None:
    from tiny import tiny_qwen

    from swce.diagnostics.attention import ResizedGrid
    from swce.diagnostics.encoder import region_features

    subject = tiny_qwen()
    inputs = tiny_inputs(subject)
    inputs["image_grid_thw"] = torch.tensor([[1, 4, 6]])
    inputs["pixel_values"] = torch.zeros(24, 3 * 2 * 14 * 14)
    state = torch.arange(24 * 16, dtype=torch.float32).reshape(24, 16)
    projected_state = torch.arange(6 * 16, dtype=torch.float32).reshape(6, 16)
    handles = [
        block.register_forward_hook(lambda _module, _args, _output: state)
        for block in subject.encoder_layers()
    ]
    projector = subject.projector()
    assert projector is not None
    handles.append(projector.register_forward_hook(lambda _module, _args, _output: projected_state))
    try:
        grid = ResizedGrid.of(subject, inputs)
        cells = grid.cells(84, 56, [28, 0, 28, 28])
        assert cells == [1]
        pooled, projected = region_features(subject, inputs, [cells], grid)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(pooled[0], state[[4, 5, 6, 7]].mean(dim=0).expand(3, -1))
    assert projected is not None
    torch.testing.assert_close(projected[0], projected_state[1])


def test_rice_region_mapping_excludes_the_leading_class_token() -> None:
    from types import SimpleNamespace

    from swce.diagnostics.attention import ResizedGrid
    from swce.diagnostics.encoder import _merged_region_indices, _resized_region_offset

    tower: Any = torch.nn.Module()
    tower.blocks = torch.nn.ModuleList([torch.nn.Identity()])
    tower.merger = torch.nn.Identity()
    tower.patch_embed = torch.nn.Identity()
    tower.pre_layernorm = torch.nn.Identity()
    tower.class_embedding = torch.nn.Parameter(torch.zeros(16))
    tower.class_pos_emb = torch.nn.Parameter(torch.zeros(1, 4))
    tower.spatial_merge_size = 2
    tower.patch_size = 14
    tower.window_size = None
    subject: Any = SimpleNamespace(
        model=tower, projector=lambda: tower.merger, encoder_layers=lambda: list(tower.blocks)
    )
    grid = ResizedGrid(4, 6, 14, 2)
    offset = _resized_region_offset(subject, {"image_grid_thw": torch.tensor([[1, 4, 6]])}, grid)
    assert offset == 1
    indices = _merged_region_indices([[1], list(range(6))], grid.merge, offset)
    assert indices[0] == [5, 6, 7, 8]
    assert indices[1] == list(range(1, 25))
    state = torch.arange(25 * 16, dtype=torch.float32).reshape(25, 16)
    torch.testing.assert_close(state[indices[1]].mean(dim=0), state[1:].mean(dim=0))
    tower.window_size = 4
    with pytest.raises(ValueError, match="unsupported merged"):
        _resized_region_offset(subject, {"image_grid_thw": torch.tensor([[1, 4, 6]])}, grid)


@pytest.mark.parametrize("shape", [[[2, 2, 2]], [[1, 4, 2]], [[1, 2, 2], [1, 2, 2]]])
def test_resized_regions_reject_mismatched_or_temporal_grids(shape: list[list[int]]) -> None:
    from tiny import tiny_qwen

    from swce.diagnostics.attention import ResizedGrid
    from swce.diagnostics.encoder import region_features

    subject = tiny_qwen()
    inputs = dict(tiny_inputs(subject), image_grid_thw=torch.tensor(shape))
    with pytest.raises(ValueError, match="one still image"):
        region_features(subject, inputs, [[0]], ResizedGrid(2, 2, 14, 2))


@pytest.mark.parametrize("source", ["encoder", "projector"])
def test_resized_regions_reject_unexpected_token_counts(source: str) -> None:
    from tiny import tiny_qwen

    from swce.diagnostics.attention import ResizedGrid
    from swce.diagnostics.encoder import region_features

    subject = tiny_qwen()
    module = subject.encoder_layers()[0] if source == "encoder" else subject.projector()
    assert module is not None
    before = dict(module._forward_hooks)
    handle = module.register_forward_hook(lambda _module, _args, output: output[:-1])
    try:
        with pytest.raises(ValueError, match=f"Cannot map {source} tokens"):
            region_features(subject, tiny_inputs(subject), [[0]], ResizedGrid(2, 2, 14, 2))
    finally:
        handle.remove()
    assert dict(module._forward_hooks) == before


@pytest.mark.parametrize("cells", [[], [[]], [[-1]], [[1]]])
def test_resized_regions_reject_cells_outside_grid(cells: list[list[int]]) -> None:
    from tiny import tiny_qwen

    from swce.diagnostics.attention import ResizedGrid
    from swce.diagnostics.encoder import region_features

    subject = tiny_qwen()
    with pytest.raises(ValueError, match="Region cells"):
        region_features(subject, tiny_inputs(subject), cells, ResizedGrid(2, 2, 14, 2))


def test_resized_regions_reject_unknown_tower_and_indivisible_grid() -> None:
    from tiny import tiny_qwen

    from swce.diagnostics.attention import ResizedGrid
    from swce.diagnostics.encoder import region_features

    subject = tiny_qwen()
    inputs = dict(tiny_inputs(subject), image_grid_thw=torch.tensor([[1, 3, 2]]))
    with pytest.raises(ValueError, match="divisible"):
        region_features(subject, inputs, [[0]], ResizedGrid(3, 2, 14, 2))
    with pytest.raises(ValueError, match="unsupported merged"):
        region_features(subject, tiny_inputs(subject), [[0]], ResizedGrid(2, 2, 14, 1))


def test_read_regions_caps_each_category_deterministically(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections import Counter

    from swce.diagnostics.attention import Grid
    from swce.diagnostics.encoder import read_regions

    subject = tiny_llava()
    inputs = tiny_inputs(subject)
    monkeypatch.setattr(HuggingFaceSubject, "template", lambda self, prompt: "template")
    monkeypatch.setattr(subject, "inputs", lambda image, text: inputs)
    monkeypatch.setattr(Grid, "of", classmethod(lambda cls, subject: Grid(28, 28, 14)))
    image = Image.new("RGB", (28, 28))
    boxes = [("dog", [0.0, 0.0, 14.0, 14.0]), ("cat", [0.0, 0.0, 28.0, 28.0])]
    training = [(image, boxes)] * 5
    first, regions = read_regions(subject, training, "train", per_category=2)
    second, repeated = read_regions(subject, training, "train", per_category=2)
    assert regions == repeated
    assert Counter(name for _, name, _ in regions) == {"dog": 2, "cat": 2}
    assert [index for index, _, _ in regions] == [0, 0, 1, 1]
    for source in first:
        np.testing.assert_array_equal(first[source], second[source])
        assert first[source].shape[0] == 4
    _, uncapped = read_regions(subject, training, "evaluation")
    assert len(uncapped) == 10
