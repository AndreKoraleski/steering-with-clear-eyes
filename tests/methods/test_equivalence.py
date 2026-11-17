"""Numerical equivalence of the ports with the official code of every method.

These tests import the official repositories, which are never part of this one. They run when
`SWCE_OFFICIAL` names a folder holding the clones `VTI`, `VISTA` and `Revis` at the commits the
method files record, and are skipped otherwise.
"""

import ast
import copy
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from tiny import WIDTH, logits, tiny_llava

METHODS = Path(__file__).parents[2] / "methods"
OFFICIAL = os.environ.get("SWCE_OFFICIAL")
needs_official = pytest.mark.skipif(OFFICIAL is None, reason="SWCE_OFFICIAL is not set")
STUBS = (
    "torchvision",
    "torchvision.transforms",
    "shortuuid",
    "qwen_vl_utils",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
)
"""Modules the official code imports and the compared functions never use."""


def port(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, METHODS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_file(repository: str, relative: str, name: str) -> ModuleType:
    assert OFFICIAL is not None
    spec = importlib.util.spec_from_file_location(name, Path(OFFICIAL) / repository / relative)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_package(repository: str, module: str) -> ModuleType:
    assert OFFICIAL is not None
    root = str(Path(OFFICIAL) / repository)
    for stub in STUBS:
        sys.modules.setdefault(stub, MagicMock())
    sys.path.insert(0, root)
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(root)


def official_function(repository: str, relative: str, name: str, **names: Any) -> Any:
    """One function of an official script, without running the imports of the script.

    The calibration script of REVIS imports a class that later versions of `transformers`
    removed, so the script cannot be imported as a whole.
    """
    assert OFFICIAL is not None
    path = Path(OFFICIAL) / repository / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    namespace: dict[str, Any] = dict(names)
    exec(compile(ast.Module(body=found, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@needs_official
def test_vti_layer() -> None:
    theirs = official_file("VTI", "vti_utils/llm_layers.py", "official_vti_layers")
    ours = port("vti")
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, WIDTH).half()
    textual = torch.randn(WIDTH)
    visual = torch.randn(5, WIDTH)
    for direction in (textual, visual):
        expected = theirs.VTILayer(direction[None], [0.9])(hidden.clone())
        assert torch.equal(ours.shift(direction, 0.9)(hidden.clone()), expected)
    single = hidden[:, :1]
    expected = theirs.VTILayer(textual[None], [0.4])(single.clone())
    assert torch.equal(ours.shift(textual, 0.4)(single.clone()), expected)


@needs_official
def test_vti_on_the_llava_architecture() -> None:
    theirs = official_file("VTI", "vti_utils/llm_layers.py", "official_vti_layers")
    ours = port("vti")
    subject = tiny_llava()
    subject.model.half()
    torch.manual_seed(0)
    textual = torch.randn(4, WIDTH)
    visual = torch.randn(3, 5, WIDTH)
    reference = copy.deepcopy(subject.model)
    theirs.add_vti_layers(reference, torch.stack([textual], dim=1), alpha=[0.9])
    theirs.add_vti_layers(
        reference.get_submodule("model.vision_tower"), torch.stack([visual], dim=1), alpha=[0.9]
    )
    edits_text = {index: ours.shift(d, 0.9) for index, d in enumerate(textual)}
    edits_vision = {index: ours.shift(d, 0.9) for index, d in enumerate(visual)}
    from swce.steering import edit_layers

    plain = logits(subject)
    with (
        edit_layers(subject.decoder_mlps(), edits_text),
        edit_layers(subject.encoder_mlps(), edits_vision),
    ):
        steered = logits(subject)
    original = subject.model
    subject.model = reference
    expected = logits(subject)
    subject.model = original
    assert not torch.equal(steered, plain)
    assert torch.equal(steered, expected)


@needs_official
def test_vti_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    theirs = official_package("VTI", "vti_utils.utils")
    ours = port("vti")
    torch.manual_seed(0)
    demonstrations, layers, tokens = 6, 4, 5
    clean = torch.randn(demonstrations, layers, tokens, WIDTH)
    masked = torch.randn(demonstrations, layers, tokens, WIDTH)
    pairs = [(masked[index], clean[index]) for index in range(demonstrations)]
    monkeypatch.setattr(theirs, "get_visual_hiddenstates", lambda *_, **__: pairs)
    expected, _ = theirs.obtain_visual_vti(None, None, rank=1)
    assert torch.allclose(ours.visual_directions(clean, masked), expected[1:], atol=1e-6)
    faithful = torch.randn(demonstrations, layers, WIDTH)
    hallucinated = torch.randn(demonstrations, layers, WIDTH)
    texts = [(hallucinated[index], faithful[index]) for index in range(demonstrations)]
    monkeypatch.setattr(theirs, "get_hiddenstates", lambda *_: texts)
    expected, _ = theirs.obtain_textual_vti(None, None, None, rank=1)
    assert torch.allclose(ours.textual_directions(faithful, hallucinated), expected[1:], atol=1e-6)
    pixels = torch.randn(3, 28, 28)
    chosen = torch.tensor([0, 3])
    assert torch.equal(ours.mask_patches(pixels, chosen), theirs.mask_patches(pixels, chosen))


@needs_official
def test_vista_layer_and_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    layers = official_package("VISTA", "llm_layers")
    vectors = official_package("VISTA", "steering_vector")
    ours = port("vista")
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, WIDTH).half()
    vector = torch.randn(WIDTH)
    for strength in (0.17, 0.01):
        expected = layers.VSVLayer(vector[None], [strength])(hidden.clone())
        assert torch.equal(ours.steer(vector, strength)(hidden.clone()), expected)
    with_image, without_image = torch.randn(4, WIDTH), torch.randn(4, WIDTH)
    monkeypatch.setattr(vectors, "get_hiddenstates", lambda *_: [(without_image, with_image)])
    expected, _ = vectors.obtain_vsv(None, None, [[None, None]], rank=1)
    assert torch.equal(ours.steering_vectors(with_image, without_image), expected)


@needs_official
def test_vista_reads_the_same_states_as_the_official_tracer() -> None:
    vectors = official_package("VISTA", "steering_vector")
    subject = tiny_llava()
    with_image = {
        "input_ids": torch.tensor([[1, 59, 59, 59, 59, 5, 6]]),
        "pixel_values": torch.zeros(1, 3, 28, 28),
    }
    without_image = {"input_ids": torch.tensor([[1, 5, 6]])}
    with torch.no_grad():
        ((expected_without, expected_with),) = vectors.get_hiddenstates(
            subject.model, [[without_image, with_image]]
        )
    assert torch.equal(subject.block_outputs(with_image), expected_with)
    assert torch.equal(subject.block_outputs(without_image), expected_without)
    states = subject.all_states(with_image)[1:, -1]
    assert torch.equal(states[:-1], expected_with[:-1])
    assert not torch.allclose(states[-1], expected_with[-1])


@needs_official
def test_vista_on_the_llava_architecture() -> None:
    layers = official_package("VISTA", "llm_layers")
    ours = port("vista")
    subject = tiny_llava()
    subject.model.half()
    torch.manual_seed(0)
    found = torch.randn(4, WIDTH)
    reference = copy.deepcopy(subject.model)
    layers.add_vsv_layers(
        reference.get_submodule("model.language_model"), torch.stack([found], dim=1), [0.17]
    )
    from swce.steering import edit_layers

    edits = {index: ours.steer(vector, 0.17) for index, vector in enumerate(found)}
    with edit_layers(subject.decoder_mlps(), edits):
        steered = logits(subject)
    original = subject.model
    subject.model = reference
    expected = logits(subject)
    subject.model = original
    assert torch.equal(steered, expected)


@needs_official
def test_revis_hook() -> None:
    theirs = official_file("Revis", "steering_utils/llm_layers.py", "official_revis_layers")
    ours = port("revis")
    torch.manual_seed(0)
    hidden = torch.randn(2, 7, WIDTH).half()
    vector = torch.randn(WIDTH) * 20
    for tau in (-0.5, 0.0, 0.2):
        hook = theirs.MADRSteeringHook_visual_only(
            vectors={"visual": vector}, layer_idx=0, alpha_vis=1.1, tau_low=tau, risk_gamma=1.0
        )
        (expected, extra) = hook.hook_fn(None, None, (hidden.clone(), "rest"))
        assert extra == "rest"
        assert torch.equal(ours.gate(vector, 1.1, tau, 1.0)(hidden.clone()), expected)


@needs_official
def test_revis_principal_direction() -> None:
    theirs = official_package("Revis", "steering_utils.vector_calculator")
    ours = port("revis")
    torch.manual_seed(0)
    samples = torch.randn(40, WIDTH) * torch.linspace(5, 0.1, WIDTH) + 3.0
    for stack in (samples, samples[:1], torch.ones(5, WIDTH)):
        expected = theirs.get_pure_direction_with_pca(stack)
        assert torch.equal(ours.principal(stack), expected)


@needs_official
def test_revis_calibration_of_one_layer(tmp_path: Path) -> None:
    calibration = official_function(
        "Revis",
        "layer_select.py",
        "process_layer_calibration",
        np=np,
        os=os,
        plt=MagicMock(),
        sns=MagicMock(),
    )
    ours = port("revis")
    torch.manual_seed(0)
    categories = ["TP", "TN", "FP", "Other"] * 10
    risks = [
        float(value) + (0.5 if category == "FP" else 0.0)
        for value, category in zip(torch.randn(40).tolist(), categories, strict=True)
    ]
    records = [
        {"risk": risk, "category": category}
        for risk, category in zip(risks, categories, strict=True)
    ]
    for percentile in (85.0, 90.0):
        expected = calibration(records, 20, str(tmp_path), percentile)
        assert expected["separability"] > 0
        found = ours.select({20: risks}, categories, percentile=percentile, min_layer=10)
        assert found == (20, pytest.approx(float(expected["tau"])))


def test_nullu_projection_equals_the_official_weight_edit() -> None:
    """The official edit of the output projection, written out, against the hook of the port."""
    ours = port("nullu")
    subject = tiny_llava()
    torch.manual_seed(0)
    bases = {layer: torch.linalg.qr(torch.randn(WIDTH, 4)).Q for layer in (2, 3)}
    reference = copy.deepcopy(subject.model)
    state = reference.state_dict()
    for key in list(state):
        # The conditions and the product below are those of `HalluEdit.edit_model`.
        if "weight" in key and "mlp" in key and "vision_tower" not in key and "down_proj" in key:
            layer = int(key.split("language_model.layers.")[1].split(".")[0])
            if layer in bases:
                null_space = torch.eye(WIDTH) - bases[layer] @ bases[layer].T
                state[key] = (state[key].T @ null_space).T.contiguous()
    reference.load_state_dict(state)
    from swce.steering import edit_layers

    edits = {layer: ours.project_out(basis) for layer, basis in bases.items()}
    with edit_layers(subject.decoder_mlps(), edits):
        steered = logits(subject)
    original = subject.model
    subject.model = reference
    expected = logits(subject)
    subject.model = original
    assert not torch.allclose(steered, logits(subject))
    assert torch.allclose(steered, expected, atol=1e-5)
