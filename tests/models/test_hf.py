import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import torch
from tiny import WIDTH, logits, tiny_llava, tiny_qwen
from torch import nn

from swce.models.hf import LLAVA_15, ONEVISION_15, QWEN3_VL, HuggingFaceSubject, module_list
from swce.steering import InapplicableError, add_vector, edit_layers

METHODS = Path(__file__).parents[2] / "methods"
SUBJECTS = [tiny_llava, tiny_qwen]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, METHODS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("build", SUBJECTS)
def test_blocks_are_found_by_name(build: Callable[[], HuggingFaceSubject]) -> None:
    subject = build()
    assert len(subject.decoder_layers()) == 4
    assert len(subject.decoder_mlps()) == 4
    assert len(subject.encoder_mlps()) == 3
    head = subject.unembedding()
    assert isinstance(head, nn.Linear)
    assert head.weight.shape == (60, WIDTH)
    with pytest.raises(ValueError, match="no list of blocks"):
        module_list(subject.model, "missing")


@pytest.mark.parametrize("build", SUBJECTS)
def test_hooks_steer_every_tower_and_leave_no_trace(
    build: Callable[[], HuggingFaceSubject],
) -> None:
    subject = build()
    before = logits(subject)
    vector = torch.ones(WIDTH)
    for modules in (subject.decoder_layers(), subject.decoder_mlps(), subject.encoder_mlps()):
        with edit_layers(modules, {0: add_vector(vector, 3.0)}):
            assert not torch.allclose(logits(subject), before)
        assert torch.equal(logits(subject), before)


@pytest.mark.parametrize("build", SUBJECTS)
def test_fixed_methods_attach_to_every_architecture(
    build: Callable[[], HuggingFaceSubject], tmp_path: Path
) -> None:
    subject = build()
    before = logits(subject)
    torch.manual_seed(1)
    revis, nullu, vti = load("revis"), load("nullu"), load("vti")
    vectors = tmp_path / "vectors.pt"
    torch.save({index: {"visual": torch.randn(WIDTH)} for index in range(5)}, vectors)
    subspaces = tmp_path / "subspaces.pt"
    torch.save({2: torch.linalg.qr(torch.randn(WIDTH, 4)).Q, 3: torch.eye(WIDTH)[:, :4]}, subspaces)
    directions = tmp_path / "directions.pt"
    torch.save({"visual": torch.randn(3, WIDTH), "textual": torch.randn(4, WIDTH)}, directions)
    steerers = [
        revis.Revis(layer=2, strength=1.1, tau=-2.0, vectors=vectors),
        nullu.Nullu(first_layer=2, last_layer=3, subspaces=subspaces),
        vti.Vti(visual=0.9, textual=0.9, directions=directions),
    ]
    for steerer in steerers:
        with steerer.attach(subject):
            assert not torch.allclose(logits(subject), before), steerer.name
        assert torch.equal(logits(subject), before), steerer.name


def test_vti_reports_a_number_of_blocks_it_was_not_fitted_for(tmp_path: Path) -> None:
    vti = load("vti")
    directions = tmp_path / "directions.pt"
    torch.save({"visual": torch.randn(3, WIDTH), "textual": torch.randn(32, WIDTH)}, directions)
    steerer = vti.Vti(visual=0.0, textual=0.9, directions=directions)
    with pytest.raises(InapplicableError, match="32 textual directions for 4 blocks"):
        steerer.attach(tiny_qwen()).__enter__()


def test_vti_cannot_be_fitted_without_a_fixed_number_of_image_tokens(tmp_path: Path) -> None:
    vti = load("vti")
    with pytest.raises(InapplicableError, match="its own number of tokens"):
        vti.fit(tiny_qwen(), tmp_path)


def test_checkpoints_are_pinned_by_full_revisions() -> None:
    for checkpoint, revision in (LLAVA_15, QWEN3_VL, ONEVISION_15):
        assert "/" in checkpoint
        assert len(revision) == 40
