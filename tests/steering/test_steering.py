from contextlib import AbstractContextManager
from pathlib import Path

import pytest
import torch
from support import ToySubject
from torch import nn

from swce.baselines import RandomDirection, Unmodified
from swce.evaluation.__main__ import load_factory
from swce.models.subject import Subject
from swce.steering import Steerer, add_vector, edit_layers

TEMPLATE = Path(__file__).parents[2] / "methods" / "template.py"


def test_hooks_edit_the_stream_and_leave_no_trace() -> None:
    subject = ToySubject()
    before = subject.hidden("prompt")
    vector = torch.ones(subject.width)
    with edit_layers(subject.decoder_layers(), {2: add_vector(vector, 2.0)}):
        steered = subject.hidden("prompt")
    assert torch.allclose(steered, before + 2.0)
    assert torch.equal(subject.hidden("prompt"), before)


def test_hooks_are_removed_after_an_error() -> None:
    subject = ToySubject()
    before = subject.hidden("prompt")
    with (
        pytest.raises(RuntimeError),
        edit_layers(subject.decoder_layers(), {0: add_vector(torch.ones(subject.width))}),
    ):
        raise RuntimeError
    assert torch.equal(subject.hidden("prompt"), before)
    with (
        pytest.raises(IndexError),
        edit_layers(subject.decoder_layers(), {3: lambda hidden: hidden}),
    ):
        pass


def test_hooks_edit_the_first_element_of_a_tuple() -> None:
    class Block(nn.Module):
        def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
            return hidden, "cache"

    block = Block()
    with edit_layers([block], {0: add_vector(torch.ones(2))}):
        hidden, cache = block(torch.zeros(1, 1, 2))
    assert torch.equal(hidden, torch.ones(1, 1, 2))
    assert cache == "cache"


def test_unmodified_changes_nothing() -> None:
    subject = ToySubject()
    before = subject.hidden("prompt")
    with Unmodified().attach(subject):
        assert torch.equal(subject.hidden("prompt"), before)
    assert Unmodified().fitted_on == ()


def test_random_direction_matches_the_norms_and_the_seed() -> None:
    control = RandomDirection(norms={0: 2.0, 2: 0.5}, width=8, seed=1)
    vectors = control.vectors()
    assert vectors[0].norm().item() == pytest.approx(2.0, rel=1e-5)
    assert vectors[2].norm().item() == pytest.approx(0.5, rel=1e-5)
    assert torch.equal(vectors[0], control.vectors()[0])
    other = RandomDirection(norms={0: 2.0, 2: 0.5}, width=8, seed=2)
    assert not torch.equal(vectors[0], other.vectors()[0])
    assert control.fingerprint != other.fingerprint
    subject = ToySubject()
    before = subject.hidden("prompt")
    with control.attach(subject):
        assert not torch.equal(subject.hidden("prompt"), before)
    assert torch.equal(subject.hidden("prompt"), before)


def test_fingerprint_covers_the_name_and_the_fields() -> None:
    class Other(Steerer):
        name = "other"
        description = "Another method."

        def attach(self, subject: Subject) -> AbstractContextManager[None]:
            raise NotImplementedError

    assert Unmodified().fingerprint == Unmodified().fingerprint
    assert Unmodified().fingerprint != Other(fitted_on=()).fingerprint
    assert Other(setting="pope").fingerprint != Other(setting="chair").fingerprint
    assert Other().fitted_on is None


def test_template_refuses_to_run_without_its_vectors(tmp_path: Path) -> None:
    steerers = load_factory(f"{TEMPLATE}:build")()
    assert [steerer.setting for steerer in steerers] == ["chair", "pope"]
    subject = ToySubject()
    with pytest.raises(FileNotFoundError):
        steerers[0].attach(subject)
    vectors = tmp_path / "vectors.pt"
    torch.save({1: torch.ones(subject.width)}, vectors)
    steerer = steerers[0].model_copy(update={"vectors": vectors})
    before = subject.hidden("prompt")
    with steerer.attach(subject):
        assert not torch.equal(subject.hidden("prompt"), before)
