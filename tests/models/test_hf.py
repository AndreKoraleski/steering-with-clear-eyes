import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tiny import WIDTH, inputs, logits, tiny_llava, tiny_qwen
from torch import Tensor, nn

from swce.models.hf import (
    LLAVA_15,
    ONEVISION_15,
    QWEN3_VL,
    HuggingFaceSubject,
    default_rope_parameters,
    module_list,
    onevision_buffers,
    onevision_code,
    onevision_config,
    require_every_weight,
    require_finite_buffers,
)
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


def test_the_original_rotary_frequencies_are_those_of_transformers_4_53() -> None:
    config = SimpleNamespace(rope_theta=10000.0, head_dim=8, hidden_size=32, num_attention_heads=2)
    frequencies, scaling = default_rope_parameters(config)
    expected = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
    assert torch.allclose(frequencies, expected)
    assert scaling == 1.0
    halved = SimpleNamespace(
        rope_theta=10000.0,
        head_dim=None,
        hidden_size=32,
        num_attention_heads=2,
        partial_rotary_factor=0.5,
    )
    assert default_rope_parameters(halved)[0].shape == (4,)


def test_the_original_rotary_embedding_exists_only_while_the_model_is_built() -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    before = dict(ROPE_INIT_FUNCTIONS)
    with onevision_code():
        assert ROPE_INIT_FUNCTIONS["default"] is (before.get("default") or default_rope_parameters)
    assert before == ROPE_INIT_FUNCTIONS
    with pytest.raises(RuntimeError), onevision_code():
        raise RuntimeError
    assert before == ROPE_INIT_FUNCTIONS


def test_the_language_configuration_gets_a_padding_index_only_when_missing() -> None:
    missing = SimpleNamespace(text_config=SimpleNamespace())
    assert onevision_config(missing).text_config.pad_token_id is None
    declared = SimpleNamespace(text_config=SimpleNamespace(pad_token_id=7))
    assert onevision_config(declared).text_config.pad_token_id == 7


class StrictForward(nn.Module):
    """A model whose forward pass names its arguments and takes no other, like old model code."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: Tensor, pixel_values: Tensor) -> Tensor:
        return input_ids


def test_inputs_a_strict_model_does_not_take_are_left_out() -> None:
    subject = tiny_llava()
    encoded = dict(inputs(subject))
    encoded["mm_token_type_ids"] = torch.zeros_like(encoded["input_ids"])
    assert set(subject.moved(encoded)) == set(encoded)
    subject.model = StrictForward()
    assert subject.accepts("input_ids")
    assert not subject.accepts("mm_token_type_ids")
    assert set(subject.moved(encoded)) == {"input_ids", "pixel_values"}


def test_a_model_missing_weights_is_refused() -> None:
    require_every_weight(
        {"missing_keys": [], "unexpected_keys": set(), "mismatched_keys": []}, "ok"
    )
    with pytest.raises(RuntimeError, match="did not load"):
        require_every_weight({"missing_keys": ["model.language_model.norm.weight"]}, "broken")
    with pytest.raises(RuntimeError, match="unexpected_keys"):
        require_every_weight({"unexpected_keys": {"model.norm.weight"}}, "broken")


def test_a_model_with_a_buffer_that_is_not_finite_is_refused() -> None:
    model = nn.Module()
    model.register_buffer("inv_freq", torch.ones(4), persistent=False)
    require_finite_buffers(model, "ok")
    model.get_buffer("inv_freq")[1] = float("nan")
    with pytest.raises(RuntimeError, match="inv_freq"):
        require_finite_buffers(model, "broken")


class RiceRotaryEmbedding(nn.Module):
    """Stand in named like the rotary embedding of the vision encoder of LLaVA-OneVision-1.5."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", frequencies, persistent=False)


class LLaVAOneVision1_5_RotaryEmbedding(nn.Module):  # noqa: N801
    """Stand in named like the rotary embedding of the language model of LLaVA-OneVision-1.5."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        frequencies, self.attention_scaling = default_rope_parameters(config)
        self.register_buffer("inv_freq", frequencies, persistent=False)
        self.original_inv_freq = self.inv_freq


def test_rotary_frequencies_left_uninitialized_are_computed_again() -> None:
    config = SimpleNamespace(rope_theta=1e6, head_dim=8, hidden_size=16, num_attention_heads=2)
    model = nn.ModuleList([LLaVAOneVision1_5_RotaryEmbedding(config), RiceRotaryEmbedding(8)])
    expected = [module.get_buffer("inv_freq").clone() for module in model]
    for module in model:
        module.get_buffer("inv_freq").fill_(float("nan"))
    onevision_buffers(model)
    for module, frequencies in zip(model, expected, strict=True):
        assert torch.equal(module.get_buffer("inv_freq"), frequencies)
    assert model[0].original_inv_freq is model[0].get_buffer("inv_freq")
