import importlib
from collections.abc import Callable, Iterator

import pytest
import torch
from tiny import logits, tiny_llava, tiny_qwen

from swce.evaluation.runner import environment
from swce.models import runtime
from swce.models.hf import HuggingFaceSubject


@pytest.fixture(autouse=True)
def restore_policy() -> Iterator[None]:
    before = runtime.attention_settings()["cudnn_sdp_enabled"]
    try:
        yield
    finally:
        torch.backends.cuda.enable_cudnn_sdp(before)


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("build", [tiny_llava, tiny_qwen])
def test_constructor_changes_only_cudnn_sdpa_and_persists(
    initial: bool, build: Callable[[], HuggingFaceSubject]
) -> None:
    torch.backends.cuda.enable_cudnn_sdp(initial)
    before = runtime.attention_settings()
    subject = build()
    expected = {**before, "cudnn_sdp_enabled": False}
    assert runtime.attention_settings() == expected
    assert subject.settings()["cudnn_sdp_enabled"] is False
    # Cover a real forward after construction, not just the constructor's scope.
    assert torch.isfinite(logits(subject)).all()
    assert runtime.attention_settings() == expected
    build()
    assert runtime.attention_settings() == expected


def test_import_and_snapshot_do_not_apply_policy() -> None:
    torch.backends.cuda.enable_cudnn_sdp(True)
    before = runtime.attention_settings()
    importlib.reload(runtime)
    assert runtime.attention_settings() == before


@pytest.mark.parametrize("enabled", [False, True])
def test_settings_and_run_metadata_read_actual_flag_without_changing_it(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    subject = tiny_qwen()
    generation = subject.settings()["generation"]
    torch.backends.cuda.enable_cudnn_sdp(enabled)
    before = runtime.attention_settings()
    rng = torch.get_rng_state().clone()
    settings = subject.settings()
    assert settings["cudnn_sdp_enabled"] is enabled
    assert settings["generation"] == generation
    # The evaluation runner merges these settings into every run's environment.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert environment(subject)["cudnn_sdp_enabled"] is enabled
    assert runtime.attention_settings() == before
    assert torch.equal(torch.get_rng_state(), rng)
