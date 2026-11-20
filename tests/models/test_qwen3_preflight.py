import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def preflight() -> ModuleType:
    path = Path(__file__).parents[2] / "experiments" / "qwen3_preflight.py"
    spec = importlib.util.spec_from_file_location("qwen3_preflight", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def answers(count: int = 40) -> list[dict[str, object]]:
    return [
        {
            "id": str(index),
            "image": "coco.jpg",
            "prompt": "Is there a cat?",
            "text": "Yes.",
            "tokens": 3,
        }
        for index in range(count)
    ]


def test_exactly_forty_identical_answers_pass(preflight: ModuleType) -> None:
    assert preflight.compare_answers(answers(), answers()) == []


@pytest.mark.parametrize("count", [0, 39, 41])
def test_an_incomplete_or_oversized_sample_is_refused(preflight: ModuleType, count: int) -> None:
    with pytest.raises(ValueError, match="exactly 40"):
        preflight.compare_answers(answers(count), answers())
    with pytest.raises(ValueError, match="exactly 40"):
        preflight.compare_answers(answers(), answers(count))


def test_duplicate_questions_are_refused(preflight: ModuleType) -> None:
    actual = answers()
    actual[-1] = actual[0].copy()
    with pytest.raises(ValueError, match="distinct POPE questions"):
        preflight.compare_answers(answers(), actual)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "other"),
        ("image", "other.jpg"),
        ("prompt", "Other?"),
        ("text", "No."),
        ("tokens", 4),
    ],
)
def test_each_changed_field_is_a_mismatch(preflight: ModuleType, field: str, value: object) -> None:
    actual = answers()
    actual[0][field] = value
    assert preflight.compare_answers(answers(), actual) == ["0"]


def test_reordered_questions_do_not_pass(preflight: ModuleType) -> None:
    actual = answers()
    actual[0], actual[1] = actual[1], actual[0]
    assert preflight.compare_answers(answers(), actual) == ["0", "1"]


def test_sampling_defaults_are_checked_not_overridden(preflight: ModuleType) -> None:
    settings = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
    before = dict(settings)
    preflight.check_sampling(settings)
    assert settings == before
    for key in settings:
        changed = {**settings, key: None}
        with pytest.raises(RuntimeError, match="Unexpected checkpoint generation"):
            preflight.check_sampling(changed)


def test_reports_are_json_and_refuse_nonfinite_values(
    preflight: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    preflight.write_json(path, {"status": "passed", "questions": 40})
    assert json.loads(path.read_text()) == {"status": "passed", "questions": 40}
    with pytest.raises(ValueError):
        preflight.write_json(path, {"invalid": float("nan")})


def test_native_reference_applies_policy_before_loading(
    preflight: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers import AutoProcessor

    events: list[str] = []

    def configure(enabled: bool) -> None:
        assert enabled is False
        events.append("runtime")

    monkeypatch.setattr(preflight.torch.backends.cuda, "enable_cudnn_sdp", configure)

    def stop_loading(*args: Any, **kwargs: Any) -> None:
        assert events == ["runtime"]
        raise RuntimeError("stop before any download or GPU work")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", stop_loading)
    with pytest.raises(RuntimeError, match="stop before any download"):
        preflight.reference_answers(None, [])


@pytest.mark.parametrize("outcome", ["passed", "text", "generation", "attention", "resume"])
def test_real_orchestration_preserves_strict_gate_and_records_policy(
    preflight: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str
) -> None:
    flags = {"cudnn_sdp_enabled": True}
    calls: list[str] = []

    def configure(enabled: bool) -> None:
        assert enabled is False
        flags["cudnn_sdp_enabled"] = enabled
        calls.append("configure")

    def environment() -> dict[str, object]:
        assert calls == ["configure"]
        assert not flags["cudnn_sdp_enabled"]
        return {"attention": dict(flags)}

    def reference(*args: Any) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert not flags["cudnn_sdp_enabled"]
        calls.append("reference")
        return answers(), {"do_sample": True}

    def subject(*args: Any) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert calls[-1] == "reference" and not flags["cudnn_sdp_enabled"]
        calls.append("subject")
        rows = answers()
        if outcome == "text":
            rows[0]["text"] = "Yes, different full answer."
        if outcome == "attention":
            flags["cudnn_sdp_enabled"] = True
        return rows, {"do_sample": outcome != "generation"}

    monkeypatch.setattr(preflight.torch.backends.cuda, "enable_cudnn_sdp", configure)
    monkeypatch.setattr(preflight, "checked_environment", environment)
    monkeypatch.setattr(preflight, "attention_settings", lambda: dict(flags))
    monkeypatch.setattr(preflight, "reference_answers", reference)
    monkeypatch.setattr(preflight, "subject_answers", subject)
    monkeypatch.setattr(preflight.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        preflight, "Task", lambda *args: SimpleNamespace(items=answers, fingerprint="test")
    )
    if outcome == "resume":
        preflight.write_json(tmp_path / "environment.json", {"attention": dict(flags)})
    argv = ["--data", str(tmp_path / "data"), "--results", str(tmp_path)]
    if outcome == "passed":
        assert preflight.main(argv) == 0
    else:
        match = "Resume environment differs" if outcome == "resume" else "comparison failed"
        with pytest.raises(RuntimeError, match=match):
            preflight.main(argv)
    report = json.loads((tmp_path / "comparison.json").read_text())
    if outcome == "resume":
        assert calls == ["configure"]
        assert report["status"] == "started"
    else:
        assert calls == ["configure", "reference", "subject"]
        assert report["status"] == ("passed" if outcome == "passed" else "failed")
        assert report["same_attention_settings"] == (outcome != "attention")
        assert report["environment"]["attention"]["cudnn_sdp_enabled"] is False
