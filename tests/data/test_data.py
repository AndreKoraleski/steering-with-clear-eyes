import urllib.request
from pathlib import Path

import pytest
from support import pin, tiny_registry

from swce.data import download
from swce.data.__main__ import main
from swce.data.download import PinError, fetch, fetch_image, file_digest, matches
from swce.data.verify import RemoteState, remote_state


def test_fetch_downloads_verifies_and_reuses(tmp_path: Path) -> None:
    source = tmp_path / "host" / "file.txt"
    source.parent.mkdir()
    source.write_text("content", encoding="utf-8")
    file = pin(source, "bench/file.txt")
    target = fetch(file, tmp_path / "data")
    assert target == tmp_path / "data" / "bench" / "file.txt"
    assert matches(target, file)
    assert file_digest(target) == file.digest
    source.unlink()
    assert fetch(file, tmp_path / "data") == target


def test_fetch_refuses_content_that_differs_from_the_pin(tmp_path: Path) -> None:
    source = tmp_path / "host" / "file.txt"
    source.parent.mkdir()
    source.write_text("content", encoding="utf-8")
    file = pin(source, "file.txt")
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(PinError, match="no longer serves"):
        fetch(file, tmp_path / "data")
    assert not list((tmp_path / "data").glob("*"))
    local = tmp_path / "data" / "file.txt"
    local.write_text("tampered", encoding="utf-8")
    with pytest.raises(PinError, match="differs from its pin"):
        fetch(file, tmp_path / "data")


def test_fetch_image(tmp_path: Path) -> None:
    registry = tiny_registry(tmp_path / "host")
    card = registry.benchmark("coco-2014")
    image = fetch_image(card, "a.jpg", tmp_path / "data")
    assert image == tmp_path / "data" / "images" / "a.jpg"
    assert image.read_bytes() == (tmp_path / "host" / "a.jpg").read_bytes()
    with pytest.raises(ValueError, match="serves no single images"):
        fetch_image(registry.benchmark("tiny"), "a.jpg", tmp_path / "data")


def test_remote_state(tmp_path: Path) -> None:
    source = tmp_path / "file.txt"
    source.write_text("content", encoding="utf-8")
    file = pin(source, "file.txt")
    assert remote_state(file) is RemoteState.OK
    source.write_text("changed", encoding="utf-8")
    assert remote_state(file) is RemoteState.CHANGED
    source.unlink()
    assert remote_state(file) is RemoteState.UNREACHABLE


def test_listing_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--data", str(tmp_path), "benchmarks"]) == 0
    assert "missing" in capsys.readouterr().out
    assert main(["tasks"]) == 0
    assert "pope-adversarial" in capsys.readouterr().out


def test_a_download_is_tried_again_after_a_failure_of_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "host" / "file.txt"
    source.parent.mkdir()
    source.write_text("content", encoding="utf-8")
    file = pin(source, "file.txt")
    real = urllib.request.urlopen
    calls: list[str] = []
    waits: list[float] = []

    def flaky(url: str, timeout: float) -> object:
        calls.append(url)
        if len(calls) < 3:
            raise OSError("getaddrinfo failed")
        return real(url, timeout=timeout)

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    monkeypatch.setattr("time.sleep", waits.append)
    assert fetch(file, tmp_path / "data").read_text(encoding="utf-8") == "content"
    assert len(calls) == 3
    assert waits == [2.0, 4.0]
    calls.clear()
    monkeypatch.setattr("urllib.request.urlopen", lambda *_, **__: flaky("", 0.0))
    monkeypatch.setattr(download, "ATTEMPTS", 2)
    with pytest.raises(OSError, match="getaddrinfo"):
        fetch(file, tmp_path / "other")
