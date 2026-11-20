import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).parents[2] / "experiments" / "qwen3_h100.sbatch"


@pytest.mark.parametrize("mode", [None, "--preflight-only", "--full", "--invalid"])
@pytest.mark.parametrize("fail_reference", [False, True])
def test_launcher_stops_at_authorized_stage(
    tmp_path: Path, mode: str | None, fail_reference: bool
) -> None:
    if any(shutil.which(command) is None for command in ("bash", "git", "flock", "tar")):
        pytest.skip("launcher test needs Unix batch tools")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "experiments").mkdir()
    (repo / "experiments" / "main.py").write_text("# mock experiment\n")
    (repo / "uv.lock").write_text("# mock lock\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "test"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    commands = tmp_path / "commands"
    commands.mkdir()
    # Fake executables never load a model, install dependencies or touch the GPU.
    for command in ("uv", "nvidia-smi"):
        path = commands / command
        path.write_text(
            "#!/usr/bin/env bash\n"
            'printf "%s\\n" "$*" >>"$TEST_CALLS"\n'
            'if [[ "$*" == *qwen3_preflight.py* && "$TEST_FAIL_REFERENCE" == 1 ]]; '
            "then exit 7; fi\n"
        )
        path.chmod(0o700)
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "SLURM_JOB_ID": "test",
        "SLURM_SUBMIT_DIR": str(repo),
        "CUDA_VISIBLE_DEVICES": "mock",
        "SWCE_WORK_ROOT": str(tmp_path / "work"),
        "SWCE_RUN_ID": "test",
        "SWCE_MIN_FREE_GIB": "1",
        "TEST_CALLS": str(calls),
        "TEST_FAIL_REFERENCE": str(int(fail_reference)),
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER), *([mode] if mode else [])],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if mode == "--invalid":
        assert result.returncode != 0 and not calls.exists()
        return
    if fail_reference:
        assert result.returncode != 0
        invoked = calls.read_text()
        assert "qwen3_preflight.py" in invoked
        assert "experiments/main.py" not in invoked
        assert "swce-diagnose" not in invoked
        return
    assert result.returncode == 0, result.stdout + result.stderr
    invoked = calls.read_text()
    assert "qwen3_preflight.py" in invoked
    assert ("experiments/main.py" in invoked) == (mode == "--full")
    assert ("swce-diagnose" in invoked) == (mode == "--full")
    if mode != "--full":
        assert "No smoke, fitting, evaluation, scoring or probes ran." in result.stdout
    execution = (tmp_path / "work/runs/test/execution.txt").read_text()
    assert execution == f"mode={mode or '--preflight-only'}\n"
