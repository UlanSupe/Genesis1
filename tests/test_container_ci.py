"""Container command line (Docker is not available in the test environment) and the CI definition."""
import re
from pathlib import Path

from traceforge.runner.container import docker_command

ROOT = Path(__file__).resolve().parents[1]


def test_docker_command_is_locked_down(tmp_path):
    script = tmp_path / "agent" / "a.py"
    cmd = docker_command(script)
    joined = " ".join(cmd)
    assert cmd[:3] == ["docker", "run", "-i"] and "--rm" in cmd
    assert cmd[cmd.index("--network") + 1] == "none" and "--read-only" in cmd
    assert cmd[cmd.index("--cap-drop") + 1] == "ALL" and "no-new-privileges" in joined
    assert cmd[cmd.index("--user") + 1] == "10001:10001"
    assert "--memory" in cmd and "--pids-limit" in cmd and "--cpus" in cmd
    mounts = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-v"]
    assert len(mounts) == 1 and mounts[0].endswith(":/agent:ro")          # only the agent's folder, read-only
    assert cmd[-1] == "/agent/a.py"


def test_nothing_of_the_project_or_datasets_is_mounted():
    joined = " ".join(docker_command(ROOT / "examples" / "custom_agent.py"))
    assert "datasets" not in joined and "private" not in joined and str(ROOT / "src") not in joined


def test_dockerfile_runs_as_an_unprivileged_user():
    d = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^USER 10001", d, re.M) and "FROM python:3.12" in d


def test_ci_covers_three_operating_systems_and_both_python_versions():
    y = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for token in ("ubuntu-latest", "windows-latest", "macos-latest", '"3.11"', '"3.12"', "pytest"):
        assert token in y
