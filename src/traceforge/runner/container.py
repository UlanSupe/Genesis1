"""OS-level isolation for untrusted agents: run the agent inside a locked-down container.

`docker_command(script)` returns a command for `SubprocessAgent(cmd=...)`: no network, read-only root filesystem,
all capabilities dropped, unprivileged user, memory / pid / cpu limits, ONLY the agent's own folder mounted
read-only. Datasets, `private/` folders and the project are not mounted, so even a native-code agent cannot see them.

NOT executed in this repository's own test environment (no Docker daemon there): the test suite checks the generated
command line, and `tools/container_smoke.sh` is the check to run where Docker is available."""
from __future__ import annotations

from pathlib import Path

IMAGE = "traceforge-agent:latest"


def docker_command(script: str | Path, *, image: str = IMAGE, memory: str = "1g", cpus: str = "1", pids: int = 64,
                   interpreter: str = "python", extra_args=()) -> list:
    script = Path(script).resolve()
    return ["docker", "run", "-i", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--user", "10001:10001", "--memory", memory, "--memory-swap", memory,
            "--cpus", cpus, "--pids-limit", str(pids), "--tmpfs", "/tmp:size=16m,noexec",
            "-v", f"{script.parent}:/agent:ro", "-w", "/agent", *extra_args, "--entrypoint", interpreter, image,
            f"/agent/{script.name}"]
