#!/usr/bin/env bash
# Needs Docker. Builds the agent image and plays one episode with examples/custom_agent.py inside the container.
set -eu
cd "$(dirname "$0")/.."
docker build -t traceforge-agent:latest .
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from traceforge.environment.audit_env import AuditEnv
from traceforge.environment.isolation import SubprocessAgent
from traceforge.runner.container import docker_command
from traceforge.runner.episode_runner import run_episode
cmd = docker_command(Path("examples/custom_agent.py"))
res = run_episode(AuditEnv("eval"), lambda env: SubprocessAgent(cmd=cmd, timeout=120), seed=7, config={"clean": True})
assert res["agent_error"] is None and res["terminated"], res
print("container episode ok:", {k: res[k] for k in ("tp", "fp", "fn", "f1", "steps")})
PY
