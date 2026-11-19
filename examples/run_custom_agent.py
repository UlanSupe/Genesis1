import sys
from pathlib import Path

from traceforge.environment.audit_env import AuditEnv
from traceforge.environment.isolation import SubprocessAgent
from traceforge.runner.episode_runner import run_episode

agent_script = Path(__file__).with_name("custom_agent.py")
result = run_episode(AuditEnv("eval"),
                     lambda env: SubprocessAgent(cmd=[sys.executable, str(agent_script)]),
                     seed=7, config={"clean": True}, label="custom")
print({k: result[k] for k in ("episode_id", "steps", "tp", "fp", "fn", "f1", "reward", "agent_error")})
