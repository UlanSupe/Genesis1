"""Step-by-step play for a HUMAN or LLM agent that can only run shell commands (one tool call per process).

The episode is generated in memory from (seed, config); the log file stores only the actions taken so far, so every call
replays them deterministically (same seed -> same episode) and applies one more step. Eval mode: the reward is never printed.
The evaluator's verdict is written to `<log>.result.json` when the episode ends; read it with `play result` (evaluator side).

  python -m traceforge play start  --log L.jsonl --seed 7 [--preset medium]
  python -m traceforge play act    --log L.jsonl --json '{"action":"list_records","table":"ledger"}'
  python -m traceforge play result --log L.jsonl

Honour-system note: the agent is expected to use ONLY these commands. The hidden truth is not in the printed observations, but
a process with file access to the repository could read it elsewhere - this is not a sandbox (see README, Isolation)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..environment.audit_env import AuditEnv
from ..exceptions import TraceForgeError


def _read(log: str):
    lines = Path(log).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise TraceForgeError("empty log: run `play start` first")
    return json.loads(lines[0]), [json.loads(x) for x in lines[1:]]


def _replay(log: str):
    head, actions = _read(log)
    env = AuditEnv("eval")
    obs, _ = env.reset(head["seed"], head["config"])
    last = (obs, False)
    for a in actions:
        obs, _, term, trunc, _ = env.step(a)
        last = (obs, term or trunc)
    return env, last[0], last[1], len(actions), head, actions


def start(log: str, seed: int, config: dict) -> str:
    p = Path(log)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"seed": seed, "config": config}) + "\n", encoding="utf-8")
    p.with_name(p.name + ".result.json").unlink(missing_ok=True)
    env = AuditEnv("eval")
    obs, _ = env.reset(seed, config)
    return json.dumps(obs, separators=(",", ":"))


def act(log: str, raw: str) -> str:
    env, obs, done, n, head, actions = _replay(log)
    if done:
        return "EPISODE ALREADY OVER"
    try:
        action = json.loads(raw)
    except json.JSONDecodeError as e:
        return f"ERROR: action is not valid JSON ({e.msg}); no step was taken"
    obs, _, term, trunc, _ = env.step(action)
    with Path(log).open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(action, separators=(",", ":")) + "\n")
    if term or trunc:
        res = env.result
        Path(log).with_name(Path(log).name + ".result.json").write_text(
            json.dumps({**asdict(res.metrics), "n_findings": res.n_findings, "n_truth": res.n_truth, "valid": res.valid,
                        "steps": n + 1}, indent=2), encoding="utf-8")
        return _brief(obs) + "\nEPISODE OVER - the evaluator has your submission."
    return _brief(obs)


def _brief(obs: dict) -> str:
    """After `start` the agent already knows the task, manifest and policy: print only what changed."""
    return json.dumps({k: obs.get(k) for k in ("last_result", "error", "steps_remaining")}, separators=(",", ":"))


def result(log: str) -> dict:
    return json.loads(Path(log).with_name(Path(log).name + ".result.json").read_text(encoding="utf-8"))
