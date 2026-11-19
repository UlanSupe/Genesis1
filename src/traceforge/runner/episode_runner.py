"""Runs agents against AuditEnv, records trajectories, collects evaluator-side results."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..environment.audit_env import AuditEnv
from ..environment.isolation import SubprocessAgent, assert_no_leak
from ..evaluation.oracle_policy import OraclePolicy
from ..exceptions import AgentProtocolError, TraceForgeError
from ..policies import REGISTRY, make_policy
from ..trajectories.recorder import TrajectoryRecorder

AGENT_NAMES = tuple(REGISTRY) + ("oracle",)


def make_agent(name: str, seed: int = 0, isolated: bool = False, verify: bool = False):
    """Factory taking the env AFTER reset (the oracle needs the evaluator's truth)."""
    def build(env: AuditEnv):
        if name == "oracle":
            if isolated:
                raise TraceForgeError("the oracle is evaluator-side and cannot run isolated")
            return OraclePolicy(env.evaluator_truth())
        return SubprocessAgent(name, seed, verify=verify) if isolated else make_policy(name, seed, verify)
    return build


def run_episode(env: AuditEnv, agent, *, seed=None, config=None, recorder: TrajectoryRecorder | None = None,
                leak_guard: bool = True, label: str = "") -> dict:
    obs, info = env.reset(seed, config)
    episode_id = info["episode_id"]
    tokens = env.leak_tokens() if leak_guard else []
    if callable(agent) and not hasattr(agent, "act"):
        agent = agent(env)
    try:
        assert_no_leak(obs, tokens)
        if recorder:
            recorder.record(episode_id, 0, None, obs, 0.0, False, False)
        done, step, agent_error = False, 0, None
        try:
            agent.reset(obs)
        except Exception as e:                          # dead on arrival -> empty submission
            agent_error = type(e).__name__
        while not done:
            try:
                if agent_error:
                    raise AgentProtocolError("agent unavailable")
                action = agent.act(obs)
            except Exception as e:                      # crash/timeout -> empty submission
                agent_error = agent_error or type(e).__name__
                action = {"action": "submit", "findings": {"episode_id": episode_id, "findings": []}}
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            assert_no_leak([obs, info], tokens)
            if recorder:
                recorder.record(episode_id, step, action, obs, reward, terminated, truncated)
            done = terminated or truncated
    finally:
        if hasattr(agent, "close"):
            agent.close()
    res = env.result
    return {"episode_id": episode_id, "policy": label or getattr(agent, "name", "agent"),
            "mode": env.mode, "steps": step, "terminated": terminated, "truncated": truncated,
            "agent_error": agent_error, **asdict(res.metrics),
            "n_findings": res.n_findings, "n_truth": res.n_truth}


def split_dirs(root: str | Path, split: str) -> list:
    base = Path(root) / split
    names = json.loads((base / "index.json").read_text(encoding="utf-8"))["episodes"]
    return [base / n for n in names]


def run_split(root, split: str, policy: str, *, mode: str = "eval", isolated: bool = False,
              traj_dir: str | Path | None = None, seed: int = 0) -> list:
    results = []
    for d in split_dirs(root, split):
        env = AuditEnv(mode)
        rec = TrajectoryRecorder(Path(traj_dir) / f"{policy}_{mode}_{d.name}.jsonl") if traj_dir else None
        try:
            results.append(run_episode(env, make_agent(policy, seed, isolated),
                                       config={"episode_dir": str(d)}, recorder=rec, label=policy))
        finally:
            if rec:
                rec.close()
    return results
