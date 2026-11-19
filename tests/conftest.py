import json
from functools import lru_cache

import pytest

from traceforge.environment.audit_env import AuditEnv
from traceforge.evaluation.oracle_policy import OraclePolicy
from traceforge.generator.builder import generate_episode

ALL4 = {"issue_counts": {"duplicate": 1, "allocation": 1, "fx": 1, "cutoff": 1}}


@lru_cache(maxsize=None)
def _cached(seed: int, cfg_json: str):
    return generate_episode(seed, json.loads(cfg_json))


def episode(seed=1, **cfg):
    """Cached episode (treat as read-only)."""
    return _cached(seed, json.dumps(cfg, sort_keys=True))


def new_env(seed=1, mode="train", **cfg):
    env = AuditEnv(mode)
    obs, info = env.reset(seed, cfg)
    return env, obs, info


def act(env, name, **kw):
    return env.step({"action": name, **kw})


def oracle_action(env, obs):
    return OraclePolicy(env.evaluator_truth()).act(obs)


def finding(t, **over):
    """Finding dict matching truth entry t (override fields via kwargs)."""
    f = {"type": t["type"], "affected_ids": list(t["affected_ids"]),
         "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
         "evidence_ids": list(t["allowed_evidence_sets"][0]), "explanation": "x"}
    f.update(over)
    return f


def submit(env, findings, episode_id=None):
    eid = episode_id or env._evaluator.episode_id
    return env.step({"action": "submit", "findings": {"episode_id": eid, "findings": findings}})


@pytest.fixture
def mixed():
    return new_env(1, **ALL4)
