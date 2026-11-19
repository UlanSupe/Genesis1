"""Regression tests for the vulnerabilities found in the final audit."""
import hashlib
import json

import pytest

from conftest import ALL4
from traceforge.config import GeneratorConfig
from traceforge.environment.audit_env import AuditEnv
from traceforge.episode_io import public_files
from traceforge.generator.builder import generate_episode
from traceforge.trajectories.reader import read_trajectory, replay
from traceforge.trajectories.recorder import TrajectoryRecorder, cap

CFG = {"difficulty": "medium"}


# ---------------------------------------------------------- SEC-1: episode_id must not reveal the seed
def test_sec1_episode_id_cannot_be_inverted_by_hashing_candidate_seeds():
    """The audit's attack: hash candidate seeds with the public config fingerprint and compare with the
    id the agent sees. Before the fix this recovered a hidden seed in milliseconds."""
    secret = 73_219
    env = AuditEnv("eval")
    obs, _ = env.reset(secret, CFG)
    visible = obs["manifest"]["episode_id"]
    fp = GeneratorConfig.from_dict(CFG).fingerprint()
    candidates = {"ep-" + hashlib.sha256(f"{s}|{fp}".encode()).hexdigest()[:12] for s in range(0, 150_000)}
    candidates |= {"ep-" + hashlib.sha256(str(s).encode()).hexdigest()[:12] for s in range(0, 150_000)}
    candidates |= {"ep-" + hashlib.md5(f"{s}".encode()).hexdigest()[:12] for s in range(0, 150_000)}
    assert visible not in candidates


def test_sec1_episode_id_is_a_digest_of_the_public_content_and_stays_deterministic():
    a, b, c = generate_episode(5, CFG), generate_episode(5, CFG), generate_episode(6, CFG)
    eid = a.public.manifest["episode_id"]
    assert eid == b.public.manifest["episode_id"] != c.public.manifest["episode_id"] and eid.startswith("ep-")
    blank = json.loads(json.dumps(a.public.manifest))
    blank["episode_id"] = ""
    pub = type(a.public)(blank, a.public.policy, a.public.ledger, a.public.invoices, a.public.payments, a.public.fx)
    h = hashlib.sha256()
    for name, text in sorted(public_files(pub).items()):
        h.update(name.encode() + text.encode())
    assert eid == "ep-" + h.hexdigest()[:12]


def test_sec1_high_entropy_seeds_are_supported_for_hidden_evaluation():
    for seed in (2 ** 64 + 12345, 2 ** 200 + 7, "a-long-random-secret-string-0123456789"):
        ep = generate_episode(seed, {"clean": True})
        assert ep.public.manifest["episode_id"].startswith("ep-") and ep.private.provenance["validation"]["ok"]


def test_sec1_agent_visible_data_never_contains_the_seed():
    secret = 987_654_321_123
    env = AuditEnv("eval")
    obs, info = env.reset(secret, ALL4)
    blob = json.dumps([obs, info, env.step({"action": "list_records", "table": "ledger"})[0]])
    assert str(secret) not in blob and "seed" not in blob.lower().replace("seeds", "")


# ---------------------------------------------------------- SEC-2: trajectory size cannot be inflated by an agent
def test_sec2_oversized_actions_are_capped_in_the_trajectory_file(tmp_path):
    huge = "A" * 5_000_000
    actions = [{"action": "get_record", "table": "ledger", "id": huge},
               {"action": "calculator", "expression": huge},
               {"action": "list_records", "table": "ledger", "cursor": huge},
               {"action": "submit", "findings": [{"type": "fx", "affected_ids": [huge] * 500, "claimed_amount_minor": 1,
                                                    "currency": huge, "evidence_ids": [], "explanation": huge}] * 400}]
    env = AuditEnv("eval")
    obs, _ = env.reset(1, ALL4)
    path = tmp_path / "t.jsonl"
    with TrajectoryRecorder(path) as rec:
        rec.record("e", 0, None, obs, 0.0, False, False)
        for i, a in enumerate(actions, 1):
            o, r, t, tr, _ = env.step(a)
            rec.record("e", i, a, o, r, t, tr)
    assert path.stat().st_size < 600_000                         # was 42 MB for this payload (and unbounded before the caps)
    assert len(read_trajectory(path)) == len(actions) + 1


def test_sec2_capping_never_changes_the_outcome_of_a_replay(tmp_path):
    huge = "A" * 100_000
    seq = [{"action": "get_record", "table": "ledger", "id": huge}, {"action": "calculator", "expression": huge},
           {"action": "list_records", "table": "payments", "cursor": huge},
           {"action": "submit", "findings": [{"type": "fx", "affected_ids": [huge], "claimed_amount_minor": 1, "currency": "EUR",
                                                "evidence_ids": [], "explanation": huge}]}]
    env = AuditEnv("train")
    obs, _ = env.reset(1, ALL4)
    path = tmp_path / "t.jsonl"
    with TrajectoryRecorder(path) as rec:
        rec.record("e", 0, None, obs, 0.0, False, False)
        for i, a in enumerate(seq, 1):
            o, r, t, tr, _ = env.step(a)
            rec.record("e", i, a, o, r, t, tr)
    assert replay(path, AuditEnv("train"), 1, ALL4)["ok"]        # capped actions reproduce the very same results


def test_sec2_cap_is_bounded_in_depth_width_and_string_length_and_leaves_normal_actions_alone():
    normal = {"action": "submit", "findings": [{"type": "fx", "affected_ids": ["PAY-1"], "claimed_amount_minor": 5, "currency": "EUR",
                                                 "evidence_ids": ["DOC-1"], "explanation": "e" * 1000}]}
    assert cap(normal) == normal
    deep = cur = []
    for _ in range(50):
        nxt = []
        cur.append(nxt)
        cur = nxt
    assert len(json.dumps(cap(deep))) < 1000
    assert len(json.dumps(cap({str(i): i for i in range(10_000)}))) < 5000
    assert len(json.dumps(cap(["x" * 10_000_000]))) < 10_000


def test_sec2_a_maximum_size_legitimate_submit_is_never_truncated():
    from traceforge.trajectories.recorder import bounded_action
    finding = {"type": "allocation", "affected_ids": ["I" * 64] * 20, "claimed_amount_minor": 10 ** 12, "currency": "EUR",
              "evidence_ids": ["E" * 64] * 10, "explanation": "x" * 1000}
    action = {"action": "submit", "findings": {"episode_id": "ep-" + "a" * 12, "findings": [dict(finding) for _ in range(30)]}}
    assert len(json.dumps(action)) < 128 * 1024
    assert bounded_action(action) == action


def test_sec3_evaluator_logging_never_carries_hidden_labels(caplog):
    import logging
    env = AuditEnv("train")
    env.reset(8, ALL4)
    tokens = env.leak_tokens() + [str(t["expected_amount_minor"]) for t in env.evaluator_truth()]
    with caplog.at_level(logging.DEBUG):
        env.step({"action": "submit", "findings": "invalid structure"})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "invalid submission" in text                      # the internal journal entry exists ...
    assert not [t for t in tokens if t and t in text]        # ... and carries no truth
