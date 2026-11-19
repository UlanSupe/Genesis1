import json

import pytest

from conftest import ALL4
from traceforge.environment.audit_env import AuditEnv
from traceforge.runner.episode_runner import make_agent, run_episode
from traceforge.trajectories.reader import check_no_leak, read_trajectory, replay
from traceforge.trajectories.recorder import TrajectoryRecorder

SEED = 987654321


def _run(tmp_path, policy="reconciliation", mode="eval", cfg=ALL4, isolated=False, name="t.jsonl"):
    env = AuditEnv(mode)
    path = tmp_path / name
    with TrajectoryRecorder(path) as rec:
        res = run_episode(env, make_agent(policy, 0, isolated), seed=SEED, config=cfg, recorder=rec, label=policy)
    return env, path, res


def test_each_line_has_exactly_the_specified_fields(tmp_path):
    _, path, _ = _run(tmp_path)
    for line in read_trajectory(path):
        assert set(line) == {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}


def test_steps_are_numbered_and_initial_observation_is_step_zero(tmp_path):
    _, path, res = _run(tmp_path)
    lines = read_trajectory(path)
    assert [l["step"] for l in lines] == list(range(len(lines))) and len(lines) == res["steps"] + 1
    assert lines[0]["action"] is None and lines[0]["observation"]["steps_remaining"] == 100
    assert lines[-1]["terminated"] and lines[-1]["action"]["action"] == "submit"


def test_trajectory_contains_no_private_or_seed_data(tmp_path):
    env, path, _ = _run(tmp_path)
    check_no_leak(path, env.leak_tokens())
    text = path.read_text(encoding="utf-8")
    for bad in (str(SEED), "expected_amount_minor", "allowed_evidence_sets", "ISS-", "clean_baseline",
                "provenance", "causal", "Users\\\\", "internal_log", '"seed"'):
        assert bad not in text, bad


def test_eval_mode_trajectory_hides_terminal_reward(tmp_path):
    env, path, res = _run(tmp_path, mode="eval")
    assert all(l["reward"] == 0.0 for l in read_trajectory(path))
    assert env.result.reward == res["reward"] != 0.0


def test_train_mode_trajectory_carries_terminal_reward(tmp_path):
    env, path, res = _run(tmp_path, mode="train")
    lines = read_trajectory(path)
    assert lines[-1]["reward"] == res["reward"] and all(l["reward"] == 0.0 for l in lines[:-1])


def test_trajectory_replays_exactly(tmp_path):
    _, path, _ = _run(tmp_path)
    out = replay(path, AuditEnv("eval"), SEED, ALL4)
    assert out["ok"] and out["mismatches"] == []


def test_replay_detects_a_tampered_trajectory(tmp_path):
    _, path, _ = _run(tmp_path)
    lines = read_trajectory(path)
    lines[3]["observation"]["last_result"]["records"][0]["debit_minor"] += 1
    path.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    assert replay(path, AuditEnv("eval"), SEED, ALL4)["mismatches"] == [3]


def test_replay_with_wrong_seed_fails(tmp_path):
    _, path, _ = _run(tmp_path)
    assert not replay(path, AuditEnv("eval"), SEED + 1, ALL4)["ok"]


def test_truncated_episode_is_recorded(tmp_path):
    env = AuditEnv("train")
    path = tmp_path / "trunc.jsonl"

    class Idle:
        def reset(self, obs): pass
        def act(self, obs): return {"action": "calculator", "expression": "1"}

    with TrajectoryRecorder(path) as rec:
        res = run_episode(env, Idle(), seed=1, config={"clean": True}, recorder=rec)
    lines = read_trajectory(path)
    assert res["truncated"] and not res["terminated"] and len(lines) == 101
    assert lines[-1]["truncated"] and lines[-1]["observation"]["steps_remaining"] == 0


def test_isolated_agent_trajectory_equals_in_process_trajectory(tmp_path):
    _, p1, _ = _run(tmp_path, name="a.jsonl")
    _, p2, _ = _run(tmp_path, isolated=True, name="b.jsonl")
    assert read_trajectory(p1) == read_trajectory(p2)


def test_random_policy_trajectory_includes_invalid_actions_and_stays_clean(tmp_path):
    env, path, _ = _run(tmp_path, policy="random")
    lines = read_trajectory(path)
    assert any(l["observation"]["error"] for l in lines)
    check_no_leak(path, env.leak_tokens())


def test_recorder_creates_parent_directories(tmp_path):
    with TrajectoryRecorder(tmp_path / "a" / "b" / "x.jsonl") as rec:
        rec.record("e", 0, None, {}, 0.0, False, False)
    assert (tmp_path / "a" / "b" / "x.jsonl").exists()
