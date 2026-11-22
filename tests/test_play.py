"""`play`: one-step-per-command interface for human/LLM agents (deterministic replay, eval mode)."""
import json

from traceforge.runner import play


def _start(tmp_path, seed=5):
    log = str(tmp_path / "l.jsonl")
    obs = json.loads(play.start(log, seed, {"difficulty": "easy"}))
    return log, obs


def test_start_prints_the_task_manifest_and_policy_but_no_truth(tmp_path):
    log, obs = _start(tmp_path)
    assert {"task", "manifest", "policy"} <= set(obs)
    text = json.dumps(obs)
    assert "truth" not in text and "expected_amount" not in text and "root_cause" not in text


def test_act_prints_only_what_changed_and_counts_steps(tmp_path):
    log, _ = _start(tmp_path)
    out = json.loads(play.act(log, '{"action":"list_records","table":"payments","limit":2}'))
    assert set(out) == {"last_result", "error", "steps_remaining"} and out["steps_remaining"] == 99
    assert len(out["last_result"]["records"]) == 2


def test_invalid_json_takes_no_step(tmp_path):
    log, _ = _start(tmp_path)
    assert play.act(log, "not json").startswith("ERROR")
    assert json.loads(play.act(log, '{"action":"list_records","table":"fx","limit":1}'))["steps_remaining"] == 99


def test_replay_is_deterministic_across_calls(tmp_path):
    a, _ = _start(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path)
    b = str(tmp_path / "x" / "other.jsonl")
    play.start(b, 5, {"difficulty": "easy"})
    act = '{"action":"list_records","table":"ledger","limit":3}'
    assert play.act(a, act) == play.act(b, act)


def test_submit_ends_the_episode_and_writes_a_verdict_but_prints_no_reward(tmp_path):
    log, _ = _start(tmp_path)
    out = play.act(log, '{"action":"submit","findings":[]}')
    assert "EPISODE OVER" in out and "reward" not in out
    assert play.act(log, '{"action":"list_records","table":"fx"}') == "EPISODE ALREADY OVER"
    res = play.result(log)
    assert res["valid"] and res["tp"] == 0 and res["fp"] == 0 and res["n_findings"] == 0 and res["steps"] == 1


def test_stored_llm_runs_replay_to_the_stored_verdicts():
    """llm_runs/: action logs of real LLM-agent episodes; replaying them reproduces the evaluator's verdicts exactly."""
    from pathlib import Path
    from traceforge.runner.play import _replay
    root = Path(__file__).resolve().parents[1] / "llm_runs"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))["rows"]
    assert len(summary) == 6
    for row in summary:
        log = root / f"llm_{row['run']}.jsonl"
        env, _, done, n, head, _ = _replay(str(log))
        m = env.result.metrics
        assert done and head["seed"] == row["seed"]
        assert (m.tp, m.fp, m.fn, m.inspection_steps) == (row["tp"], row["fp"], row["fn"], row["inspection_steps"])
        assert round(m.reward, 3) == round(row["reward"], 3)
        stored = json.loads((root / f"llm_{row['run']}.jsonl.result.json").read_text(encoding="utf-8"))
        assert stored["tp"] == m.tp and stored["fp"] == m.fp and stored["fn"] == m.fn
