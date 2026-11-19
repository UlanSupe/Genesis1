import json
import sys
import textwrap

import pytest

from conftest import ALL4, new_env
from traceforge.environment.audit_env import AuditEnv
from traceforge.environment.isolation import SubprocessAgent, assert_no_leak
from traceforge.exceptions import AgentProtocolError, LeakError, TraceForgeError
from traceforge.policies import REGISTRY
from traceforge.runner.episode_runner import make_agent, run_episode

HOSTILE = textwrap.dedent('''
    import gc, json, os, sys
    def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
    for line in sys.stdin:
        m = json.loads(line)
        if m["type"] == "close": break
        if m["type"] == "reset": send({"type": "ready"}); continue
        # try every trick to find private data inside this process / its surroundings
        dicts = sum(1 for o in gc.get_objects() if isinstance(o, dict) and "expected_amount_minor" in o)
        strs = sum(1 for o in gc.get_objects() if isinstance(o, str) and (o.startswith("ISS-") or "allowed_evidence_sets" in o))
        mods = sorted(k for k in sys.modules if k.startswith("traceforge"))
        envs = [k for k, v in os.environ.items() if "truth" in v.lower() or "episode" in v.lower()]
        files = os.listdir(os.getcwd())
        report = {"dicts": dicts, "strs": strs, "mods": mods, "envs": envs, "files": files}
        sys.stderr.write(json.dumps(report)); sys.stderr.flush()
        send({"type": "action", "action": {"action": "get_record", "table": "ledger", "id": "PROBE"}})
''')


def test_agent_process_contains_no_private_data(tmp_path):
    script = tmp_path / "hostile.py"
    script.write_text(HOSTILE, encoding="utf-8")
    env, obs, _ = new_env(1, mode="eval", **ALL4)
    agent = SubprocessAgent(cmd=[sys.executable, str(script)])
    try:
        agent.reset(obs)
        agent.act(obs)
        import time
        for _ in range(50):                       # stderr is drained by a reader thread
            if agent.stderr_tail():
                break
            time.sleep(0.05)
        report = json.loads(agent.stderr_tail())
    finally:
        agent.close()
    assert report["dicts"] == 0 and report["strs"] == 0
    assert report["envs"] == [] and report["files"] == []
    assert not any(m.startswith("traceforge.evaluation") or m.startswith("traceforge.generator") for m in report["mods"])


def test_worker_imports_neither_generator_nor_evaluator():
    import subprocess
    code = ("import sys; import traceforge.runner.agent_worker; "
            "bad=[m for m in sys.modules if m.startswith(('traceforge.generator','traceforge.evaluation','traceforge.environment.audit_env'))]; "
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": __import__("traceforge.environment.isolation", fromlist=["x"]).SRC_DIR,
                              "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}).stdout.strip()
    assert out == "[]"


def test_oracle_is_not_an_agent_policy_and_cannot_run_isolated():
    assert "oracle" not in REGISTRY
    with pytest.raises(TraceForgeError):
        run_episode(AuditEnv("eval"), make_agent("oracle", isolated=True), seed=1, config=ALL4)
    import subprocess
    p = subprocess.run([sys.executable, "-m", "traceforge.runner.agent_worker", "--policy", "oracle"],
                       capture_output=True, text=True, input="")
    assert p.returncode == 2


def test_isolated_run_matches_in_process_run():
    a = run_episode(AuditEnv("eval"), make_agent("reconciliation"), seed=4, config=ALL4)
    b = run_episode(AuditEnv("eval"), make_agent("reconciliation", isolated=True), seed=4, config=ALL4)
    for k in ("tp", "fp", "fn", "f1", "reward", "steps"):
        assert a[k] == b[k]


def test_no_private_token_in_any_observation_or_info_over_a_full_run():
    env, obs, info = new_env(2, mode="eval", **ALL4)
    tokens = env.leak_tokens()
    agent = make_agent("mark_all")(env)
    agent.reset(obs)
    done = False
    while not done:
        obs, r, t, tr, info = env.step(agent.act(obs))
        assert_no_leak([obs, info], tokens)
        done = t or tr


def test_leak_guard_detects_a_leak():
    with pytest.raises(LeakError):
        assert_no_leak({"error": "see ISS-001"}, ["ISS-001"])
    assert_no_leak({"ok": 1}, ["ISS-001"])


def test_errors_and_exceptions_do_not_expose_internals():
    env, obs, _ = new_env(1, **ALL4)
    for bad in ({"action": "get_record", "table": "ledger", "id": "\x00" * 10},
                {"action": "calculator", "expression": "__import__('os')"},
                {"action": "list_records", "table": "ledger", "cursor": {"a": 1}},
                {"action": "get_document", "doc_id": "../../private/truth.json"}):
        o, *_ = env.step(bad)
        text = json.dumps(o["error"])
        assert all(w not in text for w in ("Traceback", "truth", "private", "\\\\", "/home", ".py"))
        assert len(text) < 200


def test_crashing_agent_is_scored_as_empty_submission_not_a_crash(tmp_path):
    script = tmp_path / "dead.py"
    script.write_text("import sys\nsys.exit(3)\n")
    res = run_episode(AuditEnv("train"), lambda env: SubprocessAgent(cmd=[sys.executable, str(script)], timeout=5),
                      seed=1, config=ALL4)
    assert res["agent_error"] == "AgentProtocolError" and res["terminated"] and res["tp"] == 0 and res["fn"] == 4


def test_hanging_agent_times_out(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text("import time\ntime.sleep(60)\n")
    agent = SubprocessAgent(cmd=[sys.executable, str(script)], timeout=1.0)
    with pytest.raises(AgentProtocolError):
        agent.reset({"manifest": {}})
    agent.close()


def test_agent_cannot_reach_truth_through_the_protocol():
    env, obs, _ = new_env(1, mode="eval", **ALL4)
    for probe in ({"action": "get_truth"}, {"action": "evaluator_truth"}, {"action": "submit_oracle"},
                  {"action": "get_record", "table": "truth", "id": "ISS-001"},
                  {"action": "list_records", "table": "provenance"}):
        o, *_ = env.step(probe)
        assert o["error"] is not None and o["last_result"]["ok"] is False


def test_agent_protocol_rejects_malformed_messages():
    from traceforge.runner.agent_protocol import action_from, decode
    for bad in ("not json", "[]", '{"no_type": 1}', '{"type": 5}'):
        with pytest.raises(AgentProtocolError):
            decode(bad)
    with pytest.raises(AgentProtocolError):
        action_from({"type": "action", "action": "list"})
