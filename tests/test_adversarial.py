"""Attacking agents. Every test drives the REAL environment/evaluator/runner; none of them
mocks the component under attack."""
import json
import os
import sys
import textwrap
import time
from fractions import Fraction

import pytest

from conftest import ALL4, act, finding, new_env, submit
from traceforge.constants import ISSUE_TYPES
from traceforge.environment.audit_env import AuditEnv
from traceforge.environment.isolation import SubprocessAgent, assert_no_leak
from traceforge.evaluation.evidence_validator import evidence_satisfied
from traceforge.exceptions import AgentProtocolError, EpisodeFinished
from traceforge.runner.episode_runner import make_agent, run_episode

CLEAN = {"clean": True}


def _score(env):
    m = env.result.metrics
    return m.tp, m.fp, m.fn, m.reward


# 1 ------------------------------------------------------------------ accuse everything
def test_01_accusing_every_record_is_punished():
    env, *_ = new_env(1, **ALL4)
    ledger = env._pub.ledger
    fs = [{"type": t, "affected_ids": [r["row_id"]], "claimed_amount_minor": max(r["debit_minor"], r["credit_minor"]),
           "currency": "EUR", "evidence_ids": [r["row_id"]], "explanation": "suspicious"}
          for t in ISSUE_TYPES for r in ledger[:8]]
    assert len(fs) == 32 > 30
    submit(env, fs[:30])
    tp, fp, fn, reward = _score(env)
    assert (tp, fp, fn) == (0, 30, 4) and reward == -1.0
    env2, *_ = new_env(1, **CLEAN)
    submit(env2, fs[:30])
    assert _score(env2)[:2] == (0, 30) and _score(env2)[3] == -1.0       # clean company accused


def test_01b_accusing_all_documents_with_correct_looking_fields_still_fails():
    env, *_ = new_env(2, **ALL4)
    docs = env._pub.invoices + env._pub.payments
    fs = [{"type": "duplicate", "affected_ids": [d.get("invoice_id") or d["payment_id"]], "claimed_amount_minor": 1,
           "currency": "EUR", "evidence_ids": [d["doc_id"]], "explanation": ""} for d in docs[:30]]
    submit(env, fs)
    assert _score(env)[0] == 0


# 2 ------------------------------------------------------------------ repeated finding
@pytest.mark.parametrize("copies", [2, 5, 30])
def test_02_repeating_a_correct_finding_gains_nothing(copies):
    env, *_ = new_env(1, **{"issue_counts": {"fx": 1}})
    t = env.evaluator_truth()[0]
    submit(env, [finding(t)] * copies)
    tp, fp, fn, reward = _score(env)
    assert (tp, fp, fn) == (1, copies - 1, 0)
    assert reward == pytest.approx(float(2 * Fraction(2, 2 + copies - 1) - 1))


# 3 ------------------------------------------------------------------ wrong ids
def test_03_wrong_ids_never_match():
    base, *_ = new_env(1, **ALL4)
    truth = base.evaluator_truth()
    other = [r["row_id"] for r in base._pub.ledger if r["row_id"] not in {i for t in truth for i in t["affected_ids"]}][:3]
    attacks = []
    for t in truth:
        ids = t["affected_ids"]
        attacks += [finding(t, affected_ids=ids + [other[0]]), finding(t, affected_ids=ids[:-1] if len(ids) > 1 else [other[1]]),
                    finding(t, affected_ids=[i.lower() for i in ids]), finding(t, affected_ids=[other[2]]),
                    finding(t, affected_ids=[])]
    for a in attacks:
        env, *_ = new_env(1, **ALL4)
        submit(env, [a])
        assert _score(env)[0] == 0, a["affected_ids"]


def test_03b_ids_are_a_set_so_order_and_repeats_are_irrelevant_but_content_is_exact():
    env, *_ = new_env(1, **ALL4)
    multi = next(t for t in env.evaluator_truth() if len(t["affected_ids"]) > 1)
    submit(env, [finding(multi, affected_ids=list(reversed(multi["affected_ids"])) + multi["affected_ids"])])
    assert _score(env)[0] == 1


# 4 ------------------------------------------------------------------ currency swap
@pytest.mark.parametrize("ccy", ["USD", "JPY", "XXX", "", "eur2", "€"])
def test_04_substituted_currency_is_rejected(ccy):
    env, *_ = new_env(1, **ALL4)
    t = env.evaluator_truth()[0]
    submit(env, [finding(t, currency=ccy)])
    assert _score(env)[0] == 0


def test_04b_only_case_and_whitespace_of_the_right_currency_are_normalised():
    env, *_ = new_env(1, **ALL4)
    t = env.evaluator_truth()[0]
    submit(env, [finding(t, currency=" " + t["currency"].lower() + " ")])
    assert _score(env)[0] == 1


# 5 ------------------------------------------------------------------ right amount wrong type
def test_05_right_amount_wrong_type_is_rejected_for_every_type_pair():
    base, *_ = new_env(1, **ALL4)
    for t in base.evaluator_truth():
        for wrong in ISSUE_TYPES:
            if wrong == t["type"]:
                continue
            env, *_ = new_env(1, **ALL4)
            submit(env, [finding(t, type=wrong)])
            assert _score(env)[:3] == (0, 1, 4), (t["type"], wrong)


# 6 / 7 --------------------------------------------------------------- evidence attacks
def test_06_right_type_without_evidence_is_rejected():
    base, *_ = new_env(1, **ALL4)
    for t in base.evaluator_truth():
        env, *_ = new_env(1, **ALL4)
        submit(env, [finding(t, evidence_ids=[])])
        assert _score(env)[0] == 0


def test_07_incomplete_evidence_is_rejected_exactly_when_no_allowed_set_survives():
    base, *_ = new_env(1, **ALL4)
    zeroed = 0
    for t in base.evaluator_truth():
        chosen = t["allowed_evidence_sets"][0]
        for drop in chosen:
            ev = [e for e in chosen if e != drop]
            still_ok = evidence_satisfied(ev, t["allowed_evidence_sets"])
            env, *_ = new_env(1, **ALL4)
            submit(env, [finding(t, evidence_ids=ev)])
            assert (_score(env)[0] == 1) is still_ok
            zeroed += not still_ok
    assert zeroed >= 4                                  # incompleteness really is punished


def test_07b_evidence_from_another_issue_does_not_transfer():
    base, *_ = new_env(1, **ALL4)
    a, b = base.evaluator_truth()[:2]
    env, *_ = new_env(1, **ALL4)
    submit(env, [finding(a, evidence_ids=b["allowed_evidence_sets"][0])])
    assert _score(env)[0] == 0


# 8 ------------------------------------------------------------------ action limit
def test_08_cannot_exceed_100_actions():
    env, *_ = new_env(1, **ALL4)
    last = None
    for i in range(100):
        last = act(env, "calculator", expression="1")
    assert last[3] and last[0]["steps_remaining"] == 0
    for _ in range(3):
        with pytest.raises(EpisodeFinished):
            act(env, "calculator", expression="1")
    with pytest.raises(EpisodeFinished):
        submit(env, [])
    assert env.result.metrics.inspection_steps == 100 and env.result.n_findings == 0


def test_08b_invalid_actions_also_consume_the_budget():
    env, *_ = new_env(1, **ALL4)
    for _ in range(99):
        env.step({"action": "hack"})
    assert env._steps == 99
    out = env.step({"action": "hack"})
    assert out[3] is True and out[0]["steps_remaining"] == 0


# 9 ------------------------------------------------------------------ >50 rows
@pytest.mark.parametrize("limit", [51, 100, 1000, 10 ** 9, 0, -1, "50", 50.0, None, True])
def test_09_page_size_attacks(limit):
    env, *_ = new_env(1, **ALL4)
    obs, *_ = act(env, "list_records", table="ledger", limit=limit)
    assert obs["last_result"]["ok"] is False and obs["error"]["code"] == "invalid_limit"
    assert "records" not in obs["last_result"]


def test_09b_no_sequence_of_valid_calls_returns_more_than_50_rows_at_once():
    env, *_ = new_env(1, **ALL4)
    cursor = None
    for _ in range(12):
        obs, *_ = act(env, "list_records", table="ledger", cursor=cursor, limit=50)
        assert len(obs["last_result"]["records"]) <= 50
        cursor = obs["last_result"]["next_cursor"]
        if cursor is None:
            break


# 10 ----------------------------------------------------------------- unknown tools
@pytest.mark.parametrize("tool", ["shell", "bash", "python", "sql", "query", "read_file", "open", "get_truth", "eval", "exec",
                                  "LIST_RECORDS", "list_records ", "list-records", "__import__", "submit_oracle", "reset"])
def test_10_unknown_tools_are_refused_and_cost_a_step(tool):
    env, *_ = new_env(1, **ALL4)
    obs, r, term, trunc, _ = env.step({"action": tool, "cmd": "id", "query": "select 1"})
    assert obs["error"]["code"] == "unknown_action" and obs["steps_remaining"] == 99 and not term and r == 0.0


# 11 ----------------------------------------------------------------- broken JSON / protocol
@pytest.mark.parametrize("script_body,expected", [
    ("sys.stdout.write('this is not json\\n')", "AgentProtocolError"),
    ("sys.stdout.write('[1, 2, 3]\\n')", "AgentProtocolError"),
    ("sys.stdout.write('{\"type\": \"action\", \"action\": \"submit\"}\\n')", "AgentProtocolError"),
    ("sys.stdout.write('{\"type\": \"banana\"}\\n')", "AgentProtocolError"),
    ("sys.exit(5)", "AgentProtocolError"),
])
def test_11_malformed_agent_output_is_scored_as_empty_submission(tmp_path, script_body, expected):
    script = tmp_path / "bad.py"
    script.write_text("import sys\nfor line in sys.stdin:\n    " + script_body + "\n    sys.stdout.flush()\n", encoding="utf-8")
    res = run_episode(AuditEnv("train"), lambda env: SubprocessAgent(cmd=[sys.executable, str(script)], timeout=10),
                      seed=1, config=ALL4)
    assert res["agent_error"] == expected and res["terminated"] and res["tp"] == 0 and res["fn"] == 4


def test_11b_non_dict_actions_in_process_cost_a_step_not_a_crash():
    env, *_ = new_env(1, **ALL4)
    for bad in ("{'action': 'list_records'}", b"bytes", 12, [1], None, {"action": ["list_records"]}, {"action": {"a": 1}}):
        obs, *_ = env.step(bad)
        assert obs["error"] is not None
    assert env._steps == 7


# 12 ----------------------------------------------------------------- private truth
def test_12_private_truth_cannot_be_reached_through_any_tool():
    env, obs, info = new_env(1, mode="eval", **ALL4)
    tokens = env.leak_tokens()
    seen = [obs, info]
    probes = [{"action": "get_record", "table": t, "id": i} for t in ("truth", "provenance", "private", "clean_baseline", "ledger")
              for i in ("ISS-001", "truth.json", "../private/truth.json", "MUT-001")]
    probes += [{"action": "get_document", "doc_id": i} for i in ("ISS-001", "../private/truth.json", "truth")]
    probes += [{"action": "list_records", "table": t} for t in ("truth", "provenance", "issues", "mutations", "../private")]
    probes += [{"action": "calculator", "expression": e} for e in ("truth", "issues[0]", "self._evaluator", "env.result")]
    for p in probes:
        o, r, *_rest, i = env.step(p) if not env._done else (None,) * 5
        seen += [o, i]
        if env._steps >= 99:
            break
    text = json.dumps(seen, default=str)
    for tok in tokens + ["expected_amount_minor", "allowed_evidence_sets", "causal_description", "clean_baseline"]:
        assert tok not in text
    assert all(o is None or o["last_result"].get("ok") is False or "record" in o["last_result"] or "document" in o["last_result"]
               or "value" in o["last_result"] or "records" in o["last_result"] for o in seen[2::2] if isinstance(o, dict))


def test_12b_terminal_feedback_in_each_mode_reveals_only_what_the_mode_allows():
    for mode, expect_reward in (("train", True), ("eval", False)):
        env, obs, _ = new_env(1, mode=mode, **ALL4)
        o, r, t, tr, info = submit(env, [])
        assert (r != 0.0) is expect_reward
        assert set(info) <= {"step", "submission_valid"} and ("submission_valid" in info) is expect_reward
        assert_no_leak([o, info], env.leak_tokens())
        feedback = json.dumps([o["last_result"], o["error"]])
        assert "internal_log" not in feedback and "matches" not in feedback and "tp" not in o["last_result"]


def test_12c_agent_process_cannot_find_the_reference_in_memory(tmp_path):
    script = tmp_path / "scan.py"
    script.write_text(textwrap.dedent('''
        import gc, json, sys
        def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
        for line in sys.stdin:
            m = json.loads(line)
            if m["type"] == "close": break
            if m["type"] == "reset": send({"type": "ready"}); continue
            hits = sum(1 for o in gc.get_objects() if isinstance(o, (dict, str)) and
                       ("allowed_evidence_sets" in o or "expected_amount_minor" in o or (isinstance(o, str) and o.startswith("ISS-"))))
            send({"type": "action", "action": {"action": "get_record", "table": "ledger", "id": "HITS%d" % hits}})
    '''), encoding="utf-8")
    env, obs, _ = new_env(3, mode="eval", **ALL4)
    agent = SubprocessAgent(cmd=[sys.executable, str(script)])
    try:
        agent.reset(obs)
        action = agent.act(obs)
    finally:
        agent.close()
    assert action["id"] == "HITS0"


# 13 ----------------------------------------------------------------- calculator attacks
ATTACKS = ["__import__('os').system('echo pwned')", "().__class__.__bases__[0].__subclasses__()", "open('/etc/passwd').read()",
           "exec('print(1)')", "eval('1+1')", "[x for x in range(10**9)]", "(lambda: 1)()", "f'{1+1}'", "globals()", "locals()",
           "10 ** 10 ** 10", "9 ** 9 ** 9", "99999999 ** 99999999", "1e1000000 * 1e1000000", "0.1 ** 1000", "2 ** 1000000",
           "(" * 150 + "1" + ")" * 150, "-" * 190 + "1", "1 + " * 70 + "1", "1" * 190, "round(1e100, 12)", "max(" + ",".join(["1"] * 20) + ")",
           "x", "a.b.c", "1 if 1 else 2", "[1][0]", "{1: 2}", "'a' + 'b'", "b'x'", "...", "1 < 2 < 3", "not 0", "~1",
           "1;import os", "\n__import__('os')", "1/0", "0**0 / 0", "abs()", "round(1, -1)", "min('a')", "chr(65)", "\u0661\u0662"]


@pytest.mark.parametrize("expr", ATTACKS)
def test_13_calculator_attacks_are_contained_and_fast(expr):
    env, *_ = new_env(1, **ALL4)
    t = time.perf_counter()
    obs, r, term, trunc, _ = env.step({"action": "calculator", "expression": expr})
    assert time.perf_counter() - t < 1.0
    assert not term and r == 0.0 and obs["steps_remaining"] == 99
    if obs["error"]:
        assert obs["error"]["code"] in ("calculator_error", "internal_error") and len(obs["error"]["message"]) < 100
        assert expr[:12] not in obs["error"]["message"] or len(expr) < 3
    out = act(env, "calculator", expression="2+2")[0]                 # env still healthy
    assert out["last_result"]["value"] == "4"


def test_13b_calculator_cannot_see_or_change_the_environment():
    env, *_ = new_env(1, **ALL4)
    before = (id(env._pub), len(env._pub.ledger), env._steps)
    for expr in ("env", "self", "_pub", "__builtins__", "os.getcwd()"):
        assert act(env, "calculator", expression=expr)[0]["error"] is not None
    assert (id(env._pub), len(env._pub.ledger)) == before[:2]


# 14 ----------------------------------------------------------------- double submit
def test_14_second_submit_is_refused_and_cannot_overwrite_the_score():
    env, *_ = new_env(1, mode="train", **ALL4)
    truth = env.evaluator_truth()
    _, r1, t1, *_ = submit(env, [])                                   # bad first answer
    first = (env.result.metrics.tp, env.result.reward)
    assert t1 and r1 == -1.0
    with pytest.raises(EpisodeFinished):
        submit(env, [finding(t) for t in truth])                     # try to fix it afterwards
    assert (env.result.metrics.tp, env.result.reward) == first


def test_14b_reset_is_the_only_way_to_start_over_and_gives_a_fresh_score():
    env, *_ = new_env(1, **ALL4)
    submit(env, [])
    env.reset(1, ALL4)
    submit(env, [finding(t) for t in env.evaluator_truth()])
    assert env.result.reward == 1.0


# 15 ----------------------------------------------------------------- file system / environment
def test_15_agent_cannot_find_private_files_or_paths(tmp_path, monkeypatch):
    sandbox_tmp = tmp_path / "tmp"
    sandbox_tmp.mkdir()
    monkeypatch.setenv("TMP", str(sandbox_tmp))
    monkeypatch.setenv("TEMP", str(sandbox_tmp))
    script = tmp_path / "fs_attack.py"
    script.write_text(textwrap.dedent('''
        import json, os, sys, tempfile
        def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
        def scan(root):
            found = []
            for d, _, fs in os.walk(root):
                for f in fs:
                    try:
                        if b"expected_amount_minor" in open(os.path.join(d, f), "rb").read(): found.append(os.path.join(d, f))
                    except OSError: pass
            return found
        for line in sys.stdin:
            m = json.loads(line)
            if m["type"] == "close": break
            if m["type"] == "reset": send({"type": "ready"}); continue
            report = {"cwd_files": os.listdir("."), "tmp_hits": scan(tempfile.gettempdir()), "pythonpath": os.environ.get("PYTHONPATH"),
                      "parent_hits": scan(os.path.dirname(os.getcwd())),
                      "env": sorted(k for k, v in os.environ.items() if "Genesis" in v or "datasets" in v or "episode" in v.lower())}
            send({"type": "action", "action": {"action": "get_record", "table": "ledger", "id": "REPORT" + json.dumps(report)}})
    '''), encoding="utf-8")
    before = sorted(os.listdir(sandbox_tmp))
    env, obs, _ = new_env(5, mode="eval", **ALL4)            # generated IN MEMORY: nothing is written to disk
    agent = SubprocessAgent(cmd=[sys.executable, str(script)])
    try:
        agent.reset(obs)
        action = agent.act(obs)
    finally:
        agent.close()
    report = json.loads(action["id"][len("REPORT"):])
    assert report["pythonpath"] is None                       # custom agents are not told where the project lives
    assert report["cwd_files"] == [] and report["tmp_hits"] == [] and report["parent_hits"] == [] and report["env"] == []
    assert sorted(os.listdir(sandbox_tmp)) == before          # the evaluator never wrote private data to disk


def test_15b_observations_contain_no_filesystem_paths():
    env, obs, info = new_env(1, mode="eval", **ALL4)
    blob = json.dumps([obs, info])
    for needle in ("C:\\\\", "Users", "Genesis1", "/home/", "site-packages", ".py", "datasets"):
        assert needle not in blob


# runner-protection attacks (agent tries to hurt the evaluator process) -------------------------
def test_16_oversized_message_cannot_exhaust_runner_memory(tmp_path):
    script = tmp_path / "flood.py"
    script.write_text("import sys\nfor line in sys.stdin:\n    sys.stdout.write('x' * 3_000_000)\n    sys.stdout.flush()\n    break\n", encoding="utf-8")
    t = time.time()
    agent = SubprocessAgent(cmd=[sys.executable, str(script)], max_line=100_000, timeout=20)
    with pytest.raises(AgentProtocolError, match="too large"):
        agent.reset({"manifest": {}})
    assert time.time() - t < 15 and agent.proc.poll() is not None      # runner survived, agent killed
    agent.close()


def test_17_stderr_flood_is_bounded_and_does_not_block_the_protocol(tmp_path):
    script = tmp_path / "noisy.py"
    script.write_text(textwrap.dedent('''
        import json, sys
        for line in sys.stdin:
            m = json.loads(line)
            if m["type"] == "close": break
            sys.stderr.write("E" * 20_000_000); sys.stderr.flush()
            sys.stdout.write(json.dumps({"type": "ready"}) + "\\n"); sys.stdout.flush()
    '''), encoding="utf-8")
    agent = SubprocessAgent(cmd=[sys.executable, str(script)], timeout=30)
    try:
        agent.reset({"manifest": {}})
        assert len(agent.stderr_tail()) <= 8192
    finally:
        agent.close()


def test_18_total_episode_deadline_is_enforced(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import json, sys, time\nfor line in sys.stdin:\n    time.sleep(0.4)\n"
                      "    sys.stdout.write(json.dumps({'type': 'ready'}) + '\\n'); sys.stdout.flush()\n", encoding="utf-8")
    agent = SubprocessAgent(cmd=[sys.executable, str(script)], timeout=5, total_timeout=1.0)
    try:
        with pytest.raises(AgentProtocolError):
            for _ in range(10):
                agent._call({"type": "reset", "observation": {}})
    finally:
        agent.close()
