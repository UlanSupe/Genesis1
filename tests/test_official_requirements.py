"""Independent black-box verification of the official competition requirements (A-I).

These tests deliberately re-derive their expectations (own reward formula, real environment
episodes, real evaluator) instead of reusing helpers of the implementation under test."""
import builtins
import hashlib
import json
import os
import socket
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from conftest import ALL4
from traceforge import cli
from traceforge.constants import ACTIONS, ISSUE_TYPES
from traceforge.environment.audit_env import AuditEnv
from traceforge.episode_io import load_episode
from traceforge.generator import validation
from traceforge.generator.builder import generate_episode
from traceforge.policies import REGISTRY
from traceforge.runner.episode_runner import AGENT_NAMES, make_agent, run_episode

ROOT = Path(__file__).resolve().parents[1]


def _tree_digest(*roots) -> str:
    h = hashlib.sha256()
    for root in roots:
        for f in sorted(Path(root).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                h.update(str(f.relative_to(ROOT)).encode() + f.read_bytes())
    return h.hexdigest()


# ============================================================== A. environment contract
def test_A_reset_seed_config_returns_observation_info_and_step_returns_five_values():
    env = AuditEnv("train")
    out = env.reset(7, {"difficulty": "easy"})                      # positional (seed, config)
    assert isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict) and isinstance(out[1], dict)
    res = env.step({"action": "calculator", "expression": "1+1"})
    assert isinstance(res, tuple) and len(res) == 5
    obs, reward, terminated, truncated, info = res
    assert isinstance(obs, dict) and isinstance(reward, float) and terminated is False and truncated is False and isinstance(info, dict)
    obs, reward, terminated, truncated, info = env.step({"action": "submit", "findings": []})
    assert terminated is True and truncated is False and isinstance(reward, float)


# ============================================================== B. required tools
def test_B_exactly_the_five_required_tools_exist_and_work():
    assert set(ACTIONS) == {"list_records", "get_record", "get_document", "calculator", "submit"}
    env = AuditEnv("eval")
    obs, _ = env.reset(3, {"clean": True})
    pub = env._pub
    calls = [{"action": "list_records", "table": "ledger", "cursor": None, "limit": 5},
             {"action": "get_record", "table": "invoices", "id": pub.invoices[0]["invoice_id"]},
             {"action": "get_document", "doc_id": pub.payments[0]["doc_id"]},
             {"action": "calculator", "expression": "(2.5 + 1) * 4"}]
    for c in calls:
        o = env.step(c)[0]
        assert o["error"] is None and o["last_result"]["ok"] is True, c
    assert env.step({"action": "calculator", "expression": "(2.5 + 1) * 4"})[0]["last_result"]["value"] == "14"
    assert env.step({"action": "submit", "findings": []})[2] is True


def test_B_tools_reject_anything_else_safely():
    env = AuditEnv("eval")
    env.reset(3, {"clean": True})
    for bad in ("shell", "sql", "read_file", "open", "http_get", "python", "exec", "eval", "write_record", "delete_record"):
        o = env.step({"action": bad})[0]
        assert o["error"]["code"] == "unknown_action" and o["last_result"]["ok"] is False


# ============================================================== C. restrictions
def test_C_agent_interaction_is_read_only_no_network_no_subprocess(monkeypatch):
    """Run complete in-process episodes of every agent policy with ALL write/delete/network/process
    primitives booby-trapped: any attempt by the environment or a policy raises and fails the test."""
    before = _tree_digest(ROOT / "src", ROOT / "datasets")
    real_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(m in str(mode) for m in "wax+"):
            raise AssertionError(f"write attempt: {file}")
        return real_open(file, mode, *a, **k)

    def boom(*a, **k):
        raise AssertionError("forbidden primitive used")
    monkeypatch.setattr(builtins, "open", guarded_open)
    for name, mod in (("remove", os), ("unlink", os), ("rmdir", os), ("rename", os), ("replace", os), ("mkdir", os), ("makedirs", os),
                      ("system", os), ("popen", os)):
        monkeypatch.setattr(mod, name, boom)
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(Path, "write_text", boom)
    monkeypatch.setattr(Path, "write_bytes", boom)
    monkeypatch.setattr(Path, "unlink", boom)
    for pol in sorted(REGISTRY) + ["oracle"]:
        r = run_episode(AuditEnv("eval"), make_agent(pol, 0), seed=11, config=ALL4)
        assert r["agent_error"] is None
    # a stored episode is loaded read-only as well
    run_episode(AuditEnv("eval"), make_agent("reconciliation"), config={"episode_dir": str(ROOT / "datasets" / "dev" / "dev-002")})
    monkeypatch.undo()
    assert _tree_digest(ROOT / "src", ROOT / "datasets") == before


def test_C_limits_50_rows_and_100_steps_and_unknown_id_costs_a_step():
    env = AuditEnv("train")
    env.reset(4, ALL4)
    assert len(env.step({"action": "list_records", "table": "ledger", "limit": 50})[0]["last_result"]["records"]) == 50
    assert env.step({"action": "list_records", "table": "ledger", "limit": 51})[0]["error"]["code"] == "invalid_limit"
    assert env.step({"action": "get_record", "table": "ledger", "id": "ROW-99999"})[0]["steps_remaining"] == 97
    for _ in range(97):
        out = env.step({"action": "calculator", "expression": "1"})
    assert out[3] is True and out[0]["steps_remaining"] == 0         # truncated at exactly 100


def test_C_observations_never_contain_hidden_ground_truth():
    env = AuditEnv("eval")
    obs, info = env.reset(5, ALL4)
    tokens = env.leak_tokens() + ["expected_amount_minor", "allowed_evidence_sets", "root_cause", "causal_description"]
    seen = [obs, info]
    for table in ("ledger", "invoices", "payments", "fx"):
        seen.append(env.step({"action": "list_records", "table": table})[0])
    seen.append(env.step({"action": "get_record", "table": "ledger", "id": "truth.json"})[0])
    blob = json.dumps(seen)
    assert not [t for t in tokens if t in blob]


# ============================================================== D. anomaly types
@pytest.mark.parametrize("typ", ISSUE_TYPES)
def test_D_each_required_anomaly_type_is_injected_and_independently_detectable(typ):
    ep = generate_episode(2, {"issue_counts": {typ: 1}})
    assert [t["type"] for t in ep.private.truth] == [typ]
    found = validation.derive_issues(ep.public)           # derived from PUBLIC data only
    assert [f["type"] for f in found] == [typ]
    t = ep.private.truth[0]
    assert (found[0]["affected_ids"], found[0]["expected_amount_minor"], found[0]["currency"]) == \
           (sorted(t["affected_ids"]), t["expected_amount_minor"], t["currency"])


# ============================================================== E. evaluation
def _truth(env):
    return env.evaluator_truth()


def _finding(t, **kw):
    f = {"type": t["type"], "affected_ids": list(t["affected_ids"]), "claimed_amount_minor": t["expected_amount_minor"],
         "currency": t["currency"], "evidence_ids": list(t["allowed_evidence_sets"][0]), "explanation": "x"}
    f.update(kw)
    return f


def _score(findings, seed=21, cfg=ALL4):
    env = AuditEnv("train")
    env.reset(seed, cfg)
    env.step({"action": "submit", "findings": {"episode_id": env._evaluator.episode_id, "findings": findings(env)}})
    m = env.result.metrics
    return m.tp, m.fp, m.fn


@pytest.mark.parametrize("name,build,expected", [
    ("all exact", lambda e: [_finding(t) for t in _truth(e)], (4, 0, 0)),
    ("amount +1 minor unit accepted", lambda e: [_finding(t, claimed_amount_minor=t["expected_amount_minor"] + 1) for t in _truth(e)], (4, 0, 0)),
    ("amount -1 minor unit accepted", lambda e: [_finding(t, claimed_amount_minor=t["expected_amount_minor"] - 1) for t in _truth(e)], (4, 0, 0)),
    ("amount +2 rejected", lambda e: [_finding(t, claimed_amount_minor=t["expected_amount_minor"] + 2) for t in _truth(e)], (0, 4, 4)),
    ("wrong currency", lambda e: [_finding(t, currency="XXX") for t in _truth(e)], (0, 4, 4)),
    ("wrong type", lambda e: [_finding(t, type="duplicate" if t["type"] != "duplicate" else "fx") for t in _truth(e)], (0, 4, 4)),
    ("wrong affected ids", lambda e: [_finding(t, affected_ids=["NOPE-1"]) for t in _truth(e)], (0, 4, 4)),
    ("no evidence", lambda e: [_finding(t, evidence_ids=[]) for t in _truth(e)], (0, 4, 4)),
    ("incomplete evidence", lambda e: [_finding(t, evidence_ids=t["allowed_evidence_sets"][0][:1]) for t in _truth(e)], (0, 4, 4)),
    ("duplicates are false positives", lambda e: [_finding(t) for t in _truth(e)] * 2, (4, 4, 0)),
    ("unmatched finding is a false positive", lambda e: [_finding(t) for t in _truth(e)] + [_finding(_truth(e)[0], affected_ids=["X"])], (4, 1, 0)),
    ("missed anomalies are false negatives", lambda e: [_finding(t) for t in _truth(e)[:1]], (1, 0, 3)),
    ("empty submission", lambda e: [], (0, 0, 4)),
])
def test_E_one_to_one_matching_rules(name, build, expected):
    assert _score(build) == expected, name


def test_E_matching_is_order_independent_and_each_truth_matches_once():
    import random
    base = None
    for seed in range(12):
        def build(e, seed=seed):
            fs = [_finding(t) for t in _truth(e)] + [_finding(_truth(e)[1])] * 2
            random.Random(seed).shuffle(fs)
            return fs
        res = _score(build)
        assert res == (4, 2, 0)
        base = res
    assert base == (4, 2, 0)


# ============================================================== F. reward
def _official_reward(tp, fp, fn, steps) -> Decimal:
    denom = 2 * tp + fp + fn
    f1 = Decimal(1) if denom == 0 else Decimal(2 * tp) / Decimal(denom)
    return max(Decimal(-1), min(Decimal(1), 2 * f1 - 1 - Decimal("0.002") * steps))


def test_F_reward_formula_exact_over_the_whole_grid():
    from traceforge.evaluation.reward import f1_fraction, terminal_reward
    n = 0
    for tp in range(0, 7):
        for fp in range(0, 7):
            for fn in range(0, 7):
                for steps in (0, 1, 2, 3, 5, 10, 25, 50, 99, 100):
                    got = Decimal(repr(terminal_reward(f1_fraction(tp, fp, fn), steps)))
                    assert abs(got - _official_reward(tp, fp, fn, steps)) < Decimal("1e-12"), (tp, fp, fn, steps)
                    n += 1
    assert n == 7 * 7 * 7 * 10


@pytest.mark.parametrize("n_inspect,findings_kind", [(0, "all"), (7, "all"), (40, "half"), (99, "none"), (3, "all+fp")])
def test_F_reward_through_the_real_environment(n_inspect, findings_kind):
    env = AuditEnv("train")
    env.reset(31, ALL4)
    truth = env.evaluator_truth()
    for _ in range(n_inspect):
        env.step({"action": "calculator", "expression": "1"})
    fs = {"all": [_finding(t) for t in truth], "half": [_finding(t) for t in truth[:2]], "none": [],
          "all+fp": [_finding(t) for t in truth] + [_finding(truth[0], affected_ids=["Z"])]}[findings_kind]
    _, reward, terminated, _, _ = env.step({"action": "submit", "findings": fs})
    m = env.result.metrics
    assert terminated and m.inspection_steps == n_inspect
    assert Decimal(repr(reward)) == _official_reward(m.tp, m.fp, m.fn, n_inspect).quantize(Decimal(1).scaleb(-15)) or \
        abs(Decimal(repr(reward)) - _official_reward(m.tp, m.fp, m.fn, n_inspect)) < Decimal("1e-12")


def test_F_invalid_submission_gets_minus_one_and_exploration_gets_zero():
    env = AuditEnv("train")
    env.reset(31, ALL4)
    assert env.step({"action": "calculator", "expression": "2*2"})[1] == 0.0
    assert env.step({"action": "submit", "findings": "garbage"})[1] == -1.0


# ============================================================== G. required policies
def test_G_four_policies_and_oracle_is_evaluator_only():
    assert set(AGENT_NAMES) == {"random", "reconciliation", "mark_all", "oracle"}
    assert set(REGISTRY) == {"random", "reconciliation", "mark_all"}                 # oracle is not an agent policy
    p = subprocess.run([sys.executable, "-m", "traceforge.runner.agent_worker", "--policy", "oracle"], capture_output=True, text=True,
                       input="", env={"PYTHONPATH": str(ROOT / "src"), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
    assert p.returncode == 2                                                           # the agent worker refuses the oracle
    r = run_episode(AuditEnv("eval"), make_agent("oracle"), seed=3, config=ALL4)
    assert r["f1"] == 1.0 and r["reward"] == 1.0


def test_G_mark_all_is_penalised_and_random_is_not_favoured():
    ma = run_episode(AuditEnv("eval"), make_agent("mark_all"), seed=3, config=ALL4)
    rn = run_episode(AuditEnv("eval"), make_agent("random", 0), seed=3, config=ALL4)
    rc = run_episode(AuditEnv("eval"), make_agent("reconciliation"), seed=3, config=ALL4)
    assert ma["fp"] >= 20 and ma["reward"] < -0.9
    assert rn["steps"] <= 100 and rn["reward"] <= 0 and rn["n_findings"] <= 3        # same step cap, no free credit
    assert rc["reward"] > 0.9


# ============================================================== H. reproducibility
def test_H_fixed_seed_generation_and_stable_benchmark_and_recorded_trajectories(tmp_path):
    from traceforge.episode_io import public_files
    a, b = generate_episode(99, ALL4), generate_episode(99, ALL4)
    assert public_files(a.public) == public_files(b.public) and a.private.truth == b.private.truth
    from traceforge.runner.benchmark import run_benchmark
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("runtime_s", "env_elapsed_s")} for r in rows]
    r1, r2 = run_benchmark(("dev",), datasets=ROOT / "datasets"), run_benchmark(("dev",), datasets=ROOT / "datasets")
    assert strip(r1["rows"]) == strip(r2["rows"])
    from traceforge.trajectories.reader import read_trajectory, replay
    from traceforge.trajectories.recorder import TrajectoryRecorder
    env = AuditEnv("eval")
    with TrajectoryRecorder(tmp_path / "t.jsonl") as rec:
        run_episode(env, make_agent("reconciliation"), seed=99, config=ALL4, recorder=rec)
    lines = read_trajectory(tmp_path / "t.jsonl")
    assert len(lines) > 10 and lines[-1]["terminated"] and replay(tmp_path / "t.jsonl", AuditEnv("eval"), 99, ALL4)["ok"]


def test_H_metamorphic_transformation_preserves_correctness(tmp_path):
    from traceforge.episode_io import save_episode
    from traceforge.generator.transformations import map_findings, transform_episode
    ep = generate_episode(17, ALL4)
    new, mapping = transform_episode(ep, 5)
    answer = [_finding(t) for t in ep.private.truth]
    for tag, e, fs in (("a", ep, answer), ("b", new, map_findings(answer, mapping))):
        save_episode(e, tmp_path / tag)
        env = AuditEnv("eval")
        env.reset(config={"episode_dir": str(tmp_path / tag)})
        env.step({"action": "submit", "findings": {"episode_id": e.public.manifest["episode_id"], "findings": fs}})
        assert env.result.metrics.f1 == 1.0


# ============================================================== I. deliverables
def test_I_deliverables_exist_and_commands_work(tmp_path):
    for f in ("README.md", "ARCHITECTURE.md", "SPEC_COMPLIANCE.md", "FINAL_AUDIT.md", "JUDGE_SUMMARY.md", "DEMO_SCRIPT.md",
              "ENGINEERING_AUDIT.md", "FINAL_ENGINEERING_REPORT.md", "pyproject.toml", "requirements.txt", "FINAL_BENCHMARK.json",
              "FINAL_BENCHMARK.csv", "LICENSE"):
        assert (ROOT / f).exists(), f
    for split in ("train", "dev"):
        names = json.loads((ROOT / "datasets" / split / "index.json").read_text())["episodes"]
        assert len(names) == 10 and all((ROOT / "datasets" / split / n / "private" / "truth.json").exists() for n in names)
    p = subprocess.run([sys.executable, "-m", "traceforge", "--help"], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert all(c in p.stdout for c in ("generate", "verify", "demo", "evaluate", "inspect", "audit", "benchmark"))
    assert (ROOT / "src" / "traceforge" / "environment").is_dir() and len(list((ROOT / "tests").glob("test_*.py"))) >= 20


def test_I_final_benchmark_file_is_consistent_with_a_fresh_run():
    saved = json.loads((ROOT / "FINAL_BENCHMARK.json").read_text(encoding="utf-8"))
    assert saved["meta"]["episodes"] == len({(r["suite"], r["episode"]) for r in saved["rows"]})
    assert {r["policy"] for r in saved["rows"]} == {"random", "reconciliation", "mark_all", "oracle"}
    from traceforge.runner.benchmark import run_benchmark
    fresh = run_benchmark(("dev",), datasets=ROOT / "datasets")
    pick = lambda rows: sorted((r["suite"], r["episode"], r["policy"], r["tp"], r["fp"], r["fn"], r["reward"], r["inspection_steps"])
                               for r in rows if r["suite"] == "dev")
    assert pick(fresh["rows"]) == pick(saved["rows"])                  # the saved numbers are reproducible, not invented
