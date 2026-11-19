import pytest

from conftest import ALL4, new_env
from traceforge.environment.audit_env import AuditEnv
from traceforge.evaluation.oracle_policy import OraclePolicy
from traceforge.evaluation.metrics import summarize
from traceforge.policies import REGISTRY, make_policy
from traceforge.runner.episode_runner import make_agent, run_episode

CFGS = [ALL4, {"clean": True}, {"scenarios": ["cross_border_partial"]}, {"scenarios": ["period_end_duplicate"]},
        {"issue_counts": {"duplicate": 2, "fx": 2}}]


def _run(policy, seeds=range(1, 6), cfgs=CFGS, mode="train"):
    return [run_episode(AuditEnv(mode), make_agent(policy, 0), seed=s, config=c, label=policy)
            for s in seeds for c in cfgs]


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_policy_completes_within_step_limit_and_submits(name):
    for r in _run(name, seeds=[1, 2]):
        assert r["terminated"] and not r["truncated"] and r["steps"] <= 100 and r["agent_error"] is None
        assert r["valid"]


def test_four_policies_exist_three_agent_side_plus_evaluator_oracle():
    assert set(REGISTRY) == {"random", "reconciliation", "mark_all"}
    assert OraclePolicy([]).name == "oracle"


def test_oracle_reaches_the_maximum_reward():
    for r in _run("oracle", seeds=[1, 2, 3]):
        assert r["f1"] == 1.0 and r["reward"] == 1.0 and r["steps"] == 1


def test_random_policy_is_deterministic_per_seed_and_varies_across_seeds():
    def trace(seed):
        env = AuditEnv("eval")
        obs, _ = env.reset(1, ALL4)
        pol = make_policy("random", seed)
        pol.reset(obs)
        acts, done = [], False
        while not done:
            a = pol.act(obs)
            acts.append(a)
            obs, _, t, tr, _ = env.step(a)
            done = t or tr
        return acts
    assert trace(5) == trace(5) and trace(5) != trace(6)


def test_random_policy_does_not_touch_global_random_state():
    import random
    random.seed(7)
    state = random.getstate()
    _run("random", seeds=[1], cfgs=[ALL4])
    assert random.getstate() == state


def test_random_policy_scores_poorly_but_is_valid():
    res = _run("random")
    assert summarize(res)["mean_f1"] < 0.2 and all(r["valid"] for r in res)


def test_mark_everything_is_punished_for_false_positives():
    res = _run("mark_all")
    s = summarize(res)
    assert s["fp"] > 20 * len(res) * 0.5 and s["mean_reward"] < -0.5
    clean = [r for r in res if r["n_truth"] == 0]
    assert clean and all(r["fp"] >= 20 and r["reward"] < -0.5 for r in clean)


def test_simple_reconciliation_solves_public_data_and_beats_baselines():
    rec, mark, rnd = (summarize(_run(p)) for p in ("reconciliation", "mark_all", "random"))
    assert rec["mean_f1"] >= 0.9 > max(mark["mean_f1"], rnd["mean_f1"])
    assert rec["mean_reward"] > 0.8 > rnd["mean_reward"]


def test_reconciliation_makes_no_accusations_on_clean_companies():
    for r in _run("reconciliation", seeds=range(1, 9), cfgs=[{"clean": True, "difficulty": "hard"}]):
        assert r["fp"] == 0 and r["f1"] == 1.0


def test_reconciliation_reads_all_pages_with_few_steps():
    r = _run("reconciliation", seeds=[1], cfgs=[ALL4])[0]
    assert 10 <= r["steps"] <= 25


def test_policies_cannot_reach_evaluator_code_or_private_data():
    """Static check: no agent-side module imports evaluator/generator/env code or reads files."""
    import ast
    from pathlib import Path
    import traceforge.policies as pkg
    banned = ("traceforge.evaluation", "traceforge.generator", "traceforge.environment.audit_env",
              "traceforge.episode_io", "traceforge.runner.episode_runner")
    files = list(Path(pkg.__file__).parent.glob("*.py")) + [Path(pkg.__file__).parents[1] / "runner" / "agent_worker.py"]
    assert len(files) >= 6
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom):
                base = "traceforge." + (node.module or "") if node.level else (node.module or "")
                if node.level == 2 or (node.level == 1 and "policies" not in str(f)):
                    base = "traceforge." + (node.module or "")
                mods = [base] + [f"{base}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            assert not any(m.startswith(banned) for m in mods), (f.name, mods)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("open", "eval", "exec", "__import__"), (f.name, node.func.id)


def test_policy_act_takes_only_the_observation_and_returns_json():
    import inspect
    import json
    env, obs, _ = new_env(1, **ALL4)
    for name in REGISTRY:
        pol = make_policy(name, 0)
        assert list(inspect.signature(pol.act).parameters) == ["observation"]
        assert list(inspect.signature(pol.reset).parameters) == ["observation"]
        pol.reset(json.loads(json.dumps(obs)))                       # only plain JSON data is needed
        action = pol.act(json.loads(json.dumps(obs)))
        assert json.loads(json.dumps(action)) == action and action["action"] in ("list_records", "get_record", "get_document", "calculator", "submit")


def test_unknown_policy_name_raises():
    with pytest.raises(KeyError):
        make_policy("oracle")
