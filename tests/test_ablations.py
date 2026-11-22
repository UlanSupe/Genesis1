"""Reward sensitivity: the auditor with one capability removed must lose exactly what it should."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from traceforge.policies import ABLATIONS, REGISTRY
from traceforge.runner.ablation import ORDER, run_ablation
from traceforge.runner.episode_runner import AGENT_NAMES, run_split

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"


@pytest.fixture(scope="module")
def dev():
    return {p: run_split(DATASETS, "dev", p) for p in ("reconciliation", *ABLATIONS)}


def mean(rs, k):
    return sum(r[k] for r in rs) / len(rs)


def test_ablations_are_not_leaderboard_policies():
    assert set(REGISTRY) == {"random", "reconciliation", "mark_all"} and not set(ABLATIONS) & set(AGENT_NAMES)
    assert set(ORDER) == {"reconciliation", *ABLATIONS}


def test_explanation_text_is_not_scored(dev):
    full, bare = dev["reconciliation"], dev["no_explanation"]
    assert [(r["tp"], r["fp"], r["fn"], r["reward"]) for r in full] == [(r["tp"], r["fp"], r["fn"], r["reward"]) for r in bare]


@pytest.mark.parametrize("kind", ["duplicate", "allocation", "fx", "cutoff"])
def test_blinding_one_error_family_costs_recall_but_never_precision(dev, kind):
    rs, full = dev[f"blind_{kind}"], dev["reconciliation"]
    assert sum(r["fp"] for r in rs) == 0 and sum(r["fn"] for r in rs) > 0
    assert sum(r["tp"] for r in rs) < sum(r["tp"] for r in full) and mean(rs, "reward") < mean(full, "reward") - 0.2


@pytest.mark.parametrize("name", ["thin_evidence", "wrong_amount"])
def test_incomplete_evidence_or_wrong_amount_earns_nothing(dev, name):
    assert sum(r["tp"] for r in dev[name]) == 0 and mean(dev[name], "reward") < 0


def test_unfounded_accusations_are_punished(dev):
    full, acc = dev["reconciliation"], dev["over_accuser"]
    assert sum(r["tp"] for r in acc) == sum(r["tp"] for r in full) and sum(r["fp"] for r in acc) == 3 * len(acc)
    assert mean(acc, "reward") < mean(full, "reward") - 0.5


def test_a_smaller_inspection_budget_means_less_found(dev):
    full, b8, b4 = dev["reconciliation"], dev["budget_8"], dev["budget_4"]
    assert mean(b8, "inspection_steps") <= 8 and mean(b4, "inspection_steps") <= 4
    assert sum(r["tp"] for r in full) > sum(r["tp"] for r in b8) > sum(r["tp"] for r in b4)
    assert mean(full, "reward") > mean(b8, "reward") > mean(b4, "reward")


def test_reward_ladder_is_ordered(dev):
    r = {p: mean(rs, "reward") for p, rs in dev.items()}
    assert r["reconciliation"] > max(r[f"blind_{k}"] for k in ("duplicate", "allocation", "fx", "cutoff"))
    assert min(r[f"blind_{k}"] for k in ("duplicate", "allocation", "fx", "cutoff")) > r["budget_8"] > r["over_accuser"] - 0.2
    assert r["over_accuser"] > r["thin_evidence"]


def test_isolated_run_equals_in_process_run(dev):
    iso = run_split(DATASETS, "dev", "blind_fx", isolated=True)
    assert [(r["tp"], r["fp"], r["fn"], r["reward"]) for r in iso] == [(r["tp"], r["fp"], r["fn"], r["reward"]) for r in dev["blind_fx"]]


def test_ablations_import_nothing_from_generator_or_evaluator():
    code = ("import sys, traceforge.policies.ablations\n"
            "print(sorted(m for m in sys.modules if m.startswith('traceforge.') and m.split('.')[1] in "
            "('generator','evaluation','reporting','runner')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env={"PYTHONPATH": str(ROOT / "src"), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}).stdout.strip()
    assert out == "[]", out


def test_stored_ablation_file_is_reproducible():
    saved = json.loads((ROOT / "FINAL_ABLATION.json").read_text(encoding="utf-8"))
    fresh = run_ablation(DATASETS, ("dev", "train"))
    assert fresh["rows"] == saved["rows"]
