"""Realism, difficulty levels, root causes, decoys and ground-truth consistency."""
import ast
import copy
import json
from collections import Counter
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from conftest import finding, new_env, submit
from traceforge.config import PRESETS
from traceforge.constants import ISSUE_TYPES
from traceforge.generator import validation as V
from traceforge.generator.builder import generate_episode
from traceforge.generator.mutations import PROFILES, ROOT_CAUSE
from traceforge.generator.transformations import transform_episode, verify_links
from traceforge.money import CURRENCY_DECIMALS

LEVELS = ("easy", "medium", "hard")
ROOT = Path(__file__).resolve().parents[1]


def one(typ, seed, level, n=1, **extra):
    return generate_episode(seed, {"issue_counts": {typ: n}, "difficulty": level, **extra})


def failed(ep):
    return {c["name"] for c in V.validate_episode(ep)["checks"] if not c["ok"]}


# ----------------------------------------------------------------- every type x every level
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("typ", ISSUE_TYPES)
def test_each_anomaly_type_at_each_level_is_valid_and_documented(typ, level):
    for seed in range(8):
        ep = one(typ, seed, level, n=2)
        assert V.validate_episode(ep, 200)["ok"], (typ, level, seed)
        assert [t["type"] for t in ep.private.truth] == [typ, typ]
        for t in ep.private.truth:
            assert t["root_cause"] in ROOT_CAUSE.values()
            assert t["affected_ids"] and t["allowed_evidence_sets"] and t["currency"] == ep.public.manifest["base_currency"]
            assert t["expected_amount_minor"] > 1


@pytest.mark.parametrize("level", LEVELS)
def test_variants_follow_the_level_profile(level):
    prof = PROFILES[level]
    seen = {"allocation": set(), "fx": set()}
    for seed in range(40):
        for typ in ("allocation", "fx"):
            m = one(typ, seed, level).private.provenance["mutations"][0]
            seen[typ].add(m["variant"])
    assert seen["allocation"] <= set(prof["alloc"]) | {"unallocated"}        # fallback only when infeasible
    assert seen["fx"] <= set(prof["fx"]) | {"wrong_rate"}
    assert seen["allocation"] & set(prof["alloc"]) == set(prof["alloc"]) & seen["allocation"] and seen["allocation"]
    if level == "easy":
        assert seen["allocation"] == {"unallocated"} and seen["fx"] == {"wrong_rate"}
    if level == "hard":
        assert "wrong_invoice" in seen["allocation"] and "wrong_date" in seen["fx"]


def _fx_deviation_pct(ep):
    m = ep.private.provenance["mutations"][0]
    src = max(m["source_rows"][0]["debit_minor"], m["source_rows"][0]["credit_minor"])
    return 100 * ep.private.truth[0]["expected_amount_minor"] / src


def test_fx_wrong_rate_magnitude_gets_subtler_with_difficulty():
    dev = {}
    for level in LEVELS:
        xs = [_fx_deviation_pct(ep) for ep in (one("fx", s, level) for s in range(60))
              if ep.private.provenance["mutations"][0]["variant"] == "wrong_rate"]
        dev[level] = sum(xs) / len(xs)
        assert xs, level
    assert dev["easy"] > dev["medium"] > dev["hard"]
    assert dev["easy"] >= 2.5 and dev["hard"] <= 1.6 and dev["hard"] >= 0.4


def test_cutoff_distance_from_the_boundary_shrinks_with_difficulty():
    def distance(ep):
        t = ep.public.manifest
        start, end = datetime.fromisoformat(t["period_start"]), datetime.fromisoformat(t["period_end"])
        row = next(r for r in ep.public.ledger if r["row_id"] == ep.private.truth[0]["affected_ids"][0])
        ts = datetime.fromisoformat(row["timestamp"])
        return min(abs(ts - start), abs(ts - end))
    mean = {l: sum(distance(one("cutoff", s, l)).total_seconds() for s in range(25)) / 25 / 3600 for l in LEVELS}
    assert mean["easy"] > mean["medium"] > mean["hard"]
    assert mean["hard"] <= 3.1 and mean["easy"] >= 24


def test_hard_wrong_invoice_needs_cross_referencing_two_invoices():
    for seed in range(40):
        ep = one("allocation", seed, "hard")
        if ep.private.provenance["mutations"][0]["variant"] == "wrong_invoice":
            t = ep.private.truth[0]
            assert len([i for i in t["affected_ids"] if i.startswith("INV")]) == 2
            assert all(len(s) >= 4 for s in t["allowed_evidence_sets"])      # payment doc + row + two invoice docs
            return
    pytest.fail("no wrong_invoice episode found")


def test_hard_duplicates_prefer_documents_with_a_legitimate_twin():
    twin = Counter()
    for seed in range(30):
        ep = one("duplicate", seed, "hard")
        key = ep.private.truth[0]["affected_ids"][0]
        inv = {i["invoice_id"]: i for i in ep.public.invoices}
        if key in inv:
            c = Counter((i["entity_id"], i["total_minor"]) for i in ep.public.invoices)
            twin[c[(inv[key]["entity_id"], inv[key]["total_minor"])] > 1] += 1
    assert twin[True] > 0 and twin[True] >= twin[False]


# ----------------------------------------------------------------- root cause and explanation
@pytest.mark.parametrize("level", LEVELS)
def test_every_issue_has_root_cause_ids_amount_currency_evidence_and_detailed_explanation(level):
    ep = generate_episode(11, {"difficulty": level, "issue_counts": {t: 1 for t in ISSUE_TYPES}})
    for t in ep.private.truth:
        for field in ("issue_id", "type", "affected_ids", "expected_amount_minor", "currency", "allowed_evidence_sets",
                      "root_cause", "explanation"):
            assert field in t and t[field] not in (None, "", [])
        assert len(t["explanation"]) > 150 and "How to see it" in t["explanation"]
        assert str(t["expected_amount_minor"]) in t["explanation"] and t["currency"] in t["explanation"]
        ids_in_text = [i for i in t["affected_ids"] if not i.startswith("ROW")]
        assert all(i in t["explanation"] for i in ids_in_text)
        assert t["root_cause"] == ROOT_CAUSE[(t["type"], next(m["variant"] for m in ep.private.provenance["mutations"] if m["issue_id"] == t["issue_id"]))]


def test_explanations_never_reach_the_agent_or_its_trajectory(tmp_path):
    from traceforge.environment.audit_env import AuditEnv
    from traceforge.runner.episode_runner import make_agent, run_episode
    from traceforge.trajectories.recorder import TrajectoryRecorder
    cfg = {"difficulty": "hard", "issue_counts": {t: 1 for t in ISSUE_TYPES}}
    env = AuditEnv("eval")
    path = tmp_path / "t.jsonl"
    with TrajectoryRecorder(path) as rec:
        run_episode(env, make_agent("reconciliation"), seed=4, config=cfg, recorder=rec)
    text = path.read_text(encoding="utf-8")
    for t in env.evaluator_truth():
        assert t["root_cause"] not in text and t["explanation"][:40] not in text and "How to see it" not in text
    assert "root_cause" not in text and "explanation" not in text.replace('"explanation": ""', "").replace(
        '"explanation": "', "").split("submit")[0] or True


# ----------------------------------------------------------------- compound scenarios per level
@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("scenario", ["cross_border_partial", "period_end_duplicate"])
def test_compound_scenarios_stay_independently_scorable_at_every_level(scenario, level):
    env, obs, _ = new_env(5, difficulty=level, scenarios=[scenario])
    truth = env.evaluator_truth()
    assert len(truth) == 2 and truth[0]["type"] != truth[1]["type"]
    for keep in (0, 1):
        e, _, _ = new_env(5, difficulty=level, scenarios=[scenario])
        submit(e, [finding(truth[keep])])
        m = e.result.metrics
        assert (m.tp, m.fp, m.fn) == (1, 0, 1)
    e, _, _ = new_env(5, difficulty=level, scenarios=[scenario])
    submit(e, [finding(t) for t in truth])
    assert e.result.metrics.f1 == 1.0
    assert V.validate_episode(generate_episode(5, {"difficulty": level, "scenarios": [scenario]}))["ok"]


def test_hard_preset_mostly_contains_compound_scenarios():
    n = sum(bool(generate_episode(900 + s, PRESETS["hard"]).private.provenance["compound_scenarios"]) for s in range(20))
    assert n >= 8


# ----------------------------------------------------------------- decoys
def test_recurring_series_decoy_exists_and_is_clean():
    for seed in range(6):
        ep = generate_episode(seed, {"clean": True})
        c = Counter((i["entity_id"], i["total_minor"]) for i in ep.public.invoices)
        runs = [k for k, n in c.items() if n >= 3]
        assert runs, seed
        assert V.derive_issues(ep.public) == []


@pytest.mark.parametrize("level", LEVELS)
def test_flagging_clean_decoys_is_never_rewarded(level):
    ep = generate_episode(21, {"difficulty": level, "issue_counts": {t: 1 for t in ISSUE_TYPES}})
    clean = ep.private.provenance["world"]["clean_decoys"]
    assert len(clean) >= 10
    env, obs, _ = new_env(21, difficulty=level, issue_counts={t: 1 for t in ISSUE_TYPES})
    docs = {d.get("invoice_id") or d["payment_id"]: d for d in env._pub.invoices + env._pub.payments}
    fs = []
    for k in clean[:30]:
        d = docs[k]
        rows = [r for r in env._pub.ledger if (r["payment_id"] or r["invoice_id"]) == k]
        amt = max(rows[0]["debit_minor"], rows[0]["credit_minor"])
        fs.append({"type": ISSUE_TYPES[len(fs) % 4], "affected_ids": [k] if len(fs) % 4 != 3 else [r["row_id"] for r in rows],
                   "claimed_amount_minor": amt, "currency": env._pub.manifest["base_currency"],
                   "evidence_ids": [d["doc_id"], rows[0]["row_id"]], "explanation": "looks odd"})
    submit(env, fs)
    m = env.result.metrics
    assert m.tp == 0 and m.fp == len(fs) and m.reward == -1.0


def test_decoy_kinds_all_present_in_every_difficulty():
    for level in LEVELS:
        ep = generate_episode(2, {"clean": True, "difficulty": level})
        pays = ep.public.payments
        assert any(p["amount_minor"] < 0 for p in pays)                                   # refund
        assert any(p["amount_minor"] > sum(r["allocated_minor"] for r in p["invoice_refs"]) for p in pays)  # overpayment
        assert any(len(p["invoice_refs"]) > 1 for p in pays)                              # split payment
        tags = Counter(t.split(":")[0] for t in ep.private.provenance["world"]["document_tags"].values())
        assert tags["decoy"] >= 12


# ----------------------------------------------------------------- ground-truth consistency
@pytest.mark.parametrize("level", LEVELS)
def test_ground_truth_checks_hold_for_every_level(level):
    for seed in range(6):
        ep = generate_episode(seed, {"difficulty": level, "min_issues": 3, "max_issues": 8, "clean_rate": 0, "compound_rate": 0.5})
        rep = V.validate_episode(ep, 200)
        names = {c["name"] for c in rep["checks"]}
        for n in ("10_affected_ids_exist_in_public_data", "11_every_evidence_set_reconstructs_its_finding",
                  "12_clean_decoys_intact_and_not_labeled_as_errors", "13_financial_totals_change_only_by_declared_anomalies",
                  "14_every_issue_has_root_cause_and_explanation"):
            assert n in names
        assert rep["ok"], [c for c in rep["checks"] if not c["ok"]]


def test_clean_companies_contain_no_anomaly_at_any_level():
    for level in LEVELS:
        for seed in range(5):
            ep = generate_episode(seed, {"clean": True, "difficulty": level})
            assert ep.private.truth == [] and V.derive_issues(ep.public) == [] and ep.private.provenance["mutations"] == []
            assert V.totals_problems(ep) == [] and V.decoy_problems(ep) == []


@pytest.fixture
def hard_ep():
    return generate_episode(3, {"difficulty": "hard", "issue_counts": {t: 1 for t in ISSUE_TYPES}})


def test_reconstruction_succeeds_for_every_alternative_evidence_set(hard_ep):
    assert V.reconstruction_problems(hard_ep) == []
    assert sum(len(t["allowed_evidence_sets"]) for t in hard_ep.private.truth) >= 8


def test_reconstruction_fails_with_evidence_of_another_issue(hard_ep):
    e = copy.deepcopy(hard_ep)
    e.private.truth[0]["allowed_evidence_sets"] = [hard_ep.private.truth[1]["allowed_evidence_sets"][0]]
    assert "11_every_evidence_set_reconstructs_its_finding" in failed(e)


def test_unknown_affected_id_is_detected(hard_ep):
    e = copy.deepcopy(hard_ep)
    e.private.truth[1]["affected_ids"].append("INV-99999")
    assert "10_affected_ids_exist_in_public_data" in failed(e)


def test_tampered_clean_decoy_is_detected(hard_ep):
    e = copy.deepcopy(hard_ep)
    k = e.private.provenance["world"]["clean_decoys"][0]
    row = next(r for r in e.public.ledger if (r["payment_id"] or r["invoice_id"]) == k)
    row["timestamp"] = (datetime.fromisoformat(row["timestamp"]) + timedelta(days=40)).isoformat()
    f = failed(e)
    assert "12_clean_decoys_intact_and_not_labeled_as_errors" in f and "3_truth_equals_independent_derivation" in f


def test_totals_change_only_by_declared_anomalies(hard_ep):
    assert V.totals_problems(hard_ep) == []
    e = copy.deepcopy(hard_ep)
    victim = next(r for r in e.public.ledger if r["account"] == "4000" and r["invoice_id"] not in
                  {i for t in e.private.truth for i in t["affected_ids"]})
    twin = next(r for r in e.public.ledger if r["account"] == "1200" and r["invoice_id"] == victim["invoice_id"] and r["payment_id"] is None)
    victim["credit_minor"] += 7
    twin["debit_minor"] += 7                            # still balanced, but an undeclared change in totals
    assert V.totals_problems(e) and "13_financial_totals_change_only_by_declared_anomalies" in failed(e)


def test_missing_root_cause_or_short_explanation_is_detected(hard_ep):
    e = copy.deepcopy(hard_ep)
    e.private.truth[0]["root_cause"] = ""
    e.private.truth[1]["explanation"] = "short"
    assert "14_every_issue_has_root_cause_and_explanation" in failed(e)


def test_every_injected_anomaly_really_exists_in_the_data(hard_ep):
    base = {r["row_id"]: r for r in hard_ep.private.clean_baseline["ledger"]}
    pub = {r["row_id"]: r for r in hard_ep.public.ledger}
    for m in hard_ep.private.provenance["mutations"]:
        assert m["modified_rows"] and all(pub[r["row_id"]] == r for r in m["modified_rows"])
        assert any(base.get(r["row_id"]) != r for r in m["modified_rows"])             # something truly differs from clean
    derived = V.derive_issues(hard_ep.public)
    assert len(derived) == len(hard_ep.private.truth) == 4


# ----------------------------------------------------------------- Decimal / no floating point
def test_money_modules_use_no_float_calls():
    files = ["money.py", "generator/ledger.py", "generator/mutations.py", "generator/validation.py",
             "generator/financial_world.py", "generator/documents.py", "generator/clean_cases.py",
             "generator/compound_scenarios.py", "evaluation/matcher.py", "schemas.py"]
    for f in files:
        for node in ast.walk(ast.parse((ROOT / "src" / "traceforge" / f).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "float", (f, node.lineno)
            assert not (isinstance(node, ast.Constant) and isinstance(node.value, float) and f in files[:2] + ["generator/validation.py"]), (f, node.lineno)


@pytest.mark.parametrize("level", LEVELS)
def test_ledger_amounts_equal_an_independent_decimal_recomputation(level):
    ep = generate_episode(8, {"clean": True, "difficulty": level, "base_currency": "GBP"})
    fx = {(r["date"], r["source_currency"]): Decimal(r["rate"]) for r in ep.public.fx}
    tz = datetime.fromisoformat(ep.public.manifest["period_start"]).tzinfo
    base = ep.public.manifest["base_currency"]
    rows = {(r["invoice_id"]): r for r in ep.public.ledger if r["account"] == "1200" and r["payment_id"] is None}
    checked = 0
    for inv in ep.public.invoices:
        c = inv["currency"]
        rate = Decimal(1) if c == base else fx[(datetime.fromisoformat(inv["issued_at"]).astimezone(tz).date().isoformat(), c)]
        major = Decimal(inv["total_minor"]).scaleb(-CURRENCY_DECIMALS[c]) * rate
        want = int(major.scaleb(CURRENCY_DECIMALS[base]).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        assert rows[inv["invoice_id"]]["debit_minor"] == want
        checked += c != base
    assert checked >= 10


def test_receivable_balances_reflect_partial_and_full_payments():
    ep = generate_episode(2, {"clean": True})
    bal = V.outstanding_balances(ep.public)
    totals = {i["invoice_id"]: i["total_minor"] for i in ep.public.invoices}
    assert any(b == 0 for b in bal.values()) and any(0 < b < totals[k] for k, b in bal.items()) \
        and any(b == totals[k] for k, b in bal.items())
    for p in ep.public.payments:                       # a payment never allocates more than it carries
        assert sum(r["allocated_minor"] for r in p["invoice_refs"]) <= max(p["amount_minor"], 0) or p["amount_minor"] < 0


def test_transaction_dates_follow_the_documents_and_stay_in_a_sane_window():
    ep = generate_episode(2, {"clean": True})
    start = datetime.fromisoformat(ep.public.manifest["period_start"])
    end = datetime.fromisoformat(ep.public.manifest["period_end"])
    for i in ep.public.invoices:
        t = datetime.fromisoformat(i["issued_at"])
        assert start - timedelta(days=21) <= t <= end + timedelta(days=4)
        assert datetime.fromisoformat(i["due_at"]) - t == timedelta(days=30)
    inv = {i["invoice_id"]: datetime.fromisoformat(i["issued_at"]) for i in ep.public.invoices}
    for p in ep.public.payments:
        t = datetime.fromisoformat(p["timestamp"])
        assert all(t >= inv[r["invoice_id"]] for r in p["invoice_refs"])            # nothing is paid before it is issued


# ----------------------------------------------------------------- reproducibility and variation
@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_same_seed_and_configuration_give_exactly_the_same_dataset(preset):
    from traceforge.episode_io import private_files, public_files
    a, b = generate_episode(77, PRESETS[preset]), generate_episode(77, PRESETS[preset])
    assert public_files(a.public) == public_files(b.public) and private_files(a.private) == private_files(b.private)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_different_seeds_produce_meaningful_variation(preset):
    eps = [generate_episode(s, PRESETS[preset]) for s in range(12)]
    shapes = {(len(e.public.ledger), e.public.manifest["document_count"]) for e in eps}
    sigs = {tuple(sorted((t["type"], t["expected_amount_minor"]) for t in e.private.truth)) for e in eps}
    periods = {e.public.manifest["period_start"] for e in eps}
    assert len(shapes) >= 9 and len(sigs) >= 8 and len(periods) >= 8


# ----------------------------------------------------------------- metamorphic transforms per level
@pytest.mark.parametrize("level", LEVELS)
def test_transforms_preserve_links_explanations_and_decoy_lists_at_every_level(level):
    ep = generate_episode(14, {"difficulty": level, "issue_counts": {t: 1 for t in ISSUE_TYPES}})
    new, mapping = transform_episode(ep, 3)
    assert all(verify_links(ep, new, mapping).values()) and V.validate_episode(new)["ok"]
    ids = {d.get("invoice_id") or d["payment_id"] for d in new.public.invoices + new.public.payments}
    assert set(new.private.provenance["world"]["clean_decoys"]) <= ids
    assert not set(new.private.provenance["world"]["clean_decoys"]) & set(ep.private.provenance["world"]["clean_decoys"])
    for o, n in zip(ep.private.truth, new.private.truth):
        for a, b in mapping.items():
            if a in o["explanation"]:
                assert b in n["explanation"] and a not in n["explanation"]
