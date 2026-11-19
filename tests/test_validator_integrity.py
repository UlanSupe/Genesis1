"""The independent validator must itself be sensitive: each check is exercised by tampering
with a known-good episode and verifying that exactly the intended check fails."""
import copy

import pytest

from traceforge.generator import validation as V
from traceforge.generator.builder import generate_episode
from traceforge.schemas import PublicEpisode

CFG = {"issue_counts": {"duplicate": 1, "allocation": 1, "fx": 1, "cutoff": 1}}


@pytest.fixture
def ep():
    return copy.deepcopy(generate_episode(7, CFG))


def failed(ep):
    return {c["name"] for c in V.validate_episode(ep)["checks"] if not c["ok"]}


def test_genuine_episode_passes_every_check(ep):
    rep = V.validate_episode(ep, 200)
    assert rep["ok"] and len(rep["checks"]) >= 13


def test_debit_credit_consistency_detects_unbalanced_journal(ep):
    ep.public.ledger[3]["debit_minor"] += 1
    f = failed(ep)
    assert "2_public_data_consistent" in f and "4b_baseline_plus_manifests_equals_public_ledger" in f


def test_baseline_is_checked_against_an_independent_reconstruction(ep):
    ep.private.clean_baseline["ledger"][0]["credit_minor"] += 5
    ep.private.clean_baseline["ledger"][0]["debit_minor"] += 5
    assert "1b_baseline_equals_independent_expected_ledger" in failed(ep)


def test_currency_conversion_error_in_baseline_is_detected(ep):
    row = next(r for r in ep.private.clean_baseline["ledger"] if r["currency"] != ep.public.manifest["base_currency"]
               and r["account"] == "1200")
    row["debit_minor"] += 1
    assert "1b_baseline_equals_independent_expected_ledger" in failed(ep)


def test_reconstruction_is_sensitive_to_fx_input(ep):
    from collections import Counter
    inv = next(i for i in ep.public.invoices if i["currency"] != ep.public.manifest["base_currency"])
    day = V._inst(inv["issued_at"]).astimezone(V._inst(ep.public.manifest["period_start"]).tzinfo).date().isoformat()
    row = next(r for r in ep.public.fx if r["date"] == day and r["source_currency"] == inv["currency"])
    row["rate"] = "9" + row["rate"][1:]                                       # absurd rate on exactly that date
    exp = V.expected_ledger(PublicEpisode(ep.public.manifest, ep.public.policy, ep.private.clean_baseline["ledger"],
                                          ep.public.invoices, ep.public.payments, ep.public.fx))
    assert exp != Counter(V.row_key(x) for x in ep.private.clean_baseline["ledger"])
    assert "1b_baseline_equals_independent_expected_ledger" in failed(ep)


def test_period_cutoff_independent_detection(ep):
    cut = next(t for t in ep.private.truth if t["type"] == "cutoff")
    row = next(r for r in ep.public.ledger if r["row_id"] == cut["affected_ids"][0])
    d = [x for x in V.derive_issues(ep.public) if x["type"] == "cutoff"]
    assert d and d[0]["affected_ids"] == sorted(cut["affected_ids"]) and d[0]["expected_amount_minor"] == cut["expected_amount_minor"]
    row["timestamp"] = ep.public.manifest["period_start"]                    # nudge one leg back to a boundary instant
    assert "3_truth_equals_independent_derivation" in failed(ep) or "4b_baseline_plus_manifests_equals_public_ledger" in failed(ep)


def test_duplicate_detection_flags_a_hidden_repost(ep):
    clean_row = next(r for r in ep.public.ledger if r["account"] == "1200" and r["payment_id"] is None
                     and r["invoice_id"] not in {t["affected_ids"][0] for t in ep.private.truth})
    ep.public.ledger.append({**clean_row, "row_id": "ROW-99999"})
    ep.public.manifest["record_count"] += 1
    assert "3_truth_equals_independent_derivation" in failed(ep)


def test_unclaimed_discrepancy_is_flagged_even_when_the_ledger_stays_balanced(ep):
    pay = next(p for p in ep.public.payments if p["payment_id"] not in {i for t in ep.private.truth for i in t["affected_ids"]}
               and p["amount_minor"] > 0 and p["currency"] == ep.public.manifest["base_currency"])
    rows = [r for r in ep.public.ledger if r["payment_id"] == pay["payment_id"]]
    cash = next(r for r in rows if r["account"] == "1000")
    ar = next(r for r in rows if r["account"] == "1200")
    cash["debit_minor"] += 3
    ar["credit_minor"] += 3                                                   # balanced, but allocation now wrong
    f = failed(ep)
    assert "3_truth_equals_independent_derivation" in f and "4b_baseline_plus_manifests_equals_public_ledger" in f


def test_referential_integrity_checks(ep):
    pub = ep.public
    e = copy.deepcopy(ep)
    e.public.ledger[0]["entity_id"] = "ENT-NOPE"
    assert any("unknown entity" in p for p in V.referential_problems(e.public))
    e = copy.deepcopy(ep)
    e.public.payments[0]["invoice_refs"][0]["invoice_id"] = "INV-NOPE"
    f = failed(e)                                                             # must FAIL cleanly, not raise
    assert "2b_referential_integrity_and_balances" in f or "1b_baseline_equals_independent_expected_ledger" in f
    e = copy.deepcopy(ep)
    e.public.invoices[1]["doc_id"] = e.public.invoices[0]["doc_id"]
    assert any("duplicate document identifiers" in p for p in V.referential_problems(e.public))
    e = copy.deepcopy(ep)
    e.public.ledger[0]["account"] = "9999"
    assert any("unknown account" in p for p in V.referential_problems(e.public))
    e = copy.deepcopy(ep)
    e.public.fx = [r for r in e.public.fx if r["source_currency"] != e.public.invoices[0]["currency"]] \
        if e.public.invoices[0]["currency"] != pub.manifest["base_currency"] else e.public.fx
    foreign = [i for i in e.public.invoices if i["currency"] != pub.manifest["base_currency"]]
    assert foreign and any("no fx rate" in p for p in V.referential_problems(e.public))


def test_outstanding_balances_are_computed_in_document_currency():
    from conftest import episode
    e = episode(2, clean=True)
    bal = V.outstanding_balances(e.public)
    inv = {i["invoice_id"]: i for i in e.public.invoices}
    paid = {}
    for p in e.public.payments:
        for r in p["invoice_refs"]:
            paid[r["invoice_id"]] = paid.get(r["invoice_id"], 0) + r["allocated_minor"]
    assert all(bal[k] == inv[k]["total_minor"] - paid.get(k, 0) for k in inv)
    assert all(0 <= v <= inv[k]["total_minor"] for k, v in bal.items())
    assert any(v == 0 for v in bal.values()) and any(v == inv[k]["total_minor"] for k, v in bal.items()) \
        and any(0 < v < inv[k]["total_minor"] for k, v in bal.items())      # paid / unpaid / partially paid all occur


def test_negative_outstanding_balance_is_rejected(ep):
    p = next(p for p in ep.public.payments if p["amount_minor"] > 0)
    p["invoice_refs"][0]["allocated_minor"] = 10 ** 12
    p["amount_minor"] = 10 ** 12
    assert any("outstanding balance" in x for x in V.referential_problems(ep.public))


def test_manifest_causality_checks(ep):
    e = copy.deepcopy(ep)
    e.private.provenance["mutations"][0]["changed_amount_minor"] += 1
    assert "4c_manifest_causality" in failed(e)
    e = copy.deepcopy(ep)
    e.private.provenance["mutations"][0]["causal_description"] = ""
    assert "4c_manifest_causality" in failed(e)
    e = copy.deepcopy(ep)
    e.private.provenance["mutations"].pop()
    assert "4c_manifest_causality" in failed(e)
    e = copy.deepcopy(ep)
    e.private.provenance["mutations"][0]["modified_rows"][0]["debit_minor"] += 1
    assert "4c_manifest_causality" in failed(e)


def test_every_mutation_changed_exactly_the_documents_it_claims(ep):
    e = copy.deepcopy(ep)
    victim = next(r for r in e.public.ledger if r["account"] == "4000")
    victim["credit_minor"] += 1
    victim["debit_minor"] += 0
    assert "4_mutations_changed_exactly_the_intended_documents" in failed(e)


def test_compound_scenarios_need_two_distinct_types():
    e = copy.deepcopy(generate_episode(7, {"scenarios": ["cross_border_partial"]}))
    ids = e.private.provenance["compound_scenarios"][0]["issue_ids"]
    for t in e.private.truth:
        t["type"] = "fx"
    assert V.manifest_problems(e) and ids


def test_validator_is_deterministic_and_does_not_mutate_its_input(ep):
    import json
    before = json.dumps([ep.public.ledger, ep.private.truth], sort_keys=True)
    a, b = V.validate_episode(ep), V.validate_episode(ep)
    assert a == b and json.dumps([ep.public.ledger, ep.private.truth], sort_keys=True) == before
