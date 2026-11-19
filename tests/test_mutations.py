from collections import Counter

import pytest

from conftest import ALL4
from traceforge.constants import ISSUE_TYPES
from traceforge.generator import validation
from traceforge.generator.builder import generate_episode


def _one(typ, seed=1, **extra):
    return generate_episode(seed, {"issue_counts": {typ: 1}, **extra})


@pytest.mark.parametrize("typ", ISSUE_TYPES)
def test_each_engine_works_independently(typ):
    for seed in range(5):
        ep = _one(typ, seed)
        assert [t["type"] for t in ep.private.truth] == [typ]
        assert ep.private.provenance["validation"]["ok"]
        assert validation.derive_issues(ep.public)[0]["type"] == typ


@pytest.mark.parametrize("typ", ISSUE_TYPES)
def test_mutation_changes_only_the_target_document(typ):
    ep = _one(typ, 2)
    checks = {c["name"]: c["ok"] for c in ep.private.provenance["validation"]["checks"]}
    assert checks["4_mutations_changed_exactly_the_intended_documents"]
    assert ep.private.clean_baseline["ledger"] != ep.public.ledger


def test_duplicate_is_an_extra_posting_not_a_legit_repeat():
    ep = _one("duplicate", 3)
    t = ep.private.truth[0]
    key = t["affected_ids"][0]
    rows = [r for r in ep.public.ledger if (r["payment_id"] or r["invoice_id"]) == key]
    base = [r for r in ep.private.clean_baseline["ledger"] if (r["payment_id"] or r["invoice_id"]) == key]
    assert len(rows) == 2 * len(base)
    assert [tuple(sorted((k, v) for k, v in r.items() if k != 'row_id')) for r in rows].count(
        tuple(sorted((k, v) for k, v in base[0].items() if k != 'row_id'))) == 2


def test_duplicate_amount_equals_anchor_line_of_the_repost():
    for seed in range(8):
        ep = _one("duplicate", seed)
        t = ep.private.truth[0]
        key = t["affected_ids"][0]
        pay = key.startswith("PAY")
        anchors = [r for r in ep.public.ledger
                   if (r["account"] == "1000" and r["payment_id"] == key) or
                   (not pay and r["account"] == "1200" and r["invoice_id"] == key and r["payment_id"] is None)]
        assert len(anchors) == 2
        assert t["expected_amount_minor"] == max(anchors[0]["debit_minor"], anchors[0]["credit_minor"])


def test_allocation_variants_all_occur_across_difficulties_and_amounts_are_shortfalls():
    seen = set()
    for level in ("easy", "medium", "hard"):
        for seed in range(30):
            ep = _one("allocation", seed, difficulty=level)
            v = ep.private.provenance["mutations"][0]["variant"]
            seen.add(v)
            t = ep.private.truth[0]
            pay = next(i for i in t["affected_ids"] if i.startswith("PAY"))
            assert len(t["affected_ids"]) == (3 if v == "wrong_invoice" else 2)
            assert t["expected_amount_minor"] > 1
            base = [r for r in ep.private.clean_baseline["ledger"] if r["payment_id"] == pay]
            new = [r for r in ep.public.ledger if r["payment_id"] == pay]
            # journal stays balanced and cash is untouched: only the allocation moved
            assert sum(r["debit_minor"] for r in new) == sum(r["credit_minor"] for r in new)
            assert [r["debit_minor"] for r in base if r["account"] == "1000"] ==                    [r["debit_minor"] for r in new if r["account"] == "1000"]
    assert seen == {"unallocated", "partial", "wrong_invoice"}


def test_fx_variants_wrong_rate_and_wrong_date_both_occur():
    seen = set()
    for seed in range(40):
        ep = _one("fx", seed)
        seen.add(ep.private.provenance["mutations"][0]["variant"])
        t = ep.private.truth[0]
        assert t["currency"] == ep.public.manifest["base_currency"] and t["expected_amount_minor"] > 1
        assert any(e.startswith("FX-") for e in t["allowed_evidence_sets"][0])
    assert seen == {"wrong_rate", "wrong_date"}


def test_fx_error_requires_foreign_currency_document():
    ep = _one("fx", 4)
    doc_id = ep.private.provenance["mutations"][0]["document_ids"][0]
    doc = next(d for d in ep.public.invoices + ep.public.payments if d["doc_id"] == doc_id)
    assert doc["currency"] != ep.public.manifest["base_currency"]


def test_cutoff_rows_actually_cross_the_period_boundary():
    from datetime import datetime
    for seed in range(8):
        ep = _one("cutoff", seed)
        start = datetime.fromisoformat(ep.public.manifest["period_start"])
        end = datetime.fromisoformat(ep.public.manifest["period_end"])
        cls = lambda t: -1 if t < start else (1 if t >= end else 0)
        m = ep.private.provenance["mutations"][0]
        before = {r["timestamp"] for r in m["source_rows"]}
        after = {r["timestamp"] for r in m["modified_rows"]}
        assert {cls(datetime.fromisoformat(t)) for t in before} != {cls(datetime.fromisoformat(t)) for t in after}


def test_legitimate_boundary_and_foreign_items_are_not_flagged():
    ep = generate_episode(3, {"clean": True, "difficulty": "hard"})
    assert validation.derive_issues(ep.public) == []


def test_mutation_weights_restrict_types():
    ep = generate_episode(2, {"clean_rate": 0, "min_issues": 5, "max_issues": 5, "compound_rate": 0,
                              "mutation_weights": {"fx": 1, "duplicate": 0, "allocation": 0, "cutoff": 0}})
    assert Counter(t["type"] for t in ep.private.truth) == {"fx": 5}


def test_clean_rate_controls_clean_episodes():
    always = [generate_episode(s, {"clean_rate": 1.0}).private.truth for s in range(5)]
    never = [generate_episode(s, {"clean_rate": 0.0, "min_issues": 1}).private.truth for s in range(5)]
    assert all(t == [] for t in always) and all(len(t) >= 1 for t in never)


def test_provenance_records_causal_history():
    ep = generate_episode(1, ALL4)
    muts = ep.private.provenance["mutations"]
    assert len(muts) == 4 == len(ep.private.truth)
    for m in muts:
        for k in ("mutation_id", "issue_id", "issue_type", "source_rows", "modified_rows",
                  "causal_description", "changed_amount_minor", "document_ids", "available_evidence"):
            assert k in m
        assert m["causal_description"] and m["source_rows"] != m["modified_rows"]
    assert {t["issue_id"] for t in ep.private.truth} == {m["issue_id"] for m in muts}


def test_one_mutation_per_document():
    ep = generate_episode(1, {"issue_counts": {"duplicate": 3, "allocation": 3, "fx": 3, "cutoff": 3}})
    docs = [m["document_ids"][0] for m in ep.private.provenance["mutations"]]
    assert len(docs) == len(set(docs)) == 12
