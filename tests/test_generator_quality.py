"""Config hardening, difficulty presets, diversity, and absence of structural 'tells'."""
from collections import Counter

import pytest

from conftest import ALL4, new_env, submit
from traceforge.config import PRESETS, GeneratorConfig
from traceforge.constants import ISSUE_TYPES
from traceforge.environment.audit_env import AuditEnv
from traceforge.evaluation.evidence_validator import evidence_satisfied
from traceforge.exceptions import ConfigError
from traceforge.generator.builder import generate_episode


# ----------------------------------------------------------- config hardening (audit finding)
@pytest.mark.parametrize("bad", [
    {"mutation_weights": {t: 0 for t in ISSUE_TYPES}}, {"mutation_weights": {"fx": -1}}, {"mutation_weights": {"fx": "a"}},
    {"mutation_weights": []}, {"n_invoices": "48"}, {"n_invoices": 2}, {"n_invoices": 4.5}, {"n_invoices": True},
    {"n_entities": 10 ** 9}, {"period_start": "banana"}, {"period_start": "2024-13-01"}, {"period_start": "2024-02-15"},
    {"period_start": 20240201}, {"timezone_minutes": "x"}, {"timezone_minutes": 99999}, {"clean_rate": "a"}, {"clean_rate": 2},
    {"compound_rate": -0.1}, {"min_issues": 7, "max_issues": 3}, {"max_issues": 13}, {"min_issues": 1.5}, {"issue_counts": {"fx": 1.5}},
    {"issue_counts": {"fx": -1}}, {"issue_counts": []}, {"scenarios": "cross_border_partial"}, {"scenarios": ["nope"]},
    {"clean": "yes"}, {"episode_id": ""}, {"episode_id": 5}, {"foreign_currencies": "USD"}, {"foreign_currencies": ["EUR"]},
    {"base_currency": None}, {"difficulty": None}, {"min_ledger_rows": -5}, {"unknown_key": 1},
])
def test_bad_config_raises_config_error_never_a_raw_exception(bad):
    with pytest.raises(ConfigError):
        generate_episode(1, bad)


def test_config_must_be_an_object():
    for bad in ([1, 2], "x", 5):
        with pytest.raises(ConfigError):
            GeneratorConfig.from_dict(bad)
        with pytest.raises(ConfigError):
            AuditEnv().reset(1, bad)


def test_reset_is_gymnasium_style_and_options_alias_config():
    env = AuditEnv("train")
    a, _ = env.reset(3, {"clean": True})                       # positional options
    b, _ = env.reset(3, options={"clean": True})
    c, _ = env.reset(seed=3, config={"clean": True})
    assert a == b == c
    with pytest.raises(ConfigError):
        env.reset(3, {"clean": True}, config={"clean": True})
    for bad_seed in (1.5, True, [1], None.__class__):
        with pytest.raises(ConfigError):
            env.reset(bad_seed, {"clean": True})
    env.close()
    from traceforge.exceptions import NotReset
    with pytest.raises(NotReset):
        env.step({"action": "calculator", "expression": "1"})


# ----------------------------------------------------------- presets and difficulty
def _stats(preset, seeds=range(10)):
    docs, multi, partial, ccys, ents = [], 0, 0, [], []
    pays = 0
    for s in seeds:
        ep = generate_episode(s, {**PRESETS[preset], "clean": True})
        docs.append(ep.public.manifest["document_count"])
        inv = {i["invoice_id"]: i["total_minor"] for i in ep.public.invoices}
        paid = Counter()
        for p in ep.public.payments:
            pays += 1
            multi += len(p["invoice_refs"]) >= 2
            for r in p["invoice_refs"]:
                paid[r["invoice_id"]] += r["allocated_minor"]
        partial += sum(0 < paid[k] < inv[k] for k in paid)
        ccys.append(len({i["currency"] for i in ep.public.invoices}))
        ents.append(len({i["entity_id"] for i in ep.public.invoices}))
    return {"docs": sum(docs) / len(docs), "multi": multi / len(docs), "partial": partial / len(docs),
            "currencies": sum(ccys) / len(ccys), "entities": sum(ents) / len(ents)}


def test_difficulty_grows_with_interrelations_not_with_noise():
    easy, hard = _stats("easy"), _stats("hard")
    assert hard["docs"] > 1.5 * easy["docs"]
    assert hard["multi"] > easy["multi"]                          # more split payments to follow per episode
    assert hard["partial"] > easy["partial"]                       # more partially paid invoices
    assert hard["currencies"] > easy["currencies"] and hard["entities"] > easy["entities"]


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_is_valid_and_solvable_from_public_data(preset):
    for s in range(4):
        ep = generate_episode(200 + s, PRESETS[preset])
        assert ep.private.provenance["validation"]["ok"]
        env, obs, _ = new_env(200 + s, **PRESETS[preset])
        from traceforge.evaluation.oracle_policy import OraclePolicy
        env.step(OraclePolicy(env.evaluator_truth()).act(obs))
        assert env.result.metrics.f1 == 1.0


def test_compound_preset_always_contains_a_scenario():
    kinds = Counter()
    for s in range(12):
        ep = generate_episode(300 + s, PRESETS["compound"])
        sc = ep.private.provenance["compound_scenarios"]
        assert len(sc) >= 1
        kinds.update(x["scenario"] for x in sc)
    assert set(kinds) == {"cross_border_partial", "period_end_duplicate"}


def test_all_four_error_types_occur_in_different_presets():
    seen = {p: Counter() for p in PRESETS}
    for p in PRESETS:
        for s in range(12):
            seen[p].update(t["type"] for t in generate_episode(400 + s, PRESETS[p]).private.truth)
    for t in ISSUE_TYPES:
        assert sum(1 for p in PRESETS if seen[p][t] > 0) >= 3, (t, {p: dict(c) for p, c in seen.items()})


def test_preset_helper_and_overrides():
    cfg = GeneratorConfig.preset("hard", base_currency="JPY")
    assert cfg.difficulty == "hard" and cfg.base_currency == "JPY"
    with pytest.raises(ConfigError):
        GeneratorConfig.preset("impossible")


# ----------------------------------------------------------- diversity
def test_episodes_vary_in_size_counterparties_currencies_timezones_dates_and_amounts():
    eps = [generate_episode(s, {"clean": True}) for s in range(20)]
    rows = {len(e.public.ledger) for e in eps}
    docs = {e.public.manifest["document_count"] for e in eps}
    ents = {len({i["entity_id"] for i in e.public.invoices}) for e in eps}
    tz = {e.public.manifest["timezone"] for e in eps}
    periods = {e.public.manifest["period_start"][:7] for e in eps}
    ccy = {frozenset(i["currency"] for i in e.public.invoices) for e in eps}
    med = sorted(sorted(i["total_minor"] for i in e.public.invoices)[len(e.public.invoices) // 2] for e in eps)
    assert len(rows) >= 15 and len(docs) >= 15 and len(ents) >= 4 and len(tz) >= 3 and len(periods) >= 12 and len(ccy) >= 10
    assert med[-1] > 3 * med[0]                                    # amount scale varies per episode
    assert min(rows) >= 200


def test_repeated_amounts_and_order_of_documents_vary():
    reps = []
    for s in range(10):
        ep = generate_episode(s, {"clean": True})
        c = Counter((i["entity_id"], i["total_minor"]) for i in ep.public.invoices)
        reps.append(sum(n > 1 for n in c.values()))
    assert min(reps) >= 1 and len(set(reps)) >= 2


# ----------------------------------------------------------- no structural tells (audit finding)
def test_cross_border_scenario_has_no_constant_allocation_fingerprint():
    pct = Counter()
    for s in range(30):
        ep = generate_episode(s, {"scenarios": ["cross_border_partial"]})
        roles = ep.private.provenance["compound_scenarios"][0]["roles"]
        inv = {i["invoice_id"]: i for i in ep.public.invoices}
        for role in ("alloc_payment", "fx_payment"):
            p = next(x for x in ep.public.payments if x["payment_id"] == roles[role])
            r = p["invoice_refs"][0]
            pct[round(100 * r["allocated_minor"] / inv[r["invoice_id"]]["total_minor"])] += 1
    assert len(pct) >= 12 and max(pct.values()) / sum(pct.values()) < 0.2     # was 40%/35% spikes before the fix


def test_decoys_can_carry_errors_so_decoy_looking_is_not_a_safe_signal():
    tags = Counter()
    for s in range(30):
        ep = generate_episode(s, {"issue_counts": {"duplicate": 2, "fx": 2, "cutoff": 2, "allocation": 2}})
        dt = ep.private.provenance["world"]["document_tags"]
        docs = {d["doc_id"]: (d.get("invoice_id") or d["payment_id"]) for d in ep.public.invoices + ep.public.payments}
        tags.update(dt[docs[m["document_ids"][0]]].split(":")[0] for m in ep.private.provenance["mutations"])
    assert tags["decoy"] > 0 and tags["regular"] > 0


def test_period_end_scenario_lookalike_gap_varies():
    gaps = set()
    for s in range(20):
        ep = generate_episode(s, {"scenarios": ["period_end_duplicate"]})
        roles = ep.private.provenance["compound_scenarios"][0]["roles"]
        inv = {i["invoice_id"]: i for i in ep.public.invoices}
        from datetime import datetime
        a, b = (datetime.fromisoformat(inv[roles[r]]["issued_at"]) for r in ("duplicate_doc", "cutoff_doc"))
        gaps.add(int((b - a).total_seconds() // 3600))
    assert len(gaps) >= 8


def test_mutated_rows_are_not_localised_by_row_id():
    pos = []
    for s in range(30):
        ep = generate_episode(s, {"issue_counts": {"duplicate": 2, "allocation": 2, "fx": 2}})
        order = {r["row_id"]: i / len(ep.public.ledger) for i, r in enumerate(ep.public.ledger)}
        for m in ep.private.provenance["mutations"]:
            pos += [order[r["row_id"]] for r in m["modified_rows"]]
    assert 0.4 < sum(pos) / len(pos) < 0.6


# ----------------------------------------------------------- evidence alternatives
def test_allocation_evidence_accepts_any_single_changed_line_with_the_documents():
    found_partial = False
    for s in range(60):
        ep = generate_episode(s, {"issue_counts": {"allocation": 1}, "difficulty": "medium"})
        m = ep.private.provenance["mutations"][0]
        if m["variant"] != "partial":
            continue
        found_partial = True
        t = ep.private.truth[0]
        invoice_ids = [i for i in t["affected_ids"] if i.startswith("INV")]
        inv_docs = [i["doc_id"] for i in ep.public.invoices if i["invoice_id"] in invoice_ids]
        pay_doc = next(p["doc_id"] for p in ep.public.payments if p["payment_id"] in t["affected_ids"])
        rows = [r["row_id"] for r in m["modified_rows"]]
        assert len(t["allowed_evidence_sets"]) == 3
        assert all(evidence_satisfied([pay_doc, *inv_docs, r], t["allowed_evidence_sets"]) for r in rows)
        assert not evidence_satisfied([pay_doc, rows[0]], t["allowed_evidence_sets"])                  # cross-reference to the invoice is required
        assert not evidence_satisfied(rows + inv_docs, t["allowed_evidence_sets"])                     # the payment document is required
        break
    assert found_partial
