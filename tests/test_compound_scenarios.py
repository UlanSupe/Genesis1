import pytest

from conftest import finding, new_env, submit
from traceforge.generator import validation
from traceforge.generator.builder import generate_episode

SEEDS = range(8)


def _pay_doc(ep, pid):
    return next(p for p in ep.public.payments if p["payment_id"] == pid)


@pytest.mark.parametrize("seed", SEEDS)
def test_cross_border_partial_payment(seed):
    ep = generate_episode(seed, {"scenarios": ["cross_border_partial"]})
    types = sorted(t["type"] for t in ep.private.truth)
    assert types == ["allocation", "fx"]
    alloc = next(t for t in ep.private.truth if t["type"] == "allocation")
    fx = next(t for t in ep.private.truth if t["type"] == "fx")
    pa = next(i for i in alloc["affected_ids"] if i.startswith("PAY"))
    pf = fx["affected_ids"][0]
    assert pa != pf                                              # two separate journals
    a_doc, f_doc = _pay_doc(ep, pa), _pay_doc(ep, pf)
    base = ep.public.manifest["base_currency"]
    assert a_doc["currency"] == f_doc["currency"] != base        # cross-border
    shared = {r["invoice_id"] for r in a_doc["invoice_refs"]} & {r["invoice_id"] for r in f_doc["invoice_refs"]}
    assert shared                                                # both partial payments settle one invoice
    assert any(e.startswith("FX-") for e in fx["allowed_evidence_sets"][0])
    assert not set(alloc["allowed_evidence_sets"][0]) & set(fx["allowed_evidence_sets"][0])  # separate evidence groups
    assert ep.private.provenance["compound_scenarios"][0]["scenario"] == "cross_border_partial"


@pytest.mark.parametrize("seed", SEEDS)
def test_period_end_duplicate(seed):
    ep = generate_episode(seed, {"scenarios": ["period_end_duplicate"]})
    assert sorted(t["type"] for t in ep.private.truth) == ["cutoff", "duplicate"]
    dup = next(t for t in ep.private.truth if t["type"] == "duplicate")
    cut = next(t for t in ep.private.truth if t["type"] == "cutoff")
    roles = ep.private.provenance["compound_scenarios"][0]["roles"]
    inv = {i["invoice_id"]: i for i in ep.public.invoices}
    a, b = inv[roles["duplicate_doc"]], inv[roles["cutoff_doc"]]
    assert dup["affected_ids"] == [a["invoice_id"]]
    assert a["entity_id"] == b["entity_id"] and a["total_minor"] == b["total_minor"]   # lookalike
    assert all(i.startswith("ROW-") for i in cut["affected_ids"])
    assert dup["expected_amount_minor"] != 0 and cut["expected_amount_minor"] != 0


@pytest.mark.parametrize("scenario", ["cross_border_partial", "period_end_duplicate"])
def test_scenario_issues_are_independently_matched(scenario):
    env, obs, _ = new_env(3, scenarios=[scenario])
    truth = env.evaluator_truth()
    # report only the first issue -> exactly one TP and one FN (they do not depend on each other)
    submit(env, [finding(truth[0])])
    m = env.result.metrics
    assert (m.tp, m.fp, m.fn) == (1, 0, 1)
    env2, _, _ = new_env(3, scenarios=[scenario])
    submit(env2, [finding(truth[1])])
    assert (env2.result.metrics.tp, env2.result.metrics.fn) == (1, 1)
    env3, _, _ = new_env(3, scenarios=[scenario])
    submit(env3, [finding(t) for t in truth])
    assert (env3.result.metrics.tp, env3.result.metrics.fp, env3.result.metrics.fn) == (2, 0, 0)


@pytest.mark.parametrize("scenario", ["cross_border_partial", "period_end_duplicate"])
def test_scenario_evidence_sufficient_and_available(scenario):
    ep = generate_episode(5, {"scenarios": [scenario]})
    assert ep.private.provenance["validation"]["ok"]
    assert len(validation.derive_issues(ep.public)) == 2


def test_both_scenarios_together_and_with_other_errors():
    ep = generate_episode(2, {"scenarios": ["cross_border_partial", "period_end_duplicate"],
                              "issue_counts": {"fx": 2}})
    assert len(ep.private.truth) == 6 and ep.private.provenance["validation"]["ok"]
