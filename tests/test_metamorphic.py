"""24 paired metamorphic tests (original vs transformed) plus negative controls that prove
the link verifier is not vacuous."""
import copy
import json

import pytest

from traceforge.environment.audit_env import AuditEnv
from traceforge.episode_io import save_episode
from traceforge.generator.builder import generate_episode
from traceforge.generator.transformations import (ID_RE, check_pair, collect_ids, map_findings,
                                                  transform_episode, verify_links)
from traceforge.generator.validation import referential_problems, validate_episode
from traceforge.reporting.transformation_report import build_transformation_report, pair_specs
from traceforge.runner.episode_runner import make_agent, run_episode

SPECS = pair_specs(24)


def _answer(truth):
    return [{"type": t["type"], "affected_ids": t["affected_ids"], "claimed_amount_minor": t["expected_amount_minor"],
             "currency": t["currency"], "evidence_ids": t["allowed_evidence_sets"][0], "explanation": ""} for t in truth]


def _score(tmp, ep, tag, findings):
    save_episode(ep, tmp / tag)
    env = AuditEnv("eval")
    env.reset(config={"episode_dir": str(tmp / tag)})
    env.step({"action": "submit", "findings": {"episode_id": ep.public.manifest["episode_id"], "findings": findings}})
    return env.result.metrics


def _run(tmp, ep, tag, policy):
    save_episode(ep, tmp / tag)
    return run_episode(AuditEnv("eval"), make_agent(policy), config={"episode_dir": str(tmp / tag)})


@pytest.mark.parametrize("idx", range(24))
def test_pair(idx, tmp_path):
    seed, cfg, variant = SPECS[idx]
    ep = generate_episode(seed, cfg)
    new, mapping = transform_episode(ep, seed + 1000, **variant)

    # 1. every financial link survived (row-by-row / document-by-document isomorphism)
    links = check_pair(ep, new, mapping)
    assert all(links.values()), [k for k, v in links.items() if not v]
    # 2. the transform is itself a valid, fully consistent company
    assert validate_episode(new)["ok"] and referential_problems(new.public) == []
    # 3. identifiers really changed (unless this variant disables it) and order really changed
    if variant.get("remap_ids", True):
        assert mapping and all(a != b for a, b in mapping.items())
        assert not (set(collect_ids(ep.public)) & set(collect_ids(new.public)))
    if variant.get("shuffle", True):
        assert [r["row_id"] for r in new.public.ledger] != sorted(r["row_id"] for r in new.public.ledger)
        assert [d["doc_id"] for d in new.public.invoices] != sorted(d["doc_id"] for d in new.public.invoices)
    # 4. an answer correct for the original stays correct after translating ids
    answer = _answer(ep.private.truth)
    on_orig = _score(tmp_path, ep, "o", answer)
    on_new = _score(tmp_path, new, "n", map_findings(answer, mapping))
    assert on_orig.f1 == on_new.f1 == 1.0 and on_new.tp == len(ep.private.truth)
    # 5. negative control: the ORIGINAL ids no longer score once ids were remapped
    if mapping and ep.private.truth:
        assert _score(tmp_path, new, "s", answer).tp == 0
    # 6. oracle and the heuristic reconciliation agent behave identically on both versions
    for pol in ("oracle", "reconciliation"):
        a, b = _run(tmp_path, ep, f"{pol}a", pol), _run(tmp_path, new, f"{pol}b", pol)
        assert {k: a[k] for k in ("tp", "fp", "fn", "f1", "reward", "steps")} == {k: b[k] for k in ("tp", "fp", "fn", "f1", "reward", "steps")}
    # 7. same number of actions -> same reward (mapped oracle answer, 7 inspection steps)
    rewards = []
    for tag, e, f in (("ra", ep, answer), ("rb", new, map_findings(answer, mapping))):
        save_episode(e, tmp_path / tag)
        env = AuditEnv("train")
        env.reset(config={"episode_dir": str(tmp_path / tag)})
        for _ in range(7):
            env.step({"action": "calculator", "expression": "1"})
        rewards.append(env.step({"action": "submit", "findings": {"episode_id": e.public.manifest["episode_id"], "findings": f}})[1])
    assert rewards[0] == rewards[1] == pytest.approx(1.0 - 0.014)


def test_report_covers_all_pairs_and_is_written(tmp_path):
    out = tmp_path / "transformation_report.json"
    rep = build_transformation_report(24, out)
    assert rep["summary"]["pairs"] == 24 and rep["summary"]["failed_pairs"] == [] and rep["summary"]["passed"] == 24
    assert rep["summary"]["pairs_with_issues"] >= 20 and rep["summary"]["checks_run"] >= 24 * 10
    assert json.loads(out.read_text())["summary"] == rep["summary"]


# ----------------------------------------------------------------- negative controls
@pytest.fixture(scope="module")
def good_pair():
    ep = generate_episode(3, {"issue_counts": {"duplicate": 1, "allocation": 1, "fx": 1, "cutoff": 1}})
    new, mapping = transform_episode(ep, 9)
    return ep, new, mapping


def _broken(good_pair, mutate):
    ep, new, mapping = good_pair
    bad = copy.deepcopy(new)
    mutate(bad)
    return verify_links(ep, bad, mapping)


def test_verifier_accepts_the_genuine_transform(good_pair):
    assert all(verify_links(*good_pair).values())


@pytest.mark.parametrize("name,mutate,failing", [
    ("rewired ledger invoice link", lambda e: e.public.ledger[0].update(invoice_id="INV-00000"), "ledger_rows_isomorphic"),
    ("changed ledger amount", lambda e: e.public.ledger[1].update(debit_minor=e.public.ledger[1]["debit_minor"] + 1), "ledger_rows_isomorphic"),
    ("swapped two ledger row ids", lambda e: (e.public.ledger[0].update(row_id=e.public.ledger[1]["row_id"]),
                                              e.public.ledger[1].update(row_id=e.public.ledger[0]["row_id"])), "ledger_rows_isomorphic"),
    ("payment ref points elsewhere", lambda e: e.public.payments[0]["invoice_refs"][0].update(invoice_id=e.public.invoices[0]["invoice_id"]), "payments_isomorphic"),
    ("allocation amount changed", lambda e: e.public.payments[0]["invoice_refs"][0].update(allocated_minor=1), "payments_isomorphic"),
    ("invoice total changed", lambda e: e.public.invoices[0].update(total_minor=1), "invoices_isomorphic"),
    ("invoice entity changed", lambda e: e.public.invoices[0].update(entity_id="ENT-00000"), "invoices_isomorphic"),
    ("truth id left stale", lambda e: e.private.truth[0]["affected_ids"].append("ROW-00001"), "truth_isomorphic"),
    ("truth amount changed", lambda e: e.private.truth[0].update(expected_amount_minor=1), "truth_isomorphic"),
    ("stale original id inside provenance", lambda e: e.private.provenance.update(note="see ROW-00001 and " + sorted(collect_ids(e.public))[0]), "no_stale_ids_and_all_ids_remapped"),
])
def test_verifier_detects_corruption(good_pair, name, mutate, failing):
    res = _broken(good_pair, mutate)
    assert res[failing] is False, name


def test_transformation_is_field_aware_not_text_replacement():
    """A line-item description that LOOKS like an id must be left alone."""
    ep = generate_episode(3, {"issue_counts": {"fx": 1}})
    ep.public.invoices[0]["line_items"][0]["description"] = "ref ROW-00001 INV-0001"
    new, mapping = transform_episode(ep, 4, rename=False)
    new_inv = next(i for i in new.public.invoices if i["invoice_id"] == mapping[ep.public.invoices[0]["invoice_id"]])
    assert new_inv["line_items"][0]["description"] == "ref ROW-00001 INV-0001"


def test_ids_inside_numbers_or_fx_ids_are_not_touched():
    ep = generate_episode(3, {"issue_counts": {"fx": 1}})
    new, _ = transform_episode(ep, 5)
    assert {r["date"] for r in new.public.fx} == {r["date"] for r in ep.public.fx}
    fx_evidence = lambda e: sorted(i for t in e.private.truth for s in t["allowed_evidence_sets"] for i in s if i.startswith("FX-"))
    assert fx_evidence(ep) == fx_evidence(new) and fx_evidence(ep)


def test_transform_composes_and_stays_valid():
    ep = generate_episode(4, {"scenarios": ["cross_border_partial", "period_end_duplicate"]})
    once, m1 = transform_episode(ep, 1)
    twice, m2 = transform_episode(once, 2)
    assert validate_episode(twice)["ok"] and all(verify_links(once, twice, m2).values())
    composed = {k: m2[m1[k]] for k in m1}
    answer = _answer(ep.private.truth)
    ids = set(ID_RE.findall(json.dumps(map_findings(answer, composed))))
    assert ids <= set(collect_ids(twice.public))
