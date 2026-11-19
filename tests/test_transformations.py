import json
import re

import pytest

from conftest import ALL4
from traceforge.environment.audit_env import AuditEnv
from traceforge.episode_io import save_episode
from traceforge.generator.builder import generate_episode
from traceforge.generator.transformations import ID_RE, check_pair, transform_episode
from traceforge.generator.validation import validate_episode
from traceforge.runner.episode_runner import make_agent, run_episode


def _score(ep, policy, tmp_path, tag):
    save_episode(ep, tmp_path / tag)
    r = run_episode(AuditEnv("eval"), make_agent(policy), config={"episode_dir": str(tmp_path / tag)})
    return r["f1"], r["reward"], r["steps"], (r["tp"], r["fp"], r["fn"])


@pytest.fixture(params=[{"issue_counts": {"duplicate": 1, "allocation": 1, "fx": 1, "cutoff": 1}},
                        {"scenarios": ["cross_border_partial"]}, {"scenarios": ["period_end_duplicate"]},
                        {"clean": True}])
def pair(request):
    ep = generate_episode(6, request.param)
    new, mapping = transform_episode(ep, 77)
    return ep, new, mapping


def test_all_identifiers_are_changed(pair):
    ep, new, mapping = pair
    assert mapping and all(k != v for k, v in mapping.items())
    old_ids = {r["row_id"] for r in ep.public.ledger}
    new_ids = {r["row_id"] for r in new.public.ledger}
    assert not old_ids & new_ids and len(new_ids) == len(old_ids)


def test_references_are_updated_everywhere(pair):
    ep, new, mapping = pair
    inv = {i["invoice_id"] for i in new.public.invoices}
    docs = {d["doc_id"] for d in new.public.invoices + new.public.payments}
    pay = {p["payment_id"] for p in new.public.payments}
    for p in new.public.payments:
        assert all(r["invoice_id"] in inv for r in p["invoice_refs"])
    for r in new.public.ledger:
        assert r["invoice_id"] is None or r["invoice_id"] in inv
        assert r["payment_id"] is None or r["payment_id"] in pay
    leftovers = set(mapping) & set(ID_RE.findall(json.dumps([new.public.ledger, new.public.invoices,
                                                              new.public.payments, new.private.truth])))
    assert not leftovers and docs


def test_transform_revalidates_and_keeps_invariants(pair):
    ep, new, mapping = pair
    assert validate_episode(new)["ok"]
    assert all(check_pair(ep, new, mapping).values())


def test_order_is_actually_shuffled():
    ep = generate_episode(6, ALL4)
    new, _ = transform_episode(ep, 5)
    inv = lambda e: [d["invoice_id"] for d in e.public.invoices]
    assert inv(new) != sorted(inv(new))
    stamps = [r["timestamp"] for r in new.public.ledger]
    assert stamps != sorted(stamps)
    assert sorted(new.public.ledger, key=lambda r: r["row_id"]) != new.public.ledger


def test_truth_follows_the_mapping(pair):
    ep, new, mapping = pair
    for o, n in zip(sorted(ep.private.truth, key=lambda t: t["issue_id"]), sorted(new.private.truth, key=lambda t: t["issue_id"])):
        assert n["affected_ids"] == sorted(mapping.get(i, i) for i in o["affected_ids"])
        assert [sorted(mapping.get(i, i) for i in s) for s in o["allowed_evidence_sets"]] == n["allowed_evidence_sets"]
        assert (o["type"], o["expected_amount_minor"], o["currency"]) == (n["type"], n["expected_amount_minor"], n["currency"])


def test_oracle_gets_same_f1_and_reward_on_transformed_version(pair, tmp_path):
    ep, new, _ = pair
    a, b = _score(ep, "oracle", tmp_path, "a"), _score(new, "oracle", tmp_path, "b")
    assert a == b and a[0] == 1.0 and a[1] == 1.0


def test_heuristic_policy_is_invariant_to_ids_and_row_order(pair, tmp_path):
    ep, new, _ = pair
    assert _score(ep, "reconciliation", tmp_path, "a") == _score(new, "reconciliation", tmp_path, "b")


def test_same_number_of_actions_gives_same_reward(pair, tmp_path):
    ep, new, _ = pair
    rewards = []
    for tag, e in (("a", ep), ("b", new)):
        save_episode(e, tmp_path / tag)
        env = AuditEnv("train")
        env.reset(config={"episode_dir": str(tmp_path / tag)})
        for _ in range(9):
            env.step({"action": "calculator", "expression": "1"})
        _, r, *_ = env.step(make_agent("oracle")(env).act(env._obs(None, None)))
        rewards.append(r)
    assert rewards[0] == rewards[1] == pytest.approx(1.0 - 0.018)


def test_transform_is_deterministic_and_seed_dependent():
    ep = generate_episode(6, ALL4)
    a, _ = transform_episode(ep, 1)
    b, _ = transform_episode(ep, 1)
    c, _ = transform_episode(ep, 2)
    assert a.public == b.public and a.public != c.public


def test_individual_transformations_can_be_disabled():
    ep = generate_episode(6, ALL4)
    same_ids, _ = transform_episode(ep, 3, remap_ids=False, shuffle=True, rename=False)
    assert {r["row_id"] for r in same_ids.public.ledger} == {r["row_id"] for r in ep.public.ledger}
    only_ids, mapping = transform_episode(ep, 3, remap_ids=True, shuffle=False, rename=False)
    assert mapping and [r["timestamp"] for r in only_ids.public.ledger] == [r["timestamp"] for r in ep.public.ledger]


def test_original_episode_is_not_mutated():
    ep = generate_episode(6, ALL4)
    snapshot = json.dumps([ep.public.ledger, ep.private.truth], sort_keys=True)
    transform_episode(ep, 9)
    assert json.dumps([ep.public.ledger, ep.private.truth], sort_keys=True) == snapshot


def test_line_item_descriptions_are_renamed_consistently():
    ep = generate_episode(6, ALL4)
    new, _ = transform_episode(ep, 4)
    old = {l["description"] for i in ep.public.invoices for l in i["line_items"]}
    now = {l["description"] for i in new.public.invoices for l in i["line_items"]}
    assert len(old) == len(now)
