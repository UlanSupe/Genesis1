import pytest

from conftest import ALL4, act, episode, finding, new_env, submit
from traceforge.evaluation.evidence_validator import (check_truth_evidence, evidence_satisfied,
                                                      public_id_index)
from traceforge.generator.builder import generate_episode


def test_superset_semantics():
    sets = [["A", "B"], ["A", "C"]]
    assert evidence_satisfied(["A", "B"], sets) and evidence_satisfied(["C", "A", "Z"], sets)
    assert not evidence_satisfied(["A"], sets) and not evidence_satisfied([], sets)
    assert not evidence_satisfied(["B", "C"], sets)
    assert evidence_satisfied([" A ", "B"], sets)                      # whitespace normalised


def test_unknown_ids_are_harmless_but_never_sufficient():
    assert evidence_satisfied(["A", "B", "nonexistent-1", "nonexistent-2"], [["A", "B"]])
    assert not evidence_satisfied(["nonexistent"], [["A"]])


def test_no_allowed_sets_means_never_satisfied():
    assert not evidence_satisfied(["A"], [])


@pytest.mark.parametrize("seed", range(5))
def test_every_truth_evidence_id_exists_in_public_data(seed):
    ep = generate_episode(seed, {"issue_counts": {"duplicate": 2, "allocation": 2, "fx": 2, "cutoff": 2}})
    assert check_truth_evidence(ep.private.truth, ep.public) == []


def test_every_evidence_id_is_retrievable_through_agent_tools():
    """No early exit: ALL evidence ids of ALL issues must be fetchable with the public tools
    (several envs, each within its own 100-step budget)."""
    cfg = {"issue_counts": {"duplicate": 2, "allocation": 2, "fx": 2, "cutoff": 2}}
    env, obs, _ = new_env(2, **cfg)
    ids = sorted({i for t in env.evaluator_truth() for s in t["allowed_evidence_sets"] for i in s})
    assert len(ids) >= 20
    fetched = 0
    for start in range(0, len(ids), 90):
        env, _, _ = new_env(2, **cfg)
        for ident in ids[start:start + 90]:
            if ident.startswith("DOC-"):
                out = act(env, "get_document", doc_id=ident)
            elif ident.startswith("ROW-"):
                out = act(env, "get_record", table="ledger", id=ident)
            elif ident.startswith("FX-"):
                out = act(env, "get_record", table="fx", id=ident)
            else:
                pytest.fail(f"unexpected evidence id kind {ident}")
            assert out[0]["error"] is None, ident
            fetched += 1
    assert fetched == len(ids)


def test_validator_flags_private_or_missing_evidence():
    ep = generate_episode(1, ALL4)
    truth = [dict(t) for t in ep.private.truth]
    truth[0]["allowed_evidence_sets"] = [["ISS-001", "provenance"]]
    assert any("evidence not public" in p for p in check_truth_evidence(truth, ep.public))
    truth[0]["allowed_evidence_sets"] = [[]]
    assert any("empty" in p for p in check_truth_evidence(truth, ep.public))


def test_public_index_has_all_id_kinds():
    ids = public_id_index(episode(1, **ALL4).public)
    assert any(i.startswith(p) for p in ("ROW-", "DOC-", "INV-", "PAY-", "FX-", "ENT-") for i in ids)
    assert {"ROW", "DOC", "INV", "PAY", "FX", "ENT"} <= {i.split("-")[0] for i in ids}


def test_matching_requires_evidence_end_to_end():
    env, obs, _ = new_env(1, **ALL4)
    truth = env.evaluator_truth()
    bad = [finding(t, evidence_ids=[]) for t in truth]
    submit(env, bad)
    assert (env.result.metrics.tp, env.result.metrics.fp, env.result.metrics.fn) == (0, 4, 4)


def test_inaccessible_evidence_cannot_be_forged_from_wrong_episode():
    env_a, _, _ = new_env(1, **ALL4)
    env_b, _, _ = new_env(2, **ALL4)
    other = env_b.evaluator_truth()
    submit(env_a, [finding(t) for t in other], episode_id=env_a._evaluator.episode_id)
    assert env_a.result.metrics.tp == 0


def test_each_allowed_set_individually_suffices():
    env, _, _ = new_env(4, **{"issue_counts": {"cutoff": 1}})
    t = env.evaluator_truth()[0]
    assert len(t["allowed_evidence_sets"]) >= 2
    for ev_set in t["allowed_evidence_sets"]:
        e2, _, _ = new_env(4, **{"issue_counts": {"cutoff": 1}})
        submit(e2, [finding(t, evidence_ids=ev_set)])
        assert e2.result.metrics.tp == 1
