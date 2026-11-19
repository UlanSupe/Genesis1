"""Matcher stress tests: maximum-cardinality, one-to-one, order-independent, deterministic."""
import itertools
import random

import pytest

from traceforge.evaluation.matcher import compatible, match
from traceforge.schemas import Finding

TYPES, CCY = ("fx", "duplicate"), ("EUR", "USD")
IDSETS = (("A",), ("B",), ("A", "B"))
AMOUNTS = (100, 101, 102, 200)
EVID = (("D", "R1"), ("D", "R2"), ("D",), ("A0", "D", "R1", "R2"))


def T(i, typ="fx", ids=("A",), amt=100, ccy="EUR", ev=(("D", "R1"),)):
    return {"issue_id": f"ISS-{i:03d}", "type": typ, "affected_ids": list(ids), "expected_amount_minor": amt,
            "currency": ccy, "allowed_evidence_sets": [list(s) for s in ev]}


def F(typ="fx", ids=("A",), amt=100, ccy="EUR", ev=("D", "R1")):
    return Finding(typ, list(ids), amt, ccy, list(ev), "")


def brute_force_max(findings, truth) -> int:
    """Exhaustive maximum matching over the compatibility relation (reference oracle)."""
    adj = [[j for j, t in enumerate(truth) if compatible(f, t)] for f in findings]

    def best(i, used):
        if i == len(findings):
            return 0
        res = best(i + 1, used)
        for j in adj[i]:
            if j not in used:
                res = max(res, 1 + best(i + 1, used | {j}))
        return res
    return best(0, frozenset())


def random_instance(rng):
    truth = []
    for i in range(rng.randint(0, 5)):
        sets = rng.sample(EVID, rng.randint(1, 2))
        truth.append(T(i + 1, rng.choice(TYPES), rng.choice(IDSETS), rng.choice(AMOUNTS), rng.choice(CCY), sets))
    findings = [F(rng.choice(TYPES), rng.choice(IDSETS), rng.choice(AMOUNTS), rng.choice(CCY), rng.choice(EVID) + rng.choice(((), ("R1",), ("R2",))))
                for _ in range(rng.randint(0, 7))]
    return findings, truth


def key(f):
    return (f.type, tuple(sorted(f.affected_ids)), f.claimed_amount_minor, f.currency, tuple(sorted(f.evidence_ids)))


@pytest.mark.parametrize("seed", range(400))
def test_matches_exhaustive_maximum_and_is_consistent(seed):
    rng = random.Random(seed)
    findings, truth = random_instance(rng)
    r = match(findings, truth)
    assert r.tp == brute_force_max(findings, truth)                       # maximum cardinality
    assert len({i for i, _ in r.pairs}) == len(r.pairs) == len({t for _, t in r.pairs})   # one-to-one
    by_id = {t["issue_id"]: t for t in truth}
    assert all(compatible(findings[i], by_id[t]) for i, t in r.pairs)     # every TP satisfies ALL five conditions
    assert len(r.false_positives) == len(findings) - r.tp and len(r.false_negatives) == len(truth) - r.tp
    assert sorted(r.false_positives + [i for i, _ in r.pairs]) == list(range(len(findings)))


@pytest.mark.parametrize("seed", range(60))
def test_result_is_independent_of_finding_and_truth_order(seed):
    rng = random.Random(1000 + seed)
    findings, truth = random_instance(rng)
    base = match(findings, truth)
    base_pairs = sorted((key(findings[i]), t) for i, t in base.pairs)
    for _ in range(8):
        f2, t2 = findings[:], truth[:]
        rng.shuffle(f2)
        rng.shuffle(t2)
        r = match(f2, t2)
        assert r.tp == base.tp
        assert sorted((key(f2[i]), t) for i, t in r.pairs) == base_pairs   # same finding->issue assignment
        assert sorted(r.false_negatives) == sorted(base.false_negatives)


def test_deterministic_across_repeated_calls():
    findings, truth = random_instance(random.Random(5))
    assert [match(findings, truth).pairs for _ in range(5)] == [match(findings, truth).pairs] * 5


def test_many_predictions_for_one_error_yield_exactly_one_tp():
    for n in (2, 3, 10, 30):
        r = match([F() for _ in range(n)], [T(1)])
        assert (r.tp, len(r.false_positives), len(r.false_negatives)) == (1, n - 1, 0)


def test_lookalike_truths_each_claimed_once():
    truth = [T(i) for i in range(1, 5)]                                    # four identical-looking issues
    for n in (1, 3, 4, 6):
        r = match([F() for _ in range(n)], truth)
        assert r.tp == min(n, 4) and len({t for _, t in r.pairs}) == r.tp


def test_overlapping_evidence_sets_need_a_complete_set_each():
    t1 = T(1, ev=(("D", "R1"),))
    t2 = T(2, ev=(("D", "R2"),))
    assert match([F(ev=("D", "R1"))], [t1, t2]).pairs == [(0, "ISS-001")]
    assert match([F(ev=("D", "R1", "R2"))], [t1, t2]).tp == 1               # one prediction cannot cover both
    assert match([F(ev=("D", "R1", "R2")), F(ev=("D", "R1", "R2"))], [t1, t2]).tp == 2
    assert match([F(ev=("R1", "R2"))], [t1, t2]).tp == 0                    # shared id alone is not enough


def test_identical_amounts_different_currencies_do_not_cross_match():
    truth = [T(1, ccy="EUR", amt=500), T(2, ccy="USD", amt=500)]
    r = match([F(ccy="USD", amt=500)], truth)
    assert r.pairs == [(0, "ISS-002")]
    assert match([F(ccy="GBP", amt=500)], truth).tp == 0


def test_same_ids_different_types_are_distinct_issues():
    truth = [T(1, "fx"), T(2, "duplicate")]
    r = match([F("duplicate"), F("fx")], truth)
    assert r.tp == 2 and sorted(r.pairs) == [(0, "ISS-002"), (1, "ISS-001")]
    assert match([F("cutoff")], truth).tp == 0


def test_long_augmenting_chain_is_found():
    # p0 fits {t1,t2}, p1 fits {t2,t3}, p2 fits {t3,t4}, p3 fits {t4} -> perfect matching needs a 4-step chain
    truth = [T(1, amt=100), T(2, amt=102), T(3, amt=104), T(4, amt=106)]
    fs = [F(amt=101), F(amt=103), F(amt=105), F(amt=107)]
    for perm in itertools.permutations(range(4)):
        r = match([fs[i] for i in perm], truth)
        assert r.tp == 4, perm


def test_tolerance_chain_amounts():
    truth = [T(1, amt=100), T(2, amt=102)]
    assert match([F(amt=101)], truth).tp == 1                               # fits both, claims one
    assert match([F(amt=101), F(amt=99), F(amt=103)], truth).tp == 2        # 99 -> 100, 103 -> 102, 101 left over
    assert match([F(amt=99), F(amt=98)], truth).tp == 1                     # 98 misses both


def test_no_findings_no_truth_and_asymmetric_sizes():
    assert match([], []).tp == 0
    r = match([F()] * 3, [])
    assert len(r.false_positives) == 3 and r.false_negatives == []
    r = match([], [T(1), T(2)])
    assert r.false_positives == [] and r.false_negatives == ["ISS-001", "ISS-002"]


def test_scale_30_findings_against_12_truths_is_exact_and_fast():
    import time
    rng = random.Random(3)
    truth = [T(i, rng.choice(TYPES), rng.choice(IDSETS), 100 + i * 7, "EUR") for i in range(1, 13)]
    fs = [F(t["type"], t["affected_ids"], t["expected_amount_minor"] + rng.choice((0, 1, -1)), "EUR") for t in truth] \
        + [F("fx", ("Z",), 5) for _ in range(18)]
    t0 = time.perf_counter()
    r = match(fs, truth)
    assert time.perf_counter() - t0 < 0.2
    assert r.tp == brute_force_greedy_free_count(fs, truth)


def brute_force_greedy_free_count(fs, truth):
    # Hall-style upper bound check: matching size cannot exceed the number of distinct compatible truths
    return len({t["issue_id"] for f in fs for t in truth if compatible(f, t)})


def test_greedy_in_canonical_order_would_fail_here_so_augmenting_paths_are_required():
    """p_wide ('A0' sorts first) fits BOTH issues; p_narrow fits only the first. A greedy pass in the
    canonical order gives p_wide the first issue and strands p_narrow (TP=1); the maximum is 2."""
    t1 = T(1, ev=(("D", "R1"),))
    t2 = T(2, ev=(("D", "R2"),))
    p_wide, p_narrow = F(ev=("A0", "D", "R1", "R2")), F(ev=("D", "R1"))
    assert brute_force_max([p_wide, p_narrow], [t1, t2]) == 2
    for fs in ([p_wide, p_narrow], [p_narrow, p_wide]):
        r = match(fs, [t1, t2])
        assert r.tp == 2 and not r.false_positives and not r.false_negatives
