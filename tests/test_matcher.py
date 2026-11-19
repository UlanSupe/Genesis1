import itertools
import random

import pytest

from traceforge.evaluation.matcher import match, norm_ids
from traceforge.schemas import Finding


def T(i="ISS-001", typ="fx", ids=("PAY-1",), amt=1000, ccy="EUR", ev=(("DOC-1", "ROW-1"),)):
    return {"issue_id": i, "type": typ, "affected_ids": list(ids), "expected_amount_minor": amt,
            "currency": ccy, "allowed_evidence_sets": [list(s) for s in ev]}


def F(typ="fx", ids=("PAY-1",), amt=1000, ccy="EUR", ev=("DOC-1", "ROW-1")):
    return Finding(typ, list(ids), amt, ccy, list(ev), "")


def tp(findings, truth):
    r = match(findings, truth)
    return r.tp, len(r.false_positives), len(r.false_negatives)


def test_exact_match_is_tp():
    assert tp([F()], [T()]) == (1, 0, 0)


def test_wrong_type_is_fp_and_fn():
    assert tp([F(typ="duplicate")], [T()]) == (0, 1, 1)


def test_type_is_normalised_but_not_aliased():
    assert tp([F(typ="  FX ")], [T()]) == (1, 0, 0)
    assert tp([F(typ="fx_error")], [T()]) == (0, 1, 1)


def test_affected_ids_compared_as_sets_order_and_repeats_ignored():
    truth = [T(ids=("INV-1", "PAY-1"))]
    assert tp([F(ids=("PAY-1", "INV-1"))], truth) == (1, 0, 0)
    assert tp([F(ids=("PAY-1", "INV-1", "PAY-1", " INV-1 "))], truth) == (1, 0, 0)
    assert norm_ids(["b", "a", "a", " ", ""]) == ("a", "b")


@pytest.mark.parametrize("ids", [("PAY-1", "INV-1"), ("INV-1",), (), ("PAY-2",), ("pay-1",)])
def test_missing_extra_unknown_or_case_changed_ids_do_not_match(ids):
    assert tp([F(ids=ids)], [T()])[0] == 0


@pytest.mark.parametrize("amt,ok", [(1000, True), (999, True), (1001, True), (998, False), (1002, False), (0, False), (-1000, False)])
def test_amount_tolerance_is_one_minor_unit(amt, ok):
    assert (tp([F(amt=amt)], [T()])[0] == 1) is ok


def test_currency_must_match_but_case_is_normalised():
    assert tp([F(ccy="USD")], [T()]) == (0, 1, 1)
    assert tp([F(ccy=" eur ")], [T()]) == (1, 0, 0)


def test_evidence_must_contain_a_complete_allowed_set():
    truth = [T(ev=(("DOC-1", "ROW-1"), ("DOC-1", "ROW-2")))]
    assert tp([F(ev=("DOC-1", "ROW-1"))], truth)[0] == 1
    assert tp([F(ev=("ROW-2", "DOC-1"))], truth)[0] == 1                      # alternative set
    assert tp([F(ev=("DOC-1",))], truth)[0] == 0                              # incomplete
    assert tp([F(ev=("ROW-1", "ROW-2"))], truth)[0] == 0                      # mixes sets without doc
    assert tp([F(ev=("DOC-1", "ROW-1", "X", "Y", "Z"))], truth)[0] == 1       # extras allowed
    assert tp([F(ev=())], truth)[0] == 0
    assert tp([F(ev=("NOPE", "UNKNOWN"))], truth)[0] == 0                      # unknown ids give nothing


def test_one_to_one_duplicate_predictions_count_once():
    assert tp([F(), F(), F()], [T()]) == (1, 2, 0)


def test_one_prediction_cannot_cover_two_truths():
    truth = [T("ISS-001"), T("ISS-002")]                                     # identical twins
    assert tp([F()], truth) == (1, 0, 1)
    assert tp([F(), F()], truth) == (2, 0, 0)


def test_maximum_matching_beats_greedy_and_is_order_independent():
    t1, t2 = T("ISS-001", amt=100), T("ISS-002", amt=101)
    p_wide, p_narrow = F(amt=101), F(amt=99)          # wide fits t1 and t2, narrow fits only t1
    results = set()
    for perm in itertools.permutations([p_wide, p_narrow]):
        results.add(tp(list(perm), [t1, t2]))
        results.add(tp(list(perm), [t2, t1]))
    assert results == {(2, 0, 0)}


def test_result_is_invariant_to_submission_order_randomised():
    rng = random.Random(1)
    truth = [T(f"ISS-{i:03d}", typ=t, ids=(f"X-{i}",), amt=100 * i) for i, t in enumerate(["fx", "duplicate", "cutoff", "allocation"], 1)]
    good = [F(t["type"], t["affected_ids"], t["expected_amount_minor"], "EUR") for t in truth]
    noise = [F("fx", ("Z-1",), 5), F("fx", ("Z-1",), 5), good[0]]
    base = tp(good + noise, truth)
    for _ in range(20):
        mix = good + noise
        rng.shuffle(mix)
        assert tp(mix, truth) == base == (4, 3, 0)


def test_pairs_report_which_finding_matched_which_issue():
    r = match([F(typ="duplicate"), F()], [T()])
    assert r.pairs == [(1, "ISS-001")] and r.false_positives == [0] and r.false_negatives == []


def test_clean_episode_and_empty_submission():
    assert tp([], []) == (0, 0, 0)
    assert tp([F()], []) == (0, 1, 0)                                         # accusation on clean company
    assert tp([], [T(), T("ISS-002")]) == (0, 0, 2)


def test_all_flags_scores_badly_with_many_false_positives():
    truth = [T("ISS-001", ids=("PAY-1",))]
    spam = [F(typ=t, ids=(f"PAY-{i}",)) for i in range(1, 8) for t in ("fx", "duplicate", "cutoff", "allocation")]
    assert tp(spam, truth) == (1, 27, 0)


def test_matcher_never_mutates_inputs():
    truth, fs = [T()], [F()]
    before = (repr(truth), repr(fs))
    match(fs, truth)
    assert (repr(truth), repr(fs)) == before
