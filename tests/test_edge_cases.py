import pytest

from conftest import ALL4, act, finding, new_env, submit
from traceforge.schemas import parse_submission


def _env(**cfg):
    return new_env(1, **(cfg or ALL4))


def _score(env, findings):
    submit(env, findings)
    m = env.result.metrics
    return m.tp, m.fp, m.fn, env.result.valid


def test_empty_submit_on_clean_episode_is_perfect_minus_step_cost():
    env, obs, _ = new_env(1, clean=True)
    act(env, "calculator", expression="1")
    _, reward, *_ = submit(env, [])
    assert reward == pytest.approx(1 - 0.002) and env.result.metrics.f1 == 1.0


def test_empty_submit_with_real_errors_scores_minus_one():
    env, *_ = _env()
    _, reward, *_ = submit(env, [])
    assert reward == -1.0 and (env.result.metrics.tp, env.result.metrics.fn) == (0, 4)


def test_perfect_answer_gets_full_reward_with_zero_inspection():
    env, *_ = _env()
    assert _score(env, [finding(t) for t in env.evaluator_truth()]) == (4, 0, 0, True)
    assert env.result.reward == 1.0


def test_repeated_findings_are_false_positives_not_extra_credit():
    env, *_ = _env()
    t = env.evaluator_truth()
    assert _score(env, [finding(x) for x in t] + [finding(t[0]), finding(t[0])]) == (4, 2, 0, True)


def test_findings_without_evidence_do_not_match():
    env, *_ = _env()
    assert _score(env, [finding(t, evidence_ids=[]) for t in env.evaluator_truth()]) == (0, 4, 4, True)


def test_false_accusations_on_clean_company_are_penalised():
    env, *_ = new_env(1, clean=True)
    ledger = env._pub.ledger
    fs = [{"type": "duplicate", "affected_ids": [d["invoice_id"]], "claimed_amount_minor": 1, "currency": "EUR",
           "evidence_ids": [d["doc_id"]], "explanation": "looks odd"} for d in env._pub.invoices[:10]]
    assert _score(env, fs) == (0, 10, 0, True)
    assert env.result.reward < -0.5


def test_all_flags_case_yields_poor_reward():
    env, *_ = _env()
    fs = [{"type": t, "affected_ids": [r["row_id"]], "claimed_amount_minor": r["debit_minor"] or r["credit_minor"],
           "currency": "EUR", "evidence_ids": [r["row_id"]], "explanation": ""}
          for t in ("duplicate", "fx", "cutoff", "allocation") for r in env._pub.ledger[:7]]
    tp, fp, fn, _ = _score(env, fs)
    assert tp == 0 and fp == 28 and env.result.reward == -1.0


def test_wrong_currency_and_wrong_amount_are_rejected():
    env, *_ = _env()
    t = env.evaluator_truth()
    fs = [finding(t[0], currency="USD"), finding(t[1], claimed_amount_minor=t[1]["expected_amount_minor"] + 2),
          finding(t[2]), finding(t[3], claimed_amount_minor=t[3]["expected_amount_minor"] + 1)]
    assert _score(env, fs)[:3] == (2, 2, 2)


@pytest.mark.parametrize("mutate", [
    lambda f: f.pop("type"), lambda f: f.update(type=5), lambda f: f.update(affected_ids="ROW-1"),
    lambda f: f.update(affected_ids=[1]), lambda f: f.update(claimed_amount_minor=1.5),
    lambda f: f.update(claimed_amount_minor=True), lambda f: f.update(claimed_amount_minor="5"),
    lambda f: f.update(currency=None), lambda f: f.update(evidence_ids=["x"] * 11),
    lambda f: f.update(explanation="x" * 1001), lambda f: f.update(explanation=5),
    lambda f: f.update(affected_ids=["A"] * 21), lambda f: f.update(evidence_ids=[None]),
    lambda f: f.update(affected_ids=["x" * 65]),
])
def test_malformed_finding_invalidates_submission_with_minus_one(mutate):
    env, *_ = _env()
    f = finding(env.evaluator_truth()[0])
    mutate(f)
    _, reward, term, _, info = submit(env, [f])
    assert term and reward == -1.0 and info["submission_valid"] is False
    assert env.result.internal_log


def test_limits_exactly_at_the_boundary_are_valid():
    env, *_ = _env()
    t = env.evaluator_truth()[0]
    f = finding(t, evidence_ids=list(t["allowed_evidence_sets"][0]) + [f"E{i}" for i in range(7)],
                explanation="é" * 1000)
    assert len(f["evidence_ids"]) <= 10
    assert _score(env, [f])[3] is True


def test_thirty_findings_ok_thirty_one_invalid():
    env, *_ = _env()
    f = finding(env.evaluator_truth()[0])
    assert _score(env, [f] * 30)[3] is True
    env2, *_ = _env()
    assert _score(env2, [f] * 31)[3] is False and env2.result.reward == -1.0


@pytest.mark.parametrize("payload", [None, [], "x", 5, {}, {"episode_id": "e"}, {"findings": []},
                                     {"episode_id": "e", "findings": None}, {"episode_id": "e", "findings": {}}])
def test_parse_submission_rejects_bad_payloads(payload):
    assert parse_submission(payload, "e")[0] is None


def test_unicode_and_extra_keys_are_tolerated():
    findings, err = parse_submission({"episode_id": "e", "findings": [
        {"type": "fx", "affected_ids": ["PAY-1"], "claimed_amount_minor": 5, "currency": "EUR",
         "evidence_ids": [], "explanation": "Обоснование ✓", "confidence": 0.9}]}, "e")
    assert err is None and findings[0].explanation == "Обоснование ✓"


def test_negative_and_huge_amounts_are_valid_but_never_match():
    env, *_ = _env()
    t = env.evaluator_truth()
    fs = [finding(t[0], claimed_amount_minor=-t[0]["expected_amount_minor"]), finding(t[1], claimed_amount_minor=10 ** 30)]
    assert _score(env, fs)[:3] == (0, 2, 4)


def test_unknown_ids_in_affected_and_evidence():
    env, *_ = _env()
    t = env.evaluator_truth()
    fs = [finding(t[0], affected_ids=["INV-0000"]), finding(t[1], evidence_ids=["ROW-0", "DOC-0"])]
    assert _score(env, fs)[:3] == (0, 2, 4)


def test_submit_after_done_is_refused():
    env, *_ = _env()
    submit(env, [])
    from traceforge.exceptions import EpisodeFinished
    with pytest.raises(EpisodeFinished):
        submit(env, [])


def test_maximum_twelve_issues_can_all_be_reported():
    cfg = {"issue_counts": {"duplicate": 3, "allocation": 3, "fx": 3, "cutoff": 3}}
    env, *_ = new_env(3, **cfg)
    assert _score(env, [finding(t) for t in env.evaluator_truth()]) == (12, 0, 0, True)


def test_tiny_and_huge_configurations_are_valid():
    env, *_ = new_env(1, clean=True, n_invoices=4, n_entities=4, min_ledger_rows=1)
    assert len(env._pub.ledger) < 200            # below the standard minimum
    env, *_ = new_env(1, **{**ALL4, "n_invoices": 150})
    assert len(env._pub.ledger) > 500 and len(env.evaluator_truth()) == 4


def test_jpy_base_and_year_end_period_with_all_error_types():
    env, *_ = new_env(2, **{**ALL4, "base_currency": "JPY", "period_start": "2025-12-01"})
    assert _score(env, [finding(t) for t in env.evaluator_truth()]) == (4, 0, 0, True)


def test_hard_difficulty_clean_episode_is_clean():
    env, *_ = new_env(1, clean=True, difficulty="hard")
    assert env.evaluator_truth() == []
