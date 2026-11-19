"""Property-based checks with deterministic seeds (no hypothesis dependency): random but
reproducible configurations, invariants that must hold for EVERY generated episode."""
import random
from collections import Counter
from decimal import Decimal

import pytest

from conftest import new_env, submit
from traceforge.config import PRESETS
from traceforge.constants import ISSUE_TYPES
from traceforge.environment.calculator import evaluate
from traceforge.generator import validation
from traceforge.generator.builder import generate_episode
from traceforge.money import convert_minor

BASES = ["EUR", "USD", "GBP", "CHF", "JPY", "SEK", "CAD", "KWD"]


def random_config(rng: random.Random) -> dict:
    kind = rng.randrange(4)
    cfg = {"base_currency": rng.choice(BASES), "difficulty": rng.choice(["easy", "medium", "hard"]),
           "timezone_minutes": rng.choice([None, -720, -300, 0, 60, 330, 840])}
    if kind == 0:
        cfg["clean"] = True
    elif kind == 1:
        cfg["issue_counts"] = {t: rng.randint(0, 2) for t in ISSUE_TYPES}
    elif kind == 2:
        cfg["scenarios"] = rng.sample(["cross_border_partial", "period_end_duplicate"], rng.randint(1, 2))
        cfg["issue_counts"] = {rng.choice(ISSUE_TYPES): rng.randint(0, 2)}
    else:
        cfg.update(PRESETS[rng.choice(list(PRESETS))])
    if rng.random() < 0.3:
        cfg["n_invoices"] = rng.randint(20, 120)
    return cfg


SEEDS = range(60)


@pytest.fixture(scope="module")
def episodes():
    out = []
    for s in SEEDS:
        cfg = random_config(random.Random(s))
        out.append((s, cfg, generate_episode(s, cfg)))
    return out


def test_every_random_episode_passes_all_independent_validation_checks(episodes):
    for s, cfg, ep in episodes:
        rep = validation.validate_episode(ep, 200)
        assert rep["ok"], (s, cfg, [c for c in rep["checks"] if not c["ok"]])


def test_before_mutation_company_is_consistent_for_every_episode(episodes):
    for s, cfg, ep in episodes:
        base = type(ep.public)(ep.public.manifest, ep.public.policy, ep.private.clean_baseline["ledger"],
                               ep.public.invoices, ep.public.payments, ep.public.fx)
        assert validation.derive_issues(base) == [], s
        assert validation.expected_ledger(base) == Counter(validation.row_key(r) for r in base.ledger), s


def test_after_mutation_every_discrepancy_is_declared_and_nothing_else(episodes):
    for s, cfg, ep in episodes:
        derived = validation.derive_issues(ep.public)
        assert len(derived) == len(ep.private.truth) == len(ep.private.provenance["mutations"]), (s, cfg)
        changed = {r["row_id"] for r in validation.replay_manifests(ep.private.clean_baseline["ledger"], ep.private.provenance["mutations"])}
        assert changed == {r["row_id"] for r in ep.public.ledger}


def test_requested_issue_counts_are_realised_exactly(episodes):
    checked = 0
    for s, cfg, ep in episodes:
        if "issue_counts" in cfg and "scenarios" not in cfg:
            got = Counter(t["type"] for t in ep.private.truth)
            assert got == Counter({k: v for k, v in cfg["issue_counts"].items() if v}), (s, cfg)
            checked += 1
    assert checked >= 5


def test_global_ledger_balance_and_non_negativity(episodes):
    for s, cfg, ep in episodes:
        assert sum(r["debit_minor"] for r in ep.public.ledger) == sum(r["credit_minor"] for r in ep.public.ledger)
        assert all(r["debit_minor"] >= 0 and r["credit_minor"] >= 0 for r in ep.public.ledger)
        assert all(v >= 0 for v in validation.outstanding_balances(ep.public).values()), s


def test_clean_episodes_have_no_issue_and_oracle_gets_perfect_score_everywhere(episodes):
    for s, cfg, ep in list(episodes)[:25]:
        env, obs, _ = new_env(s, **cfg)
        truth = env.evaluator_truth()
        assert truth == ep.private.truth
        submit(env, [{"type": t["type"], "affected_ids": t["affected_ids"], "claimed_amount_minor": t["expected_amount_minor"],
                      "currency": t["currency"], "evidence_ids": t["allowed_evidence_sets"][0], "explanation": ""} for t in truth])
        assert env.result.metrics.f1 == 1.0 and (not truth or env.result.metrics.tp == len(truth))


def test_generation_is_deterministic_for_random_configs():
    for s in range(15):
        cfg = random_config(random.Random(500 + s))
        a, b = generate_episode(s, cfg), generate_episode(s, cfg)
        assert a.public == b.public and a.private.truth == b.private.truth


def test_evidence_ids_are_always_public_and_dedicated_per_issue(episodes):
    for s, cfg, ep in episodes:
        assert all(c["ok"] for c in ep.private.provenance["validation"]["checks"])
        assert len({t["issue_id"] for t in ep.private.truth}) == len(ep.private.truth)


@pytest.mark.parametrize("seed", range(40))
def test_money_conversion_laws(seed):
    rng = random.Random(seed)
    src, dst = rng.sample(BASES, 2)
    rate = f"{rng.randint(1, 400000000) / 10 ** rng.randint(3, 8):.8f}"
    a, b = rng.randint(1, 10 ** 8), rng.randint(1, 10 ** 8)
    assert convert_minor(a, src, src, rate) == a                                   # identity
    assert convert_minor(-a, src, dst, rate) == -convert_minor(a, src, dst, rate)  # sign symmetry
    assert convert_minor(a, src, dst, rate) <= convert_minor(a + b, src, dst, rate)  # monotone
    assert abs(convert_minor(a + b, src, dst, rate) - convert_minor(a, src, dst, rate) - convert_minor(b, src, dst, rate)) <= 1
    exact = Decimal(a).scaleb(-({"JPY": 0, "KWD": 3}.get(src, 2))) * Decimal(rate)
    assert abs(Decimal(convert_minor(a, src, dst, rate)).scaleb(-({"JPY": 0, "KWD": 3}.get(dst, 2))) - exact) <= Decimal("0.5").scaleb(-({"JPY": 0, "KWD": 3}.get(dst, 2)))


@pytest.mark.parametrize("seed", range(40))
def test_calculator_matches_decimal_arithmetic(seed):
    rng = random.Random(seed)
    a, b = (f"{rng.randint(-10 ** 6, 10 ** 6) / 10 ** rng.randint(0, 6)}" for _ in range(2))
    da, db = Decimal(a), Decimal(b)
    assert Decimal(evaluate(f"({a}) + ({b})")) == da + db
    assert Decimal(evaluate(f"({a}) * ({b})")) == da * db
    assert evaluate(f"({a}) + ({b})") == evaluate(f"({b}) + ({a})")
    if db != 0 and da != 0:
        assert Decimal(evaluate(f"(({a}) * ({b})) / ({b})")) == da


@pytest.mark.parametrize("seed", range(10))
def test_pagination_reconstructs_every_table_for_random_page_sizes(seed):
    rng = random.Random(seed)
    cfg = {"difficulty": rng.choice(["easy", "hard"]), "clean": True}
    for table in ("ledger", "invoices", "payments", "fx"):
        env, *_ = new_env(seed, **cfg)                       # fresh 100-step budget per table
        limit, cursor, got, pages = rng.randint(20, 50), None, [], 0
        while True:
            obs, *_ = env.step({"action": "list_records", "table": table, "cursor": cursor, "limit": limit})
            r = obs["last_result"]
            got += r["records"]
            pages += 1
            cursor = r["next_cursor"]
            if cursor is None:
                break
        assert len(got) == r["total"] and pages == -(-r["total"] // limit)       # complete and exact page count
        assert got == env._tables[table]                                           # same records, same order
