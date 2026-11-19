import json
import random
from collections import Counter
from datetime import datetime
from decimal import Decimal

import pytest

from conftest import ALL4, episode
from traceforge.config import GeneratorConfig
from traceforge.episode_io import load_episode, public_files, save_episode
from traceforge.exceptions import ConfigError
from traceforge.generator import validation
from traceforge.generator.builder import generate_episode
from traceforge.money import convert_minor, fmt_major


def test_default_episode_meets_size_minimums():
    ep = generate_episode(1, {"clean": True})
    assert len(ep.public.ledger) >= 200
    assert ep.public.manifest["document_count"] >= 20
    assert ep.public.manifest["record_count"] == len(ep.public.ledger)


def test_same_seed_and_config_is_byte_identical():
    a, b = generate_episode(5, ALL4), generate_episode(5, ALL4)
    assert public_files(a.public) == public_files(b.public)
    assert a.private.truth == b.private.truth and a.private.provenance == b.private.provenance


def test_different_seed_gives_different_episode():
    assert public_files(episode(1, **ALL4).public) != public_files(episode(2, **ALL4).public)


def test_generation_ignores_and_preserves_global_random_state():
    random.seed(123)
    before = random.getstate()
    a = generate_episode(9, ALL4)
    assert random.getstate() == before
    random.seed(999)
    assert public_files(generate_episode(9, ALL4).public) == public_files(a.public)


def test_clean_episode_has_no_truth_and_no_derivable_issue():
    for seed in range(6):
        ep = generate_episode(seed, {"clean": True})
        assert ep.private.truth == []
        assert validation.derive_issues(ep.public) == []
        assert ep.private.clean_baseline["ledger"] == ep.public.ledger


def test_money_is_integer_minor_units_everywhere():
    ep = episode(1, **ALL4)
    assert all(type(r["debit_minor"]) is int and type(r["credit_minor"]) is int for r in ep.public.ledger)
    assert all(type(d["total_minor"]) is int for d in ep.public.invoices)
    assert all(type(d["amount_minor"]) is int for d in ep.public.payments)
    assert "." not in json.dumps([d["total_minor"] for d in ep.public.invoices])


def test_fx_rates_are_decimal_strings_with_8_places():
    ep = episode(1)
    for r in ep.public.fx:
        assert isinstance(r["rate"], str) and Decimal(r["rate"]) > 0
        assert len(r["rate"].split(".")[1]) == 8


def test_every_journal_balances_before_and_after_injection():
    for ep in (episode(1, clean=True), episode(1, **ALL4)):
        sums = {}
        for r in ep.public.ledger:
            k = r["payment_id"] or r["invoice_id"]
            d, c = sums.get(k, (0, 0))
            sums[k] = (d + r["debit_minor"], c + r["credit_minor"])
        assert all(d == c for d, c in sums.values())


def test_invoice_totals_and_payment_allocations_consistent():
    ep = episode(3, clean=True)
    for i in ep.public.invoices:
        assert i["total_minor"] == sum(l["amount_minor"] for l in i["line_items"])
    inv = {i["invoice_id"]: i for i in ep.public.invoices}
    for p in ep.public.payments:
        alloc = sum(r["allocated_minor"] for r in p["invoice_refs"])
        assert alloc <= p["amount_minor"] if p["amount_minor"] > 0 else alloc == p["amount_minor"]
        assert all(inv[r["invoice_id"]]["currency"] == p["currency"] for r in p["invoice_refs"])


def test_timestamps_are_iso8601_with_offsets_and_multiple_timezones():
    ep = episode(2, difficulty="medium")
    stamps = [r["timestamp"] for r in ep.public.ledger] + [p["timestamp"] for p in ep.public.payments]
    offsets = {datetime.fromisoformat(s).utcoffset() for s in stamps}
    assert all(datetime.fromisoformat(s).tzinfo is not None for s in stamps)
    assert len(offsets) >= 3


def test_multiple_currencies_present():
    ep = episode(2, difficulty="medium")
    assert len({d["currency"] for d in ep.public.invoices}) >= 3


def test_missing_references_are_null_not_empty_string():
    ep = episode(1, **ALL4)
    assert any(r["invoice_id"] is None for r in ep.public.ledger)
    assert all(r["invoice_id"] != "" and r["payment_id"] != "" for r in ep.public.ledger)


def test_zero_and_three_decimal_currencies_round_half_up():
    assert convert_minor(1000, "JPY", "EUR", "0.0065") == 650           # 1000 JPY -> 6.50 EUR
    assert convert_minor(1005, "USD", "JPY", "100") == 1005             # 10.05 USD * 100 = 1005 JPY
    assert convert_minor(1, "EUR", "JPY", "50") == 1                    # 0.5 JPY rounds half up to 1
    assert convert_minor(5, "EUR", "KWD", "0.1") == 5                   # 0.005 KWD = 5 fils
    assert convert_minor(125, "USD", "EUR", "0.5") == 63                # 62.5 -> 63 (half up)
    assert convert_minor(-125, "USD", "EUR", "0.5") == -63              # symmetric for refunds
    assert fmt_major(-1234, "EUR") == "-12.34" and fmt_major(7, "JPY") == "7" and fmt_major(5, "KWD") == "0.005"


@pytest.mark.parametrize("base", ["JPY", "KWD", "USD", "CHF"])
def test_other_base_currencies_validate(base):
    ep = generate_episode(4, {**ALL4, "base_currency": base})
    assert ep.private.provenance["validation"]["ok"]


def test_config_validation_rejects_bad_input():
    with pytest.raises(ConfigError):
        GeneratorConfig.from_dict({"nope": 1})
    with pytest.raises(ConfigError):
        GeneratorConfig.from_dict({"issue_counts": {"fx": 13}})
    with pytest.raises(ConfigError):
        GeneratorConfig.from_dict({"issue_counts": {"banana": 1}})
    with pytest.raises(ConfigError):
        GeneratorConfig.from_dict({"difficulty": "extreme"})
    with pytest.raises(ConfigError):
        GeneratorConfig.from_dict({"base_currency": "XXX"})


def test_zero_to_twelve_issues_supported():
    assert generate_episode(1, {"clean": True}).private.truth == []
    twelve = generate_episode(1, {"issue_counts": {"duplicate": 3, "allocation": 3, "fx": 3, "cutoff": 3}})
    assert len(twelve.private.truth) == 12 and twelve.private.provenance["validation"]["ok"]


def test_record_counts_are_configurable():
    small = generate_episode(1, {"clean": True, "n_invoices": 6, "min_ledger_rows": 30, "n_entities": 4})
    big = generate_episode(1, {"clean": True, "n_invoices": 120, "min_ledger_rows": 30})
    # the guaranteed decoy kinds put a floor under the size; the configured counts still drive it
    assert len(small.public.ledger) < 260 and len(big.public.ledger) > 500 and 2 * len(small.public.ledger) < len(big.public.ledger)


def test_decoys_are_present_in_every_episode():
    ep = episode(1, clean=True)
    inv, pay = ep.public.invoices, ep.public.payments
    # repeated amounts on one counterparty (distinct invoices)
    same = Counter((i["entity_id"], i["total_minor"]) for i in inv)
    assert any(n >= 2 for n in same.values())
    assert any(p["amount_minor"] < 0 for p in pay)                               # refund
    assert any(len(p["invoice_refs"]) >= 2 for p in pay)                         # split payment
    paid = Counter()
    for p in pay:
        for r in p["invoice_refs"]:
            paid[r["invoice_id"]] += r["allocated_minor"]
    totals = {i["invoice_id"]: i["total_minor"] for i in inv}
    assert any(0 < paid[k] < totals[k] for k in paid)                            # partial payment
    assert any(p["amount_minor"] > sum(r["allocated_minor"] for r in p["invoice_refs"]) for p in pay)  # unapplied


def test_timezone_boundary_decoys_exist_and_are_clean():
    ep = episode(1, clean=True)
    start = datetime.fromisoformat(ep.public.manifest["period_start"])
    end = datetime.fromisoformat(ep.public.manifest["period_end"])
    hits = 0
    for p in ep.public.payments:
        t = datetime.fromisoformat(p["timestamp"])
        near = min(abs((t - start).total_seconds()), abs((t - end).total_seconds())) <= 5400
        if near and t.date() != t.astimezone(start.tzinfo).date():
            hits += 1
    assert hits >= 1 and validation.derive_issues(ep.public) == []


def test_row_ids_follow_final_order_and_do_not_reveal_injection():
    ep = episode(1, **ALL4)
    ids = [r["row_id"] for r in ep.public.ledger]
    assert ids == [f"ROW-{i:05d}" for i in range(1, len(ids) + 1)]
    stamps = [datetime.fromisoformat(r["timestamp"]) for r in ep.public.ledger]
    assert stamps == sorted(stamps)


def test_public_private_separation_on_disk(tmp_path):
    ep = episode(1, **ALL4)
    save_episode(ep, tmp_path / "e")
    pub = sorted(p.name for p in (tmp_path / "e" / "public").iterdir())
    priv = sorted(p.name for p in (tmp_path / "e" / "private").iterdir())
    assert pub == ["fx.csv", "invoices.jsonl", "ledger.csv", "manifest.json", "payments.jsonl", "policy.json"]
    assert priv == ["clean_baseline.json", "provenance.json", "truth.json"]
    for p in (tmp_path / "e" / "public").iterdir():
        text = p.read_text(encoding="utf-8")
        assert "ISS-" not in text and "MUT-" not in text and "expected_amount" not in text


def test_save_load_roundtrip(tmp_path):
    ep = episode(1, **ALL4)
    save_episode(ep, tmp_path / "e")
    back = load_episode(tmp_path / "e")
    assert back.public == ep.public and back.private.truth == ep.private.truth
    assert load_episode(tmp_path / "e", private=False).private is None


def test_manifest_has_required_fields():
    m = episode(1).public.manifest
    for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end",
              "timezone", "record_count", "document_count", "rounding_rules"):
        assert k in m
    assert m["rounding_rules"]["mode"] == "ROUND_HALF_UP"


def test_period_boundaries_february_leap_year_and_december():
    feb = generate_episode(1, {"clean": True, "period_start": "2024-02-01", "timezone_minutes": 60})
    assert feb.public.manifest["period_end"].startswith("2024-03-01")
    dec = generate_episode(1, {"clean": True, "period_start": "2025-12-01"})
    assert dec.public.manifest["period_end"].startswith("2026-01-01")


def test_extreme_timezone_offsets_validate():
    for tz in (-720, 840):
        ep = generate_episode(2, {**ALL4, "timezone_minutes": tz})
        assert ep.private.provenance["validation"]["ok"]


def test_validator_detects_tampered_ledger():
    ep = generate_episode(1, ALL4)
    ep.public.ledger[0]["debit_minor"] += 7                     # unbalance a journal
    report = validation.validate_episode(ep)
    assert not report["ok"]


def test_validator_detects_unavailable_evidence():
    ep = generate_episode(1, ALL4)
    ep.private.truth[0]["allowed_evidence_sets"][0].append("ROW-99999")
    failed = [c["name"] for c in validation.validate_episode(ep)["checks"] if not c["ok"]]
    assert "6_evidence_available_to_agent" in failed


def test_validator_detects_unlisted_error():
    ep = generate_episode(1, {"clean": True})
    inv = next(r for r in ep.public.ledger if r["account"] == "1200" and r["payment_id"] is None)
    inv["credit_minor"], inv["debit_minor"] = 0, inv["debit_minor"] + 5     # corrupt silently
    assert not validation.validate_episode(ep)["ok"]


def test_period_end_is_exclusive_in_derivation():
    ep = generate_episode(1, {"clean": True})
    start = datetime.fromisoformat(ep.public.manifest["period_start"])
    end_s = ep.public.manifest["period_end"]
    inv = next(i for i in ep.public.invoices if start <= datetime.fromisoformat(i["issued_at"]) < datetime.fromisoformat(end_s))
    for r in ep.public.ledger:
        if r["invoice_id"] == inv["invoice_id"] and r["payment_id"] is None:
            r["timestamp"] = end_s                                # exactly at period_end -> next period
    found = validation.derive_issues(ep.public)
    assert [f["type"] for f in found] == ["cutoff"]
