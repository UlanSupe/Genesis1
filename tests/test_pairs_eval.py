"""Paired transformations, hidden-evaluation generation, the `amount` field alias."""
import json
from pathlib import Path

import pytest

from conftest import ALL4, finding, new_env, submit
from traceforge.episode_io import load_episode, save_episode
from traceforge.generator.builder import generate_episode
from traceforge.generator.pairs import build_pairs, eval_seed, generate_eval
from traceforge.generator.validation import validate_episode
from traceforge.runner.paired import run_paired, summarize, write_paired

STORED = Path(__file__).resolve().parents[1] / "datasets"
PAIRS = STORED / "pairs"


# --- finding.amount alias (the official field list says `amount`) ---------------------------------
@pytest.mark.parametrize("key", ["claimed_amount_minor", "amount", "amount_minor"])
def test_amount_aliases_are_accepted(key):
    env, obs, _ = new_env(1, mode="eval", **ALL4)
    f = finding(env.evaluator_truth()[0])
    f[key] = f.pop("claimed_amount_minor")
    _, _, terminated, *_ = submit(env, [f])
    assert terminated and env.result.valid and env.result.metrics.tp >= 1


def test_amount_must_still_be_an_int():
    env, *_ = new_env(1, mode="eval", **ALL4)
    f = finding(env.evaluator_truth()[0])
    f["amount"] = f.pop("claimed_amount_minor") + 0.5
    submit(env, [f])
    assert not env.result.valid and env.result.reward == -1.0


# --- build_pairs -------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("src")
    names = []
    for i, cfg in enumerate([ALL4, {"clean": True}]):
        n = f"s-{i:03d}"
        save_episode(generate_episode(10 + i, {**cfg, "episode_id": n}), root / "train" / n)
        names.append(n)
    (root / "train" / "index.json").write_text(json.dumps({"episodes": names}), encoding="utf-8")
    return root


def test_build_pairs_is_deterministic_and_valid(tiny_source, tmp_path):
    a = build_pairs(tiny_source, "train", tmp_path / "a", seed=3)
    build_pairs(tiny_source, "train", tmp_path / "b", seed=3)
    assert a["pairs"] == 2
    for name in ("pair-train-000", "pair-train-001"):
        da, db = (tmp_path / x / "train" / name for x in "ab")
        for rel in ("public/ledger.csv", "public/invoices.jsonl", "private/truth.json"):
            assert (da / rel).read_bytes() == (db / rel).read_bytes()
        assert all(c["ok"] for c in validate_episode(load_episode(da))["checks"])


def test_twin_has_same_economics_but_different_ids(tiny_source, tmp_path):
    build_pairs(tiny_source, "train", tmp_path, seed=1)
    orig = load_episode(tiny_source / "train" / "s-000")
    twin = load_episode(tmp_path / "train" / "pair-train-000")
    ot, tt = orig.private.truth, twin.private.truth
    assert [t["expected_amount_minor"] for t in ot] == [t["expected_amount_minor"] for t in tt]
    assert [t["type"] for t in ot] == [t["type"] for t in tt]
    assert {i for t in ot for i in t["affected_ids"]}.isdisjoint({i for t in tt for i in t["affected_ids"]})
    assert twin.public.manifest["episode_id"] == "pair-train-000"


def test_pair_index_layout_and_private_mapping(tiny_source, tmp_path):
    build_pairs(tiny_source, "train", tmp_path, seed=1)
    idx = json.loads((tmp_path / "train" / "index.json").read_text())
    assert idx["episodes"] == ["pair-train-000", "pair-train-001"]
    assert idx["originals"] == {"pair-train-000": "s-000", "pair-train-001": "s-001"}
    meta = json.loads((tmp_path / "train" / "pair-train-000" / "private" / "transformation.json").read_text())
    assert meta["original"] == "s-000" and meta["id_mapping"]
    assert "transformation.json" not in " ".join(p.name for p in (tmp_path / "train" / "pair-train-000" / "public").iterdir())


# --- paired evaluation -------------------------------------------------------------------------------
def test_paired_eval_oracle_is_fully_robust_and_random_is_not(tiny_source, tmp_path):
    build_pairs(tiny_source, "train", tmp_path, seed=1)
    res = run_paired(tiny_source, tmp_path, "train", ["oracle", "reconciliation"], mode="eval")
    s = {r["policy"]: r for r in res["summary"]}
    assert s["oracle"]["robustness"] == 1.0 and s["oracle"]["consistent_rate"] == 1.0
    assert s["reconciliation"]["consistent_rate"] == 1.0
    j, c = write_paired(res, tmp_path / "out")
    assert json.loads(j.read_text())["summary"] == res["summary"] and c.read_text().splitlines()[0].startswith("split,pair")


def test_summarize_math():
    rows = [{"policy": "p", "consistent": True, "orig_f1": 1.0, "twin_f1": 0.5, "orig_reward": 1, "twin_reward": 0, "abs_delta_f1": 0.5},
            {"policy": "p", "consistent": False, "orig_f1": 1.0, "twin_f1": 1.0, "orig_reward": 1, "twin_reward": 1, "abs_delta_f1": 0.0}]
    s = summarize(rows)[0]
    assert s["consistent_rate"] == 0.5 and s["mean_abs_delta_f1"] == 0.25 and s["robustness"] == 0.75


# --- hidden evaluation set -----------------------------------------------------------------------------
def test_eval_seed_depends_on_secret_and_index():
    assert eval_seed("a", 0) == eval_seed("a", 0)
    assert len({eval_seed("a", 0), eval_seed("a", 1), eval_seed("b", 0)}) == 3
    assert 0 <= eval_seed("a", 0) < 2 ** 64


def test_generate_eval_is_deterministic_per_secret_valid_and_paired(tmp_path):
    a = generate_eval(tmp_path / "a", count=4, secret="topsecret")
    generate_eval(tmp_path / "b", count=4, secret="topsecret")
    c = generate_eval(tmp_path / "c", count=4, secret="other")
    assert a["episodes"] == 4 and a["secret"] == "topsecret" and a["pairs"]["pairs"] == 4
    for i in range(4):
        n = f"eval-{i:03d}"
        pa, pb = tmp_path / "a" / "eval" / n, tmp_path / "b" / "eval" / n
        assert (pa / "public" / "ledger.csv").read_bytes() == (pb / "public" / "ledger.csv").read_bytes()
        assert all(x["ok"] for x in validate_episode(load_episode(pa))["checks"])
    assert (tmp_path / "a" / "eval" / "eval-000" / "public" / "ledger.csv").read_bytes() != \
           (tmp_path / "c" / "eval" / "eval-000" / "public" / "ledger.csv").read_bytes()
    assert c["secret"] == "other"


def test_generate_eval_without_secret_draws_a_fresh_one(tmp_path):
    a = generate_eval(tmp_path / "a", count=1, with_pairs=False)
    b = generate_eval(tmp_path / "b", count=1, with_pairs=False)
    assert a["secret"] != b["secret"] and len(a["secret"]) == 32 and "pairs" not in a


def test_generate_eval_no_secret_in_public_files(tmp_path):
    r = generate_eval(tmp_path, count=2, secret="S3CR3T-xyz", with_pairs=False)
    blob = b"".join(p.read_bytes() for p in (tmp_path / "eval").rglob("*") if p.is_file())
    assert b"S3CR3T-xyz" not in blob and r["secret"] == "S3CR3T-xyz"


# --- stored pair datasets ------------------------------------------------------------------------------
@pytest.mark.parametrize("split", ["train", "dev"])
def test_stored_pairs_match_regeneration(split, tmp_path):
    assert (PAIRS / split / "index.json").exists(), "run `python -m traceforge pairs`"
    build_pairs(STORED, split, tmp_path, seed=0)
    idx = json.loads((PAIRS / split / "index.json").read_text())
    assert len(idx["episodes"]) == 10
    for n in idx["episodes"]:
        for rel in ("public/ledger.csv", "public/invoices.jsonl", "public/payments.jsonl", "private/truth.json"):
            assert (PAIRS / split / n / rel).read_bytes() == (tmp_path / split / n / rel).read_bytes()


def test_stored_dev_pairs_oracle_robustness_is_one():
    res = run_paired(STORED, PAIRS, "dev", ["oracle"], mode="eval")
    assert res["summary"][0]["robustness"] == 1.0 and res["summary"][0]["pairs"] == 10


# --- representation change: same instants, other UTC offsets ----------------------------------------------
def _doc_instants(ep):
    from datetime import datetime
    inv = {d["invoice_id"]: (datetime.fromisoformat(d["issued_at"]), datetime.fromisoformat(d["due_at"])) for d in ep.public.invoices}
    pay = {d["payment_id"]: datetime.fromisoformat(d["timestamp"]) for d in ep.public.payments}
    return inv, pay


def test_retime_changes_offsets_but_not_instants(tiny_source):
    from traceforge.generator.transformations import transform_episode
    orig = load_episode(tiny_source / "train" / "s-000")
    new, mapping = transform_episode(orig, 5)
    io, po = _doc_instants(orig)
    inn, pn = _doc_instants(new)
    assert {mapping[k]: v for k, v in io.items()} == inn and {mapping[k]: v for k, v in po.items()} == pn   # same instants
    changed = sum(a.utcoffset() != b.utcoffset() for (a, _), (b, _) in zip(io.values(), [inn[mapping[k]] for k in io]))
    assert changed == len(io)                                                                              # every offset differs
    assert new.private.provenance["transformation"]["retime"] is True


def test_retime_can_be_switched_off(tiny_source):
    from traceforge.generator.transformations import transform_episode
    orig = load_episode(tiny_source / "train" / "s-000")
    new, mapping = transform_episode(orig, 5, retime=False)
    assert {d["issued_at"] for d in orig.public.invoices} == {d["issued_at"] for d in new.public.invoices}
    assert new.private.provenance["transformation"]["retime"] is False


def test_verify_links_catches_a_shifted_instant(tiny_source):
    import copy
    from traceforge.generator.transformations import check_pair, transform_episode
    orig = load_episode(tiny_source / "train" / "s-000")
    new, mapping = transform_episode(orig, 5)
    assert check_pair(orig, new, mapping)["retime_applied_when_requested"]
    bad = copy.deepcopy(new)
    bad.public.payments[0]["timestamp"] = "2000-01-01T00:00:00+00:00"          # a different instant, not a different offset
    assert not check_pair(orig, bad, mapping)["payments_isomorphic"]


def test_paired_eval_separates_a_parser_of_instants_from_a_wall_clock_reader():
    res = run_paired(STORED, PAIRS, "dev", ["reconciliation", "naive_clock"], mode="eval")
    s = {r["policy"]: r for r in res["summary"]}
    assert s["reconciliation"]["consistent_rate"] == 1.0 and s["naive_clock"]["consistent_rate"] <= 0.2
