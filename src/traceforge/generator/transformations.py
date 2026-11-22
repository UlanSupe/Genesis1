"""Metamorphic Transformation Engine: economically equivalent re-labelling/re-ordering.

Operations (each can be switched off):
  * remap_ids : DOC/INV/PAY/ROW/ENT identifiers are re-assigned FIELD BY FIELD (ledger.row_id,
                ledger.entity_id/invoice_id/payment_id, invoices.*, payments.invoice_refs[].invoice_id,
                truth ids/evidence, baseline, provenance). Free text in provenance is rewritten
                token-wise using ONLY known ids.
  * shuffle   : ledger rows, invoices, payments and fx rows are reordered.
  * rename    : neutral line-item descriptions are permuted.
  * retime    : every DOCUMENT timestamp (issued_at, due_at, payment timestamp) is re-expressed at another UTC
                offset - the same instant, a different representation. An agent that compares strings or
                assumes the company offset breaks; one that parses instants does not. Ledger rows stay in the
                company offset (policy), so the economics and the period/FX-date rules are untouched.

`verify_links` then PROVES the transformation preserved every financial link instead of
trusting the substitution: row-by-row and document-by-document isomorphism under the
mapping, injectivity, no stale ids, referential integrity, and truth isomorphism."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
import random
import re
from collections import Counter

from ..schemas import Episode, PublicEpisode
from .documents import DESCRIPTIONS
from .validation import referential_problems, validate_episode

ID_RE = re.compile(r"\b(?:DOC|INV|PAY|ROW|ENT)-\d+\b")


def _prefix(i: str) -> str:
    return i.split("-")[0]


def collect_ids(pub: PublicEpisode, baseline_ledger=()) -> dict:
    """All identifiers, taken from STRUCTURED fields only."""
    ids = set()
    for r in list(pub.ledger) + list(baseline_ledger):
        ids |= {r["row_id"], r["entity_id"]} | {x for x in (r["invoice_id"], r["payment_id"]) if x}
    for d in pub.invoices:
        ids |= {d["doc_id"], d["invoice_id"], d["entity_id"]}
    for d in pub.payments:
        ids |= {d["doc_id"], d["payment_id"]} | {r["invoice_id"] for r in d["invoice_refs"]}
    return {i: _prefix(i) for i in ids}


def build_mapping(ids: dict, rng: random.Random) -> dict:
    mapping = {}
    for prefix in sorted(set(ids.values())):
        olds = sorted(i for i, p in ids.items() if p == prefix)
        for old, n in zip(olds, rng.sample(range(10_000, 99_999), len(olds))):
            mapping[old] = f"{prefix}-{n}"
    return mapping


def _row(r: dict, m) -> dict:
    return {**r, "row_id": m(r["row_id"]), "entity_id": m(r["entity_id"]),
            "invoice_id": m(r["invoice_id"]), "payment_id": m(r["payment_id"])}


def _text(s: str, mapping: dict) -> str:
    return mapping[s] if s in mapping else ID_RE.sub(lambda t: mapping.get(t.group(0), t.group(0)), s)


def _walk(obj, mapping: dict):
    if isinstance(obj, str):
        return _text(obj, mapping)
    if isinstance(obj, list):
        return [_walk(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {_text(k, mapping): _walk(v, mapping) for k, v in obj.items()}
    return obj


OFFSETS_MIN = (-480, -420, -360, -300, -240, -180, 0, 60, 120, 180, 240, 330, 345, 480, 540, 600, 720)


def _reexpress(ts: str, rng: random.Random) -> str:
    """Same instant, another fixed UTC offset (never the one it already has)."""
    t = datetime.fromisoformat(ts)
    cur = int(t.utcoffset().total_seconds() // 60)
    off = rng.choice([o for o in OFFSETS_MIN if o != cur])
    return t.astimezone(timezone(timedelta(minutes=off))).isoformat()


def _retime(pub: PublicEpisode, rng: random.Random) -> None:
    for d in pub.invoices:
        d["issued_at"], d["due_at"] = _reexpress(d["issued_at"], rng), _reexpress(d["due_at"], rng)
    for d in pub.payments:
        d["timestamp"] = _reexpress(d["timestamp"], rng)


def transform_episode(ep: Episode, seed: int, *, remap_ids=True, shuffle=True, rename=True, retime=True):
    """Returns (new_episode, id_mapping)."""
    rng = random.Random(f"transform|{seed}")
    pub, priv = copy.deepcopy(ep.public), copy.deepcopy(ep.private)
    mapping = build_mapping(collect_ids(pub, priv.clean_baseline["ledger"]), rng) if remap_ids else {}
    m = lambda x: mapping.get(x, x) if x is not None else None

    pub.ledger = [_row(r, m) for r in pub.ledger]
    pub.invoices = [{**d, "doc_id": m(d["doc_id"]), "invoice_id": m(d["invoice_id"]), "entity_id": m(d["entity_id"])}
                    for d in pub.invoices]
    pub.payments = [{**d, "doc_id": m(d["doc_id"]), "payment_id": m(d["payment_id"]),
                     "invoice_refs": [{**r, "invoice_id": m(r["invoice_id"])} for r in d["invoice_refs"]]}
                    for d in pub.payments]
    for t in priv.truth:
        t["affected_ids"] = sorted(m(i) for i in t["affected_ids"])
        t["allowed_evidence_sets"] = [sorted(m(i) for i in s) for s in t["allowed_evidence_sets"]]
        if "explanation" in t:                       # free text: only known ids are rewritten
            t["explanation"] = _text(t["explanation"], mapping)
    priv.clean_baseline = {**priv.clean_baseline, "ledger": [_row(r, m) for r in priv.clean_baseline["ledger"]]}
    priv.provenance = _walk(priv.provenance, mapping)
    swap = {}
    if rename:
        swap = dict(zip(DESCRIPTIONS, rng.sample(DESCRIPTIONS, len(DESCRIPTIONS))))
        for inv in pub.invoices:
            for line in inv["line_items"]:
                line["description"] = swap.get(line["description"], line["description"])
    if retime:
        _retime(pub, rng)
    if shuffle:
        for table in (pub.ledger, pub.invoices, pub.payments, pub.fx):
            rng.shuffle(table)
    pub.manifest["episode_id"] = f"{pub.manifest['episode_id']}-x{seed}"
    priv.provenance["transformation"] = {"seed": seed, "remap_ids": remap_ids, "shuffle": shuffle,
                                         "rename": rename, "retime": retime, "description_map": swap}
    return Episode(pub, priv), mapping


def map_findings(findings: list, mapping: dict) -> list:
    """Translate an answer written for the original episode into the transformed one."""
    m = lambda x: mapping.get(x, x)
    return [{**f, "affected_ids": [m(i) for i in f["affected_ids"]],
             "evidence_ids": [m(i) for i in f["evidence_ids"]]} for f in findings]


def _economic_rows(pub: PublicEpisode) -> Counter:
    return Counter((r["timestamp"], r["account"], r["debit_minor"], r["credit_minor"], r["currency"])
                   for r in pub.ledger)


_INV_SKIP = ("doc_id", "entity_id", "invoice_id", "line_items", "issued_at", "due_at")


def _offsets_changed(orig: Episode, new: Episode, mapping: dict) -> bool:
    """True when the transformation really re-expressed document instants at other offsets."""
    m = lambda x: mapping.get(x, x)
    old = {m(d["invoice_id"]): d["issued_at"] for d in orig.public.invoices}
    return any(datetime.fromisoformat(d["issued_at"]).utcoffset() != datetime.fromisoformat(old[d["invoice_id"]]).utcoffset()
               for d in new.public.invoices)


def verify_links(orig: Episode, new: Episode, mapping: dict) -> dict:
    """Prove that every financial link survived. Returns {check: bool}."""
    m = lambda x: mapping.get(x, x) if x is not None else None
    swap = new.private.provenance["transformation"]["description_map"]
    po, pn = orig.public, new.public
    inst = lambda ts: datetime.fromisoformat(ts)

    rows_n = {r["row_id"]: r for r in pn.ledger}
    ledger_ok = len(rows_n) == len(po.ledger) == len(pn.ledger) and all(
        rows_n.get(m(r["row_id"])) == _row(r, m) for r in po.ledger)
    inv_n = {d["invoice_id"]: d for d in pn.invoices}
    inv_ok = len(inv_n) == len(po.invoices) == len(pn.invoices) and all(
        inv_n.get(m(d["invoice_id"])) is not None
        and {k: v for k, v in inv_n[m(d["invoice_id"])].items() if k not in _INV_SKIP}
        == {k: v for k, v in d.items() if k not in _INV_SKIP}
        and inst(inv_n[m(d["invoice_id"])]["issued_at"]) == inst(d["issued_at"])
        and inst(inv_n[m(d["invoice_id"])]["due_at"]) == inst(d["due_at"])
        and inv_n[m(d["invoice_id"])]["doc_id"] == m(d["doc_id"]) and inv_n[m(d["invoice_id"])]["entity_id"] == m(d["entity_id"])
        and Counter((swap.get(l["description"], l["description"]), l["amount_minor"]) for l in d["line_items"])
        == Counter((l["description"], l["amount_minor"]) for l in inv_n[m(d["invoice_id"])]["line_items"])
        for d in po.invoices)
    pay_n = {d["payment_id"]: d for d in pn.payments}
    pay_ok = len(pay_n) == len(po.payments) == len(pn.payments) and all(
        pay_n.get(m(d["payment_id"])) is not None
        and pay_n[m(d["payment_id"])]["doc_id"] == m(d["doc_id"])
        and (inst(pay_n[m(d["payment_id"])]["timestamp"]), pay_n[m(d["payment_id"])]["currency"], pay_n[m(d["payment_id"])]["amount_minor"])
        == (inst(d["timestamp"]), d["currency"], d["amount_minor"])
        and Counter((r["invoice_id"], r["allocated_minor"]) for r in pay_n[m(d["payment_id"])]["invoice_refs"])
        == Counter((m(r["invoice_id"]), r["allocated_minor"]) for r in d["invoice_refs"])
        for d in po.payments)
    new_ids = set(collect_ids(pn, new.private.clean_baseline["ledger"]))
    blob = json.dumps([pn.ledger, pn.invoices, pn.payments, new.private.truth, new.private.provenance,
                       new.private.clean_baseline])
    seen = set(ID_RE.findall(blob))
    stale = (set(mapping) - set(mapping.values())) & seen if mapping else set()
    truth_o = sorted(orig.private.truth, key=lambda t: t["issue_id"])
    truth_n = sorted(new.private.truth, key=lambda t: t["issue_id"])
    return {
        "mapping_injective": len(set(mapping.values())) == len(mapping),
        "mapping_preserves_id_kind": all(_prefix(a) == _prefix(b) for a, b in mapping.items()),
        "ledger_rows_isomorphic": ledger_ok,
        "invoices_isomorphic": inv_ok,
        "payments_isomorphic": pay_ok,
        "no_stale_ids_and_all_ids_remapped": not stale and (not mapping or new_ids == set(mapping.values())),
        "truth_isomorphic": [(t["issue_id"], t["type"], t["expected_amount_minor"], t["currency"],
                              sorted(m(i) for i in t["affected_ids"]),
                              [sorted(m(i) for i in s) for s in t["allowed_evidence_sets"]]) for t in truth_o]
        == [(t["issue_id"], t["type"], t["expected_amount_minor"], t["currency"], t["affected_ids"],
             t["allowed_evidence_sets"]) for t in truth_n],
        "baseline_isomorphic": sorted(_row(r, m)["row_id"] for r in orig.private.clean_baseline["ledger"])
        == sorted(r["row_id"] for r in new.private.clean_baseline["ledger"]),
        "referential_integrity": referential_problems(pn) == [],
        "economic_ledger_equal": _economic_rows(po) == _economic_rows(pn),
        "retime_applied_when_requested": (not new.private.provenance["transformation"].get("retime", False))
        or not po.invoices or _offsets_changed(orig, new, mapping),
    }


def check_pair(orig: Episode, new: Episode, mapping: dict) -> dict:
    """verify_links plus full re-validation of the transformed episode."""
    out = verify_links(orig, new, mapping)
    out["transform_revalidates"] = validate_episode(new)["ok"]
    return out
