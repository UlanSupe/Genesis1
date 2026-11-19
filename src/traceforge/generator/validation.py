"""Independent validator.

`derive_issues` re-derives every discrepancy from PUBLIC data only (documents + fx +
policy) with its own implementation of the posting rules. A generated episode is
valid iff this independent derivation equals the recorded truth one-to-one, so every
truth issue is objectively checkable from public data and nothing else is wrong."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal

from ..constants import (ACC_AR, ACC_CASH, ACC_UNAPPLIED, ACCOUNTS, ALLOCATION, AMOUNT_TOLERANCE_MINOR,
                         CUTOFF, DUPLICATE, FX, LEDGER_FIELDS, MAX_ISSUES)
from ..evaluation.evidence_validator import check_truth_evidence
from ..exceptions import EpisodeValidationError
from ..money import convert_minor
from ..schemas import Episode, PublicEpisode


def _inst(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _key(r: dict) -> str:
    return r["payment_id"] or r["invoice_id"]


def _amt(r: dict) -> int:
    return max(r["debit_minor"], r["credit_minor"])


def _expect(doc: dict, base: str, rate: str):
    """(anchor amount, {invoice_id: signed base amount}, unapplied base amount)."""
    c = doc["currency"]
    if "total_minor" in doc:
        return convert_minor(doc["total_minor"], c, base, rate), {}, 0
    exp = {r["invoice_id"]: convert_minor(r["allocated_minor"], c, base, rate)
           for r in doc["invoice_refs"]}
    rem = doc["amount_minor"] - sum(r["allocated_minor"] for r in doc["invoice_refs"])
    un = convert_minor(rem, c, base, rate) if rem else 0
    return abs(sum(exp.values()) + un), exp, un


def derive_issues(pub: PublicEpisode) -> list:
    base = pub.policy["base_currency"]
    start, end = _inst(pub.policy["period"]["start"]), _inst(pub.policy["period"]["end"])
    tz = start.tzinfo
    fxmap = {(r["date"], r["source_currency"]): r["rate"] for r in pub.fx}
    cls = lambda t: -1 if t < start else (1 if t >= end else 0)
    by_key = defaultdict(list)
    for r in pub.ledger:
        by_key[_key(r)].append(r)
    out = []
    for doc in list(pub.invoices) + list(pub.payments):
        is_inv = "total_minor" in doc
        key = doc["invoice_id"] if is_inv else doc["payment_id"]
        inst = _inst(doc["issued_at"] if is_inv else doc["timestamp"])
        rows = by_key.get(key, [])
        is_anchor = (lambda r: r["account"] == ACC_AR and r["payment_id"] is None) if is_inv \
            else (lambda r: r["account"] == ACC_CASH)
        anchors = [r for r in rows if is_anchor(r)]
        if not anchors:
            raise EpisodeValidationError(f"{key}: no journal")
        mk = lambda typ, aff, amt: out.append({"type": typ, "affected_ids": sorted(set(aff)),
                                               "expected_amount_minor": amt, "currency": base})
        if len(anchors) > 1:
            mk(DUPLICATE, [key], sum(_amt(a) for a in anchors) - _amt(anchors[0]))
        seen, view = set(), []
        for r in rows:                              # collapse exact reposts
            sig = tuple(r[f] for f in LEDGER_FIELDS if f != "row_id")
            if sig not in seen:
                seen.add(sig)
                view.append(r)
        ccy = doc["currency"]
        rate = "1" if ccy == base else fxmap.get((inst.astimezone(tz).date().isoformat(), ccy))
        if rate is None:
            raise EpisodeValidationError(f"{key}: no fx rate")
        cash, exp, un = _expect(doc, base, rate)
        a0 = next(r for r in view if is_anchor(r))
        fx_bad = ccy != base and _amt(a0) != cash
        if fx_bad:
            mk(FX, [key], abs(_amt(a0) - cash))
        elif not is_inv:
            act, unact = defaultdict(int), 0
            for r in view:
                if r["account"] == ACC_AR and r["invoice_id"]:
                    act[r["invoice_id"]] += r["credit_minor"] - r["debit_minor"]
                elif r["account"] == ACC_UNAPPLIED:
                    unact += r["credit_minor"] - r["debit_minor"]
            diffs = {i: exp.get(i, 0) - act.get(i, 0) for i in set(exp) | set(act)}
            diffs = {i: d for i, d in diffs.items() if d}
            if diffs or un != unact:
                short = sum(d for d in diffs.values() if d > 0)
                excess = -sum(d for d in diffs.values() if d < 0)
                mk(ALLOCATION, [key, *diffs], short or excess or abs(un - unact))
        bad = [r for r in rows if cls(_inst(r["timestamp"])) != cls(inst)]
        if bad:
            mk(CUTOFF, [r["row_id"] for r in bad],
               max(sum(r["debit_minor"] for r in bad), sum(r["credit_minor"] for r in bad)))
    return sorted(out, key=lambda i: (i["type"], i["affected_ids"]))


def consistency_problems(pub: PublicEpisode) -> list:
    """Structural/accounting consistency of a public data set (no truth involved)."""
    p, ids = [], [r["row_id"] for r in pub.ledger]
    if len(set(ids)) != len(ids):
        p.append("duplicate row_id")
    by_key = defaultdict(list)
    for r in pub.ledger:
        by_key[_key(r)].append(r)
    for k, rows in by_key.items():
        if sum(r["debit_minor"] for r in rows) != sum(r["credit_minor"] for r in rows):
            p.append(f"{k}: unbalanced journal")
        if any(r["debit_minor"] < 0 or r["credit_minor"] < 0 for r in rows):
            p.append(f"{k}: negative ledger amount")
    inv = {i["invoice_id"]: i for i in pub.invoices}
    for i in pub.invoices:
        if i["total_minor"] != sum(l["amount_minor"] for l in i["line_items"]):
            p.append(f"{i['invoice_id']}: total != sum of lines")
    for d in pub.payments:
        alloc = sum(r["allocated_minor"] for r in d["invoice_refs"])
        ok = alloc <= d["amount_minor"] if d["amount_minor"] > 0 else alloc == d["amount_minor"]
        if not ok:
            p.append(f"{d['payment_id']}: bad allocation sum")
        for r in d["invoice_refs"]:
            if r["invoice_id"] not in inv or inv[r["invoice_id"]]["currency"] != d["currency"]:
                p.append(f"{d['payment_id']}: bad invoice ref")
    for r in pub.ledger:
        for f in ("invoice_id", "payment_id"):
            if r[f] is not None and r[f] not in inv and f == "invoice_id":
                p.append(f"{r['row_id']}: unknown invoice")
    if len(pub.invoices) + len(pub.payments) != pub.manifest["document_count"] or \
            len(pub.ledger) != pub.manifest["record_count"]:
        p.append("manifest counts do not match data")
    return p


_ROW_KEY = ("timestamp", "account", "debit_minor", "credit_minor", "currency", "entity_id", "invoice_id", "payment_id")


def row_key(r: dict) -> tuple:
    return tuple(r[f] for f in _ROW_KEY)


def expected_ledger(pub: PublicEpisode) -> Counter:
    """Independent reconstruction of the CLEAN ledger from documents + fx + policy alone
    (own implementation of the posting rules, deliberately not sharing code with the
    generator): multiset of row keys without row ids."""
    base = pub.policy["base_currency"]
    tz = _inst(pub.policy["period"]["start"]).tzinfo
    fx = {(r["date"], r["source_currency"]): r["rate"] for r in pub.fx}
    ent = {i["invoice_id"]: i["entity_id"] for i in pub.invoices}
    out = Counter()
    for d in list(pub.invoices) + list(pub.payments):
        inv = "total_minor" in d
        t = _inst(d["issued_at"] if inv else d["timestamp"])
        c = d["currency"]
        rate = "1" if c == base else fx[(t.astimezone(tz).date().isoformat(), c)]
        cv = lambda v: convert_minor(v, c, base, rate)
        ts = t.astimezone(tz).isoformat()
        if inv:
            a = cv(d["total_minor"])
            out[(ts, ACC_AR, a, 0, c, d["entity_id"], d["invoice_id"], None)] += 1
            out[(ts, "4000", 0, a, c, d["entity_id"], d["invoice_id"], None)] += 1
            continue
        refs, pid = d["invoice_refs"], d["payment_id"]
        e = ent[refs[0]["invoice_id"]]
        parts = [(r["invoice_id"], cv(r["allocated_minor"])) for r in refs]
        rem = d["amount_minor"] - sum(r["allocated_minor"] for r in refs)
        if rem:
            parts.append((None, cv(rem)))
        if d["amount_minor"] > 0:
            out[(ts, ACC_CASH, sum(v for _, v in parts), 0, c, e, None, pid)] += 1
            for i, v in parts:
                out[(ts, ACC_AR if i else ACC_UNAPPLIED, 0, v, c, e, i, pid)] += 1
        else:
            out[(ts, ACC_CASH, 0, -sum(v for _, v in parts), c, e, None, pid)] += 1
            for i, v in parts:
                out[(ts, ACC_AR, -v, 0, c, e, i, pid)] += 1
    return out


def replay_manifests(baseline_ledger: list, mutations: list) -> list:
    """Apply the mutation manifests (modified_rows keyed by row_id) to the clean baseline."""
    rows = {r["row_id"]: r for r in baseline_ledger}
    for m in mutations:
        for r in m["modified_rows"]:
            rows[r["row_id"]] = r
    return list(rows.values())


def outstanding_balances(pub: PublicEpisode) -> dict:
    """invoice_id -> total - sum(allocations) in the invoice currency (never negative in a
    valid world; refunds re-open balances)."""
    paid = defaultdict(int)
    for p in pub.payments:
        for r in p["invoice_refs"]:
            paid[r["invoice_id"]] += r["allocated_minor"]
    return {i["invoice_id"]: i["total_minor"] - paid[i["invoice_id"]] for i in pub.invoices}


def referential_problems(pub: PublicEpisode) -> list:
    p = []
    dup = lambda xs: len(xs) != len(set(xs))
    docs = list(pub.invoices) + list(pub.payments)
    if dup([d["doc_id"] for d in docs]) or dup([i["invoice_id"] for i in pub.invoices]) \
            or dup([x["payment_id"] for x in pub.payments]):
        p.append("duplicate document identifiers")
    inv = {i["invoice_id"]: i for i in pub.invoices}
    pay = {x["payment_id"]: x for x in pub.payments}
    ents = {}
    for i in pub.invoices:
        if ents.setdefault(i["entity_id"], i["currency"]) != i["currency"]:
            p.append(f"{i['entity_id']}: mixed currencies")
        if _inst(i["due_at"]) <= _inst(i["issued_at"]):
            p.append(f"{i['invoice_id']}: due_at not after issued_at")
    for d in pub.payments:
        ids = [r["invoice_id"] for r in d["invoice_refs"]]
        if dup(ids) or not ids:
            p.append(f"{d['payment_id']}: empty or repeated invoice refs")
        _inst(d["timestamp"])
    fxm = {(r["date"], r["source_currency"]) for r in pub.fx}
    base, tz = pub.policy["base_currency"], _inst(pub.policy["period"]["start"]).tzinfo
    for d in docs:
        t = _inst(d.get("issued_at") or d["timestamp"])
        if d["currency"] != base and (t.astimezone(tz).date().isoformat(), d["currency"]) not in fxm:
            p.append(f"{d.get('invoice_id') or d['payment_id']}: no fx rate for its date")
    for r in pub.fx:
        if not r["rate"].replace(".", "", 1).isdigit() or len(r["rate"].split(".")[-1]) != 8 or Decimal(r["rate"]) <= 0:
            p.append(f"bad fx rate {r['date']} {r['source_currency']}")
    for r in pub.ledger:
        if r["entity_id"] not in ents:
            p.append(f"{r['row_id']}: unknown entity")
        if r["account"] not in ACCOUNTS:
            p.append(f"{r['row_id']}: unknown account")
        if (r["debit_minor"] > 0) == (r["credit_minor"] > 0):
            p.append(f"{r['row_id']}: exactly one of debit/credit must be positive")
        if r["invoice_id"] is not None and r["invoice_id"] not in inv:
            p.append(f"{r['row_id']}: unknown invoice")
        if r["payment_id"] is not None and r["payment_id"] not in pay:
            p.append(f"{r['row_id']}: unknown payment")
        if r["payment_id"] is not None and r["payment_id"] in pay and pay[r["payment_id"]]["currency"] != r["currency"]:
            p.append(f"{r['row_id']}: currency differs from its payment")
    p += [f"{k}: outstanding balance {v} outside [0, total]" for k, v in outstanding_balances(pub).items()
          if v < 0 or v > inv[k]["total_minor"]]
    return p[:50]


def manifest_problems(ep: Episode) -> list:
    """Every recorded issue has exactly one mutation manifest whose rows really are in the
    public ledger and whose amounts agree with the truth."""
    pub, priv = ep.public, ep.private
    p, pubrows = [], {r["row_id"]: r for r in pub.ledger}
    base = {r["row_id"]: r for r in priv.clean_baseline["ledger"]}
    muts = priv.provenance["mutations"]
    truth = {t["issue_id"]: t for t in priv.truth}
    if sorted(m["issue_id"] for m in muts) != sorted(truth):
        p.append("issues and mutation manifests are not one-to-one")
    for m in muts:
        t = truth.get(m["issue_id"])
        if t and (t["expected_amount_minor"] != m["changed_amount_minor"] or t["type"] != m["issue_type"]):
            p.append(f"{m['mutation_id']}: manifest disagrees with truth")
        if not m["causal_description"]:
            p.append(f"{m['mutation_id']}: no causal description")
        for r in m["modified_rows"]:
            if pubrows.get(r["row_id"]) != r:
                p.append(f"{m['mutation_id']}: modified row {r['row_id']} not in the public ledger")
        for r in m["source_rows"]:
            if r["row_id"] in base and base[r["row_id"]] != r:
                p.append(f"{m['mutation_id']}: source row {r['row_id']} differs from baseline")
    for sc in priv.provenance["compound_scenarios"]:
        kinds = {truth[i]["type"] for i in sc["issue_ids"] if i in truth}
        if len(sc["issue_ids"]) != 2 or len(kinds) != 2:
            p.append(f"scenario {sc['scenario']}: needs two issues of different types")
    return p[:20]


def reconstruct_from_evidence(pub: PublicEpisode, evidence_ids) -> list:
    """What an auditor can derive from ONE evidence set: take only the documents named by the
    evidence (directly, or through a cited ledger row's journal) with their journals, plus the
    public fx reference table, and run the independent detector on that subset."""
    docs = {d["doc_id"]: d for d in list(pub.invoices) + list(pub.payments)}
    rows = {r["row_id"]: r for r in pub.ledger}
    keys = set()
    for e in evidence_ids:
        if e in docs:
            keys.add(docs[e].get("invoice_id") or docs[e]["payment_id"])
        elif e in rows:
            keys.add(_key(rows[e]))
    sub = PublicEpisode(pub.manifest, pub.policy, [r for r in pub.ledger if _key(r) in keys],
                        [d for d in pub.invoices if d["invoice_id"] in keys],
                        [d for d in pub.payments if d["payment_id"] in keys], pub.fx)
    return derive_issues(sub)


def reconstruction_problems(ep: Episode) -> list:
    """Every allowed evidence set of every issue must, on its own, reproduce exactly that issue
    (type, affected ids, amount, currency) and nothing that is not a declared issue."""
    declared = {(t["type"], tuple(sorted(t["affected_ids"])), t["expected_amount_minor"], t["currency"])
                for t in ep.private.truth}
    p = []
    for t in ep.private.truth:
        want = (t["type"], tuple(sorted(t["affected_ids"])), t["expected_amount_minor"], t["currency"])
        for s in t["allowed_evidence_sets"]:
            got = {(i["type"], tuple(i["affected_ids"]), i["expected_amount_minor"], i["currency"])
                   for i in reconstruct_from_evidence(ep.public, s)}
            if want not in got:
                p.append(f"{t['issue_id']}: evidence {s[:3]}... does not reproduce the issue")
            elif got - declared:
                p.append(f"{t['issue_id']}: evidence set yields undeclared issue(s) {sorted(got - declared)[:1]}")
    return p[:10]


def affected_id_problems(ep: Episode) -> list:
    from ..evaluation.evidence_validator import public_id_index
    known = public_id_index(ep.public)
    return [f"{t['issue_id']}: unknown affected id {i}" for t in ep.private.truth for i in t["affected_ids"]
            if i not in known][:10]


def decoy_problems(ep: Episode) -> list:
    """Decoys that no mutation touched must be intact: their journals equal the independent
    expected posting, and no duplicate/fx issue names them as the affected document."""
    pub, priv = ep.public, ep.private
    clean = set(priv.provenance["world"].get("clean_decoys", []))
    p = []
    by_key = defaultdict(Counter)
    for r in pub.ledger:
        by_key[_key(r)][row_key(r)] += 1
    for d in list(pub.invoices) + list(pub.payments):
        k = d.get("invoice_id") or d["payment_id"]
        if k not in clean:
            continue
        one = PublicEpisode(pub.manifest, pub.policy, [], [d] if "total_minor" in d else [],
                            [d] if "payment_id" in d else [], pub.fx)
        if "payment_id" in d:                       # entity of a payment comes from its invoices
            one.invoices = [i for i in pub.invoices if i["invoice_id"] in {r["invoice_id"] for r in d["invoice_refs"]}]
            exp = expected_ledger(one) - expected_ledger(PublicEpisode(pub.manifest, pub.policy, [], one.invoices, [], pub.fx))
        else:
            exp = expected_ledger(one)
        if exp != by_key[k]:
            p.append(f"decoy {k}: journal differs from its expected posting")
    for t in priv.truth:
        if t["type"] in (DUPLICATE, FX) and clean & set(t["affected_ids"]):
            p.append(f"{t['issue_id']}: a clean decoy is named as the erroneous document")
    return p[:10]


def totals_problems(ep: Episode) -> list:
    """Financial totals may differ from the clean baseline ONLY by the declared anomalies:
    per-account totals (excluding fx-mutated journals, whose delta depends on the wrong rate) and
    the in-period debit total must move exactly as the mutation manifests predict."""
    pub, priv = ep.public, ep.private
    start, end = _inst(pub.policy["period"]["start"]), _inst(pub.policy["period"]["end"])
    in_period = lambda ts: start <= _inst(ts) < end
    docs = {d["doc_id"]: (d.get("invoice_id") or d["payment_id"], d) for d in list(pub.invoices) + list(pub.payments)}
    fx_keys = {docs[m["document_ids"][0]][0] for m in priv.provenance["mutations"] if m["issue_type"] == FX}

    def totals(ledger):
        acc, per = defaultdict(lambda: [0, 0]), 0
        for r in ledger:
            if _key(r) in fx_keys:
                continue
            acc[r["account"]][0] += r["debit_minor"]
            acc[r["account"]][1] += r["credit_minor"]
            per += r["debit_minor"] * in_period(r["timestamp"])
        return {k: tuple(v) for k, v in acc.items() if v != [0, 0]}, per

    (b_acc, b_per), (p_acc, p_per) = totals(priv.clean_baseline["ledger"]), totals(pub.ledger)
    pred, pred_per = defaultdict(lambda: [0, 0]), 0
    for m in priv.provenance["mutations"]:
        key, doc = docs[m["document_ids"][0]]
        amount, kind = m["changed_amount_minor"], m["issue_type"]
        if kind == DUPLICATE:
            for r in m["source_rows"]:
                pred[r["account"]][0] += r["debit_minor"]
                pred[r["account"]][1] += r["credit_minor"]
                pred_per += r["debit_minor"] * in_period(r["timestamp"])
        elif kind == ALLOCATION and m["variant"] in ("unallocated", "partial"):
            pred[ACC_AR][1] -= amount
            pred[ACC_UNAPPLIED][1] += amount
        elif kind == CUTOFF:
            before = in_period(m["source_rows"][0]["timestamp"])
            after = in_period(m["modified_rows"][0]["timestamp"])
            pred_per += amount * (after - before)
    exp = {k: (b_acc.get(k, (0, 0))[0] + pred[k][0], b_acc.get(k, (0, 0))[1] + pred[k][1])
           for k in set(b_acc) | set(pred)}
    exp = {k: v for k, v in exp.items() if v != (0, 0)}
    p = []
    if exp != p_acc:
        p.append(f"per-account totals differ from baseline+declared effects: {sorted(set(exp.items()) ^ set(p_acc.items()))[:2]}")
    if p_per != b_per + pred_per:
        p.append(f"in-period debit total {p_per} != baseline {b_per} + declared {pred_per}")
    if sum(r["debit_minor"] for r in pub.ledger) != sum(r["credit_minor"] for r in pub.ledger):
        p.append("ledger does not balance")
    return p


def _row_sets(ledger: list) -> dict:
    d = defaultdict(Counter)
    for r in ledger:
        d[_key(r)][tuple(r[f] for f in LEDGER_FIELDS)] += 1
    return d


def validate_episode(ep: Episode, min_rows: int | None = None) -> dict:
    """Run every independent check. A check that cannot even be evaluated (e.g. because the
    data is referentially broken) FAILS with the reason instead of raising."""
    pub, priv = ep.public, ep.private
    checks = []

    def check(name, compute):
        try:
            problems = compute()
        except Exception as e:                       # broken data must yield FAIL, never a crash
            problems = [f"check could not be evaluated: {type(e).__name__}: {str(e)[:80]}"]
        checks.append({"name": name, "ok": not problems, "detail": problems[:5]})

    base_pub = PublicEpisode(pub.manifest, pub.policy, priv.clean_baseline["ledger"],
                             pub.invoices, pub.payments, pub.fx)
    base_pub.manifest = dict(pub.manifest, record_count=len(base_pub.ledger))

    def baseline_vs_expected():
        exp, got = expected_ledger(base_pub), Counter(row_key(r) for r in priv.clean_baseline["ledger"])
        return ([f"expected-only {sorted(map(str, exp - got))[:1]}"] * bool(exp - got)
                + [f"baseline-only {sorted(map(str, got - exp))[:1]}"] * bool(got - exp))

    def replay_check():
        replayed = Counter(row_key(r) for r in replay_manifests(priv.clean_baseline["ledger"], priv.provenance["mutations"]))
        return [] if replayed == Counter(row_key(r) for r in pub.ledger) else ["unexplained difference"]

    def truth_vs_derivation():
        derived = {(i["type"], tuple(i["affected_ids"]), i["expected_amount_minor"], i["currency"])
                   for i in derive_issues(pub)}
        recorded = {(t["type"], tuple(sorted(t["affected_ids"])), t["expected_amount_minor"], t["currency"])
                    for t in priv.truth}
        return ([f"derived-only {sorted(derived - recorded)[:2]}"] * bool(derived - recorded)
                + [f"truth-only {sorted(recorded - derived)[:2]}"] * bool(recorded - derived)
                + ["duplicate truth entries"] * (len(recorded) != len(priv.truth)))

    def changed_docs():
        doc_key = {d["doc_id"]: (d.get("invoice_id") or d.get("payment_id"))
                   for d in list(pub.invoices) + list(pub.payments)}
        a, b = _row_sets(priv.clean_baseline["ledger"]), _row_sets(pub.ledger)
        changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        expected = {doc_key[m["document_ids"][0]] for m in priv.provenance["mutations"]}
        return [f"changed {sorted(changed ^ expected)[:3]}"] * (changed != expected)

    def private_markers():
        blob = " ".join(str(x) for x in (pub.ledger[:50], pub.invoices[:20], pub.manifest, pub.policy))
        return [m for m in ("ISS-", "MUT-", "truth", "provenance") if m in blob]

    def sizes():
        size = []
        if len(pub.invoices) + len(pub.payments) < 20:
            size.append("fewer than 20 documents")
        if min_rows and len(pub.ledger) < min_rows:
            size.append(f"fewer than {min_rows} ledger rows")
        return size

    check("1_clean_baseline_consistent", lambda: consistency_problems(base_pub)
          + [f"baseline has discrepancy {i}" for i in derive_issues(base_pub)])
    check("2_public_data_consistent", lambda: consistency_problems(pub))
    check("1b_baseline_equals_independent_expected_ledger", baseline_vs_expected)
    check("2b_referential_integrity_and_balances", lambda: referential_problems(pub))
    check("4b_baseline_plus_manifests_equals_public_ledger", replay_check)
    check("4c_manifest_causality", lambda: manifest_problems(ep))
    check("3_truth_equals_independent_derivation", truth_vs_derivation)
    check("4_mutations_changed_exactly_the_intended_documents", changed_docs)
    check("5_expected_amounts_exceed_tolerance",
          lambda: [t["issue_id"] for t in priv.truth if t["expected_amount_minor"] <= AMOUNT_TOLERANCE_MINOR])
    check("6_evidence_available_to_agent", lambda: check_truth_evidence(priv.truth, pub))
    check("7_issue_count_bounds", lambda: [] if len(priv.truth) <= MAX_ISSUES else ["too many issues"])
    check("8_no_private_markers_in_public", private_markers)
    check("9_size_requirements", sizes)
    check("10_affected_ids_exist_in_public_data", lambda: affected_id_problems(ep))
    check("11_every_evidence_set_reconstructs_its_finding", lambda: reconstruction_problems(ep))
    check("12_clean_decoys_intact_and_not_labeled_as_errors", lambda: decoy_problems(ep))
    check("13_financial_totals_change_only_by_declared_anomalies", lambda: totals_problems(ep))
    check("14_every_issue_has_root_cause_and_explanation",
          lambda: [t["issue_id"] for t in priv.truth if not t.get("root_cause") or len(t.get("explanation", "")) < 80])
    return {"ok": all(c["ok"] for c in checks), "checks": checks}
