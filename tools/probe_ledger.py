import sys; sys.path.insert(0, "src")
from collections import Counter
from datetime import datetime
from traceforge.generator.builder import generate_episode
from traceforge.money import convert_minor

def expected(pub):
    base = pub.policy["base_currency"]; tz = datetime.fromisoformat(pub.policy["period"]["start"]).tzinfo
    fx = {(r["date"], r["source_currency"]): r["rate"] for r in pub.fx}
    ent = {i["invoice_id"]: i["entity_id"] for i in pub.invoices}
    out = Counter()
    for d in pub.invoices + pub.payments:
        inv = "total_minor" in d
        t = datetime.fromisoformat(d["issued_at"] if inv else d["timestamp"])
        c = d["currency"]; rate = "1" if c == base else fx[(t.astimezone(tz).date().isoformat(), c)]
        cv = lambda v: convert_minor(v, c, base, rate); ts = t.astimezone(tz).isoformat()
        if inv:
            a = cv(d["total_minor"]); e = d["entity_id"]
            out[(ts, "1200", a, 0, c, e, d["invoice_id"], None)] += 1; out[(ts, "4000", 0, a, c, e, d["invoice_id"], None)] += 1
        else:
            e = ent[d["invoice_refs"][0]["invoice_id"]]; ref = d["invoice_refs"]; pid = d["payment_id"]
            parts = [(r["invoice_id"], cv(r["allocated_minor"])) for r in ref]
            rem = d["amount_minor"] - sum(r["allocated_minor"] for r in ref)
            if rem: parts.append((None, cv(rem)))
            if d["amount_minor"] > 0:
                out[(ts, "1000", sum(v for _, v in parts), 0, c, e, None, pid)] += 1
                for i, v in parts: out[(ts, "1200" if i else "2300", 0, v, c, e, i, pid)] += 1
            else:
                out[(ts, "1000", 0, -sum(v for _, v in parts), c, e, None, pid)] += 1
                for i, v in parts: out[(ts, "1200", -v, 0, c, e, i, pid)] += 1
    return out
key = lambda r: (r["timestamp"], r["account"], r["debit_minor"], r["credit_minor"], r["currency"], r["entity_id"], r["invoice_id"], r["payment_id"])
bad_base = bad_apply = 0
for s in range(60):
    ep = generate_episode(s, {"difficulty": ["easy", "medium", "hard"][s % 3], "base_currency": ["EUR", "JPY", "USD"][s % 3], "issue_counts": {"duplicate": 2, "allocation": 2, "fx": 2, "cutoff": 2}})
    base = Counter(key(r) for r in ep.private.clean_baseline["ledger"])
    exp = expected(ep.public)
    if base != exp: bad_base += 1; print("baseline != expected", s, list((base - exp).items())[:1], list((exp - base).items())[:1])
    led = {r["row_id"]: r for r in ep.private.clean_baseline["ledger"]}
    for m in ep.private.provenance["mutations"]:
        for r in m["modified_rows"]: led[r["row_id"]] = r
    if sorted(led) != sorted(r["row_id"] for r in ep.public.ledger) or Counter(key(r) for r in led.values()) != Counter(key(r) for r in ep.public.ledger):
        bad_apply += 1; print("baseline+manifests != public", s)
print("independent expected-ledger mismatches:", bad_base, "| manifest replay mismatches:", bad_apply)
