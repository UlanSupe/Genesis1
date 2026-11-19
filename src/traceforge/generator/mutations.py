"""Four independent mutation engines. Each mutates the journal of ONE document
(a document is mutated at most once), keeps every journal balanced, and records the
truth (issue) with a root cause and a detailed explanation, plus causal provenance.

DIFFICULTY PROFILES decide which variants occur and how subtle they are (never whether
the data is solvable - the independent validator proves every episode is):

  easy    clear anomalies, direct evidence: an allocation parked in unapplied receipts, a
          3-5 % wrong rate, a posting at least a day in the wrong period.
  medium  cross-referencing: partial allocations, a rate of the wrong date (needs the fx
          table), a posting 4-24 h across the edge.
  hard    indirect: an allocation applied to a different invoice (two invoices to compare),
          0.5-1.5 % rate differences, wrong-date rates of neighbouring days, postings only
          15 min - 3 h across the edge, reposts of documents that have a legitimate lookalike."""
from __future__ import annotations

import random
from collections import Counter
from datetime import timedelta
from decimal import Decimal

from ..constants import (ACC_AR, ACC_UNAPPLIED, ALLOCATION, CUTOFF, DUPLICATE, FX,
                         ISSUE_TYPES, LEDGER_FIELDS)
from ..exceptions import GenerationError
from ..money import fmt_rate
from .financial_world import World, iso
from .ledger import journal_spec

PROFILES = {
    "easy": dict(alloc=("unallocated",), fx=("wrong_rate",), rate_bp=(300, 500), shift_min=(1440, 3600), lookalike=False, max_k=6),
    "medium": dict(alloc=("unallocated", "partial"), fx=("wrong_rate", "wrong_date"), rate_bp=(150, 300), shift_min=(240, 1440), lookalike=False, max_k=6),
    "hard": dict(alloc=("partial", "wrong_invoice"), fx=("wrong_date", "wrong_rate"), rate_bp=(50, 150), shift_min=(15, 180), lookalike=True, max_k=2),
}

ROOT_CAUSE = {
    (DUPLICATE, "repost"): "journal_posted_twice",
    (ALLOCATION, "unallocated"): "allocation_not_applied",
    (ALLOCATION, "partial"): "allocation_partially_applied",
    (ALLOCATION, "wrong_invoice"): "allocation_applied_to_wrong_invoice",
    (FX, "wrong_rate"): "wrong_fx_rate_used",
    (FX, "wrong_date"): "fx_rate_of_wrong_date_used",
    (CUTOFF, "period_shift"): "booked_in_wrong_reporting_period",
}


def snap(row: dict) -> dict:
    d = {k: row[k] for k in LEDGER_FIELDS}
    d["row_id"] = row["_uid"]                    # symbolic until final ids are assigned
    return d


class MutCtx:
    def __init__(self, world: World, journals: dict, rng: random.Random):
        self.world, self.journals, self.rng = world, journals, rng
        self.profile = PROFILES[world.cfg.difficulty]
        self.docs = {World.doc_key(d): d for d in world.invoices + world.payments}
        self.inv_doc = {i["invoice_id"]: i["doc_id"] for i in world.invoices}
        self.used: set = set()
        self.issues: list = []
        self.mutations: list = []
        self.scenarios: list = []

    def clone(self, row: dict) -> dict:
        c = dict(row)
        c["_uid"], c["_tb"] = self.world.uid(), self.rng.random()
        return c

    def record(self, typ, doc, affected, amount, evidence, before, after, desc, variant,
               how: str, scenario=None) -> str:
        n = len(self.issues) + 1
        iid = f"ISS-{n:03d}"
        cause = ROOT_CAUSE[(typ, variant)]
        explanation = (f"{desc} How to see it: {how} Expected amount: {amount} {self.world.base} minor units "
                       f"(currency {self.world.base}).")
        self.issues.append({"issue_id": iid, "type": typ, "affected_ids": affected,
                            "expected_amount_minor": amount, "currency": self.world.base,
                            "allowed_evidence_sets": evidence, "root_cause": cause,
                            "explanation": explanation})
        self.mutations.append({
            "mutation_id": f"MUT-{n:03d}", "issue_id": iid, "issue_type": typ,
            "variant": variant, "scenario": scenario, "document_ids": [doc["doc_id"]],
            "source_rows": before, "modified_rows": after, "changed_amount_minor": amount,
            "root_cause": cause, "causal_description": desc, "available_evidence": evidence})
        self.used.add(World.doc_key(doc))
        return iid

    def free_docs(self) -> list:
        """Every document may carry an error except scaffold documents reserved for compound
        scenarios. Decoys are NOT immune: otherwise 'looks like a decoy' would be a free tell."""
        return [d for d in self.docs.values()
                if not d["_tag"].startswith("compound") and World.doc_key(d) not in self.used]


# ---------------------------------------------------------------- A. DUPLICATE
def cand_duplicate(ctx: MutCtx) -> list:
    pool = ctx.free_docs()
    if ctx.profile["lookalike"]:                  # hard: repost a document that has a legitimate twin
        twins = Counter((i["entity_id"], i["total_minor"]) for i in ctx.world.invoices)
        like = [d for d in pool if "total_minor" in d and twins[(d["entity_id"], d["total_minor"])] > 1]
        pool = like or pool
    return pool


def apply_duplicate(ctx: MutCtx, doc: dict, scenario=None) -> str:
    key = World.doc_key(doc)
    rows = ctx.journals[key]
    before = [snap(r) for r in rows]
    copies = [ctx.clone(r) for r in rows]
    rows.extend(copies)
    amt = max(rows[0]["debit_minor"], rows[0]["credit_minor"])
    n = len(copies)                      # any leg: (original line, its repost) is valid evidence
    ev = [[doc["doc_id"], rows[i]["_uid"], copies[i]["_uid"]] for i in range(n)]
    desc = (f"The journal of {key} ({doc['doc_id']}) was posted twice with identical accounts, amounts "
            f"and timestamp; the extra posting overstates the ledger by {amt} minor units.")
    how = (f"the ledger holds two identical postings for {key} while only one source document {doc['doc_id']} "
           f"exists (compare the anchor lines {rows[0]['_uid']} and {copies[0]['_uid']}); a legitimate "
           f"repeat would have its own document id.")
    return ctx.record(DUPLICATE, doc, [key], amt, ev, before, [snap(c) for c in copies], desc,
                      "repost", how, scenario)


# ------------------------------------------------- B. PARTIAL / MISSING ALLOCATION
def cand_allocation(ctx: MutCtx) -> list:
    return [d for d in ctx.free_docs() if "payment_id" in d and d["amount_minor"] > 0
            and d["invoice_refs"]]


def apply_allocation(ctx: MutCtx, doc: dict, variant=None, scenario=None) -> str:
    key, rng, world = World.doc_key(doc), ctx.rng, ctx.world
    rows = ctx.journals[key]
    ars = [r for r in rows if r["account"] == ACC_AR and r["invoice_id"]]
    refs = {r["invoice_id"] for r in doc["invoice_refs"]}
    others = [i["invoice_id"] for i in world.invoices
              if i["entity_id"] == doc["_entity"] and i["invoice_id"] not in refs]
    feasible = ["unallocated"] + (["partial"] if any(r["credit_minor"] >= 4 for r in ars) else []) \
        + (["wrong_invoice"] if others else [])
    if variant is None:
        preferred = [v for v in ctx.profile["alloc"] if v in feasible]
        variant = rng.choice(preferred or feasible)
    if variant not in feasible:
        raise GenerationError(f"allocation variant {variant} infeasible")
    if variant == "partial":
        ars = [r for r in ars if r["credit_minor"] >= 4]
    r = rng.choice(ars)
    inv_x, amt0 = r["invoice_id"], r["credit_minor"]
    before = [snap(r)]
    base = world.base
    if variant == "unallocated":
        r["account"], r["invoice_id"] = ACC_UNAPPLIED, None
        after, affected, amount, ev = [snap(r)], [key, inv_x], amt0, [[doc["doc_id"], r["_uid"]]]
        desc = (f"Allocation of {key} to {inv_x} was never applied: the line was parked in "
                f"unapplied receipts ({ACC_UNAPPLIED}), leaving the invoice open.")
        how = (f"{doc['doc_id']} lists {inv_x} in invoice_refs, but ledger line {r['_uid']} of {key} credits "
               f"account {ACC_UNAPPLIED} with no invoice_id instead of {ACC_AR} for {inv_x}.")
    elif variant == "partial":
        s = rng.randint(max(2, amt0 // 4), max(2, amt0 * 3 // 5))
        r["credit_minor"] -= s
        extra = ctx.clone(r)
        extra.update(account=ACC_UNAPPLIED, invoice_id=None, credit_minor=s)
        rows.append(extra)
        after, affected, amount = [snap(r), snap(extra)], [key, inv_x], s
        ev = [[doc["doc_id"], r["_uid"], extra["_uid"]], [doc["doc_id"], r["_uid"]],
              [doc["doc_id"], extra["_uid"]]]          # either changed line shows the gap
        desc = (f"Only part of the {key} allocation to {inv_x} was applied; {s} minor units "
                f"were left in unapplied receipts.")
        how = (f"the AR credit {r['_uid']} for {inv_x} is smaller than the converted allocation in {doc['doc_id']}; "
               f"the difference sits in unapplied line {extra['_uid']} ({ACC_UNAPPLIED}).")
    else:
        inv_y = rng.choice(others)
        r["invoice_id"] = inv_y
        after, affected, amount, ev = [snap(r)], sorted([key, inv_x, inv_y]), amt0, [[doc["doc_id"], r["_uid"]]]
        desc = f"Allocation of {key} was applied to {inv_y} instead of {inv_x} ({doc['doc_id']})."
        how = (f"{doc['doc_id']} allocates to {inv_x}, but ledger line {r['_uid']} carries invoice_id {inv_y}; "
               f"compare both invoices: {inv_x} is left open and {inv_y} is over-credited.")
    # cross-referencing: the documents of every affected invoice are part of the evidence
    inv_docs = [ctx.inv_doc[i] for i in affected if i in ctx.inv_doc]
    ev = [sorted(set(s) | set(inv_docs)) for s in ev]
    how += f" Invoice documents {', '.join(inv_docs)} show which invoices are involved."
    return ctx.record(ALLOCATION, doc, sorted(affected), amount, ev, before, after, desc, variant, how, scenario)


# --------------------------------------------------------------- C. FX RATE / DATE
def cand_fx(ctx: MutCtx) -> list:
    return [d for d in ctx.free_docs() if d["currency"] != ctx.world.base]


def apply_fx(ctx: MutCtx, doc: dict, variant=None, scenario=None) -> str:
    key, rng, world, prof = World.doc_key(doc), ctx.rng, ctx.world, ctx.profile
    rows, ccy, base = ctx.journals[key], doc["currency"], world.base
    inst = World.doc_instant(doc)
    rate, date_ok, fx_ok = world.rate_for(ccy, inst)
    anchor0 = max(rows[0]["debit_minor"], rows[0]["credit_minor"])
    thr = max(3, anchor0 // 400)                    # difference must clearly exceed the tolerance

    def anchor_of(r2):
        s = journal_spec(doc, base, r2)[0]
        return max(s[3], s[4])

    if variant is None:
        variant = rng.choice(prof["fx"])
    options = []
    if variant == "wrong_date":
        ks = [k for k in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6) if abs(k) <= prof["max_k"]]
        for k in ks:
            alt = world.rate_for(ccy, inst, k)
            if alt and abs(anchor_of(alt[0]) - anchor0) >= thr:
                options.append(("wrong_date", alt[0], f"used the rate of {alt[1]} instead of {date_ok}", alt[1]))
        if not options:
            for k in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6):      # neighbouring days too flat: widen
                alt = world.rate_for(ccy, inst, k)
                if alt and abs(anchor_of(alt[0]) - anchor0) >= thr:
                    options.append(("wrong_date", alt[0], f"used the rate of {alt[1]} instead of {date_ok}", alt[1]))
    if not options:
        if variant == "wrong_date" and scenario:
            raise GenerationError("no feasible wrong-date rate")
        variant = "wrong_rate"
        for _ in range(20):
            pct = Decimal(rng.randint(*prof["rate_bp"])) / Decimal(10000) * rng.choice((-1, 1))
            r2 = fmt_rate(Decimal(rate) * (1 + pct))
            if abs(anchor_of(r2) - anchor0) >= thr:
                options.append(("wrong_rate", r2, f"used rate {r2} instead of {rate} ({date_ok})", None))
                break
        if not options:
            raise GenerationError("no feasible wrong rate")
    var, r2, how_used, wrong_day = rng.choice(options)
    before = [snap(r) for r in rows]
    spec = journal_spec(doc, base, r2)
    for r, (_, _, _, dr, cr) in zip(rows, spec):
        r["debit_minor"], r["credit_minor"] = dr, cr
    booked = max(rows[0]["debit_minor"], rows[0]["credit_minor"])
    amount = abs(booked - anchor0)
    ev = [[doc["doc_id"], r["_uid"], fx_ok] for r in rows]       # any line of the journal
    desc = f"{key} ({ccy}) was converted with a wrong FX input: {how_used}."
    how = (f"policy prescribes the rate of {date_ok} ({fx_ok}); converting the document amount with it gives "
           f"{anchor0} but the ledger anchor line {rows[0]['_uid']} carries {booked}"
           f"{' (the booked amount matches the rate of ' + wrong_day + ')' if wrong_day else ''}.")
    return ctx.record(FX, doc, [key], amount, ev, before, [snap(r) for r in rows], desc, var, how, scenario)


# ------------------------------------------------------------------ D. PERIOD CUTOFF
def cutoff_dir(world: World, inst):
    """(boundary, sign) such that boundary + sign*shift crosses the period edge, or None."""
    s, e, four = world.start, world.end, timedelta(days=4)
    if s <= inst < e:
        if e - inst <= four:
            return e, 1
        return (s, -1) if inst - s <= four else None
    if e <= inst <= e + four:
        return e, -1
    if s - four <= inst < s:
        return s, 1
    return None


def cand_cutoff(ctx: MutCtx) -> list:
    return [d for d in ctx.free_docs() if cutoff_dir(ctx.world, World.doc_instant(d))]


def apply_cutoff(ctx: MutCtx, doc: dict, scenario=None) -> str:
    key, rng, world = World.doc_key(doc), ctx.rng, ctx.world
    rows = ctx.journals[key]
    inst = World.doc_instant(doc)
    boundary, sign = cutoff_dir(world, inst)
    new = boundary + sign * timedelta(minutes=rng.randint(*ctx.profile["shift_min"]))
    before = [snap(r) for r in rows]
    for r in rows:
        r["timestamp"] = iso(new.astimezone(world.tz))
    amount = max(sum(r["debit_minor"] for r in rows), sum(r["credit_minor"] for r in rows))
    side = lambda t: "before the period" if t < world.start else ("after the period" if t >= world.end else "inside the period")
    desc = (f"{key} ({doc['doc_id']}) is dated {iso(inst)} ({side(inst)}) but its journal was booked at "
            f"{rows[0]['timestamp']} ({side(new)}), in a different reporting period.")
    how = (f"compare the document timestamp with the booking timestamp of the ledger lines against the period "
           f"[{iso(world.start)}, {iso(world.end)}) (end exclusive, instants compared across offsets).")
    return ctx.record(CUTOFF, doc, sorted(r["_uid"] for r in rows), amount,
                      [[doc["doc_id"], r["_uid"]] for r in rows], before, [snap(r) for r in rows],
                      desc, "period_shift", how, scenario)


ENGINES = {DUPLICATE: (cand_duplicate, apply_duplicate), ALLOCATION: (cand_allocation, apply_allocation),
           FX: (cand_fx, apply_fx), CUTOFF: (cand_cutoff, apply_cutoff)}
assert set(ENGINES) == set(ISSUE_TYPES)


def apply_random(ctx: MutCtx, typ: str) -> bool:
    cands, apply = ENGINES[typ]
    pool = cands(ctx)
    if not pool:
        return False
    apply(ctx, ctx.rng.choice(pool))
    return True
