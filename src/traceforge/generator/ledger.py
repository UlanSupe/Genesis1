"""Double-entry journals. Ledger amounts are always in the BASE currency; the
`currency` column records the currency of the source document.

Posting rules (rounding ROUND_HALF_UP per ledger line, rate date = company-timezone
calendar date of the document timestamp):

  invoice          Dr 1200 AR  total          | Cr 4000 Revenue  total
  payment (>0)     Dr 1000 Cash (sum of lines)| Cr 1200 AR per invoice_ref
                                              | Cr 2300 Unapplied (amount - allocated)
  refund  (<0)     Cr 1000 Cash (sum of lines)| Dr 1200 AR per invoice_ref

Row 0 of every posting is its anchor (AR debit of an invoice / the cash line of a payment).
"""
from __future__ import annotations

import random

from ..constants import ACC_AR, ACC_CASH, ACC_REVENUE, ACC_UNAPPLIED
from ..money import convert_minor
from .financial_world import World, iso


def journal_spec(doc: dict, base: str, rate: str) -> list:
    """[(account, invoice_id, payment_id, debit, credit)] in base-currency minor units."""
    ccy = doc["currency"]
    conv = lambda v: convert_minor(v, ccy, base, rate)
    if "total_minor" in doc:
        amt, iid = conv(doc["total_minor"]), doc["invoice_id"]
        return [(ACC_AR, iid, None, amt, 0), (ACC_REVENUE, iid, None, 0, amt)]
    pid, refunds = doc["payment_id"], doc["amount_minor"] < 0
    parts = [(r["invoice_id"], conv(r["allocated_minor"])) for r in doc["invoice_refs"]]
    rem = doc["amount_minor"] - sum(r["allocated_minor"] for r in doc["invoice_refs"])
    if rem:
        parts.append((None, conv(rem)))
    cash = sum(abs(v) for _, v in parts) if not refunds else sum(-v for _, v in parts)
    if not refunds:
        spec = [(ACC_CASH, None, pid, cash, 0)]
        spec += [(ACC_AR if i else ACC_UNAPPLIED, i, pid, 0, v) for i, v in parts]
    else:
        spec = [(ACC_CASH, None, pid, 0, cash)]
        spec += [(ACC_AR, i, pid, -v, 0) for i, v in parts]
    return spec


def posting_rows(world: World, rng: random.Random, doc: dict) -> list:
    inst = World.doc_instant(doc)
    rate = world.rate_for(doc["currency"], inst)[0]
    ts = iso(inst.astimezone(world.tz))
    rows = []
    for acc, iid, pid, dr, cr in journal_spec(doc, world.base, rate):
        rows.append({"_uid": world.uid(), "_tb": rng.random(), "row_id": None, "timestamp": ts,
                     "account": acc, "debit_minor": dr, "credit_minor": cr,
                     "currency": doc["currency"], "entity_id": doc["_entity"],
                     "invoice_id": iid, "payment_id": pid})
    return rows


def build_journals(world: World, rng: random.Random) -> dict:
    """{journal key (invoice_id or payment_id): [row dicts]} for the clean history."""
    return {World.doc_key(d): posting_rows(world, rng, d) for d in world.invoices + world.payments}
