"""Invoice / payment factories and the regular (non-decoy) payment behaviour."""
from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime, timedelta

from ..money import decimals
from .financial_world import World, iso, tz_of

DESCRIPTIONS = ("Consulting services", "Software subscription", "Hardware supply",
                "Logistics fee", "Maintenance contract", "Training package",
                "Licence renewal", "Cloud hosting", "Audit support", "Marketing campaign")
SCALE = {"JPY": 140}          # typical magnitude of amounts in whole currency units


def rand_lines(world: World, rng: random.Random, ccy: str) -> list:
    d, lines = decimals(ccy), []
    for _ in range(rng.randint(1, 3)):
        major = max(1, rng.randint(150, 6000) * SCALE.get(ccy, 1) * world.scale_pct // 100)
        frac = rng.randint(0, 10 ** d - 1) if d else 0
        lines.append({"description": rng.choice(DESCRIPTIONS), "amount_minor": major * 10 ** d + frac})
    return lines


def make_invoice(world: World, entity: dict, issued: datetime, tag="regular", lines=None,
                 rng: random.Random | None = None) -> dict:
    lines = lines or rand_lines(world, rng, entity["currency"])
    key = world.key()
    inv = {"_key": key, "_tag": tag, "_entity": entity["entity_id"], "_paid": 0, "_pays": [],
           "doc_id": None, "invoice_id": key, "entity_id": entity["entity_id"],
           "issued_at": iso(issued.astimezone(world.tz)),
           "due_at": iso((issued + timedelta(days=30)).astimezone(world.tz)),
           "currency": entity["currency"], "total_minor": sum(l["amount_minor"] for l in lines),
           "line_items": lines}
    world.invoices.append(inv)
    return inv


def make_payment(world: World, entity: dict, when: datetime, allocs: list, tag="regular",
                 amount_minor: int | None = None) -> dict:
    """allocs: [(invoice_doc, allocated_minor)]; amount defaults to the allocation sum
    (a larger amount leaves a legitimate unapplied remainder)."""
    key = world.key()
    amount = sum(a for _, a in allocs) if amount_minor is None else amount_minor
    pay = {"_key": key, "_tag": tag, "_entity": entity["entity_id"], "doc_id": None,
           "payment_id": key, "timestamp": iso(when.astimezone(tz_of(entity["tz_minutes"]))),
           "currency": entity["currency"], "amount_minor": amount,
           "invoice_refs": [{"invoice_id": inv["_key"], "allocated_minor": a} for inv, a in allocs]}
    for inv, a in allocs:
        inv["_paid"] += a
        inv["_pays"].append(pay)
    world.payments.append(pay)
    return pay


def _pay_time(world: World, rng, after: datetime, lo=3, hi=38):
    t = after + timedelta(days=rng.randint(lo, hi), minutes=rng.randint(0, 1439))
    return t if t <= world.window_end else None


def plan_payments(world: World, rng: random.Random, invoices: list) -> None:
    """Realistic regular behaviour: full, two-part, partial, split and unpaid."""
    splits = defaultdict(list)
    for inv in invoices:
        ent = world.entity(inv["entity_id"])
        issued, total = World.doc_instant(inv), inv["total_minor"]
        r = rng.random() * 100
        full, two, part, split = world.mix
        if r < full:
            t = _pay_time(world, rng, issued)
            if t:
                make_payment(world, ent, t, [(inv, total)])
        elif r < two:
            first = total * rng.randint(30, 70) // 100
            t1 = _pay_time(world, rng, issued)
            t2 = _pay_time(world, rng, t1, 3, 12) if t1 else None
            if t1:
                make_payment(world, ent, t1, [(inv, first)])
            if t2:
                make_payment(world, ent, t2, [(inv, total - first)])
        elif r < part:
            t = _pay_time(world, rng, issued)
            if t:
                make_payment(world, ent, t, [(inv, total * rng.randint(30, 80) // 100)])
        elif r < split:
            splits[inv["entity_id"]].append(inv)
    for eid, group in splits.items():
        ent = world.entity(eid)
        for i in range(0, len(group), 3):
            chunk = group[i:i + 3]
            t = _pay_time(world, rng, max(World.doc_instant(c) for c in chunk))
            if t:
                make_payment(world, ent, t, [(c, c["total_minor"]) for c in chunk])


def add_regular(world: World, rng: random.Random, n: int) -> list:
    out = []
    lo, hi = world.start - timedelta(days=20), world.end + timedelta(days=3)
    span = int((hi - lo).total_seconds() // 60)
    for _ in range(n):
        ent = rng.choice(world.entities)
        out.append(make_invoice(world, ent, lo + timedelta(minutes=rng.randrange(span)), rng=rng))
    plan_payments(world, rng, out)
    return out


def add_boundary_regulars(world: World, rng: random.Random, n_each: int) -> None:
    """Ordinary documents near period edges (candidates for cut-off errors)."""
    windows = [(world.end - timedelta(hours=72), world.end - timedelta(hours=1)),
               (world.start + timedelta(hours=1), world.start + timedelta(hours=72)),
               (world.start - timedelta(hours=72), world.start - timedelta(hours=1)),
               (world.end + timedelta(hours=1), world.end + timedelta(hours=60))]
    out = []
    for lo, hi in windows:
        span = int((hi - lo).total_seconds() // 60)
        for _ in range(n_each):
            out.append(make_invoice(world, rng.choice(world.entities),
                                    lo + timedelta(minutes=rng.randrange(span)), rng=rng))
    plan_payments(world, rng, out)
