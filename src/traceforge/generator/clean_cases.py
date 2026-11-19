"""Legitimate-but-suspicious situations (decoys). Tagged "decoy": never mutated."""
from __future__ import annotations

import random
from datetime import timedelta

from .documents import make_invoice, make_payment, rand_lines
from .financial_world import World


def _rand_time(world: World, rng, lo_days: int, hi_days: int):
    lo = world.start + timedelta(days=lo_days)
    span = (hi_days - lo_days) * 1440
    return lo + timedelta(minutes=rng.randrange(span))


def tz_boundary(world: World, rng: random.Random, n: int) -> None:
    """Full payment landing within ~1h of a period edge by the company clock, from a
    counterparty whose local calendar date differs (naive date handling mislabels it)."""
    far = [e for e in world.entities if abs(e["tz_minutes"] - world.tz_minutes) >= 240]
    far.sort(key=lambda e: e["currency"] == world.base)        # foreign first
    for i in range(n):
        if not far:
            return
        ent = far[i % len(far)]
        behind = ent["tz_minutes"] < world.tz_minutes
        edge = world.end if i % 2 == 0 else world.start
        when = edge + timedelta(minutes=rng.randint(30, 90) * (1 if behind else -1))
        inv = make_invoice(world, ent, when - timedelta(days=rng.randint(6, 14),
                                                        minutes=rng.randint(0, 600)),
                           "decoy", rng=rng)
        make_payment(world, ent, when, [(inv, inv["total_minor"])], "decoy")


def repeat_amount(world: World, rng: random.Random, n: int) -> None:
    """Two genuine invoices to one counterparty with identical lines and totals."""
    for _ in range(n):
        ent = rng.choice(world.entities)
        lines = rand_lines(world, rng, ent["currency"])
        t = _rand_time(world, rng, -8, 20)
        a = make_invoice(world, ent, t, "decoy", [dict(l) for l in lines])
        b = make_invoice(world, ent, t + timedelta(days=rng.randint(2, 6)), "decoy",
                         [dict(l) for l in lines])
        for inv in (a, b):
            make_payment(world, ent, World.doc_instant(inv) + timedelta(days=rng.randint(5, 14),
                                                                        minutes=rng.randint(0, 900)),
                         [(inv, inv["total_minor"])], "decoy")


def same_amount_other_entity(world: World, rng: random.Random, n: int) -> None:
    for _ in range(n):
        ent_a = rng.choice(world.entities)
        peers = [e for e in world.entities if e["currency"] == ent_a["currency"] and e is not ent_a]
        if not peers:
            continue
        ent_b = rng.choice(peers)
        lines = rand_lines(world, rng, ent_a["currency"])
        t = _rand_time(world, rng, -8, 20)
        for ent, dt in ((ent_a, t), (ent_b, t + timedelta(hours=rng.randint(0, 8)))):
            inv = make_invoice(world, ent, dt, "decoy", [dict(l) for l in lines])
            make_payment(world, ent, dt + timedelta(days=rng.randint(4, 12)),
                         [(inv, inv["total_minor"])], "decoy")


def recurring_series(world: World, rng: random.Random, n: int) -> None:
    """Subscription-like series: 3-4 invoices to one counterparty with the same lines and total at a
    fixed cadence, each paid in full. Repeated amounts, repeated dates of month - and all legitimate."""
    for _ in range(n):
        ent = rng.choice(world.entities)
        lines = rand_lines(world, rng, ent["currency"])
        cadence = rng.choice((5, 7, 10))            # 3 invoices always fit inside the window
        t = _rand_time(world, rng, -6, 8)
        for k in range(rng.randint(3, 4)):
            when = t + timedelta(days=cadence * k)
            if when > world.end + timedelta(days=3):
                break
            inv = make_invoice(world, ent, when, "decoy", [dict(l) for l in lines])
            make_payment(world, ent, when + timedelta(days=rng.randint(5, 12), minutes=rng.randint(0, 900)),
                         [(inv, inv["total_minor"])], "decoy")


def split_payments(world: World, rng: random.Random, n: int) -> None:
    """One payment settling 2-3 invoices of the same counterparty (several invoice_refs)."""
    for _ in range(n):
        ent = rng.choice(world.entities)
        k = rng.randint(2, 3)
        t0 = _rand_time(world, rng, -8, 18)
        invs = [make_invoice(world, ent, t0 + timedelta(days=2 * j, minutes=rng.randint(0, 600)), "decoy", rng=rng)
                for j in range(k)]
        when = max(World.doc_instant(i) for i in invs) + timedelta(days=rng.randint(4, 12), minutes=rng.randint(0, 900))
        make_payment(world, ent, when, [(i, i["total_minor"]) for i in invs], "decoy")


def refunds(world: World, rng: random.Random, n: int) -> None:
    """Payment followed by an equal negative payment (looks like a mirrored duplicate)."""
    pool = [i for i in world.invoices if i["_tag"] == "regular" and len(i["_pays"]) == 1
            and i["_paid"] == i["total_minor"] and len(i["_pays"][0]["invoice_refs"]) == 1]
    rng.shuffle(pool)
    done = 0
    for inv in pool:
        if done >= n:
            break
        pay = inv["_pays"][0]
        t = World.doc_instant(pay) + timedelta(days=rng.randint(3, 14), minutes=rng.randint(0, 900))
        if t > world.window_end:
            continue
        amt = inv["total_minor"] if rng.random() < 0.6 else inv["total_minor"] // 2
        make_payment(world, world.entity(inv["entity_id"]), t, [(inv, -amt)], "decoy")
        done += 1


def overpayments(world: World, rng: random.Random, n: int) -> None:
    """Customer pays slightly more than invoiced; the remainder is legitimately unapplied."""
    for _ in range(n):
        ent = rng.choice(world.entities)
        inv = make_invoice(world, ent, _rand_time(world, rng, -8, 24), "decoy", rng=rng)
        extra = max(1, inv["total_minor"] * rng.randint(1, 3) // 100)
        make_payment(world, ent, World.doc_instant(inv) + timedelta(days=rng.randint(4, 12)),
                     [(inv, inv["total_minor"])], "decoy", amount_minor=inv["total_minor"] + extra)


def add_decoys(world: World, rng: random.Random, P: dict) -> None:
    """Decoy volumes vary around the difficulty level (never below 1, so every kind of
    legitimate-but-suspicious situation exists in every episode)."""
    n = {k: max(1, P[k] + rng.randint(-1, 2)) for k in ("tz", "repeat", "sameamt", "refunds", "over", "recurring", "split")}
    tz_boundary(world, rng, n["tz"])
    repeat_amount(world, rng, n["repeat"])
    same_amount_other_entity(world, rng, n["sameamt"])
    refunds(world, rng, n["refunds"])
    overpayments(world, rng, n["over"])
    recurring_series(world, rng, n["recurring"])
    split_payments(world, rng, n["split"])
