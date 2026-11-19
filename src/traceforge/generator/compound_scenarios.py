"""Two compound scenarios. Each yields two SEPARATE truth issues (own evidence,
own amount) on different journals that share a counterparty/invoice story."""
from __future__ import annotations

import random
from datetime import timedelta

from .documents import make_invoice, make_payment, rand_lines
from .financial_world import World
from .mutations import MutCtx, apply_allocation, apply_cutoff, apply_duplicate, apply_fx


def add_scaffold(name: str, world: World, rng: random.Random) -> None:
    tag = f"compound:{name}"
    if name == "cross_border_partial":
        # foreign invoice settled by two partial payments
        ent = rng.choice([e for e in world.entities if e["currency"] != world.base])
        issued = world.start + timedelta(days=rng.randint(0, 10), minutes=rng.randint(0, 1439))
        inv = make_invoice(world, ent, issued, tag, rng=rng)
        t = inv["total_minor"]       # shares are random: no constant "scenario fingerprint" for agents to learn
        p1 = make_payment(world, ent, issued + timedelta(days=rng.randint(3, 6), minutes=rng.randint(0, 900)),
                          [(inv, t * rng.randint(20, 45) // 100)], tag)
        p2 = make_payment(world, ent, issued + timedelta(days=rng.randint(8, 12), minutes=rng.randint(0, 900)),
                          [(inv, t * rng.randint(20, 45) // 100)], tag)
        roles = {"invoice": inv, "alloc_payment": p1, "fx_payment": p2}
    else:
        # two lookalike invoices (same entity, same lines) just before period end
        ent = rng.choice(world.entities)
        lines = rand_lines(world, rng, ent["currency"])
        t1 = world.end - timedelta(hours=rng.randint(30, 70), minutes=rng.randint(0, 59))
        a = make_invoice(world, ent, t1, tag, [dict(l) for l in lines])
        b = make_invoice(world, ent, t1 + timedelta(hours=rng.randint(5, 24)), tag, [dict(l) for l in lines])
        roles = {"duplicate_doc": a, "cutoff_doc": b}
    world._roles_by_key[name] = {role: d["_key"] for role, d in roles.items()}


def apply_scenario(ctx: MutCtx, name: str) -> None:
    roles, docs = ctx.world.roles[name], ctx.docs
    if name == "cross_border_partial":
        v = ctx.rng.choice(["partial", "unallocated"])
        ids = [apply_allocation(ctx, docs[roles["alloc_payment"]], v, name),
               apply_fx(ctx, docs[roles["fx_payment"]], "wrong_date", name)]
        text = "Foreign-currency invoice settled in two partial payments: one payment " \
               "was mis-allocated, the other was converted with the wrong date's rate."
    else:
        ids = [apply_duplicate(ctx, docs[roles["duplicate_doc"]], name),
               apply_cutoff(ctx, docs[roles["cutoff_doc"]], name)]
        text = "Two lookalike invoices at period end: one journal is posted twice, " \
               "the other is booked into the next period."
    ctx.scenarios.append({"scenario": name, "roles": dict(roles), "issue_ids": ids,
                          "description": text})
