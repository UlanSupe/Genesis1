"""World container, FX table and top-level clean-history assembly."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from ..config import GeneratorConfig
from ..exceptions import GenerationError
from ..money import fmt_rate
from ..schemas import fx_id
from .company import COMPANY_TZ_CHOICES, make_company, make_entities

USD_PER = {"USD": Decimal("1"), "EUR": Decimal("1.08"), "GBP": Decimal("1.27"),
           "CHF": Decimal("1.13"), "JPY": Decimal("0.0067"), "SEK": Decimal("0.095"),
           "CAD": Decimal("0.73"), "KWD": Decimal("3.25")}
FOREIGN_POOL = ("USD", "GBP", "CHF", "JPY", "SEK", "CAD", "KWD")
N_FOREIGN = {"easy": 2, "medium": 3, "hard": 4}
# per-difficulty decoy volumes
# mix = cumulative % thresholds for regular invoices: full | two-part | single partial | split
# (the remainder stays unpaid). More partial/split payments = more links to follow.
DIFF = {
    "easy":   dict(tz=2, repeat=1, sameamt=1, refunds=1, over=1, recurring=1, split=1, boundary=2,
                   n_inv=(30, 45), n_ent=(5, 7), mix=(55, 65, 75, 85)),
    "medium": dict(tz=4, repeat=2, sameamt=2, refunds=2, over=2, recurring=1, split=1, boundary=3,
                   n_inv=(38, 62), n_ent=(6, 10), mix=(45, 60, 72, 85)),
    "hard":   dict(tz=6, repeat=3, sameamt=3, refunds=4, over=3, recurring=2, split=2, boundary=4,
                   n_inv=(36, 46), n_ent=(8, 14), mix=(30, 48, 65, 85)),
}
FX_LEAD_DAYS, FX_TAIL_DAYS = 30, 15


def tz_of(minutes: int) -> timezone:
    return timezone(timedelta(minutes=minutes))


def iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def tz_string(minutes: int) -> str:
    sign = "-" if minutes < 0 else "+"
    return f"{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


@dataclass
class World:
    cfg: GeneratorConfig
    base: str
    tz_minutes: int
    start: datetime
    end: datetime                      # exclusive
    company: dict
    entities: list
    fx_rows: list
    fx_map: dict                       # (date, ccy) -> rate string
    invoices: list = field(default_factory=list)
    payments: list = field(default_factory=list)
    roles: dict = field(default_factory=dict)      # scenario -> {role: business id}
    scale_pct: int = 100                           # per-episode amount scale
    mix: tuple = (45, 60, 72, 85)
    _roles_by_key: dict = field(default_factory=dict)
    _k: int = 0
    _u: int = 0

    @property
    def tz(self) -> timezone:
        return tz_of(self.tz_minutes)

    @property
    def window_end(self) -> datetime:
        return self.end + timedelta(days=8)

    def key(self) -> str:
        self._k += 1
        return f"K{self._k:05d}"

    def uid(self) -> str:
        self._u += 1
        return f"@u{self._u:06d}"

    def entity(self, entity_id: str) -> dict:
        return next(e for e in self.entities if e["entity_id"] == entity_id)

    def rate_for(self, ccy: str, instant: datetime, shift_days: int = 0):
        """(rate, date, fx_id) per policy: calendar date of the document timestamp in
        the company timezone. Returns None when the table has no such date."""
        d = (instant.astimezone(self.tz).date() + timedelta(days=shift_days)).isoformat()
        if ccy == self.base:
            return "1", d, None
        rate = self.fx_map.get((d, ccy))
        return None if rate is None else (rate, d, f"FX-{d}-{ccy}-{self.base}")

    @staticmethod
    def doc_instant(doc: dict) -> datetime:
        return datetime.fromisoformat(doc["issued_at"] if "total_minor" in doc else doc["timestamp"])

    @staticmethod
    def doc_key(doc: dict) -> str:
        return doc["invoice_id"] if "total_minor" in doc else doc["payment_id"]


def make_fx(rng: random.Random, base: str, foreign: list, d0: date, d1: date):
    rows, fmap = [], {}
    for ccy in foreign:
        value = (USD_PER[ccy] / USD_PER[base])
        day = d0
        while day <= d1:
            # integer basis-point random walk: bit-for-bit reproducible across platforms
            value = value * (Decimal(10000 + rng.randint(-70, 70)) / Decimal(10000))
            rate = fmt_rate(value)
            rows.append({"date": day.isoformat(), "source_currency": ccy,
                         "target_currency": base, "rate": rate})
            fmap[(day.isoformat(), ccy)] = rate
            day += timedelta(days=1)
    return rows, fmap


def count_rows(world: World) -> int:
    n = 2 * len(world.invoices)
    for p in world.payments:
        alloc = sum(r["allocated_minor"] for r in p["invoice_refs"])
        n += 1 + len(p["invoice_refs"]) + (1 if p["amount_minor"] > alloc else 0)
    return n


def build_world(rng: random.Random, cfg: GeneratorConfig, scenarios=()) -> World:
    # local imports keep the module graph acyclic
    from . import clean_cases, compound_scenarios, documents

    base = cfg.base_currency
    if cfg.foreign_currencies is not None:
        foreign = list(cfg.foreign_currencies)
    else:
        pool = [c for c in FOREIGN_POOL if c != base]
        foreign = rng.sample(pool, N_FOREIGN[cfg.difficulty])
        if cfg.difficulty == "hard" and "JPY" not in foreign and "JPY" != base:
            foreign[-1] = "JPY"            # exercise zero-decimal rounding
    if scenarios and "cross_border_partial" in scenarios and not foreign:
        raise GenerationError("cross_border_partial needs a foreign currency")
    tz_minutes = cfg.timezone_minutes if cfg.timezone_minutes is not None \
        else rng.choice(COMPANY_TZ_CHOICES)
    tz = tz_of(tz_minutes)
    if cfg.period_start:
        y, m = int(cfg.period_start[:4]), int(cfg.period_start[5:7])
    else:
        y, m = rng.randint(2023, 2026), rng.randint(1, 12)
    start = datetime(y, m, 1, tzinfo=tz)
    end = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=tz)
    fx_rows, fx_map = make_fx(rng, base, foreign,
                              (start - timedelta(days=FX_LEAD_DAYS)).date(),
                              (end + timedelta(days=FX_TAIL_DAYS)).date())
    P = DIFF[cfg.difficulty]
    n_inv = cfg.n_invoices or rng.randint(*P["n_inv"])
    n_ent = cfg.n_entities or rng.randint(*P["n_ent"])
    world = World(cfg, base, tz_minutes, start, end, make_company(rng),
                  make_entities(rng, n_ent, base, foreign, tz_minutes), fx_rows, fx_map,
                  scale_pct=rng.choice((40, 100, 250)), mix=P["mix"])

    documents.add_regular(world, rng, n_inv)
    documents.add_boundary_regulars(world, rng, P["boundary"])
    clean_cases.add_decoys(world, rng, P)
    for name in scenarios:
        compound_scenarios.add_scaffold(name, world, rng)
    while count_rows(world) < cfg.min_ledger_rows:      # top up to the published minimum
        documents.add_regular(world, rng, 4)
    _finalize_ids(world)
    return world


def _finalize_ids(world: World) -> None:
    """Assign final DOC/INV/PAY ids in chronological order and rewire references."""
    docs = [(World.doc_instant(d), d["_key"], d) for d in world.invoices + world.payments]
    docs.sort(key=lambda t: (t[0], t[1]))
    inv_map, ni, np_ = {}, 0, 0
    for n, (_, key, d) in enumerate(docs, 1):
        d["doc_id"] = f"DOC-{n:04d}"
        if "total_minor" in d:
            ni += 1
            inv_map[key] = d["invoice_id"] = f"INV-{ni:04d}"
        else:
            np_ += 1
            d["payment_id"] = f"PAY-{np_:04d}"
    pay_map = {d["_key"]: d["payment_id"] for d in world.payments}
    for p in world.payments:
        for r in p["invoice_refs"]:
            r["invoice_id"] = inv_map[r["invoice_id"]]
    world.invoices.sort(key=lambda d: d["invoice_id"])
    world.payments.sort(key=lambda d: d["payment_id"])
    for scen, roles in world._roles_by_key.items():
        world.roles[scen] = {role: inv_map.get(k) or pay_map[k] for role, k in roles.items()}
