"""Synthetic company and counterparties."""
from __future__ import annotations

import random

COMPANY_TZ_CHOICES = (0, 60, 120, 180, 330)
TZ_POOL = (-480, -420, -360, -300, -240, 0, 60, 120, 180, 330, 480, 540, 600, 720)
_A = ("Aurora", "Borealis", "Cobalt", "Delta", "Ember", "Falcon", "Granite", "Helix",
      "Indigo", "Juniper", "Kestrel", "Lumen", "Meridian", "Nimbus", "Orion", "Prism")
_B = ("Trading", "Logistics", "Systems", "Foods", "Energy", "Textiles", "Labs", "Partners")


def make_company(rng: random.Random) -> dict:
    return {"name": f"{rng.choice(_A)} {rng.choice(_B)} Ltd"}


def _pick_tz(rng, company_tz: int, side: int) -> int:
    """side=-1: >=6h behind the company, +1: >=6h ahead, 0: anything."""
    pool = [t for t in TZ_POOL if side == 0 or (t - company_tz) * side >= 360]
    return rng.choice(pool or list(TZ_POOL))


def make_entities(rng: random.Random, n: int, base: str, foreign: list, company_tz: int) -> list:
    """Entities 0/1 are foreign counterparties far behind/ahead of the company clock
    (timezone decoys); the remainder mixes base-currency and foreign customers."""
    out = []
    for i in range(n):
        if foreign and i < 2:
            ccy, tz = foreign[i % len(foreign)], _pick_tz(rng, company_tz, -1 if i == 0 else 1)
        elif foreign and rng.random() > 0.35:
            ccy, tz = rng.choice(foreign), _pick_tz(rng, company_tz, 0)
        else:
            ccy, tz = base, _pick_tz(rng, company_tz, 0) if i < 2 else company_tz
        out.append({"entity_id": f"ENT-{i + 1:03d}", "currency": ccy, "tz_minutes": tz,
                    "name": f"Counterparty {rng.choice(_A)} {i + 1}"})
    return out
