"""Integer minor-unit money and exact FX conversion (no floats anywhere)."""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, localcontext

from .constants import CURRENCY_DECIMALS


def decimals(ccy: str) -> int:
    return CURRENCY_DECIMALS[ccy]


def convert_minor(amount_minor: int, src: str, dst: str, rate: str) -> int:
    """Convert integer minor units src->dst. rate = dst units per 1 src unit
    (decimal string). Rounding: ROUND_HALF_UP on |amount|, sign restored."""
    if src == dst:
        return amount_minor
    with localcontext() as ctx:
        ctx.prec = 60
        major = Decimal(abs(amount_minor)).scaleb(-decimals(src)) * Decimal(rate)
        out = int(major.scaleb(decimals(dst)).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return -out if amount_minor < 0 else out


def fmt_rate(value: Decimal) -> str:
    """Canonical rate string: 8 decimal places, plain notation."""
    return format(value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP), "f")


def fmt_major(minor: int, ccy: str) -> str:
    d = decimals(ccy)
    sign = "-" if minor < 0 else ""
    s = str(abs(minor)).rjust(d + 1, "0")
    return f"{sign}{s[:-d]}.{s[-d:]}" if d else f"{sign}{s}"
