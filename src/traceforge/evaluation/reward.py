"""Official reward.

    F1 = 2*TP / (2*TP + FP + FN)            (F1 = 1 if there is no truth and no prediction)
    terminal reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)

Computed with exact fractions so equal inputs give bit-identical rewards."""
from __future__ import annotations

from fractions import Fraction

from ..constants import INVALID_REWARD, STEP_COST


def f1_fraction(tp: int, fp: int, fn: int) -> Fraction:
    denom = 2 * tp + fp + fn
    return Fraction(1) if denom == 0 else Fraction(2 * tp, denom)


def terminal_reward(f1: Fraction, inspection_steps: int) -> float:
    r = 2 * f1 - 1 - STEP_COST * inspection_steps
    return float(max(Fraction(-1), min(Fraction(1), r)))


def invalid_reward() -> float:
    return INVALID_REWARD
