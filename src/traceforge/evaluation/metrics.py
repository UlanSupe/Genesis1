"""Metrics. Conventions for empty sets: precision = 1 if there are no predictions and
no truth, else 0 when there are no predictions; recall = 1 when there is no truth."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .reward import f1_fraction


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    reward: float
    inspection_steps: int
    elapsed_time: float
    valid: bool = True


def precision_recall(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r


def make_metrics(tp, fp, fn, reward, inspection_steps, elapsed, valid=True) -> Metrics:
    p, r = precision_recall(tp, fp, fn)
    return Metrics(tp, fp, fn, p, r, float(f1_fraction(tp, fp, fn)), reward,
                   inspection_steps, round(elapsed, 6), valid)


def summarize(results: list) -> dict:
    """Mean of numeric metrics over a list of Metrics/dicts."""
    rows = [asdict(r) if isinstance(r, Metrics) else r for r in results]
    if not rows:
        return {"episodes": 0}
    keys = ("precision", "recall", "f1", "reward", "inspection_steps", "elapsed_time")
    out = {"episodes": len(rows), **{f"mean_{k}": sum(r[k] for r in rows) / len(rows) for k in keys}}
    out.update({k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn")})
    return out
