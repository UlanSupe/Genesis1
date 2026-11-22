"""Reward-sensitivity study: the reconciliation auditor with one capability removed (see policies/ablations.py).

Every number comes from real evaluator runs on the stored datasets (eval mode, identical episodes for all variants)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..policies.ablations import ABLATIONS
from .episode_runner import run_split

ORDER = ("reconciliation", "no_explanation", "blind_duplicate", "blind_allocation", "blind_fx", "blind_cutoff",
         "budget_8", "over_accuser", "thin_evidence", "wrong_amount", "budget_4")
COLUMNS = ("split", "policy", "episodes", "tp", "fp", "fn", "mean_f1", "mean_reward", "mean_inspection_steps")
WHAT = {"reconciliation": "full auditor (reference)", "no_explanation": "empty explanation text (not scored)",
        "blind_duplicate": "never reports duplicates", "blind_allocation": "never reports allocation errors",
        "blind_fx": "never reports fx errors", "blind_cutoff": "never reports cut-off errors",
        "budget_8": "reads only 8 pages of data", "budget_4": "reads only 4 pages of data",
        "over_accuser": "correct + 3 unfounded accusations per episode", "thin_evidence": "cites only the source document",
        "wrong_amount": "claimed amount off by 5 minor units (tolerance 1)"}


def run_ablation(datasets="datasets", splits=("dev", "train"), *, isolated: bool = False) -> dict:
    rows = []
    for split in splits:
        for pol in ORDER:
            rs = run_split(datasets, split, pol, mode="eval", isolated=isolated)
            n = len(rs)
            rows.append({"split": split, "policy": pol, "episodes": n, "tp": sum(r["tp"] for r in rs),
                         "fp": sum(r["fp"] for r in rs), "fn": sum(r["fn"] for r in rs),
                         "mean_f1": round(sum(r["f1"] for r in rs) / n, 4), "mean_reward": round(sum(r["reward"] for r in rs) / n, 4),
                         "mean_inspection_steps": round(sum(r["inspection_steps"] for r in rs) / n, 2)})
    assert set(ORDER) - {"reconciliation"} == set(ABLATIONS)
    return {"mode": "eval", "splits": list(splits), "rows": rows}


def write_ablation(result: dict, out_dir, name: str = "ablation") -> tuple:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(result["rows"])
    return out / f"{name}.json", out / f"{name}.csv"


def render(result: dict) -> str:
    lines = []
    for split in result["splits"]:
        lines += [f"[{split}]", f"{'variant':<16} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6} {'reward':>7} {'steps':>6}  what changed"]
        for r in (r for r in result["rows"] if r["split"] == split):
            lines.append(f"{r['policy']:<16} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['mean_f1']:>6.3f} "
                         f"{r['mean_reward']:>7.3f} {r['mean_inspection_steps']:>6.1f}  {WHAT[r['policy']]}")
        lines.append("")
    return "\n".join(lines)
