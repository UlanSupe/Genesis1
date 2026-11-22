"""Paired (metamorphic) evaluation: does the agent find the ERROR, or memorise ids / row positions?

For every pair (original, transformed twin with identical economics) the same policy runs under
identical conditions. Reported per policy:
  consistent_rate    share of pairs with identical TP/FP/FN (the answer survived the transformation)
  mean_abs_delta_f1  mean |F1(original) - F1(twin)|        (0 = fully robust)
  robustness         1 - mean_abs_delta_f1
An id-/position-memorising agent scores well on the originals and collapses on the twins."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from ..environment.audit_env import AuditEnv
from .episode_runner import AGENT_NAMES, make_agent, run_episode, split_dirs

COLUMNS = ("split", "pair", "original", "policy", "orig_tp", "orig_fp", "orig_fn", "orig_f1", "orig_reward", "orig_steps",
           "twin_tp", "twin_fp", "twin_fn", "twin_f1", "twin_reward", "twin_steps", "consistent", "abs_delta_f1")


def run_paired(source_root, pairs_root, split: str, policies=AGENT_NAMES, *, mode: str = "eval", isolated: bool = False,
               policy_seed: int = 0) -> dict:
    pdir = Path(pairs_root) / split
    originals = json.loads((pdir / "index.json").read_text(encoding="utf-8"))["originals"]
    src = {d.name: d for d in split_dirs(source_root, split)}
    rows = []
    for twin_dir in split_dirs(pairs_root, split):
        orig_name = originals[twin_dir.name]
        for pol in policies:
            res = {}
            for tag, d in (("orig", src[orig_name]), ("twin", twin_dir)):
                res[tag] = run_episode(AuditEnv(mode), make_agent(pol, policy_seed, isolated and pol != "oracle"),
                                       config={"episode_dir": str(d)}, label=pol)
            o, t = res["orig"], res["twin"]
            rows.append({"split": split, "pair": twin_dir.name, "original": orig_name, "policy": pol,
                         "orig_tp": o["tp"], "orig_fp": o["fp"], "orig_fn": o["fn"], "orig_f1": o["f1"], "orig_reward": o["reward"],
                         "orig_steps": o["inspection_steps"], "twin_tp": t["tp"], "twin_fp": t["fp"], "twin_fn": t["fn"],
                         "twin_f1": t["f1"], "twin_reward": t["reward"], "twin_steps": t["inspection_steps"],
                         "consistent": (o["tp"], o["fp"], o["fn"]) == (t["tp"], t["fp"], t["fn"]),
                         "abs_delta_f1": round(abs(o["f1"] - t["f1"]), 6)})
    return {"mode": mode, "split": split, "rows": rows, "summary": summarize(rows)}


def summarize(rows: list) -> list:
    out = []
    for pol in sorted({r["policy"] for r in rows}):
        s = [r for r in rows if r["policy"] == pol]
        n = len(s)
        delta = sum(r["abs_delta_f1"] for r in s) / n
        out.append({"policy": pol, "pairs": n, "consistent_rate": round(sum(r["consistent"] for r in s) / n, 4),
                    "mean_f1_original": round(sum(r["orig_f1"] for r in s) / n, 4),
                    "mean_f1_twin": round(sum(r["twin_f1"] for r in s) / n, 4),
                    "mean_abs_delta_f1": round(delta, 4), "robustness": round(1 - delta, 4),
                    "mean_reward_original": round(sum(r["orig_reward"] for r in s) / n, 4),
                    "mean_reward_twin": round(sum(r["twin_reward"] for r in s) / n, 4)})
    return out


def write_paired(result: dict, out_dir: str | Path, name: str = "paired_eval") -> tuple:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(result["rows"])
    return out / f"{name}.json", out / f"{name}.csv"


def render(result: dict) -> str:
    lines = [f"{'policy':<15} {'pairs':>5} {'consistent':>10} {'F1 orig':>8} {'F1 twin':>8} {'|dF1|':>7} {'robustness':>10}"]
    for s in result["summary"]:
        lines.append(f"{s['policy']:<15} {s['pairs']:>5} {s['consistent_rate']:>10.3f} {s['mean_f1_original']:>8.3f} "
                     f"{s['mean_f1_twin']:>8.3f} {s['mean_abs_delta_f1']:>7.3f} {s['robustness']:>10.3f}")
    return "\n".join(lines)
