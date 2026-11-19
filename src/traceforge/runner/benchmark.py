"""Benchmark runner: every policy runs on IDENTICAL episodes; results go to JSON and CSV.

Nothing is tuned or hard-coded, and no policy is assumed to win: the table is whatever
the run produced. `elapsed` is wall time of the whole episode including agent compute."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from ..config import PRESETS
from ..environment.audit_env import AuditEnv
from .episode_runner import AGENT_NAMES, make_agent, run_episode, split_dirs

COLUMNS = ("suite", "episode", "level", "policy", "issues", "tp", "fp", "fn", "precision", "recall", "f1", "reward",
           "inspection_steps", "steps", "findings", "runtime_s", "env_elapsed_s", "agent_error")


def suite_episodes(suite: str, datasets: str | Path, seeds: int) -> list:
    """[(label, config_for_env_reset, seed)] - the same list is replayed for every policy."""
    if suite in ("train", "dev"):
        return [(d.name, {"episode_dir": str(d)}, None) for d in split_dirs(datasets, suite)]
    if suite == "presets":
        return [(f"{name}-{i}", {**PRESETS[name]}, 5000 + 17 * i + len(name))
                for name in PRESETS for i in range(seeds)]
    raise ValueError(f"unknown suite {suite!r}")


def level_of(label: str, config: dict) -> str:
    """Difficulty level of a benchmark episode (evaluator side: stored episodes record it in their
    private provenance; generated ones carry it in the preset name)."""
    if "episode_dir" in config:
        prov = json.loads((Path(config["episode_dir"]) / "private" / "provenance.json").read_text(encoding="utf-8"))
        return prov["world"].get("difficulty", "?") + ("+compound" if prov["compound_scenarios"] else "")
    return label.split("-")[0]


def run_benchmark(suites=("dev", "presets"), policies=AGENT_NAMES, *, datasets="datasets", seeds: int = 5,
                  mode: str = "eval", isolated: bool = False, policy_seed: int = 0) -> dict:
    rows = []
    for suite in suites:
        for label, config, seed in suite_episodes(suite, datasets, seeds):
            for pol in policies:
                t0 = time.perf_counter()
                r = run_episode(AuditEnv(mode), make_agent(pol, policy_seed, isolated and pol != "oracle"),
                                seed=seed, config=config, label=pol)
                rows.append({"suite": suite, "episode": label, "level": level_of(label, config), "policy": pol, "issues": r["n_truth"],
                             "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "precision": r["precision"],
                             "recall": r["recall"], "f1": r["f1"], "reward": r["reward"],
                             "inspection_steps": r["inspection_steps"], "steps": r["steps"],
                             "findings": r["n_findings"], "runtime_s": round(time.perf_counter() - t0, 4),
                             "env_elapsed_s": r["elapsed_time"], "agent_error": r["agent_error"]})
    return {"meta": _meta(suites, datasets, seeds, mode, isolated, policy_seed, policies, rows),
            "mode": mode, "isolated": isolated, "policy_seed": policy_seed, "rows": rows,
            "summary": summarize_rows(rows), "by_level": summarize_by_level(rows)}


def _meta(suites, datasets, seeds, mode, isolated, policy_seed, policies, rows) -> dict:
    """Everything needed to reproduce / audit a saved benchmark (no timestamps, no local paths)."""
    import hashlib
    import platform
    from .. import __version__
    h = hashlib.sha256()
    for suite in suites:
        if suite in ("train", "dev"):
            for d in split_dirs(datasets, suite):
                for f in sorted((d / "public").iterdir()):
                    h.update(f.name.encode() + f.read_bytes())
    return {"traceforge_version": __version__, "python": platform.python_version(), "platform": platform.system(),
            "suites": list(suites), "presets_seeds_per_preset": seeds, "policies": list(policies), "mode": mode,
            "isolated": isolated, "policy_seed": policy_seed,
            "episodes": len({(r["suite"], r["episode"]) for r in rows}), "runs": len(rows),
            "stored_dataset_public_sha256": h.hexdigest() if any(x in ("train", "dev") for x in suites) else None,
            "command": f"python -m traceforge benchmark --suite {'all' if len(suites) == 3 else ','.join(suites)} "
                       f"--seeds {seeds} --mode {mode}" + (" --isolated" if isolated else "")}


def summarize_rows(rows: list) -> list:
    out = []
    for suite in sorted({r["suite"] for r in rows}):
        for pol in sorted({r["policy"] for r in rows}):
            sel = [r for r in rows if r["suite"] == suite and r["policy"] == pol]
            if not sel:
                continue
            n = len(sel)
            mean = lambda k: round(sum(r[k] for r in sel) / n, 4)
            out.append({"suite": suite, "policy": pol, "episodes": n, "tp": sum(r["tp"] for r in sel),
                        "fp": sum(r["fp"] for r in sel), "fn": sum(r["fn"] for r in sel),
                        "mean_precision": mean("precision"), "mean_recall": mean("recall"), "mean_f1": mean("f1"),
                        "mean_reward": mean("reward"), "mean_inspection_steps": mean("inspection_steps"),
                        "mean_findings": mean("findings"), "mean_runtime_s": mean("runtime_s"),
                        "agent_errors": sum(bool(r["agent_error"]) for r in sel)})
    return out


def summarize_by_level(rows: list) -> list:
    out = []
    for level in sorted({r["level"] for r in rows}):
        for pol in sorted({r["policy"] for r in rows}):
            sel = [r for r in rows if r["level"] == level and r["policy"] == pol]
            if not sel:
                continue
            n = len(sel)
            mean = lambda k: round(sum(r[k] for r in sel) / n, 4)
            out.append({"level": level, "policy": pol, "episodes": n, "issues": sum(r["issues"] for r in sel),
                        "tp": sum(r["tp"] for r in sel), "fp": sum(r["fp"] for r in sel), "fn": sum(r["fn"] for r in sel),
                        "mean_precision": mean("precision"), "mean_recall": mean("recall"), "mean_f1": mean("f1"),
                        "mean_reward": mean("reward"), "mean_inspection_steps": mean("inspection_steps")})
    return out


def write_benchmark(result: dict, out_dir: str | Path, name: str = "benchmark") -> tuple:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(result["rows"])
    with (out / f"{name}_by_level.csv").open("w", encoding="utf-8", newline="") as fh:
        cols = list(result["by_level"][0]) if result["by_level"] else []
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(result["by_level"])
    with (out / f"{name}_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        cols = list(result["summary"][0]) if result["summary"] else []
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(result["summary"])
    return out / f"{name}.json", out / f"{name}.csv"


def render_summary(result: dict) -> str:
    lines = [f"{'suite':<8} {'policy':<15} {'eps':>3} {'TP':>4} {'FP':>4} {'FN':>4} {'prec':>6} {'rec':>6} "
             f"{'F1':>6} {'reward':>7} {'steps':>6} {'find':>5} {'sec':>6}"]
    for s in result["summary"]:
        lines.append(f"{s['suite']:<8} {s['policy']:<15} {s['episodes']:>3} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} "
                     f"{s['mean_precision']:>6.3f} {s['mean_recall']:>6.3f} {s['mean_f1']:>6.3f} "
                     f"{s['mean_reward']:>7.3f} {s['mean_inspection_steps']:>6.1f} {s['mean_findings']:>5.1f} "
                     f"{s['mean_runtime_s']:>6.3f}")
    lines += ["", f"{'level':<16} {'policy':<15} {'eps':>3} {'issues':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'prec':>6} {'rec':>6} "
                  f"{'F1':>6} {'reward':>7} {'steps':>6}"]
    for s in result.get("by_level", []):
        lines.append(f"{s['level']:<16} {s['policy']:<15} {s['episodes']:>3} {s['issues']:>6} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} "
                     f"{s['mean_precision']:>6.3f} {s['mean_recall']:>6.3f} {s['mean_f1']:>6.3f} {s['mean_reward']:>7.3f} "
                     f"{s['mean_inspection_steps']:>6.1f}")
    return "\n".join(lines)
