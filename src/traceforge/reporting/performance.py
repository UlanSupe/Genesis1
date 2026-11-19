"""Performance probe: generation / reset / list_records / get_document / matcher / memory
at several ledger sizes. Measured on the machine running it; numbers are not claims about
other hardware."""
from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from pathlib import Path

from ..environment.audit_env import AuditEnv
from ..episode_io import save_episode
from ..evaluation.matcher import match
from ..generator.builder import generate_episode
from ..runner.episode_runner import make_agent, run_episode
from ..schemas import Finding

SIZES = (200, 500, 1000, 2000)


def _ms(fn, n: int) -> float:
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 4)


def probe(cfg: dict, label: str, tmp: Path) -> dict:
    tracemalloc.start()
    t = time.perf_counter()
    ep = generate_episode(1, cfg)
    gen_s = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    path = save_episode(ep, tmp / label)
    env = AuditEnv("eval")
    reset_dir = _ms(lambda: env.reset(config={"episode_dir": str(path)}), 3)
    reset_seed = _ms(lambda: env.reset(1, cfg), 3)
    env.reset(config={"episode_dir": str(path)})
    docs = [d["doc_id"] for d in env._pub.invoices + env._pub.payments]
    lst = _ms(lambda: env._execute({"action": "list_records", "table": "ledger", "cursor": "c0", "limit": 50}), 50)
    get = _ms(lambda: [env._execute({"action": "get_document", "doc_id": d}) for d in docs[:50]], 5) / min(50, len(docs))
    truth = ep.private.truth
    fs = [Finding(t["type"], t["affected_ids"], t["expected_amount_minor"], t["currency"],
                  t["allowed_evidence_sets"][0], "") for t in truth] + \
         [Finding("fx", [f"PAY-{i}"], i, "EUR", [], "") for i in range(30 - len(truth))]
    mt = _ms(lambda: match(fs[:30], truth), 200)
    t = time.perf_counter()
    r = run_episode(AuditEnv("eval"), make_agent("reconciliation"), config={"episode_dir": str(path)})
    return {"label": label, "ledger_rows": len(ep.public.ledger), "documents": ep.public.manifest["document_count"],
            "issues": len(truth), "generate_s": round(gen_s, 3), "reset_from_dir_ms": reset_dir,
            "reset_generate_ms": reset_seed, "list_records_50_ms": lst, "get_document_ms": round(get, 4),
            "matcher_30x12_ms": mt, "reconciliation_episode_s": round(time.perf_counter() - t, 3),
            "reconciliation_f1": r["f1"], "peak_python_mem_MB_generate": round(peak / 1e6, 1)}


def _fit(issues: dict, key: str, target: int) -> dict:
    """Smallest config (n_invoices grown by ~10%) whose episode reaches the target size."""
    n = max(4, target // 6)
    while True:
        cfg = {**issues, "n_invoices": n, "min_ledger_rows": 0, "difficulty": "medium"}
        ep = generate_episode(1, cfg)
        size = len(ep.public.ledger) if key == "rows" else ep.public.manifest["document_count"]
        if size >= target or n > 2000:
            return cfg
        n = max(n + 2, int(n * 1.1))


def run_performance(out_dir: str | Path | None = None, tmp: Path | None = None, sizes=SIZES) -> dict:
    import tempfile
    issues = {"issue_counts": {"duplicate": 2, "allocation": 2, "fx": 2, "cutoff": 2}}
    with tempfile.TemporaryDirectory() as t:
        base = tmp or Path(t)
        rows = [probe(_fit(issues, "rows", n), f"rows~{n}", base) for n in sizes]
        rows.append(probe(_fit(issues, "docs", 200), "docs~200", base))
    report = {"rows": rows, "note": "median of repeated calls; generation includes full independent validation"}
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "performance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def render(report: dict) -> str:
    head = f"{'case':<9} {'rows':>5} {'docs':>5} {'gen s':>7} {'reset(dir) ms':>14} {'list50 ms':>10} {'get_doc ms':>11} " \
           f"{'match ms':>9} {'recon s':>8} {'MB':>6}"
    lines = [head]
    for r in report["rows"]:
        lines.append(f"{r['label']:<9} {r['ledger_rows']:>5} {r['documents']:>5} {r['generate_s']:>7.3f} "
                     f"{r['reset_from_dir_ms']:>14.2f} {r['list_records_50_ms']:>10.3f} {r['get_document_ms']:>11.4f} "
                     f"{r['matcher_30x12_ms']:>9.3f} {r['reconciliation_episode_s']:>8.3f} {r['peak_python_mem_MB_generate']:>6.1f}")
    return "\n".join(lines)
