"""`traceforge audit`: automated quality audit of architecture and data.

Every finding is computed from the code base / generated data at run time."""
from __future__ import annotations

import ast
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from ..config import PRESETS
from ..constants import ISSUE_TYPES
from ..environment.audit_env import AuditEnv
from ..environment.calculator import evaluate
from ..episode_io import private_files, public_files
from ..exceptions import CalculatorError
from ..generator.builder import generate_episode
from ..generator.validation import validate_episode

PKG = Path(__file__).resolve().parents[1]
SRC = PKG.parent


def _section(name, fn):
    t = time.monotonic()
    try:
        ok, detail, data = fn()
    except Exception as e:
        ok, detail, data = False, f"{type(e).__name__}: {e}", {}
    return {"name": name, "ok": bool(ok), "detail": detail, "data": data, "seconds": round(time.monotonic() - t, 2)}


def _py_files():
    return sorted(p for p in PKG.rglob("*.py"))


# ----------------------------------------------------------------------------- sections
def data_sweep(n_seeds: int = 6):
    """Validate episodes across presets, single types and scenarios; build a preset x type matrix."""
    cfgs = {**{f"preset:{k}": v for k, v in PRESETS.items()},
            **{f"only:{t}": {"issue_counts": {t: 2}} for t in ISSUE_TYPES},
            "scenario:cross_border_partial": {"scenarios": ["cross_border_partial"]},
            "scenario:period_end_duplicate": {"scenarios": ["period_end_duplicate"]},
            "clean": {"clean": True, "difficulty": "hard"}}
    matrix, failures, n, tags, rows = {}, [], 0, Counter(), []
    for name, cfg in cfgs.items():
        counts = Counter()
        for seed in range(n_seeds):
            ep = generate_episode(9000 + seed, cfg)
            n += 1
            rep = validate_episode(ep)
            if not rep["ok"]:
                failures.append((name, seed, [c["name"] for c in rep["checks"] if not c["ok"]]))
            counts.update(t["type"] for t in ep.private.truth)
            rows.append(len(ep.public.ledger))
            dt = ep.private.provenance["world"]["document_tags"]
            docs = {d["doc_id"]: (d.get("invoice_id") or d["payment_id"]) for d in ep.public.invoices + ep.public.payments}
            for m in ep.private.provenance["mutations"]:
                tags[dt[docs[m["document_ids"][0]]].split(":")[0]] += 1
            floats = [x for tbl in (ep.public.ledger, ep.public.invoices, ep.public.payments)
                      for r in tbl for k, x in r.items() if k.endswith("_minor") and isinstance(x, float)]
            if floats:
                failures.append((name, seed, "float money"))
        matrix[name] = {t: counts[t] for t in ISSUE_TYPES}
    covered = {t: [k for k, v in matrix.items() if v[t] and k.startswith("preset:")] for t in ISSUE_TYPES}
    ok = not failures and all(len(v) >= 2 for v in covered.values()) and min(rows) >= 200
    return ok, (f"{n} episodes, {sum(1 for _ in failures)} validation failures; types seen in >=2 presets: "
                f"{ {t: len(v) for t, v in covered.items()} }; min ledger rows {min(rows)}"), \
        {"matrix": matrix, "failures": failures, "mutated_document_tags": dict(tags)}


def architecture():
    problems, forbidden, global_random, network, third_party = [], [], [], [], []
    net = {"socket", "ssl", "http", "urllib", "ftplib", "smtplib", "xmlrpc", "requests", "httpx", "aiohttp", "telnetlib", "asyncio"}
    for f in _py_files():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else                 ([node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
            for n in names:
                top = n.split(".")[0]
                if top in net:
                    network.append(f"{f.relative_to(PKG)}:{node.lineno} {n}")
                if top != "traceforge" and top not in sys.stdlib_module_names:
                    third_party.append(f"{f.relative_to(PKG)}:{node.lineno} {n}")
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in ("eval", "exec", "compile", "__import__"):
                    forbidden.append(f"{f.relative_to(PKG)}:{node.lineno} {fn.id}")
                if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "random" \
                        and fn.attr != "Random":
                    global_random.append(f"{f.relative_to(PKG)}:{node.lineno} random.{fn.attr}")
                if isinstance(fn, ast.Attribute) and fn.attr in ("system", "popen") and isinstance(fn.value, ast.Name) \
                        and fn.value.id == "os":
                    problems.append(f"{f.relative_to(PKG)}:{node.lineno} os.{fn.attr}")
    code = ("import sys; import traceforge.runner.agent_worker; import traceforge.policies; "
            "print(sorted(m for m in sys.modules if m.startswith(('traceforge.generator', 'traceforge.evaluation', "
            "'traceforge.environment.audit_env', 'traceforge.episode_io', 'traceforge.reporting'))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": str(SRC), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}).stdout.strip()
    from ..policies import REGISTRY
    ok = not (problems or forbidden or global_random or network or third_party) and out == "[]" and "oracle" not in REGISTRY
    return ok, (f"no eval/exec/compile: {not forbidden}; no global random state: {not global_random}; "
                f"no network modules: {not network}; standard library only: {not third_party}; "
                f"agent process imports evaluator/generator modules: {out}; oracle registered for agents: {'oracle' in REGISTRY}"), \
        {"forbidden_calls": forbidden, "global_random": global_random, "other": problems,
         "network_imports": network, "third_party_imports": third_party,
         "python_files": len(_py_files()), "largest_files": sorted(
             ((len(f.read_text(encoding='utf-8').splitlines()), str(f.relative_to(PKG))) for f in _py_files()), reverse=True)[:3]}


def tells():
    """Statistical check that generation does not leave structural fingerprints."""
    pct = Counter()
    for s in range(30):
        ep = generate_episode(s, {"scenarios": ["cross_border_partial"]})
        inv = {i["invoice_id"]: i for i in ep.public.invoices}
        roles = ep.private.provenance["compound_scenarios"][0]["roles"]
        for role in ("alloc_payment", "fx_payment"):
            p = next(x for x in ep.public.payments if x["payment_id"] == roles[role])
            r = p["invoice_refs"][0]
            pct[round(100 * r["allocated_minor"] / inv[r["invoice_id"]]["total_minor"])] += 1
    top_share = max(pct.values()) / sum(pct.values())
    pos, tags = [], Counter()
    for s in range(40):
        ep = generate_episode(s, {"issue_counts": {"duplicate": 2, "allocation": 2, "fx": 2}})
        order = {r["row_id"]: i / len(ep.public.ledger) for i, r in enumerate(ep.public.ledger)}
        for m in ep.private.provenance["mutations"]:
            pos += [order[r["row_id"]] for r in m["modified_rows"]]
        dt = ep.private.provenance["world"]["document_tags"]
        docs = {d["doc_id"]: (d.get("invoice_id") or d["payment_id"]) for d in ep.public.invoices + ep.public.payments}
        tags.update(dt[docs[m["document_ids"][0]]].split(":")[0] for m in ep.private.provenance["mutations"])
    mean_pos = sum(pos) / len(pos)
    decoy_share = tags["decoy"] / sum(tags.values())
    ok = top_share < 0.2 and len(pct) >= 12 and 0.4 < mean_pos < 0.6 and decoy_share > 0.05
    return ok, (f"cross-border allocation shares: {len(pct)} distinct whole-percent values (>=12 required), top value {top_share:.0%} (<20% required); "
                f"mean ledger position of mutated rows {mean_pos:.2f} (0.4-0.6 required); "
                f"share of mutated documents that are decoys {decoy_share:.0%} (>5% required)"), \
        {"allocation_buckets": dict(sorted(pct.items())), "mutated_tags": dict(tags)}


def reproducibility_across_processes():
    code = ("import sys,hashlib;sys.path.insert(0,sys.argv[1]);from traceforge.generator.builder import generate_episode as g;"
            "from traceforge.episode_io import public_files as p, private_files as q;e=g(3,{'difficulty':'hard','issue_counts':{'duplicate':2,'allocation':2,'fx':2,'cutoff':2}});"
            "h=hashlib.sha256();[h.update((k+v).encode()) for f in (p(e.public),q(e.private)) for k,v in sorted(f.items())];print(h.hexdigest())")
    digests = {}
    for hs in ("0", "1", "4242", "random"):
        r = subprocess.run([sys.executable, "-c", code, str(SRC)], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": hs})
        digests[hs] = r.stdout.strip()
    return len(set(digests.values())) == 1 and all(digests.values()), \
        f"same digest under {len(digests)} PYTHONHASHSEED values / separate processes: {len(set(digests.values())) == 1}", digests


def _junk(rng, depth=0):
    c = rng.random()
    if c < .15:
        return None
    if c < .3:
        return rng.choice([0, 1, -1, 50, 51, 10 ** 100, 3.5, True])
    if c < .5:
        return rng.choice(["", "ledger", "c50", "x" * 500, "ROW-00001", "\x00", "1+1", "__import__('os')"])
    if c < .7 and depth < 3:
        return [_junk(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    if depth < 3:
        return {rng.choice(["action", "table", "id", "cursor", "limit", "doc_id", "expression", "findings", "type"]):
                _junk(rng, depth + 1) for _ in range(rng.randint(0, 4))}
    return "z"


def fuzz_env(n: int = 2000):
    rng = random.Random(7)
    env = AuditEnv("eval")
    cfg = {"issue_counts": {"fx": 1, "duplicate": 1}}
    env.reset(1, cfg)
    tokens, crashes, leaks, steps = env.leak_tokens(), 0, 0, 0
    for _ in range(n):
        if env._done:
            env.reset(1, cfg)
            tokens = env.leak_tokens()
        a = _junk(rng)
        if rng.random() < .5:
            a = {"action": rng.choice(["list_records", "get_record", "get_document", "calculator", "submit"]),
                 **{k: _junk(rng) for k in ("table", "id", "cursor", "limit", "doc_id", "expression", "findings") if rng.random() < .5}}
        try:
            o, r, t, tr, i = env.step(a)
            steps += 1
            leaks += any(tok in json.dumps([o, i], default=str) for tok in tokens)
        except Exception:
            crashes += 1
    return crashes == 0 and leaks == 0, f"{steps} garbage actions: {crashes} unhandled exceptions, {leaks} leaks", \
        {"actions": steps, "crashes": crashes, "leaks": leaks}


def fuzz_calculator(n: int = 1500):
    rng = random.Random(11)
    alphabet = list("0123456789.+-*/()%e ") + ["**", "//", "round(", "abs(", "min(", "max(", "x", "_", "'", ",", "1e99", "9**9"]
    slow, bad, accepted = [], [], 0
    for _ in range(n):
        expr = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60)))
        t = time.perf_counter()
        try:
            evaluate(expr)
            accepted += 1
        except CalculatorError:
            pass
        except Exception as e:
            bad.append((expr, type(e).__name__))
        if time.perf_counter() - t > 0.25:
            slow.append(expr)
    return not bad and not slow, f"{n} random expressions: {accepted} evaluated, {len(bad)} unexpected exceptions, {len(slow)} slow (>0.25s)", \
        {"unexpected": bad[:5], "slow": slow[:5]}


def run_audit(out_dir: str | Path = "reports", performance: bool = True) -> dict:
    from .performance import run_performance
    secs = [_section("data sweep (validator on every preset/type/scenario)", data_sweep),
            _section("architecture scan", architecture),
            _section("structural tells", tells),
            _section("cross-process reproducibility", reproducibility_across_processes),
            _section("environment fuzzing", fuzz_env),
            _section("calculator fuzzing", fuzz_calculator)]
    if performance:
        def perf():
            r = run_performance(out_dir)
            worst = max(x["generate_s"] for x in r["rows"])
            return worst < 30, f"{len(r['rows'])} sizes up to {max(x['ledger_rows'] for x in r['rows'])} rows; worst generation {worst}s", r
        secs.append(_section("performance", perf))
    report = {"ok": all(s["ok"] for s in secs), "sections": secs}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def render(report: dict) -> str:
    lines = [f"AUDIT: {'PASS' if report['ok'] else 'FAIL'}", ""]
    for s in report["sections"]:
        lines.append(f"[{'PASS' if s['ok'] else 'FAIL'}] {s['name']} ({s['seconds']}s)\n       {s['detail']}")
    return "\n".join(lines)
