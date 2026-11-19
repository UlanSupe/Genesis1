"""`traceforge verify`: full verification of the project.

Runs the pytest suite (once, with JUnit output), derives SPEC_COMPLIANCE.md from the real
test results, and runs the end-to-end checks (generator, matcher/reward, reproducibility,
isolation, 24 metamorphic pairs, datasets, audit, benchmark). Every number is measured at
run time; nothing is hard-coded."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..constants import ISSUE_TYPES, SCENARIOS
from ..environment.audit_env import AuditEnv
from ..episode_io import load_episode, private_files, public_files
from ..evaluation.evaluator import Evaluator
from ..generator import validation
from ..generator.builder import dataset_specs, generate_dataset, generate_episode
from ..runner.benchmark import run_benchmark, summarize_rows, write_benchmark
from ..runner.episode_runner import make_agent, run_episode
from ..trajectories.reader import check_no_leak, read_trajectory, replay
from ..trajectories.recorder import TrajectoryRecorder
from . import audit, spec_compliance
from .transformation_report import build_transformation_report


def _section(name, fn):
    t = time.monotonic()
    try:
        ok, detail = fn()
    except Exception as e:                                  # a crashing check is a failed check
        ok, detail = False, f"{type(e).__name__}: {e}"
    return {"name": name, "ok": bool(ok), "detail": detail, "seconds": round(time.monotonic() - t, 2)}


def _digest(ep) -> str:
    h = hashlib.sha256()
    for files in (public_files(ep.public), private_files(ep.private)):
        for k in sorted(files):
            h.update(k.encode() + files[k].encode())
    return h.hexdigest()


def run_pytest(root: Path, junit: Path):
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider",
                        f"--junitxml={junit}"], cwd=root, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = [l for l in p.stdout.strip().splitlines() if l.strip()][-1:] or ["no output"]
    return p.returncode == 0, tail[0]


def junit_totals(path: Path) -> dict:
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = sum(any(c.tag in ("failure", "error") for c in tc) for tc in cases)
    skipped = sum(any(c.tag == "skipped" for c in tc) for tc in cases)
    return {"total": len(cases), "failed": failed, "skipped": skipped, "passed": len(cases) - failed - skipped}


def _generator():
    bad, n = [], 0
    for typ in ISSUE_TYPES:
        for seed in range(6):
            ep = generate_episode(seed, {"issue_counts": {typ: 1}})
            n += 1
            ok = [t["type"] for t in ep.private.truth] == [typ] and ep.private.provenance["validation"]["ok"]
            if not ok:
                bad.append((typ, seed))
    for sc in SCENARIOS:
        for seed in range(6):
            ep = generate_episode(seed, {"scenarios": [sc]})
            n += 1
            ok = len(ep.private.truth) == 2 and ep.private.provenance["validation"]["ok"]
            if not ok or len({t["type"] for t in ep.private.truth}) != 2:
                bad.append((sc, seed))
    for seed in range(10):
        ep = generate_episode(seed, {"clean": True})
        n += 1
        if ep.private.truth or validation.derive_issues(ep.public) or len(ep.public.ledger) < 200:
            bad.append(("clean", seed))
    return not bad, f"{n} episodes (4 types x6, 2 scenarios x6, 10 clean) validated; failures: {bad or 'none'}"


def _matcher_reward():
    ev = Evaluator([{"issue_id": "ISS-001", "type": "duplicate", "affected_ids": ["INV-1"],
                     "expected_amount_minor": 100, "currency": "EUR",
                     "allowed_evidence_sets": [["DOC-1", "ROW-1", "ROW-2"]]}], "e")
    f = {"type": "duplicate", "affected_ids": ["INV-1"], "claimed_amount_minor": 101,
         "currency": "EUR", "evidence_ids": ["DOC-1", "ROW-1", "ROW-2"], "explanation": ""}
    cases = {
        "exact": ([f], 1.0 - 0.002 * 5),
        "empty_submit_with_truth": ([], -1.0),
        "wrong_currency": ([{**f, "currency": "USD"}], -1.0),
        "amount_off_by_2": ([{**f, "claimed_amount_minor": 102}], -1.0),
        "missing_evidence": ([{**f, "evidence_ids": ["DOC-1"]}], -1.0),
        "duplicate_prediction": ([f, f], 2 * (2 / 3) - 1 - 0.01),
    }
    bad = []
    for name, (fs, want) in cases.items():
        got = ev.evaluate({"episode_id": "e", "findings": fs}, 5).reward
        if abs(got - want) > 1e-9:
            bad.append((name, got, want))
    inv = ev.evaluate({"episode_id": "x", "findings": []}, 5).reward
    if inv != -1.0:
        bad.append(("invalid_structure", inv, -1.0))
    return not bad, f"{len(cases) + 1} matcher/reward cases; failures: {bad or 'none'}"


def _reproducibility():
    cfg = {"issue_counts": {"duplicate": 1, "allocation": 1, "fx": 1, "cutoff": 1}}
    a, b, c = generate_episode(42, cfg), generate_episode(42, cfg), generate_episode(43, cfg)
    return _digest(a) == _digest(b) and _digest(a) != _digest(c), \
        f"seed 42 twice -> identical sha256 {_digest(a)[:16]}...; seed 43 differs"


def _leaks_and_isolation():
    cfg = {"issue_counts": {"duplicate": 1, "fx": 1, "cutoff": 1}}
    out, rows = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for pol in ("random", "reconciliation", "mark_all"):
            for iso in (False, True):
                env = AuditEnv("eval")
                path = Path(tmp) / f"{pol}_{iso}.jsonl"
                with TrajectoryRecorder(path) as rec:
                    r = run_episode(env, make_agent(pol, 0, iso), seed=3, config=cfg, recorder=rec, label=pol)
                check_no_leak(path, env.leak_tokens())
                steps = read_trajectory(path)
                rows.append((pol, iso, r["f1"], r["agent_error"], steps[-1]["reward"]))
                if r["agent_error"] or steps[-1]["reward"] != 0.0:
                    out.append((pol, iso))
                if not replay(path, AuditEnv("eval"), 3, cfg)["ok"]:
                    out.append((pol, iso, "replay"))
        inproc = {p: f for p, i, f, *_ in rows if not i}
        for p, i, f, *_ in rows:
            if i and f != inproc[p]:
                out.append((p, "isolated result differs"))
    return not out, f"{len(rows)} runs (in-process + isolated subprocess), trajectories leak-scanned " \
                    f"and replayed; eval-mode reward hidden from agent; problems: {out or 'none'}"


def _metamorphic(out_dir: Path):
    rep = build_transformation_report(24, out_dir / "transformation_report.json")
    s = rep["summary"]
    return s["failed_pairs"] == [] and s["passed"] == s["pairs"] == 24, \
        f"{s['passed']}/{s['pairs']} original/transformed pairs passed ({s['checks_run']} checks, " \
        f"{s['total_ids_remapped']} ids remapped); saved to transformation_report.json"


def _datasets(root: Path):
    ds = root / "datasets"
    if not (ds / "train" / "index.json").exists():
        return False, "datasets/ missing - run `python -m traceforge generate`"
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        generate_dataset(tmp)
        for split in ("train", "dev"):
            for name, *_ in dataset_specs(split):
                a, b = load_episode(ds / split / name), load_episode(Path(tmp) / split / name)
                if _digest(a) != _digest(b):
                    bad.append(f"{split}/{name} differs from regeneration")
                if not validation.validate_episode(a)["ok"]:
                    bad.append(f"{split}/{name} fails validation")
    return not bad, f"20 stored episodes revalidated and regenerated bit-for-bit; problems: {bad or 'none'}"


def _audit(out_dir: Path):
    rep = audit.run_audit(out_dir, performance=True)
    failed = [s["name"] for s in rep["sections"] if not s["ok"]]
    return rep["ok"], f"{len(rep['sections'])} audit sections; failed: {failed or 'none'} (report: audit_report.json)"


def _benchmark(root: Path, out_dir: Path):
    res = run_benchmark(("dev", "presets"), datasets=root / "datasets", seeds=3)
    write_benchmark(res, out_dir)
    s = {(x["suite"], x["policy"]): x for x in res["summary"]}
    errors = sum(x["agent_errors"] for x in res["summary"])
    ok = all(s[("dev", "oracle")]["mean_f1"] == 1.0 and s[("presets", "oracle")]["mean_f1"] == 1.0 for _ in (0,)) and errors == 0
    line = "; ".join(f"{x['suite']}/{x['policy']}: F1 {x['mean_f1']:.3f} reward {x['mean_reward']:.3f}" for x in res["summary"])
    return ok, f"{len(res['rows'])} runs on identical episodes (JSON+CSV written); {line}"


def run_verification(root: str | Path = ".", run_tests: bool = True, out_dir: str | Path = "reports") -> dict:
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    secs = []
    if run_tests:
        junit = out_dir / "pytest_junit.xml"
        secs.append(_section("pytest suite", lambda: run_pytest(root, junit)))
        if junit.exists():
            totals = junit_totals(junit)
            rows = spec_compliance.evaluate(spec_compliance.parse_junit(junit))
            (root / "SPEC_COMPLIANCE.md").write_text(spec_compliance.render(rows, totals), encoding="utf-8")
            secs[0]["detail"] += f" | {totals['passed']} passed, {totals['failed']} failed, {totals['skipped']} skipped"
            secs.append(_section("specification compliance (SPEC_COMPLIANCE.md)", lambda: (
                all(r["status"] == "PASS" for r in rows),
                f"{sum(r['status'] == 'PASS' for r in rows)}/{len(rows)} requirements PASS; "
                f"not PASS: {[r['requirement'][:40] for r in rows if r['status'] != 'PASS'] or 'none'}")))
    secs += [_section("generator: types, compound scenarios, clean episodes", _generator),
             _section("matcher and reward", _matcher_reward),
             _section("reproducibility", _reproducibility),
             _section("isolation, leaks, trajectories", _leaks_and_isolation),
             _section("metamorphic pairs (24)", lambda: _metamorphic(out_dir)),
             _section("datasets", lambda: _datasets(root)),
             _section("audit (architecture, data, tells, fuzzing, performance)", lambda: _audit(out_dir)),
             _section("benchmark (all policies, identical episodes)", lambda: _benchmark(root, out_dir))]
    report = {"ok": all(s["ok"] for s in secs), "sections": secs}
    (out_dir / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "verification_report.md").write_text(render(report), encoding="utf-8")
    return report


def render(report: dict) -> str:
    lines = ["# TraceForge verification report", "",
             f"Overall: **{'PASS' if report['ok'] else 'FAIL'}**", "",
             "| # | Check | Result | Seconds |", "|---|---|---|---|"]
    for i, s in enumerate(report["sections"], 1):
        lines.append(f"| {i} | {s['name']} | {'PASS' if s['ok'] else 'FAIL'} | {s['seconds']} |")
    lines.append("")
    for s in report["sections"]:
        lines += [f"## {s['name']} - {'PASS' if s['ok'] else 'FAIL'}", "", str(s["detail"]), ""]
    return "\n".join(lines)
