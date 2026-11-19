"""Mutation check of the TEST SUITE: inject one known bug at a time into a scratch copy of
the code base and verify that at least one test fails ("killed"). A surviving mutant means
the tests would not notice that bug.

    python tools/mutation_check.py [--out reports/mutation_check.json]
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (id, file, old, new, description)
MUTANTS = [
    ("M01", "evaluation/matcher.py", "<= tol\n", "<= tol + 5\n", "amount tolerance widened"),
    ("M02", "evaluation/matcher.py", "norm_ids(f.affected_ids) == tuple(sorted(set(t[\"affected_ids\"])))",
     "set(norm_ids(f.affected_ids)) <= set(t[\"affected_ids\"])", "affected_ids subset instead of equality"),
    ("M03", "evaluation/evidence_validator.py", "any(set(s) <= ev for s in allowed_sets)",
     "any(set(s) & ev for s in allowed_sets)", "evidence: any overlap instead of full set"),
    ("M04", "evaluation/matcher.py", "and f.currency.strip().upper() == t[\"currency\"]", "", "currency not compared"),
    ("M05", "constants.py", "STEP_COST = Fraction(1, 500)", "STEP_COST = Fraction(1, 400)", "step cost 0.0025"),
    ("M06", "evaluation/reward.py", "Fraction(1) if denom == 0 else", "Fraction(0) if denom == 0 else", "F1 of empty/empty = 0"),
    ("M07", "constants.py", "MAX_PAGE = 50", "MAX_PAGE = 60", "page limit 60"),
    ("M08", "environment/audit_env.py", "if self.mode == \"train\":", "if True:", "eval mode leaks reward"),
    ("M09", "environment/calculator.py", "abs(b) > 12", "abs(b) > 12000", "exponent limit removed"),
    ("M10", "environment/pagination.py", "end = offset + limit", "end = offset + limit - 1", "page off-by-one"),
    ("M11", "generator/validation.py", "t >= end else 0", "t > end else 0", "period end inclusive"),
    ("M12", "evaluation/matcher.py", "if j not in owner or augment(owner[j], seen):", "if j not in owner:",
     "greedy matching (no augmenting paths)"),
    ("M13", "environment/audit_env.py", "inspection = self._steps - 1", "inspection = self._steps", "submit counted as inspection"),
    ("M14", "constants.py", "INVALID_REWARD = -1.0", "INVALID_REWARD = 0.0", "invalid submission reward 0"),
    ("M15", "environment/isolation.py", "raise LeakError(f\"private token #{i} found in agent-visible data\")", "pass",
     "leak guard disabled"),
    ("M16", "generator/mutations.py", "    rows.extend(copies)\n", "", "duplicate mutation does nothing"),
    ("M17", "generator/transformations.py", "    if shuffle:\n", "    if False:\n", "shuffle disabled"),
    ("M18", "policies/mark_all_policy.py", "if len(out) >= 30:", "if len(out) >= 31:", "mark_all exceeds 30 findings"),
    ("M19", "environment/audit_env.py", "truncated = not terminated and self._steps >= MAX_STEPS",
     "truncated = not terminated and self._steps > MAX_STEPS", "truncation one step late"),
    ("M20", "environment/audit_env.py", "\"records\": copy.deepcopy(page)", "\"records\": page",
     "list_records returns live references"),
    ("M21", "environment/audit_env.py", "\"record\": copy.deepcopy(rec)", "\"record\": rec",
     "get_record returns live reference"),
    ("M22", "environment/calculator.py", "rounding=ROUND_HALF_UP)", "rounding=\"ROUND_HALF_EVEN\")", "calculator rounds half-even"),
    ("M23", "money.py", "quantize(Decimal(1), rounding=ROUND_HALF_UP)", "quantize(Decimal(1), rounding=\"ROUND_HALF_EVEN\")",
     "money conversion rounds half-even"),
    ("M24", "environment/actions.py", "_need(isinstance(name, str) and name in ACTIONS", "_need(isinstance(name, str)",
     "unknown actions accepted"),
    ("M25", "generator/mutations.py", "thr = max(3, anchor0 // 400)", "thr = 0", "fx errors may be below tolerance"),
    ("M26", "trajectories/recorder.py", "\"reward\": reward, \"terminated\": terminated, \"truncated\": truncated},",
     "\"reward\": reward, \"terminated\": terminated},", "trajectory drops truncated"),
    ("M27", "environment/audit_env.py", "info[\"submission_valid\"] = self._result.valid",
     "info[\"submission_valid\"] = self._result.valid; info[\"debug\"] = self._evaluator.leak_tokens()", "info leaks issue ids"),
    ("M28", "schemas.py", "if payload.get(\"episode_id\") != episode_id:", "if False:", "episode_id not checked"),
    ("M29", "environment/isolation.py", "if len(line) > self.max_line:", "if False:", "stdout message cap removed"),
    ("M30", "environment/isolation.py", "deque(maxlen=STDERR_TAIL)", "deque()", "stderr tail unbounded"),
    ("M31", "environment/isolation.py", "if builtin:", "if True:", "project path leaked to custom agents"),
    ("M32", "config.py", "and sum(self.mutation_weights.values()) > 0", "and True", "zero mutation weights accepted"),
    ("M33", "generator/mutations.py", "if not d[\"_tag\"].startswith(\"compound\") and World.doc_key(d) not in self.used]",
     "if d[\"_tag\"] == \"regular\" and World.doc_key(d) not in self.used]", "decoys immune to mutation again"),
    ("M34", "generator/compound_scenarios.py", "[(inv, t * rng.randint(20, 45) // 100)], tag)\n        p2",
     "[(inv, t * 40 // 100)], tag)\n        p2", "constant 40% scenario fingerprint"),
    ("M35", "generator/validation.py", "[] if replayed == Counter(row_key(r) for r in pub.ledger) else [\"unexplained difference\"]",
     "[]", "manifest replay check disabled"),
    ("M36", "generator/validation.py", "out[(ts, ACC_AR, -v, 0, c, e, i, pid)] += 1", "out[(ts, ACC_AR, v, 0, c, e, i, pid)] += 1",
     "expected ledger wrong for refunds"),
    ("M37", "generator/transformations.py", "\"ledger_rows_isomorphic\": ledger_ok,", "\"ledger_rows_isomorphic\": True,",
     "link verifier ignores ledger rows"),
    ("M38", "generator/transformations.py", "\"evidence_ids\": [m(i) for i in f[\"evidence_ids\"]]}", "\"evidence_ids\": f[\"evidence_ids\"]}",
     "map_findings leaves evidence ids stale"),
    ("M39", "generator/builder.py", "while count_rows(world) < cfg.min_ledger_rows:", "while False:", "minimum ledger size not enforced"),
    ("M40", "environment/audit_env.py", "config = options if options is not None else config", "config = config",
     "options alias ignored"),
    ("M41", "reporting/spec_compliance.py", "elif any(r is False for r in runs):", "elif False:", "failing tests not reflected in spec matrix"),
    ("M42", "generator/financial_world.py", "scale_pct=rng.choice((40, 100, 250)), mix=P[\"mix\"])",
     "scale_pct=rng.choice((40, 100, 250)), mix=(45, 60, 72, 85))", "difficulty ignores payment mix"),
    ("M43", "generator/clean_cases.py", "n = {k: max(1, P[k] + rng.randint(-1, 2)) for k in (\"tz\", \"repeat\", \"sameamt\", \"refunds\", \"over\")}",
     "n = {k: P[k] for k in (\"tz\", \"repeat\", \"sameamt\", \"refunds\", \"over\")}", "decoy volume constant"),
    ("M44", "environment/pagination.py", "or int(m.group(1)) > len(records)", "or False", "out-of-range cursor accepted"),
    ("M45", "evaluation/matcher.py", "order = sorted(range(len(findings)), key=lambda i: (",
     "order = sorted(range(len(findings)), key=lambda i: (i,) + (", "matching depends on submission order"),
    ("M46", "generator/mutations.py", "ev = [[doc[\"doc_id\"], r[\"_uid\"], extra[\"_uid\"]], [doc[\"doc_id\"], r[\"_uid\"]],\n              [doc[\"doc_id\"], extra[\"_uid\"]]]",
     "ev = [[doc[\"doc_id\"], r[\"_uid\"], extra[\"_uid\"]]]", "allocation evidence alternatives removed"),
    ("M47", "generator/validation.py", "for k, v in outstanding_balances(pub).items()", "for k, v in {}.items()",
     "outstanding balance check disabled"),
    ("M49", "generator/mutations.py", "rate_bp=(50, 150)", "rate_bp=(300, 500)", "hard fx errors as obvious as easy ones"),
    ("M50", "generator/mutations.py", "shift_min=(15, 180)", "shift_min=(1440, 3600)", "hard cutoff errors as obvious as easy ones"),
    ("M51", "generator/mutations.py", "lookalike=True", "lookalike=False", "hard duplicates ignore lookalike documents"),
    ("M52", "generator/validation.py", "            if want not in got:", "            if False:", "evidence reconstruction check disabled"),
    ("M53", "generator/validation.py", "    if exp != p_acc:", "    if False:", "account totals check disabled"),
    ("M54", "generator/validation.py", "        if exp != by_key[k]:", "        if False:", "clean decoy integrity check disabled"),
    ("M55", "generator/validation.py", "for i in t[\"affected_ids\"]\n            if i not in known][:10]", "for i in t[\"affected_ids\"]\n            if False][:10]",
     "affected id existence check disabled"),
    ("M56", "generator/clean_cases.py", "cadence = rng.choice((5, 7, 10))", "cadence = rng.choice((40,))", "recurring series cut short"),
    ("M57", "generator/clean_cases.py", "    split_payments(world, rng, n[\"split\"])", "    pass", "guaranteed split-payment decoy removed"),
    ("M58", "generator/mutations.py", "\"allowed_evidence_sets\": evidence, \"root_cause\": cause,\n                            \"explanation\": explanation})",
     "\"allowed_evidence_sets\": evidence, \"root_cause\": cause,\n                            \"explanation\": \"x\"})", "explanation text dropped"),
    ("M59", "runner/benchmark.py", "    return label.split(\"-\")[0]", "    return \"?\"", "benchmark level breakdown broken"),
    ("M60", "generator/mutations.py", "ev = [sorted(set(s) | set(inv_docs)) for s in ev]", "ev = ev", "allocation evidence lacks invoice documents"),
    ("M61", "config.py", "\"compound_rate\": 0.6}", "\"compound_rate\": 0.0}", "hard preset never compound"),
    ("M62", "generator/mutations.py", "rng.randint(*ctx.profile[\"shift_min\"])", "rng.randint(120, 3600)", "cutoff shift ignores difficulty"),
    ("M63", "generator/mutations.py", "self.profile = PROFILES[world.cfg.difficulty]", "self.profile = PROFILES[\"medium\"]", "difficulty profile ignored"),
    ("M48", "runner/episode_runner.py", "agent_error = type(e).__name__\n        while not done:", "raise\n        while not done:",
     "agent crash during reset not contained"),
]


def run(mutant, scratch):
    mid, rel, old, new, desc = mutant
    target = scratch / "src" / "traceforge" / rel
    text = target.read_text(encoding="utf-8")
    if old not in text:
        return {"id": mid, "description": desc, "status": "INVALID (pattern not found)"}
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-rf", "-o", "addopts=", "-p", "no:cacheprovider",
                            "--tb=no", "-W", "ignore"], cwd=scratch, capture_output=True, text=True, timeout=300)
        killed = p.returncode != 0
        tail = [l for l in p.stdout.splitlines() if l.startswith("FAILED") or "passed" in l or "failed" in l]
        by = tail[0].split(" - ")[0].replace("FAILED ", "") if tail and tail[0].startswith("FAILED") else ""
    except subprocess.TimeoutExpired:
        killed, by = True, "timeout (infinite loop = detected)"
    finally:
        target.write_text(text, encoding="utf-8")
    return {"id": mid, "file": rel, "description": desc, "status": "KILLED" if killed else "SURVIVED",
            "killed_by": by, "seconds": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/mutation_check.json")
    ap.add_argument("--only", nargs="*")
    ns = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "proj"
        top_skip = {".git", "reports", "trajectories", "graphify-out", "tools"}
        skip = shutil.ignore_patterns("__pycache__", "*.egg-info", ".pytest_cache")
        shutil.copytree(ROOT, scratch, ignore=lambda d, names: set(skip(d, names)) | (
            top_skip & set(names) if Path(d) == ROOT else set()))
        results = []
        for m in MUTANTS:
            if ns.only and m[0] not in ns.only:
                continue
            r = run(m, scratch)
            results.append(r)
            print(f"{r['id']} {r['status']:<9} {r['description']}  {r.get('killed_by', '')}", flush=True)
    killed = sum(r["status"] == "KILLED" for r in results)
    summary = {"mutants": len(results), "killed": killed,
               "survived": [r["id"] for r in results if r["status"] == "SURVIVED"],
               "invalid": [r["id"] for r in results if r["status"].startswith("INVALID")], "results": results}
    out = ROOT / ns.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nkilled {killed}/{len(results)}; survived: {summary['survived']}; invalid: {summary['invalid']}")


if __name__ == "__main__":
    main()
