"""`traceforge demo`: one reproducible, fully real walk-through.

Generates a synthetic company (fixed seed), lets an auditor agent investigate it ONLY through
the public tools, then lets the evaluator compare the findings with the hidden ground truth.
Everything printed and written (terminal, JSON, HTML, trajectory) comes from this execution.

The private reference is never shown in the normal run. `instructor=True` adds a clearly
separated section with the truth (never visible to the agent)."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .. import __version__
from ..constants import ACTIONS, MAX_PAGE, MAX_STEPS, STEP_COST
from ..config import GeneratorConfig
from ..environment.audit_env import AuditEnv
from ..environment.isolation import assert_no_leak
from ..episode_io import public_files
from ..runner.episode_runner import make_agent
from ..trajectories.recorder import TrajectoryRecorder
from .terminal import Term

DEMO_SEED = 7
# Two compound scenarios = four interacting anomalies, medium difficulty (cross-referencing needed).
DEMO_CONFIG = {"difficulty": "medium", "base_currency": "EUR",
               "scenarios": ["cross_border_partial", "period_end_duplicate"], "episode_id": "demo-001"}
TOTAL_SECTIONS = 6

TOOL_PURPOSE = {
    "list_records": "page through a table (ledger, invoices, payments, fx); <= 50 rows per page",
    "get_record": "fetch one record by its id",
    "get_document": "fetch a source document (invoice or payment) by doc_id",
    "calculator": "safe Decimal arithmetic (AST whitelist, no code execution)",
    "submit": "send structured findings; ends the episode",
}


def _brief(action: dict, obs: dict) -> str:
    lr, name = obs["last_result"] or {}, action.get("action", "?")
    if obs["error"]:
        return f"ERROR {obs['error']['code']}"
    if name == "list_records":
        return f"{action.get('table')}: {len(lr['records'])} rows (total {lr['total']}, next {lr['next_cursor']})"
    if name == "get_record":
        rec = lr["record"]
        return f"{action.get('table')} {action.get('id')}" + (f" rate {rec['rate']}" if "rate" in rec else "")
    if name == "get_document":
        d = lr["document"]
        if "total_minor" in d:
            return f"invoice {d['invoice_id']} {d['currency']} total {d['total_minor']}"
        return f"payment {d['payment_id']} {d['currency']} amount {d['amount_minor']}, {len(d['invoice_refs'])} invoice ref(s)"
    if name == "calculator":
        return f"{action.get('expression')} = {lr.get('value')}"
    return str(lr.get("status"))


def _scenario_line(config: dict) -> str:
    if "episode_dir" in config:
        return "stored episode (configuration not shown)"
    cfg = GeneratorConfig.from_dict(config)
    parts = [f"difficulty {cfg.difficulty}"]
    if cfg.scenarios:
        parts.append("compound scenarios: " + ", ".join(cfg.scenarios) + " (two interacting anomalies each)")
    if cfg.issue_counts:
        parts.append("extra anomalies: " + ", ".join(f"{n}x {k}" for k, n in cfg.issue_counts.items() if n))
    if cfg.clean:
        parts.append("clean company (no anomalies)")
    return "; ".join(parts) + "  [configuration known to the evaluator, NOT to the agent]"


def _facts(pub) -> dict:
    inv, pay = pub.invoices, pub.payments
    ccy = Counter(d["currency"] for d in inv + pay)
    return {"entities": len({d["entity_id"] for d in inv}), "invoices": len(inv), "payments": len(pay),
            "ledger_rows": len(pub.ledger), "fx_rows": len(pub.fx), "currency_docs": dict(sorted(ccy.items())),
            "refunds": sum(p["amount_minor"] < 0 for p in pay), "split_payments": sum(len(p["invoice_refs"]) > 1 for p in pay),
            "partial_or_open_invoices": None}


def run_demo(policy: str = "reconciliation", seed: int = DEMO_SEED, config: dict | None = None, isolated: bool = False,
             instructor: bool = False, verify: bool = True, color: bool | None = None, term: Term | None = None,
             report_dir: str | Path | None = "reports", traj_dir: str | Path | None = "trajectories",
             benchmark: bool = True, datasets: str | Path = "datasets", benchmark_seeds: int = 3) -> dict:
    from ..exceptions import TraceForgeError
    if policy == "oracle" and isolated:
        raise TraceForgeError("the oracle is evaluator-side and cannot run in an isolated agent process")
    t = term or Term(color=color)
    started = time.perf_counter()
    config = dict(DEMO_CONFIG if config is None else config)
    env = AuditEnv("eval")
    obs, info = env.reset(seed, config)
    pub, manifest, pol_doc = env._pub, obs["manifest"], obs["policy"]
    tokens = env.leak_tokens()
    digest = hashlib.sha256("".join(k + v for k, v in sorted(public_files(pub).items())).encode()).hexdigest()
    facts = _facts(pub)
    use_verify = verify and policy == "reconciliation"
    agent_label = policy + (" + evidence verification" if use_verify else "")

    # ------------------------------------------------------------------ 1. company and data
    t.banner("TRACEFORGE - evidence-first financial audit gym",
             "An RL environment where an agent audits a synthetic company and is scored against a hidden ground truth.")
    t.section(1, TOTAL_SECTIONS, "Generated synthetic company (public view)")
    t.kv([("Episode", f"{manifest['episode_id']} (schema {manifest['schema_version']}), seed {seed}"),
          ("Base currency", f"{manifest['base_currency']}   company time zone {manifest['timezone']}"),
          ("Reporting period", f"{manifest['period_start'][:10]} .. {manifest['period_end'][:10]} (end exclusive)"),
          ("Counterparties", str(facts["entities"])),
          ("Documents", f"{manifest['document_count']} ({facts['invoices']} invoices, {facts['payments']} payments; "
                        f"{facts['refunds']} refunds, {facts['split_payments']} split payments)"),
          ("Ledger entries", f"{manifest['record_count']} double-entry lines, amounts in {manifest['base_currency']} minor units"),
          ("Currencies", ", ".join(f"{c} ({n} docs)" for c, n in facts["currency_docs"].items())),
          ("FX table", f"{facts['fx_rows']} daily rates"),
          ("Scenario", _scenario_line(config)),
          ("Ground truth", "hidden: kept only by the evaluator process (private/ folder), never sent to the agent")])
    t.write("")
    t.write("  Sample ledger lines:")
    t.table(["row_id", "timestamp", "acct", "debit", "credit", "ccy", "invoice", "payment"],
            [[r["row_id"], r["timestamp"][:16], r["account"], r["debit_minor"], r["credit_minor"], r["currency"],
              r["invoice_id"] or "-", r["payment_id"] or "-"] for r in pub.ledger[:4]], "llrrrlll")
    foreign = next((p for p in pub.payments if p["currency"] != manifest["base_currency"] and len(p["invoice_refs"]) >= 1), pub.payments[0])
    t.write("")
    t.write("  Sample payment document (source of truth for the allocation):")
    t.table(["doc_id", "payment_id", "timestamp", "ccy", "amount_minor", "invoice_refs"],
            [[foreign["doc_id"], foreign["payment_id"], foreign["timestamp"][:19], foreign["currency"], foreign["amount_minor"],
              ", ".join(f"{r['invoice_id']}:{r['allocated_minor']}" for r in foreign["invoice_refs"])]], "llllrl")

    # ------------------------------------------------------------------ 2. environment contract
    t.section(2, TOTAL_SECTIONS, "Environment contract the agent works within")
    t.table(["tool", "what it does"], [[a, TOOL_PURPOSE[a]] for a in ACTIONS], max_col=80)
    t.kv([("Limits", f"{MAX_STEPS} actions including submit, {MAX_PAGE} rows per page, errors still cost a step"),
          ("Reward", "clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1); one-to-one matching of findings to truth"),
          ("Mode", "eval: the agent never sees its reward; the evaluator reports it after the episode"),
          ("Agent", agent_label + (" (separate OS process)" if isolated else ""))])

    # ------------------------------------------------------------------ 3. investigation
    t.section(3, TOTAL_SECTIONS, "Agent investigation (every step is a real tool call)")
    agent = make_agent(policy, 0, isolated, use_verify)(env)
    steps, usage, findings, last_action = [], Counter(), [], None
    rec = TrajectoryRecorder(Path(traj_dir) / f"demo_seed{seed}.jsonl") if traj_dir else None
    traj_path = str(rec.path) if rec else None
    try:
        assert_no_leak(obs, tokens)
        if rec:
            rec.record(manifest["episode_id"], 0, None, obs, 0.0, False, False)
        agent.reset(obs)
        done, step = False, 0
        while not done:
            action = agent.act(obs)
            last_action = action
            obs, reward, term_, trunc, step_info = env.step(action)
            step += 1
            assert_no_leak([obs, step_info], tokens)
            if rec:
                rec.record(manifest["episode_id"], step, action, obs, reward, term_, trunc)
            name = action.get("action", "?")
            usage[name] += 1
            brief = _brief(action, obs)
            steps.append({"step": step, "tool": name, "summary": brief, "error": bool(obs["error"])})
            t.write(f"  {t.style(t.progress(step, MAX_STEPS), 'dim')}  {t.style(name.ljust(13), 'cyan')} {brief}")
            done = term_ or trunc
    finally:
        if hasattr(agent, "close"):
            agent.close()
        if rec:
            rec.close()
    if last_action["action"] == "submit":
        sub = last_action["findings"]
        findings = sub["findings"] if isinstance(sub, dict) else sub
    t.write("")
    t.table(["tool", "calls"], [[a, usage[a]] for a in ACTIONS if usage[a]], "lr")

    # ------------------------------------------------------------------ 4. findings
    res = env.result
    mt = res.metrics
    status = {m["finding"]: "MATCH" for m in res.matches}
    t.section(4, TOTAL_SECTIONS, f"SUBMITTED FINDINGS ({len(findings)}) and the evidence behind them")
    t.table(["#", "type", "affected ids", "amount", "ccy", "evidence ids"],
            [[i + 1, f["type"], ", ".join(f["affected_ids"]), f["claimed_amount_minor"], f["currency"],
              ", ".join(f["evidence_ids"])] for i, f in enumerate(findings)], "lllrll", max_col=44)
    for i, f in enumerate(findings):
        t.write(f"  {i + 1}. {f['explanation']}")

    # ------------------------------------------------------------------ 5. evaluation
    f1 = Fraction(mt.f1).limit_denominator(10 ** 6)
    t.section(5, TOTAL_SECTIONS, "Evaluator: one-to-one comparison with the hidden ground truth")
    t.table(["#", "type", "result"],
            [[i + 1, f["type"], t.ok("TRUE POSITIVE") if status.get(i) else t.bad("FALSE POSITIVE")] for i, f in enumerate(findings)])
    t.write(f"  injected issues that were NOT found (false negatives): {mt.fn}")
    t.write("")
    t.table(["metric", "value"], [["true positives (TP)", mt.tp], ["false positives (FP)", mt.fp], ["false negatives (FN)", mt.fn],
                                  ["precision", f"{mt.precision:.3f}"], ["recall", f"{mt.recall:.3f}"], ["F1", f"{mt.f1:.3f}"],
                                  ["inspection steps", mt.inspection_steps], ["reward", f"{mt.reward:.3f}"]], "lr")
    t.write(f"  reward = clip(2*{mt.f1:.3f} - 1 - {float(STEP_COST):.3f}*{mt.inspection_steps}, -1, 1) = "
            + t.style(f"{mt.reward:.3f}", "bold", "green" if mt.reward > 0 else "red"))

    # ------------------------------------------------------------------ truth (instructor only)
    instructor_data = None
    if instructor:
        t.write("")
        t.write(t.style("*" * t.width, "yellow"))
        t.write(t.style("INSTRUCTOR MODE - PRIVATE REFERENCE (never visible to the agent)", "bold", "yellow"))
        t.write(t.style("*" * t.width, "yellow"))
        matched = {m["issue_id"] for m in res.matches}
        truth = env.evaluator_truth()
        t.table(["issue", "type", "affected ids", "amount", "root cause", "found?"],
                [[x["issue_id"], x["type"], ", ".join(x["affected_ids"]), x["expected_amount_minor"], x["root_cause"],
                  "FOUND" if x["issue_id"] in matched else "MISSED"] for x in truth], "lllrll", max_col=40)
        for x in truth:
            t.write(f"  {x['issue_id']}: {x['explanation']}")
        instructor_data = [{k: x[k] for k in ("issue_id", "type", "affected_ids", "expected_amount_minor", "currency",
                                              "root_cause", "explanation")} | {"found": x["issue_id"] in matched} for x in truth]

    # ------------------------------------------------------------------ 6. benchmark + outputs
    result = {
        "version": __version__, "python": platform.python_version(), "platform": platform.system(),
        "seed": seed, "config": config, "episode_id": manifest["episode_id"], "public_data_sha256": digest,
        "manifest": manifest, "facts": facts, "agent": {"policy": policy, "verify": use_verify, "isolated": isolated},
        "steps": steps, "tool_usage": dict(usage),
        "findings": [{**f, "status": "TP" if status.get(i) else "FP"} for i, f in enumerate(findings)],
        "evaluation": {"tp": mt.tp, "fp": mt.fp, "fn": mt.fn, "precision": mt.precision, "recall": mt.recall, "f1": mt.f1,
                       "reward": mt.reward, "inspection_steps": mt.inspection_steps, "valid": res.valid},
        "trajectory": traj_path,
        "reproduce": "python -m traceforge demo" if config == DEMO_CONFIG and seed == DEMO_SEED
        else f"python -m traceforge demo --seed {seed}",
    }
    t.section(6, TOTAL_SECTIONS, "Baseline policies on the same environment (real benchmark run)")
    if benchmark:
        from ..runner.benchmark import run_benchmark
        suites = ("dev", "presets") if (Path(datasets) / "dev" / "index.json").exists() else ("presets",)
        t0 = time.perf_counter()
        bench = run_benchmark(suites, datasets=datasets, seeds=benchmark_seeds)
        result["benchmark"] = {"suites": list(suites), "seeds_per_preset": benchmark_seeds, "episodes": len({(r["suite"], r["episode"]) for r in bench["rows"]}),
                               "runs": len(bench["rows"]), "summary": bench["summary"], "by_level": bench["by_level"],
                               "seconds": round(time.perf_counter() - t0, 2)}
        agg = {}
        for r in bench["rows"]:
            a = agg.setdefault(r["policy"], {"n": 0, "tp": 0, "fp": 0, "fn": 0, "p": 0.0, "r": 0.0, "f1": 0.0, "rw": 0.0, "st": 0})
            a["n"] += 1
            for k, v in (("tp", r["tp"]), ("fp", r["fp"]), ("fn", r["fn"]), ("p", r["precision"]), ("r", r["recall"]), ("f1", r["f1"]),
                         ("rw", r["reward"]), ("st", r["inspection_steps"])):
                a[k] += v
        result["benchmark"]["overall"] = {p: {"episodes": a["n"], "tp": a["tp"], "fp": a["fp"], "fn": a["fn"],
                                              "precision": a["p"] / a["n"], "recall": a["r"] / a["n"], "f1": a["f1"] / a["n"],
                                              "reward": a["rw"] / a["n"], "inspection_steps": a["st"] / a["n"]} for p, a in agg.items()}
        t.write(f"  {result['benchmark']['episodes']} episodes x 4 policies = {len(bench['rows'])} runs "
                f"(suites: {', '.join(suites)}; {result['benchmark']['seconds']}s). The oracle is evaluator-side only.")
        order = ["random", "reconciliation", "mark_all", "oracle"]
        t.table(["policy", "precision", "recall", "F1", "reward", "inspection steps"],
                [[p, f"{agg_['precision']:.3f}", f"{agg_['recall']:.3f}", f"{agg_['f1']:.3f}", f"{agg_['reward']:.3f}",
                  f"{agg_['inspection_steps']:.1f}"] for p in order for agg_ in [result["benchmark"]["overall"][p]]], "lrrrrr")
    else:
        t.write("  (skipped: --no-benchmark)")
    result["instructor_included"] = bool(instructor)
    if instructor:
        result["instructor"] = instructor_data
    result["runtime_seconds"] = round(time.perf_counter() - started, 2)

    outputs = []
    if report_dir:
        from .html_report import write_report
        outputs.append(("HTML report", str(write_report(result, Path(report_dir) / "demo_report.html"))))
        jp = Path(report_dir) / "demo_result.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        outputs.append(("JSON result", str(jp)))
    if traj_path:
        outputs.append(("Agent trajectory", traj_path))
    t.write("")
    t.write(t.style("Summary", "bold"))
    t.kv([("Verdict", f"{mt.tp} of {mt.tp + mt.fn} injected anomalies found, {mt.fp} false alarm(s); F1 {mt.f1:.3f}, "
                      f"reward {mt.reward:.3f} after {mt.inspection_steps} inspection steps"),
          ("Reproduce", result["reproduce"] + "   (same seed + config + code => identical episode and metrics)"),
          ("Public data sha256", digest[:32] + "...")] + outputs)
    return result
