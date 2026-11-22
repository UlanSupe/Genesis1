"""Command line interface: generate | verify | demo | evaluate | inspect."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PRESETS
from .constants import ISSUE_TYPES, SCENARIOS
from .exceptions import TraceForgeError


def _config_arg(ns) -> dict:
    from .config import PRESETS
    cfg = json.loads(Path(ns.config).read_text(encoding="utf-8")) if getattr(ns, "config", None) else {}
    cfg = {k: v for k, v in cfg.items() if v is not None}
    if getattr(ns, "preset", None):
        cfg = {**PRESETS[ns.preset], **cfg}
    for key in ("difficulty", "base_currency"):
        if getattr(ns, key, None):
            cfg[key] = getattr(ns, key)
    if getattr(ns, "scenario", None):
        cfg["scenarios"] = [ns.scenario]
    if getattr(ns, "clean", False):
        cfg["clean"] = True
    return cfg


def cmd_generate(ns) -> int:
    from .episode_io import save_episode
    from .generator.builder import generate_dataset, generate_episode
    if ns.seed is not None:                       # single episode
        ep = generate_episode(ns.seed, _config_arg(ns))
        dest = save_episode(ep, Path(ns.out) / (ns.name or f"episode_seed{ns.seed}"))
        print(f"episode {ep.public.manifest['episode_id']}: {len(ep.public.ledger)} ledger rows, "
              f"{ep.public.manifest['document_count']} documents -> {dest}")
        return 0
    splits = ("train", "dev") if ns.split == "all" else (ns.split,)
    index = generate_dataset(ns.out, splits)
    for split, names in index.items():
        print(f"{split}: {len(names)} episodes -> {Path(ns.out) / split}")
    return 0


def cmd_verify(ns) -> int:
    from .reporting.verification_report import render, run_verification
    report = run_verification(Path(ns.root), run_tests=not ns.skip_tests, out_dir=ns.out)
    print(render(report))
    print(f"report written to {Path(ns.out) / 'verification_report.md'}")
    return 0 if report["ok"] else 1


def cmd_pairs(ns) -> int:
    from .generator.pairs import build_pairs
    splits = ("train", "dev") if ns.split == "all" else (ns.split,)
    for sp in splits:
        r = build_pairs(ns.source, sp, ns.out, ns.seed)
        print(f"{sp}: {r['pairs']} paired transformations -> {r['out']}")
    return 0


def cmd_paired_eval(ns) -> int:
    from .runner.episode_runner import AGENT_NAMES
    from .runner.paired import render, run_paired, write_paired
    policies = AGENT_NAMES if ns.policy == "all" else (ns.policy,)
    result = run_paired(ns.source, ns.pairs, ns.split, policies, mode=ns.mode, isolated=ns.isolated, policy_seed=ns.seed)
    j, c = write_paired(result, ns.out, ns.name)
    print(render(result))
    print()
    print(f"{len(result['rows'])} paired runs -> {j} , {c}")
    return 0


def cmd_generate_eval(ns) -> int:
    from .generator.pairs import generate_eval
    res = generate_eval(ns.out, ns.count, ns.secret, not ns.no_pairs)
    print(f"{res['episodes']} hidden evaluation episodes -> {res['out']}")
    if "pairs" in res:
        print(f"{res['pairs']['pairs']} paired transformations -> {res['pairs']['out']}")
    if ns.secret_out:
        Path(ns.secret_out).write_text(res["secret"] + "\n", encoding="utf-8")
        print(f"secret written to {ns.secret_out}  (keep it private; the same secret regenerates the same set)")
    elif not ns.secret:
        print(f"SECRET (shown once, store it privately): {res['secret']}")
    return 0


def cmd_audit(ns) -> int:
    from .reporting import audit
    report = audit.run_audit(ns.out, performance=not ns.skip_performance)
    print(audit.render(report))
    if not ns.skip_performance:
        from .reporting.performance import render
        print("\n" + render(next(s for s in report["sections"] if s["name"] == "performance")["data"]))
    print(f"\nreport written to {Path(ns.out) / 'audit_report.json'}")
    return 0 if report["ok"] else 1


def cmd_benchmark(ns) -> int:
    from .runner.benchmark import render_summary, run_benchmark, write_benchmark
    from .runner.episode_runner import AGENT_NAMES
    policies = AGENT_NAMES if ns.policy == "all" else (ns.policy,)
    suites = ("dev", "train", "presets") if ns.suite == "all" else (ns.suite,)
    result = run_benchmark(suites, policies, datasets=ns.datasets, seeds=ns.seeds, mode=ns.mode,
                           isolated=ns.isolated, policy_seed=ns.seed)
    j, c = write_benchmark(result, ns.out, ns.name)
    print(render_summary(result))
    print(f"\n{len(result['rows'])} episode runs -> {j} , {c}")
    return 0


def cmd_ablation(ns) -> int:
    from .runner.ablation import render, run_ablation, write_ablation
    splits = ("dev", "train") if ns.split == "all" else (ns.split,)
    result = run_ablation(ns.datasets, splits, isolated=ns.isolated)
    print(render(result))
    j, c = write_ablation(result, ns.out, ns.name)
    print(f"-> {j} , {c}")
    return 0


def cmd_play(ns) -> int:
    from .runner import play
    if ns.cmd == "start":
        print(play.start(ns.log, ns.seed, _config_arg(ns)))
    elif ns.cmd == "act":
        print(play.act(ns.log, ns.json))
    else:
        print(json.dumps(play.result(ns.log), indent=2))
    return 0


def cmd_demo(ns) -> int:
    from .reporting.demo_report import DEMO_SEED, run_demo
    cfg = _config_arg(ns)
    if ns.episode:
        if not (Path(ns.episode) / "public" / "manifest.json").exists():
            raise TraceForgeError(f"episode directory not found or incomplete: {ns.episode} (expected <dir>/public/manifest.json)")
        cfg = {"episode_dir": ns.episode}
    color = True if ns.color else (False if ns.no_color else None)
    out = None if ns.no_report else ns.report_dir
    run_demo(ns.policy, DEMO_SEED if ns.seed is None else ns.seed, cfg or None, ns.isolated, ns.instructor,
             verify=not ns.no_verify, color=color, report_dir=out, traj_dir=None if ns.no_report else ns.trajectories,
             benchmark=not ns.no_benchmark, datasets=ns.datasets)
    return 0


def cmd_evaluate(ns) -> int:
    from .evaluation.metrics import summarize
    from .runner.episode_runner import AGENT_NAMES, run_split
    policies = AGENT_NAMES if ns.policy == "all" else (ns.policy,)
    splits = ("train", "dev") if ns.split == "all" else (ns.split,)
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'split':<6} {'policy':<15} {'eps':>3} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6} "
          f"{'reward':>7} {'steps':>6}")
    allres = []
    for split in splits:
        for pol in policies:
            res = run_split(ns.datasets, split, pol, mode=ns.mode, isolated=ns.isolated and pol != "oracle",
                            traj_dir=None if ns.no_trajectories else ns.trajectories, seed=ns.seed)
            s = summarize(res)
            print(f"{split:<6} {pol:<15} {s['episodes']:>3} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} "
                  f"{s['mean_f1']:>6.3f} {s['mean_reward']:>7.3f} {s['mean_inspection_steps']:>6.1f}")
            allres.append({"split": split, "policy": pol, "summary": s, "episodes": res})
    (out / "evaluation.json").write_text(json.dumps(allres, indent=2), encoding="utf-8")
    print(f"details: {out / 'evaluation.json'}")
    return 0


def cmd_inspect(ns) -> int:
    from .episode_io import load_episode
    ep = load_episode(ns.path, private=ns.instructor)
    pub = ep.public
    m = pub.manifest
    print(json.dumps(m, indent=2))
    print(f"\nledger rows: {len(pub.ledger)}  invoices: {len(pub.invoices)}  payments: "
          f"{len(pub.payments)}  fx rows: {len(pub.fx)}")
    rows = {"ledger": pub.ledger, "invoices": pub.invoices, "payments": pub.payments, "fx": pub.fx}[ns.table]
    print(f"\nfirst {ns.limit} rows of {ns.table}:")
    for r in rows[:ns.limit]:
        print(" ", json.dumps(r, sort_keys=True))
    if ns.instructor:
        print("\n*** INSTRUCTOR MODE: private reference (never give this to an agent) ***")
        for t in ep.private.truth:
            print(" ", json.dumps(t, sort_keys=True))
        if not ep.private.truth:
            print("  (clean episode)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="traceforge", description="TraceForge: evidence-first financial audit gym")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate the train/dev datasets or a single episode")
    g.add_argument("--out", default="datasets")
    g.add_argument("--split", choices=("train", "dev", "all"), default="all")
    g.add_argument("--seed", type=int, help="generate ONE episode with this seed instead of a dataset")
    g.add_argument("--name", help="folder name for --seed mode")
    g.add_argument("--config", help="JSON file with generator config")
    g.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    g.add_argument("--base-currency", dest="base_currency")
    g.add_argument("--scenario", choices=SCENARIOS)
    g.add_argument("--preset", choices=sorted(PRESETS), help="easy|medium|hard|compound")
    g.add_argument("--clean", action="store_true")
    g.set_defaults(fn=cmd_generate)

    pr = sub.add_parser("pairs", help="build the paired (metamorphic) twin of every episode of a split")
    pr.add_argument("--source", default="datasets")
    pr.add_argument("--split", choices=("train", "dev", "eval", "all"), default="all")
    pr.add_argument("--out", default="datasets/pairs")
    pr.add_argument("--seed", type=int, default=0)
    pr.set_defaults(fn=cmd_pairs)

    pe = sub.add_parser("paired-eval", help="run policies on originals and their twins; report robustness to re-labelling/re-ordering")
    pe.add_argument("--source", default="datasets")
    pe.add_argument("--pairs", default="datasets/pairs")
    pe.add_argument("--split", default="dev")
    pe.add_argument("--policy", default="all", help="all | random | reconciliation | mark_all | oracle | an ablation name (e.g. naive_clock)")
    pe.add_argument("--mode", choices=("train", "eval"), default="eval")
    pe.add_argument("--isolated", action="store_true")
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--out", default="reports")
    pe.add_argument("--name", default="paired_eval")
    pe.set_defaults(fn=cmd_paired_eval)

    ge = sub.add_parser("generate-eval", help="organizer tool: hidden evaluation episodes from a SECRET (+ their paired twins)")
    ge.add_argument("--out", default="eval_hidden")
    ge.add_argument("--count", type=int, default=20)
    ge.add_argument("--secret", help="organizer secret (default: random 128-bit, printed once)")
    ge.add_argument("--secret-out", help="write the secret to this file instead of printing it")
    ge.add_argument("--no-pairs", action="store_true")
    ge.set_defaults(fn=cmd_generate_eval)

    a = sub.add_parser("audit", help="automated audit of architecture, data quality, tells, fuzzing, performance")
    a.add_argument("--out", default="reports")
    a.add_argument("--skip-performance", action="store_true")
    a.set_defaults(fn=cmd_audit)

    b = sub.add_parser("benchmark", help="run baseline policies on identical episodes; write JSON and CSV")
    b.add_argument("--suite", choices=("dev", "train", "presets", "all"), default="all")
    b.add_argument("--policy", default="all", choices=("all", "random", "reconciliation", "mark_all", "oracle"))
    b.add_argument("--seeds", type=int, default=5, help="episodes per preset in the presets suite")
    b.add_argument("--mode", choices=("train", "eval"), default="eval")
    b.add_argument("--datasets", default="datasets")
    b.add_argument("--out", default="reports")
    b.add_argument("--isolated", action="store_true")
    b.add_argument("--seed", type=int, default=0, help="seed of the random policy")
    b.add_argument("--name", default="benchmark", help="output file stem (e.g. FINAL_BENCHMARK)")
    b.set_defaults(fn=cmd_benchmark)

    pl = sub.add_parser("play", help="one step per command for a human/LLM agent (eval mode, deterministic replay)")
    pl.add_argument("cmd", choices=("start", "act", "result"))
    pl.add_argument("--log", required=True)
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--preset", choices=("easy", "medium", "hard", "compound"))
    pl.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    pl.add_argument("--json", default="{}", help="the action as a JSON object (for `act`)")
    pl.set_defaults(fn=cmd_play)

    ab = sub.add_parser("ablation", help="reward-sensitivity study: the auditor with one capability removed (real runs)")
    ab.add_argument("--split", choices=("dev", "train", "all"), default="all")
    ab.add_argument("--datasets", default="datasets")
    ab.add_argument("--out", default="reports")
    ab.add_argument("--name", default="ablation")
    ab.add_argument("--isolated", action="store_true")
    ab.set_defaults(fn=cmd_ablation)

    v = sub.add_parser("verify", help="run tests and end-to-end checks, write a report")
    v.add_argument("--root", default=".")
    v.add_argument("--out", default="reports")
    v.add_argument("--skip-tests", action="store_true")
    v.set_defaults(fn=cmd_verify)

    d = sub.add_parser("demo", help="fixed-seed demonstration: company, agent investigation, evaluation, benchmark, HTML report")
    d.add_argument("--policy", default="reconciliation", choices=("random", "reconciliation", "mark_all", "oracle"))
    d.add_argument("--seed", type=int, default=None, help="default: the fixed demo seed (7)")
    d.add_argument("--episode", help="stored episode directory instead of generating")
    d.add_argument("--config")
    d.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    d.add_argument("--base-currency", dest="base_currency")
    d.add_argument("--scenario", choices=SCENARIOS)
    d.add_argument("--preset", choices=sorted(PRESETS))
    d.add_argument("--clean", action="store_true")
    d.add_argument("--isolated", action="store_true", help="run the agent in a separate process")
    d.add_argument("--instructor", action="store_true", help="also print the private reference")
    d.add_argument("--no-verify", action="store_true", help="skip the agent's tool-based evidence verification phase")
    d.add_argument("--no-benchmark", action="store_true", help="skip the baseline comparison")
    d.add_argument("--no-report", action="store_true", help="do not write the HTML/JSON report or the trajectory")
    d.add_argument("--report-dir", default="reports")
    d.add_argument("--trajectories", default="trajectories")
    d.add_argument("--datasets", default="datasets")
    g2 = d.add_mutually_exclusive_group()
    g2.add_argument("--color", action="store_true", help="force ANSI colour")
    g2.add_argument("--no-color", action="store_true", help="plain text (also: NO_COLOR environment variable)")
    d.set_defaults(fn=cmd_demo)

    e = sub.add_parser("evaluate", help="evaluate policies on the stored datasets")
    e.add_argument("--policy", default="all", choices=("all", "random", "reconciliation", "mark_all", "oracle"))
    e.add_argument("--split", choices=("train", "dev", "all"), default="dev")
    e.add_argument("--mode", choices=("train", "eval"), default="eval")
    e.add_argument("--datasets", default="datasets")
    e.add_argument("--trajectories", default="trajectories")
    e.add_argument("--no-trajectories", action="store_true")
    e.add_argument("--out", default="reports")
    e.add_argument("--isolated", action="store_true")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(fn=cmd_evaluate)

    i = sub.add_parser("inspect", help="look at a stored episode (public view by default)")
    i.add_argument("path")
    i.add_argument("--table", choices=("ledger", "invoices", "payments", "fx"), default="ledger")
    i.add_argument("--limit", type=int, default=5)
    i.add_argument("--instructor", action="store_true", help="also show private truth")
    i.set_defaults(fn=cmd_inspect)
    return ap


def run(argv=None) -> int:
    ns = build_parser().parse_args(argv)
    try:
        return ns.fn(ns)
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename}", file=sys.stderr)
        return 2
    except TraceForgeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


main = run
