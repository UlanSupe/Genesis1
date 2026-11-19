"""Self-contained HTML report (no scripts needed, no external assets, standard library only).

Every number and chart is rendered from the `result` dict produced by a real `demo` run
(its episode metrics and its real benchmark run). Nothing is hard-coded or placeholder; a JSON
data island with the same data is embedded so the figures can be audited."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

POLICY_ORDER = ["random", "reconciliation", "mark_all", "oracle"]
POLICY_NOTE = {"random": "random exploration, random guesses", "reconciliation": "rule-based reconciliation of all tables",
               "mark_all": "accuses the most suspicious documents of every type", "oracle": "evaluator-only: reads the truth"}
CATEGORIES = [
    ("duplicate", "Journal of one document is posted twice", "repost", "document + the two identical anchor lines"),
    ("allocation", "A payment is not (fully) applied to the invoices it names", "unallocated / partial / wrong_invoice",
     "payment document + changed ledger lines + the affected invoice documents"),
    ("fx", "Wrong exchange rate or the rate of the wrong date", "wrong_rate / wrong_date",
     "document + a ledger line + the fx record the policy prescribes"),
    ("cutoff", "Journal booked in a different reporting period than its document", "period_shift",
     "document + a mis-dated ledger line"),
]
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2430;--mute:#5d6b7c;--line:#dfe3e8;--accent:#2457c5;--good:#1f8a4c;--bad:#c23b3b;--warn:#b7791f}
@media (prefers-color-scheme: dark){:root{--bg:#10151c;--card:#18202a;--ink:#e6ebf1;--mute:#93a1b3;--line:#2a3441;--accent:#7aa2ff;--good:#4cc27d;--bad:#ff7b7b;--warn:#f0b455}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:15px;margin:18px 0 6px}.sub{color:var(--mute);margin:0 0 14px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.chips span{display:inline-block;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:2px 10px;margin:2px 6px 2px 0;font-size:13px;color:var(--mute)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.metric{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.metric b{display:block;font-size:24px}.metric small{color:var(--mute)}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:14px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:rgba(127,127,127,.08);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.tp{color:var(--good);font-weight:600}.fp{color:var(--bad);font-weight:600}code,.mono{font-family:ui-monospace,Consolas,monospace;font-size:13px}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin:10px 0}.box{flex:1 1 200px;border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:var(--card)}
.box.priv{border-color:var(--bad)}.box.pub{border-color:var(--good)}.box b{display:block;margin-bottom:4px}.arrow{align-self:center;color:var(--mute);font-size:20px}
.steps{display:flex;flex-wrap:wrap;gap:3px}.st{font-size:12px;padding:1px 6px;border-radius:4px;border:1px solid var(--line);background:var(--card)}.st.list_records{border-color:var(--accent)}.st.get_document,.st.get_record{border-color:var(--good)}.st.calculator{border-color:var(--warn)}.st.submit{background:var(--accent);color:#fff;border-color:var(--accent)}
svg text{fill:var(--ink);font:12px system-ui,sans-serif}svg .mute{fill:var(--mute)}.legend{color:var(--mute);font-size:13px}.warn{border-color:var(--warn);background:rgba(183,121,31,.08)}
.tw{overflow-x:auto}@media (max-width:720px){.arrow{display:none}.flow .box{flex-basis:100%}h1{font-size:23px}}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;font-size:13px}
"""


def _e(x) -> str:
    return escape(str(x), quote=True)


def _f(x, d=3) -> str:
    return f"{x:.{d}f}"


def hbar_chart(title: str, data: list, lo: float, hi: float, fmt=lambda v: f"{v:.3f}", color="var(--accent)") -> str:
    """Horizontal bars for [(label, value)] on the axis [lo, hi] (a zero line is drawn when lo < 0)."""
    w, row, left, right = 560, 26, 130, 70
    plot = w - left - right
    zero = left + plot * (0 - lo) / (hi - lo)
    h = row * len(data) + 28
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{_e(title)}">',
           f'<text x="0" y="14" style="font-weight:600">{_e(title)}</text>']
    for i, (label, v) in enumerate(data):
        y = 24 + i * row
        x0 = left + plot * (min(v, 0) - lo) / (hi - lo)
        x1 = left + plot * (max(v, 0) - lo) / (hi - lo)
        c = "var(--bad)" if v < 0 else color
        out.append(f'<text x="0" y="{y + 14}">{_e(label)}</text>')
        out.append(f'<rect x="{x0:.1f}" y="{y}" width="{max(x1 - x0, 1):.1f}" height="{row - 8}" rx="3" fill="{c}"/>')
        out.append(f'<text x="{max(x1, zero) + 6:.1f}" y="{y + 13}" class="mute">{_e(fmt(v))}</text>')
    if lo < 0:
        out.append(f'<line x1="{zero:.1f}" y1="20" x2="{zero:.1f}" y2="{h}" stroke="var(--mute)" stroke-width="1"/>')
    out.append("</svg>")
    return "".join(out)


def _table(headers: list, rows: list, num: tuple = ()) -> str:
    th = "".join(f'<th class="{"n" if i in num else ""}">{_e(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"n" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


def render_html(r: dict) -> str:
    ev, m, fa = r["evaluation"], r["manifest"], r["facts"]
    bench = r.get("benchmark")
    steps_html = "".join(f'<span class="st {_e(s["tool"])}" title="{_e(s["summary"])}">{s["step"]}&nbsp;{_e(s["tool"])}</span>' for s in r["steps"])
    usage = [(k, v) for k, v in sorted(r["tool_usage"].items(), key=lambda kv: -kv[1])]
    p = []
    p.append(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>TraceForge demo report</title><style>{CSS}</style></head><body><main>")
    p.append("<h1>TraceForge - evidence-first financial audit gym</h1>"
             "<p class='sub'>Demonstration report generated from a real run: every number and chart below comes from "
             "that execution.</p>")
    p.append("<div class='chips'>" + "".join(f"<span>{_e(c)}</span>" for c in (
        f"seed {r['seed']}", f"episode {r['episode_id']}", f"code v{r['version']}", f"python {r['python']}",
        f"agent: {r['agent']['policy']}{' + verification' if r['agent']['verify'] else ''}", "mode: eval")) + "</div>")

    p.append("<h2>1. What this is</h2><div class='card'><p><b>TraceForge</b> is an offline, reproducible environment for studying and "
             "training AI agents that perform <b>financial audit</b>. It generates a synthetic company with a consistent history "
             "(invoices, payments, partial payments, refunds, multi-currency FX, reporting periods), injects verifiable accounting errors, "
             "gives the agent a narrow set of investigation tools and scores its structured findings against a hidden ground truth "
             "with strict one-to-one matching.</p><p>The problem matters because a wrong accusation is as costly as a missed error: "
             "the reward punishes false alarms, and legitimate look-alikes (refunds, repeated amounts, split payments, boundary-time "
             "payments from other time zones) are built into every company.</p></div>")

    p.append("<h2>2. Architecture</h2><div class='flow'>"
             "<div class='box'><b>Generator</b>seed + config &rarr; clean company &rarr; independent validation &rarr; injected anomalies</div><div class='arrow'>&rarr;</div>"
             "<div class='box priv'><b>Private (evaluator process)</b>truth, provenance, clean baseline; matcher, reward</div><div class='arrow'>&harr;</div>"
             "<div class='box'><b>Environment</b>reset / step; 5 tools; limits; JSON observations only</div><div class='arrow'>&harr;</div>"
             "<div class='box pub'><b>Agent (separate process)</b>sees public data through tools; submits findings</div></div>"
             "<p class='legend'>Red = never visible to the agent. Green = what the agent sees. See ARCHITECTURE.md for the full diagram.</p>")

    p.append("<h2>3. Generated scenario</h2>")
    p.append("<div class='cards'>" + "".join(f"<div class='metric'><b>{_e(v)}</b><small>{_e(k)}</small></div>" for k, v in (
        ("ledger lines", m["record_count"]), ("documents", m["document_count"]), ("invoices", fa["invoices"]), ("payments", fa["payments"]),
        ("counterparties", fa["entities"]), ("refunds (decoys)", fa["refunds"]), ("split payments", fa["split_payments"]),
        ("fx rates", fa["fx_rows"]))) + "</div>")
    p.append(f"<p class='sub'>Base currency {_e(m['base_currency'])}, time zone {_e(m['timezone'])}, period {_e(m['period_start'][:10])} "
             f"to {_e(m['period_end'][:10])} (end exclusive). Documents by currency: "
             f"{_e(', '.join(f'{c}: {n}' for c, n in fa['currency_docs'].items()))}. Configuration: <code>{_e(json.dumps(r['config'], sort_keys=True))}</code></p>")

    p.append("<h2>4. Anomaly categories</h2>" + _table(["type", "what goes wrong", "variants", "evidence an auditor must cite"],
             [[f"<code>{_e(a)}</code>", _e(b), _e(c), _e(d)] for a, b, c, d in CATEGORIES]) +
             "<p class='legend'>Two compound scenarios combine them: <i>cross_border_partial</i> (a mis-allocated partial payment + a wrong-date FX rate on a foreign invoice) and "
             "<i>period_end_duplicate</i> (a reposted invoice + a lookalike invoice booked into the next period). "
             "Each anomaly is a separate truth record, scored independently. Difficulty levels easy/medium/hard change the variants and their subtlety.</p>")

    p.append("<h2>5. How the agent investigated</h2>")
    p.append(f"<div class='steps'>{steps_html}</div><p class='legend'>{len(r['steps'])} actions "
             f"({ev['inspection_steps']} inspection steps + submit); hover a step for its result.</p>")
    p.append(hbar_chart("Tool calls in this episode", usage, 0, max(v for _, v in usage) * 1.1, fmt=lambda v: f"{int(v)}"))
    rows = []
    for i, f in enumerate(r["findings"], 1):
        rows.append([i, f"<code>{_e(f['type'])}</code>", f"<span class='mono'>{_e(', '.join(f['affected_ids']))}</span>",
                     f"{f['claimed_amount_minor']} {_e(f['currency'])}", f"<span class='mono'>{_e(', '.join(f['evidence_ids']))}</span>",
                     f"<span class='{'tp' if f['status'] == 'TP' else 'fp'}'>{'TRUE POSITIVE' if f['status'] == 'TP' else 'FALSE POSITIVE'}</span>"])
    p.append("<h3>Findings, evidence and evaluator verdict</h3>" + _table(["#", "type", "affected ids", "amount", "evidence ids", "evaluator"], rows, (0, 3)))
    p.append("<p class='legend'>The evaluator accepts a finding only if type, affected ids (as a set), amount (+-1 minor unit), currency and a complete allowed evidence set all match, "
             "and each truth record can be claimed once (maximum bipartite matching).</p>")

    p.append("<h2>6. Evaluation of this episode</h2><div class='cards'>" + "".join(
        f"<div class='metric'><b>{_e(v)}</b><small>{_e(k)}</small></div>" for k, v in (
            ("precision", _f(ev["precision"])), ("recall", _f(ev["recall"])), ("F1", _f(ev["f1"])),
            ("reward", _f(ev["reward"])), ("inspection steps", ev["inspection_steps"]),
            ("TP / FP / FN", f"{ev['tp']} / {ev['fp']} / {ev['fn']}"))) + "</div>"
             f"<p class='sub'>reward = clip(2&middot;F1 &minus; 1 &minus; 0.002&middot;steps, &minus;1, 1) = 2&middot;{_f(ev['f1'])} &minus; 1 &minus; 0.002&middot;{ev['inspection_steps']} = <b>{_f(ev['reward'])}</b></p>")

    if bench:
        ov = bench["overall"]
        order = [x for x in POLICY_ORDER if x in ov]
        p.append(f"<h2>7. Baseline policy comparison (real benchmark run)</h2><p class='sub'>{bench['episodes']} episodes "
                 f"({_e(' + '.join(bench['suites']))}; presets: {bench['seeds_per_preset']} seeds each), {bench['runs']} runs, all policies on identical episodes, eval mode. "
                 f"Means over episodes; the oracle runs on the evaluator side only and is never exposed to an agent.</p>")
        p.append("<div class='card'>" + hbar_chart("F1 (mean)", [(x, ov[x]["f1"]) for x in order], 0, 1) +
                 hbar_chart("Reward (mean)", [(x, ov[x]["reward"]) for x in order], -1, 1) +
                 hbar_chart("Precision (mean)", [(x, ov[x]["precision"]) for x in order], 0, 1) +
                 hbar_chart("Recall (mean)", [(x, ov[x]["recall"]) for x in order], 0, 1) +
                 hbar_chart("Inspection steps (mean)", [(x, ov[x]["inspection_steps"]) for x in order], 0,
                            max(ov[x]["inspection_steps"] for x in order) * 1.15 or 1, fmt=lambda v: f"{v:.1f}") + "</div>")
        p.append(_table(["policy", "idea", "episodes", "TP", "FP", "FN", "precision", "recall", "F1", "reward", "inspection steps"],
                        [[f"<b>{_e(x)}</b>", _e(POLICY_NOTE[x]), ov[x]["episodes"], ov[x]["tp"], ov[x]["fp"], ov[x]["fn"], _f(ov[x]["precision"]),
                          _f(ov[x]["recall"]), _f(ov[x]["f1"]), _f(ov[x]["reward"]), f"{ov[x]['inspection_steps']:.1f}"] for x in order], tuple(range(2, 11))))
        lv = bench["by_level"]
        if lv:
            p.append("<h3>By difficulty level</h3>" + _table(["level", "policy", "episodes", "issues", "TP", "FP", "FN", "F1", "reward", "steps"],
                     [[_e(x["level"]), _e(x["policy"]), x["episodes"], x["issues"], x["tp"], x["fp"], x["fn"], _f(x["mean_f1"]), _f(x["mean_reward"]),
                       f"{x['mean_inspection_steps']:.1f}"] for x in lv if x["policy"] in ("reconciliation", "mark_all", "random", "oracle")], tuple(range(2, 10))))
        p.append("<div class='card warn'><b>Reading this honestly.</b> The reconciliation baseline is a transparent rule-based auditor; because every error is "
                 "derivable from public data (enforced by an independent validator), it can solve the episodes. The figures say nothing about "
                 "LLM agents. Harder levels cost it more inspection steps, which lowers its reward but not its accuracy.</div>")

    p.append("<h2>8. Reproducibility</h2>" + _table(["item", "value"], [
        ["command", f"<code>{_e(r['reproduce'])}</code>"], ["seed", r["seed"]], ["configuration", f"<code>{_e(json.dumps(r['config'], sort_keys=True))}</code>"],
        ["episode id", _e(r["episode_id"])], ["public data sha256", f"<code>{_e(r['public_data_sha256'])}</code>"],
        ["code version / python / platform", _e(f"{r['version']} / {r['python']} / {r['platform']}")],
        ["agent trajectory (agent view only)", _e(r.get("trajectory") or "not recorded")],
        ["run time", _e(f"{r['runtime_seconds']} s (including the benchmark)")]]) +
        "<p class='legend'>The same seed, configuration and code produce byte-identical episodes (checked by tests). Run time is the only non-deterministic figure.</p>")

    if r.get("instructor"):
        rows = [[_e(x["issue_id"]), f"<code>{_e(x['type'])}</code>", f"<span class='mono'>{_e(', '.join(x['affected_ids']))}</span>", x["expected_amount_minor"],
                 _e(x["root_cause"]), _e(x["explanation"]), "FOUND" if x["found"] else "MISSED"] for x in r["instructor"]]
        p.append("<h2>Instructor section - hidden ground truth</h2><div class='card warn'>Generated with <code>--instructor</code>. "
                 "This is the private reference; it is never available to the agent.</div>" +
                 _table(["issue", "type", "affected ids", "amount", "root cause", "explanation", "result"], rows, (3,)))

    data = {"episode": {k: r[k] for k in ("seed", "episode_id", "config", "evaluation", "tool_usage")},
            "benchmark": None if not bench else {"overall": bench["overall"], "summary": bench["summary"], "by_level": bench["by_level"]}}
    p.append("<script type='application/json' id='report-data'>" + _island(data) + "</script>")
    p.append("</main></body></html>")
    return "".join(p)


def _island(data: dict) -> str:
    """JSON for a <script type=application/json> block: <, > and & become unicode escapes (still valid JSON)."""
    return json.dumps(data, sort_keys=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def write_report(result: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result), encoding="utf-8", newline="\n")
    return path
