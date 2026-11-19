# DEMO_SCRIPT — 3-minute presentation

**Preparation (once, ~15 s, offline):** `pip install -e .` (or `set PYTHONPATH=src`), then run the demo once so the files exist:

```bash
python -m traceforge demo
```

Open two windows: the terminal (a dark theme with colour looks best; `--no-color` works everywhere) and `reports/demo_report.html` in a browser.
Everything on screen is real: same seed → same company, same findings, same metrics (only the runtime differs).

---

## 0:00 – 0:30  The problem (browser: report §1 "What this is")

> "Auditing is detective work: a payment allocated to the wrong invoice, an invoice posted twice, an FX rate from the wrong day, a booking in the wrong period.
> Real data can't be shared and has no ground truth. **TraceForge** generates a synthetic company with a *consistent* history, injects errors we **know**, and scores an
> agent's findings against a hidden ground truth. A false accusation costs as much as a missed error — so the reward punishes both."

Point at: the four anomaly categories (§4) and the sentence about look-alike decoys (refunds, repeated amounts, split payments, boundary-time payments).

## 0:30 – 1:00  Architecture (browser: report §2, or ARCHITECTURE.md Mermaid diagram)

> "The truth lives only in the **evaluator process**. The agent is a separate process that sees public data only through five tools — list, get record, get document, a safe
> calculator and submit — with 100 steps, 50 rows per page. Every observation is leak-scanned. The oracle policy exists only on the evaluator side."

## 1:00 – 2:00  Live run (terminal)

```bash
python -m traceforge demo
```

Walk down the screen while it prints (it finishes in about two seconds):

1. **[1/6] Company** — "172 documents, 353 ledger lines, four currencies, seven counterparties; the line at the bottom says the ground truth is hidden." Mention the scenario line: *two compound scenarios — four interacting anomalies*.
2. **[3/6] Investigation** — "18 `list_records` pages to read the books, then it *verifies*: it opens four source documents, reads one FX record and uses the calculator to recompute a converted amount. 24 inspection steps in total."
3. **[4/6] Findings** — "Four findings, each with its evidence ids: a duplicate, a cut-off, an allocation, an FX error — the evidence is what the evaluator checks, not the explanation text."

## 2:00 – 2:30  The evaluator (terminal, sections 5)

> "One-to-one matching against the hidden truth: type, affected ids, amount within one minor unit, currency, and a complete evidence set must all match.
> **4 true positives, 0 false positives, 0 missed — precision, recall and F1 all 1.000.** Reward = 2·F1 − 1 − 0.002·steps = **0.952**; every extra look at the data costs reward."

Optional 10 seconds: `python -m traceforge demo --instructor` — "here is what the agent never sees": the four root causes, e.g. *journal posted twice*, *FX rate of the wrong date used*, and *booked in the wrong reporting period*.

## 2:30 – 3:00  Baselines, honesty, reproducibility (terminal section 6 + browser §7–8)

> "Same environment, four baseline policies, 22 episodes, 88 runs. **Random** F1 0.000, reward −1.000. **Mark-everything** accuses 659 times for 1 true hit: F1 0.003, reward −0.997.
> The **rule-based reconciliation** agent finds all 78 issues (F1 1.000, reward 0.963, 18.5 steps), and the evaluator-only **oracle** gets the maximum, 1.000 — proving the top score is reachable.
> To be clear: a rule-based solver *can* solve it, because every error is derivable from public data — that is what the independent validator guarantees. This measures an environment, not a model.
> Same seed, same config, same code: byte-identical episode — the SHA-256 is in the report, and 24 metamorphic pairs (renamed ids, shuffled records) confirm the answers survive."

Close with: `python -m traceforge verify` — "one command: the full test-suite, a spec-compliance matrix generated from real test results, the audit and the benchmark."

---

## Backup answers

* *"Could the agent peek at the answer?"* — No: separate process, scrubbed environment, empty working directory, leak scanner on every message; the oracle is not in the agent registry. A file-system-capable hostile agent could read stored `private/` folders — use in-memory episodes or a container (documented limitation).
* *"Is this just random errors?"* — No: each anomaly has a root cause, expected amount and evidence sets; an independent validator proves every episode is solvable from public data and that nothing else is wrong.
* *"How do you know the score isn't gameable by accusing everything?"* — `mark_all` does exactly that: F1 0.003.
* *"What's missing?"* — No LLM agent was run; four error families; fixed UTC offsets; see FINAL_ENGINEERING_REPORT.md §5.

## Commands cheat-sheet

```bash
python -m traceforge demo                  # the whole demonstration (+ reports/demo_report.html, trajectory)
python -m traceforge demo --instructor     # also show the hidden truth (clearly separated)
python -m traceforge demo --no-color       # plain text
python -m traceforge benchmark --seeds 8   # all policies, JSON + CSV, by difficulty level
python -m traceforge verify                # tests + spec matrix + audit + metamorphic pairs + benchmark
```
