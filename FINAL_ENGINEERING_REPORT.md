# FINAL_ENGINEERING_REPORT

All numbers below were produced by commands run on this machine (Windows 11, Python 3.12.0) and are
reproducible with the commands at the end. Nothing is estimated or rounded in our favour.

## 1. What this stage improved

| Area | Before | Now |
|---|---|---|
| Truth records | type / ids / amount / currency / evidence | + `root_cause` code and a detailed `explanation` (what happened, which records show it, how the amount follows). Private; the evaluator ignores them; agents never see them (test + trajectory scan) |
| Difficulty | only volume (more decoys/documents) | **Profiles** that change *which* anomalies occur and how subtle they are: easy = direct (`unallocated`, 3–5 % wrong rate, ≥ 1 day cut-off); medium = cross-referencing (`partial`, `wrong_date`, 4–24 h); hard = indirect (`wrong_invoice` needing two invoices, 0.5–1.5 % rates, 15 min–3 h cut-off, reposts of documents with a legitimate twin); compound = hard + two interacting scenarios (hard preset is compound ~60 % of the time) |
| Evidence | document + ledger rows | allocation evidence now **cross-references** the invoice documents of every affected invoice; convention published in `policy.finding_conventions` |
| Decoys | repeats, same amounts, refunds, overpayments, boundary payments | + **recurring series** (3–4 identical invoices at a fixed cadence) and a **guaranteed split-payment decoy**; counts vary per episode; decoys can also carry errors (no "looks like a decoy ⇒ safe" shortcut) |
| Ground-truth validation | derive == truth, baseline rebuild, manifest replay, integrity | + affected ids exist; **every evidence set alone reconstructs exactly its finding** (detector run only on the cited documents); **clean decoys intact**; **totals change only by declared anomalies** (per account and in-period debit total); root cause/explanation present. 14 checks per episode |
| Floating point | one `float()` on fx rates in the validator | removed; AST test forbids `float()` in the money/generator/matcher modules; ledger amounts are re-computed independently with `Decimal` in a test |
| Benchmark | per suite/policy | + per-difficulty-level table (`benchmark_by_level.csv`) |

Also fixed in this stage (found by re-running the mutation check): **four test gaps** — the greedy-matching bug
survived the matcher tests (canonical ordering makes greedy optimal on every hand-written case), returned
records were not proven to be copies, and a path-leak test passed only by accident of the temp-directory name.
Each now has a dedicated test (see `ENGINEERING_AUDIT.md`, stage 2).

## 2. Tests

| Measure | Result |
|---|---|
| pytest suite (`python -m traceforge verify`) | **1278 passed, 0 failed, 0 skipped** (158 s) after the demo stage (1254 before) |
| Specification matrix (`SPEC_COMPLIANCE.md`) | **56 / 56 requirements PASS** (49 before the demo stage), 0 FAIL, 0 UNVERIFIED (a requirement is PASS only if every cited test ran and passed) |
| Verify sections | 10 / 10 PASS |
| Metamorphic pairs | **24 / 24 pairs pass**, 384 checks, 15,816 identifiers remapped; 10 corruption controls prove the link verifier detects broken links |
| Audit | 7 / 7 sections PASS (66-episode validator sweep with 0 failures; architecture scan; structural tells; cross-process determinism under 4 `PYTHONHASHSEED` values; 2000 garbage actions; 1500 calculator fuzz cases; performance) |
| Datasets | 20 / 20 stored episodes validate and regenerate bit-for-bit |
| Mutation check of the test-suite (this stage's 15 new mutants + the 4 earlier survivors) | **19 / 19 killed**, 0 survived (`reports/mutation_check.json`). The first full run (stage 1, 28 mutants) had no recorded killer names for 24 of them; the partial rerun with names showed 5 survivors, of which 4 were real test gaps (now fixed) and 1 (M18) is an *equivalent* mutant (the policy truncates to 30 findings anyway). A complete rerun of all 63 mutants was **not** done after the final code; only the 19 listed |

## 3. Benchmark (all four baseline policies, identical episodes, eval mode)

`python -m traceforge benchmark --seeds 8` — 208 episode runs. Oracle runs evaluator-side only.

By suite:

| suite | policy | eps | TP | FP | FN | precision | recall | F1 | reward | inspection steps |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | random | 10 | 0 | 30 | 30 | 0.000 | 0.200 | 0.000 | −1.000 | 20.0 |
| dev | reconciliation | 10 | 30 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.964 | 18.0 |
| dev | mark_all | 10 | 0 | 300 | 30 | 0.000 | 0.200 | 0.000 | −1.000 | 18.0 |
| dev | oracle | 10 | 30 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0 |
| train | random | 10 | 0 | 30 | 28 | 0.000 | 0.200 | 0.000 | −1.000 | 20.0 |
| train | reconciliation | 10 | 28 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.961 | 19.7 |
| train | mark_all | 10 | 1 | 299 | 27 | 0.003 | 0.208 | 0.005 | −0.995 | 19.7 |
| train | oracle | 10 | 28 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0 |
| presets (32 generated) | random | 32 | 0 | 96 | 117 | 0.000 | 0.031 | 0.000 | −1.000 | 20.0 |
| presets | reconciliation | 32 | 117 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.961 | 19.5 |
| presets | mark_all | 32 | 1 | 959 | 116 | 0.001 | 0.039 | 0.002 | −0.998 | 19.5 |
| presets | oracle | 32 | 117 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0 |

By difficulty level (all suites pooled):

| level | policy | eps | issues | TP | FP | FN | F1 | reward | steps |
|---|---|---|---|---|---|---|---|---|---|
| easy | reconciliation | 14 | 24 | 24 | 0 | 0 | 1.000 | 0.975 | 12.4 |
| medium | reconciliation | 10 | 26 | 26 | 0 | 0 | 1.000 | 0.966 | 17.1 |
| hard | reconciliation | 15 | 75 | 75 | 0 | 0 | 1.000 | 0.952 | 24.1 |
| compound (preset) | reconciliation | 8 | 38 | 38 | 0 | 0 | 1.000 | 0.951 | 24.6 |
| easy / medium / hard / compound | mark_all | 14 / 10 / 15 / 8 | 24 / 26 / 75 / 38 | 0 / 0 / 1 / 1 | 420 / 300 / 449 / 239 | — | 0.000 / 0.000 / 0.003 / 0.007 | ≈ −1.0 | 12–25 |
| easy / medium / hard / compound | random | 14 / 10 / 15 / 8 | same | 0 | 42 / 30 / 45 / 24 | — | 0.000 | −1.000 | 20.0 |
| all levels (incl. dataset episodes with a compound scenario) | oracle | 52 | 175 | 175 | 0 | 0 | 1.000 | 1.000 | 0.0 |

(The full per-level table, including the dataset episodes that mix a compound scenario, is in `reports/benchmark_by_level.csv`.)

How to read this honestly:
* The simple reconciliation policy solves every episode at every level. That follows from the design requirement that every
  error be derivable from public data (the validator proves it); the harder levels cost it **more steps** (12 → 24), which lowers
  its reward (0.975 → 0.951), but not accuracy. We did not tune the policy or the data to hide this.
* Mark-everything and random are punished as intended; mark-everything occasionally scores a true positive by chance (2 of 175 issues).
* These baselines say nothing about LLM agents or about model improvement; no model was run.

## 4. Performance (measured, this machine)

| case | ledger rows | documents | generate (incl. full validation) | reset from disk | list_records(50) | get_document | matcher (30×12) | reconciliation episode |
|---|---|---|---|---|---|---|---|---|
| ~200 rows | 277 | 134 | 0.13 s | 3.0 ms | 0.19 ms | 0.007 ms | 0.11 ms | 0.010 s |
| ~500 rows | 545 | 265 | 0.25 s | 4.7 ms | 0.17 ms | 0.006 ms | 0.10 ms | 0.014 s |
| ~1000 rows | 1156 | 566 | 0.51 s | 7.7 ms | 0.17 ms | 0.006 ms | 0.11 ms | 0.027 s |
| ~2000 rows | 2106 | 1038 | 0.86 s | 14.3 ms | 0.17 ms | 0.006 ms | 0.11 ms | 0.047 s |
| ~200 documents | 413 | 200 | 0.19 s | 4.0 ms | 0.19 ms | 0.007 ms | 0.11 ms | 0.012 s |

Peak Python memory while generating 2,106 rows: 6.2 MB (tracemalloc). Generation time grows roughly linearly (tested).

## 5. Known limitations / incomplete

* **The rule-based baseline is perfect at all levels** (see above). Difficulty for LLM agents comes from step budget, data volume and the number of links to follow; there is no evidence yet how real agents behave.
* The anomaly families are the four required ones; each has 1–3 variants. No FX gain/loss accounting: ledger lines are converted per document, so the receivable control account is not netted in base currency (outstanding balances are checked in document currency instead).
* Fixed UTC offsets (no DST/IANA zones); daily FX rates including weekends.
* Isolation is process-level. It does not stop an agent with file-system access from reading stored `private/` folders; `RLIMIT_*` limits exist only on POSIX. `AuditEnv` is the evaluator's object: do not hand it to untrusted in-process code.
* Verified only on Windows 11 / Python 3.12.0; Python 3.11 is declared (`requires-python`) but never executed.
* The mutation check covers 19 mutants after the final code (63 exist); a full rerun takes ~1 hour and was not repeated.
* Quality of explanations is validated structurally (present, mentions ids and amount, ≥ 80 chars, ids remapped consistently), not linguistically.

## 6. Commands to verify

```bash
pip install -e .                       # or: set PYTHONPATH=src
python -m traceforge generate          # regenerate datasets/train and datasets/dev
python -m traceforge verify            # pytest + spec matrix + audit + 24 metamorphic pairs + benchmark; writes SPEC_COMPLIANCE.md
python -m traceforge audit             # architecture / data / tells / fuzzing / performance
python -m traceforge benchmark --seeds 8
python -m traceforge demo --preset hard --seed 7 --instructor
python tools/mutation_check.py --only M12 M49 M52   # optional: does the test-suite notice injected bugs?
python -m pytest                       # 1278 tests
```

---

## 7. Demonstration stage (this addendum)

Added: a one-command fixed-seed demo (`python -m traceforge demo`), a verifying reconciliation agent, a terminal layer that works without
colour, a self-contained HTML report, `ARCHITECTURE.md` (Mermaid), `DEMO_SCRIPT.md`, 24 new tests (`tests/test_demo.py`).
Measured (`python -m traceforge demo`, seed 7, config = two compound scenarios at medium difficulty):

| item | result |
|---|---|
| company | 172 documents, 353 ledger lines, 4 currencies, 7 counterparties |
| agent tool use | list_records 18, get_document 4, get_record 1, calculator 1, submit 1 (24 inspection steps) |
| evaluator | TP 4, FP 0, FN 0; precision 1.000, recall 1.000, F1 1.000; reward 0.952 |
| same agent without the verification phase | 18 inspection steps, reward 0.964 (identical findings): the verification costs exactly 6 steps = 0.012 reward |
| baseline benchmark inside the demo (22 episodes: dev + 3 seeds x 4 presets; 88 runs) | random F1 0.000 / reward −1.000; reconciliation F1 1.000 / reward 0.963 / 18.5 steps; mark_all F1 0.003 / reward −0.997; oracle F1 1.000 / reward 1.000 |
| verify after this stage | 10 / 10 sections PASS; 1278 passed, 0 failed; 56 / 56 spec requirements PASS |

Defects found and fixed while building it (all caught by the new tests or by looking at the output): coloured table cells were measured
including their escape codes (misaligned tables); `--color` was ignored when piped on Windows; the JSON data island in the HTML contained raw `<`;
the oracle-isolated error appeared only after several sections had printed; narrow-screen horizontal overflow in the report. A first full test run
showed one failure that was a race with my own mid-edit changes; the rerun and the final run are clean.

Not done / limitations of the demo: the `verify` phase of the agent only re-checks FX findings with the calculator (other findings only re-read their document); the HTML report was
visually checked in an embedded browser at desktop and phone widths, not across browsers; the benchmark inside the demo uses 3 seeds per preset (fast), so its numbers differ slightly from the 8-seed benchmark above.
