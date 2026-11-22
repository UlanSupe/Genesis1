# FINAL_AUDIT

Independent final audit of TraceForge (security, evaluation integrity, requirements, reproducibility, repository hygiene).
Status legend: **PASS** = verified by execution or inspection in this audit; **PARTIAL** = implemented, with a stated gap;
**FAIL** = missing/incorrect; **NOT VERIFIED** = insufficient evidence. Nothing is marked PASS from documentation alone.

## 0. Scope and honest caveats

* The "official competition specification" was **not available as a separate document**: requirements were taken from the task statements given to this project. Where those statements leave a parameter open (the numeric amount tolerance, the exact definition of `inspection_steps`), the implementation's published choice is verified and the gap is stated (rows E3, F2).
* Environment: Windows 11, Python 3.12.0 only. Python 3.11 is registered on this machine but its executable does not exist, so runtime support on 3.11 could **not** be tested (all 93 Python files do parse under the 3.11 grammar: `ast.parse(feature_version=(3, 11))`). Linux/macOS were not available.
* Starting state (before any change in this audit): **1278 passed, 0 failed** (`python -m pytest`).

## 1. Findings of this audit

| # | Severity | Finding | Evidence | Fix | Regression test |
|---|---|---|---|---|---|
| F-1 | **High** | **The `episode_id` an agent sees was `sha256(seed\|config)`.** An agent could hash candidate seeds, recover a "hidden" seed (reproduced: seed 73,219 recovered in 0.06 s) and, because the generator is importable, regenerate the episode **and its ground truth**. | `/tmp` probe reproduced the recovery before the fix; after the fix the same attack finds nothing | `generator/builder.py`: a generated episode id is now a digest of the episode's **public content**, not of the seed | `test_final_security.py::test_sec1_*` (4 tests, incl. the exact attack with three hash families over 450,000 candidates) |
| F-2 | Medium | **Trajectory files could be inflated by the agent.** Actions were written verbatim; a 5 MB field in each of 100 steps filled the disk. A first fix (per-field caps) still allowed **42 MB for four actions** because the bounds multiply (100 findings × 100 ids × 4 KB). | measured: 42,121,540 bytes for 4 hostile actions | `trajectories/recorder.py`: per-field caps **plus** an overall 128 KB per-action budget (a legitimate maximum action is ≈ 94 KB) | `test_final_security.py::test_sec2_*` (4 tests: size bound, replay equality of capped actions, maximum legitimate action untouched) |
| F-3 | Info | **Seed secrecy is inherent to an open, deterministic generator.** With the code and a guessable seed anyone can regenerate an episode and its truth. | design review | documented: hidden evaluation must use high-entropy seeds (≥ 64 bits; tested up to 2²⁰⁰ and string seeds) | `test_sec1_high_entropy_seeds_are_supported_for_hidden_evaluation` |
| F-4 | Low | Submission clutter: 6.3 MB generated trajectories, a 7.3 MB developer-tool folder (`graphify-out/`), stale reports, caches, `egg-info`. | `du` | removed / ignored (see §8) | — |
| F-5 | Low | Evaluator logging: confirmed that the internal journal line for an invalid submission carries no truth. | inspection + test | none needed | `test_sec3_evaluator_logging_never_carries_hidden_labels` |

No Critical issue was found. Earlier-stage findings (isolation DoS, structural tells, config validation, test gaps) are in `ENGINEERING_AUDIT.md`.

## 2. Requirement matrix

Columns: implementation | verification (command or test) | actual result | status. All tests are in `tests/`; "suite" means `python -m pytest`.

| ID | Requirement | Implementation | Verification | Actual result | Status |
|---|---|---|---|---|---|
| A1 | `reset(seed, config)` → `(observation, info)` | `environment/audit_env.py::reset` | `test_official_requirements.py::test_A_*` | returns a 2-tuple of dicts; positional `(seed, config)` works | **PASS** |
| A2 | `step(action)` → `(obs, reward, terminated, truncated, info)` | `audit_env.py::step` | same | 5-tuple with float reward, bool flags; submit → terminated | **PASS** |
| B1 | Tools list_records, get_record, get_document, calculator, submit | `environment/actions.py`, `audit_env.py` | `test_B_exactly_the_five_required_tools_exist_and_work` | exactly these 5; each returns `ok` | **PASS** |
| B2 | Unknown/invalid tools rejected safely | `actions.py::parse_action` | `test_B_tools_reject_anything_else_safely`, `test_adversarial.py::test_10_*`, 3000+2000 fuzz actions | all rejected with a bounded error, 0 crashes | **PASS** |
| C1 | Read-only agent interaction | no write path in `environment/`, `policies/` | `test_C_agent_interaction_is_read_only_no_network_no_subprocess` (file writes, deletes, sockets, subprocess booby-trapped during complete episodes of all 4 policies; src+datasets hash unchanged) | no forbidden call raised; hashes equal | **PASS** |
| C2 | ≤ 50 records per page | `environment/pagination.py` | `test_C_limits_*`, `test_adversarial.py::test_09_*` | 50 ok, 51/0/-1/huge/non-int rejected | **PASS** |
| C3 | ≤ 100 steps incl. submit | `audit_env.py` | `test_C_limits_*`, `test_adversarial.py::test_08_*` | truncated at exactly 100; 101st step refused | **PASS** |
| C4 | No shell / SQL / filesystem / internet **through the tool interface** | the 5-tool API; AST-whitelist calculator | `test_B_tools_reject_*`, `test_adversarial.py::test_13_*` (42 calculator attacks), audit AST scan (no eval/exec/network/third-party) | all contained, each <1 s | **PASS** |
| C5 | Same restrictions for an **arbitrary agent process** (OS-level sandbox) | `environment/isolation.py`: separate process, scrubbed env, empty cwd, no PYTHONPATH for custom agents, bounded I/O, deadline, POSIX rlimits | `test_adversarial.py::test_12c/15/16/17/18`, `test_isolation.py` | agent process holds no private data; runner survives hostile output; but there is **no network/file-system sandbox**: a hostile process can read OS files (e.g. stored `private/`), use the network, spawn processes. POSIX rlimits untested here | **PARTIAL** |
| C6 | No hidden-truth leakage (observations, info, errors, logs, trajectories) | `AuditEnv`, `assert_no_leak`, recorder | `test_C_observations_never_contain_*`, `test_isolation.py::test_no_private_token_*`, `test_demo.py::test_normal_demo_never_exposes_*`, `test_sec3_*`, fuzz | 0 leaks in all scans; seed not in observations (`test_sec1_agent_visible_data_*`) | **PASS** |
| C7 | Agent cannot reach `truth.json` / `provenance.json` via tools, process memory, cwd or environment | evaluator-only objects; scrubbed process | `test_adversarial.py::test_12_*`, `::test_12c_*`, `::test_15_*` | gc/`sys.modules`/env/cwd/tmp scan finds nothing (in-memory episodes) | **PASS** |
| D1 | Duplicate invoices/postings | `generator/mutations.py::apply_duplicate` | `test_D_*[duplicate]`, `test_mutations.py` | injected; independent detector recovers (type, ids, amount, currency) | **PASS** |
| D2 | Missing/partial payment allocation | `apply_allocation` | `test_D_*[allocation]` | same | **PASS** |
| D3 | Incorrect FX rate/date | `apply_fx` | `test_D_*[fx]` | same | **PASS** |
| D4 | Incorrect reporting period | `apply_cutoff` | `test_D_*[cutoff]` | same | **PASS** |
| E1 | Correct anomaly type required | `evaluation/matcher.py` | `test_E_one_to_one_matching_rules["wrong type"]` | 0 TP / 4 FP / 4 FN | **PASS** |
| E2 | Correct affected ids required (set equality) | matcher | `["wrong affected ids"]`, `test_adversarial.py::test_03_*` | rejected | **PASS** |
| E3 | Amount within the specified tolerance | `constants.AMOUNT_TOLERANCE_MINOR = 1` | `["amount +1/−1 accepted"]`, `["amount +2 rejected"]`, mutants M01 | ±1 minor unit accepted, ±2 rejected; the brief gives no numeric tolerance, so the value is this implementation's published choice | **PARTIAL** |
| E4 | Correct currency required | matcher | `["wrong currency"]`, `test_04_*` | rejected (case/space normalised only) | **PASS** |
| E5 | Complete valid evidence set required | `evidence_validator.py` | `["no evidence"]`, `["incomplete evidence"]`, `test_07_*` | partial evidence rejected; any complete allowed set accepted | **PASS** |
| E6 | One-to-one, deterministic, order-independent matching | Kuhn maximum bipartite matching + canonical order | `test_matcher_stress.py` (471 cases, incl. exhaustive search), `test_E_matching_is_order_independent_*` | maximum cardinality; identical result under 12 shuffles | **PASS** |
| E7 | Duplicate and unmatched findings are false positives | matcher | `["duplicates are false positives"]` (4,4,0), `["unmatched finding..."]` (4,1,0) | as required | **PASS** |
| E8 | Missed anomalies are false negatives | matcher | `["missed anomalies..."]` (1,0,3), `["empty submission"]` (0,0,4) | as required | **PASS** |
| F1 | `reward = clip(2·F1 − 1 − 0.002·inspection_steps, −1, 1)` | `evaluation/reward.py` (exact fractions) | `test_F_reward_formula_exact_over_the_whole_grid` (490 combinations vs independent Decimal formula), `test_F_reward_through_the_real_environment`, mutants M05/M06 | exact agreement; formula unchanged from the original | **PASS** |
| F2 | Definition of `inspection_steps` | actions before `submit`, invalid actions included; truncation = 100 | `test_adversarial.py::test_08*`, `test_environment.py` | consistent and published; cannot be checked against an official definition that was not provided | **PARTIAL** |
| F3 | Invalid submission → −1; exploration → 0 | `schemas.parse_submission`, `audit_env.py` | `test_F_invalid_submission_gets_minus_one_*` | as required | **PASS** |
| G1 | Random policy | `policies/random_policy.py` | `test_G_*`, benchmark | seeded, same step cap, no truth access | **PASS** |
| G2 | Simple reconciliation | `policies/reconciliation_policy.py` | `test_policies.py::test_policies_cannot_reach_evaluator_code_*` (static), benchmark | uses only observations/tools; F1 1.000 | **PASS** |
| G3 | Mark-all | `policies/mark_all_policy.py` | `test_G_mark_all_is_penalised_*` | ≥ 20 FP per episode, reward < −0.9 | **PASS** |
| G4 | Oracle, evaluator only | `evaluation/oracle_policy.py` | `test_G_four_policies_and_oracle_is_evaluator_only` | not in agent registry; agent worker exits 2; cannot run isolated | **PASS** |
| H1 | Fixed-seed generation | `generator/builder.py` | `test_H_fixed_seed_*`, `test_reproducibility.py` | byte-identical public+private files | **PASS** |
| H2 | Stable benchmark | `runner/benchmark.py` | `test_H_*`, `test_I_final_benchmark_file_is_consistent_with_a_fresh_run` | two runs equal apart from timing; saved file reproduced | **PASS** |
| H3 | Metamorphic transformations | `generator/transformations.py` | `test_H_metamorphic_*`, `test_metamorphic.py` (24 pairs + 10 corruption controls) | correct answers survive id remap + shuffle | **PASS** |
| H4 | Recorded, replayable trajectories | `trajectories/` | `test_H_*`, `test_trajectories.py`, `test_sec2_*` | replay exact; agent-view only | **PASS** |
| H5 | Determinism across processes | audit | `python -m traceforge audit` | same digest under 4 `PYTHONHASHSEED` values | **PASS** |
| I1 | Source code | `src/traceforge/` | `compileall`, suite | no broken imports | **PASS** |
| I2 | Tests | `tests/` (29 files, 1366 tests) | suite | see §6 | **PASS** |
| I3 | CLI | `cli.py` | `--help`, `test_integration.py`, `test_demo.py` | generate, verify, demo, evaluate, inspect, audit, benchmark all run | **PASS** |
| I4 | README | `README.md` | inspection + quick-start run (§7) | present, commands verified | **PASS** |
| I5 | Demo | `reporting/demo_report.py` | `python -m traceforge demo`, `test_demo.py` (24) | 4/4 anomalies, F1 1.000, reward 0.958, HTML report | **PASS** |
| I6 | Benchmark output | `FINAL_BENCHMARK.json/.csv` | `python -m traceforge benchmark --suite all --seeds 8 --out . --name FINAL_BENCHMARK`; consistency test | 52 episodes × 4 policies = 208 runs | **PASS** |
| I7 | Documentation | README, ARCHITECTURE, DEMO_SCRIPT, DESIGN, SPEC_COMPLIANCE, ENGINEERING_AUDIT, FINAL_ENGINEERING_REPORT, FINAL_AUDIT, JUDGE_SUMMARY | `test_I_deliverables_exist_and_commands_work` | present | **PASS** |
| I8 | 10 train + 10 dev episodes | `datasets/` | same + `test_integration.py` | 10 + 10, each validated and regenerated bit-for-bit | **PASS** |
| J1 | Runs offline, no external services | stdlib only | audit AST scan, booby-trapped sockets during episodes, `test_demo.py::test_demo_works_with_the_standard_library_only` (`python -S`) | no network/third-party imports; `dependencies = []` | **PASS** |
| J2 | Python ≥ 3.11 | `requires-python` | grammar check only | all 93 files parse under 3.11; **runtime on 3.11 not run** (no interpreter) | **NOT VERIFIED** |
| J3 | Non-Windows operating systems | `os.name`-guarded code | none available | not run | **NOT VERIFIED** |
| J4 | Clean-environment install, tests, benchmark and demo | §7 | fresh virtual environment from a stripped copy | Fresh virtual environment (Python 3.12.0, contained only pip) built from a stripped copy of 512 submission files (re-run after the addendum changes) (no `reports/`, no cached trajectories, no tooling folders) by `tools/clean_env_check.sh`: `pip install -r requirements.txt` and `pip install -e .` exit 0; installed packages = traceforge, pytest and pytest's dependencies only; full suite **1366 passed, 0 failed**; benchmark 208 runs **identical** to `FINAL_BENCHMARK.json` (same dataset digest); demo 4/4 anomalies, reward 0.958. Windows / Python 3.12 only; needs network for `pip` | **PASS** |
| J5 | Conformance to the original official specification text | — | the document was not provided | requirements verified against the task statements only | **NOT VERIFIED** |
| K1 | No secrets / credentials | repository | regex scan for key/token/password patterns | none | **PASS** |
| K2 | No absolute local paths | repository | scan of `src tests datasets configs examples tools *.md pyproject.toml` | none, except the substring "Genesis" inside a test that checks environment variables | **PASS** |
| K3 | No machine-specific dependencies | repository | inspection | Windows VT (`ctypes`) and POSIX `resource` are guarded by `os.name` | **PASS** |
| K4 | No broken imports / missing dependencies | repository | `compileall -W error`, suite, clean-env run | none | **PASS** |

## 3. Security review (checklist)

| Item | Result | Evidence |
|---|---|---|
| Agent cannot access `truth.json` / `provenance.json` | PASS (tool API and agent-process memory/cwd/env); hostile FS-capable agents can read stored `private/` (see C5) | `test_12`, `test_12c`, `test_15` |
| Hidden evaluator data never enters observations | PASS | leak scan over full runs; 5000 garbage actions; `test_C_observations_*` |
| Debug output does not expose labels | PASS | `test_sec3_*`; error texts are fixed and never echo input |
| Trajectories store no hidden information | PASS | `test_trajectories.py`, `test_demo.py`, seed absent; now size-bounded (F-2) |
| Tool arguments validated; unknown actions rejected | PASS | `actions.py`; 35+ invalid-argument cases; fuzz |
| Malformed input does not crash the environment | PASS | 3000 + 2000 random actions, 0 unhandled exceptions |
| Calculator cannot execute code | PASS | AST whitelist; 42 attacks + 1500 random expressions, all contained |
| No agent action can modify source or datasets | PASS for the tool API (booby-trap + hash test); a hostile **process** can write files it has OS permission for (C5) | `test_C_agent_interaction_*` |
| No external network needed | PASS | AST scan, booby-trapped sockets |
| Agent cannot learn the hidden seed from what it sees | **PASS after fix F-1** (was a High vulnerability) | `test_sec1_*` |
| Runner protected against a hostile agent | PASS | oversized output killed, stderr bounded, deadlines (`test_16/17/18`) |

## 4. Evaluation integrity

Adversarial evaluator tests (`test_adversarial.py` 106 cases, `test_matcher_stress.py` 471, `test_official_requirements.py::test_E_*` 14) check: determinism and order independence (12 shuffles; 60×8 random shuffles), maximum matching against exhaustive search (400 random instances; plus a constructed case where a greedy canonical-order pass gives TP=1 vs the true maximum 2), duplicate submissions (k copies → 1 TP, k−1 FP), partial evidence rejected, wrong currency/amount/type/ids rejected, false positives and false negatives counted, reward from the official formula. Expected values were never edited to make a test pass; the one place an expectation changed (allocation evidence now includes invoice documents) was a deliberate, documented contract change in an earlier stage.

## 5. Benchmark verification

Source: `FINAL_BENCHMARK.json` / `.csv` (+ `_summary.csv`, `_by_level.csv`), produced by `python -m traceforge benchmark --suite all --seeds 8 --out . --name FINAL_BENCHMARK`; metadata records code version, Python, command and a digest of the stored dataset. 52 episodes (10 train, 10 dev, 32 generated), 175 injected issues, 208 runs, 0 agent errors. Same episode list, eval mode and 100-step cap for all policies (checked: identical episode sets). Oracle: evaluator-side only (G4). Mark-all: 1558 FP, reward −0.997. Random: 156 FP, 0 TP, reward −1.000 (same step cap, no truth access, no special treatment). Reconciliation: 175/175, F1 1.000, reward 0.968, uses only tool observations (static test + G2). Per-policy table: `JUDGE_SUMMARY.md` §10. Macro averages count an empty episode with no findings as precision = recall = 1.

## 6. Final test results

Final run (`python -m traceforge verify`, which runs `python -m pytest` once): **1366 tests: 1366 passed, 0 failed, 0 skipped** (after the addendum below; the audit itself ended at 1324). The 1366-test suite also passed in the clean environment (1366 / 0 / 0). Before this audit the suite had 1278 tests; this audit added 46 (37 official-requirement tests, 9 security regression tests), the addendum 24 (8 sandbox, 16 pairs/eval/alias).

| Area | Files | Tests |
|---|---|---|
| Evaluator: matcher, stress (exhaustive reference), reward, evidence | `test_matcher`, `test_matcher_stress`, `test_reward`, `test_evidence` | 535 |
| Environment contract / unit: environment, pagination, calculator, edge cases, policies, trajectories | `test_environment`, `test_pagination`, `test_calculator`, `test_edge_cases`, `test_policies`, `test_trajectories` | 195 |
| Security / adversarial (15 attack scenarios + runner protection + final-audit regressions) | `test_adversarial`, `test_isolation`, `test_final_security` | 126 |
| Generator, data and property-based checks | `test_generator`, `test_generator_quality`, `test_mutations`, `test_compound_scenarios`, `test_validator_integrity`, `test_property`, `test_realism_levels` | 300 |
| Reproducibility | `test_reproducibility` | 8 |
| Metamorphic | `test_metamorphic`, `test_transformations` | 72 |
| Integration, CLI, reports, demo | `test_integration`, `test_reports`, `test_demo` | 51 |
| Official requirements A-I (independent black-box) | `test_official_requirements` | 37 |

Other verify sections, all PASS: spec matrix 67/67, generator sweep (46 episodes), matcher/reward cases, reproducibility, isolation + leaks + replay, 24 metamorphic pairs (384 checks), datasets (20/20 regenerate bit-for-bit), audit (architecture scan, 66-episode validator sweep, structural tells, cross-process determinism, 2000-action fuzz, 1500-expression calculator fuzz, performance), benchmark. Not rerun in this audit: the full mutation check of the test-suite (last full result in `FINAL_ENGINEERING_REPORT.md`: 19/19 targeted mutants killed).

## 7. Clean-environment verification and judge quick start

Exact sequence a judge can use (Windows or POSIX; Python >= 3.11, tested on 3.12.0):

```bash
python -m venv .venv                      # optional but recommended; activate it
pip install -r requirements.txt           # only pytest==9.1.1 (the package itself has no dependencies)
pip install -e .                          # or, without installing: set PYTHONPATH=src
python -m pytest                          # 1366 tests, about 2.5-3 minutes
python -m traceforge benchmark --suite all --seeds 8     # all 4 policies -> reports/benchmark.json / .csv
python -m traceforge demo                 # fixed-seed demonstration -> reports/demo_report.html
python -m traceforge verify               # everything above plus spec matrix, audit, metamorphic pairs
```

This sequence was executed in a freshly created virtual environment from a copy containing only the submission files (`bash tools/clean_env_check.sh`; log summarised in row J4, re-run on the final state): install exit 0, `pytest` 1366 passed, benchmark identical to `FINAL_BENCHMARK.json`, demo 4/4 with reward 0.958. Not verified: other operating systems and Python 3.11 (rows J2, J3). The demo, benchmark and tests use no previously generated local artefacts (the stripped copy had none).

## 8. Repository cleanup

Removed or excluded: generated trajectories (6.3 MB), stale and probe reports, `__pycache__`, `.pytest_cache`, `src/traceforge.egg-info`; the developer-tooling folder `graphify-out/` (7.3 MB) is git-ignored and **not part of the submission**. Added `requirements.txt` (runtime dependencies: none; `pytest==9.1.1` for the tests) and updated `.gitignore`. Datasets, tests and documentation were not touched. Scans: no secrets, no absolute local paths (one benign test string), no file above 300 KB outside the excluded folders. `datasets/*/private/` is **intentionally present** (open train/dev sets); hidden evaluation must use fresh high-entropy episodes.

## 9. Verdict

**READY WITH LIMITATIONS.**

Requirement matrix (section 2, 52 rows): **46 PASS, 3 PARTIAL, 0 FAIL, 3 NOT VERIFIED.**
* PARTIAL: C5 (no OS-level sandbox for a hostile agent *process*), E3 (the numeric amount tolerance is this implementation's published choice, not given in the brief), F2 (definition of `inspection_steps` is this implementation's published choice).
* NOT VERIFIED: J2 (Python 3.11 runtime: no interpreter available; grammar-checked only), J3 (non-Windows systems), J5 (the original official specification text was not provided).
* No FAIL. No Critical vulnerability. One **High** vulnerability (F-1, seed recoverable from the episode id) and one Medium (F-2, trajectory inflation) were found, fixed and covered by regression tests.

Why not simply "READY": the three PARTIAL items and three NOT VERIFIED items are real, stated limits, and the benchmark cannot yet separate strong agents (the transparent rule-based baseline is perfect and no learned agent was run). The evidence supports a correct, reproducible, offline environment with a verified evaluator; it does not support claims about model improvement or about hostile-agent containment beyond the tool interface.

## Addendum - post-audit improvements (not part of the original audit run)

* **C5 (agent-process sandbox)**: new `runner/sandbox.py` (irreversible audit hook; built-in policies run under it by default, custom Python agents via `python -m traceforge.runner.sandbox agent.py`). `tests/test_sandbox.py` runs a hostile agent twice: without the sandbox all 12 attacks succeed (including deleting a file), with it all are blocked and nothing on disk changes; a legitimate agent and its score are unaffected. Status: **PASS for Python agents, still PARTIAL overall** - audit hooks are not a security boundary (native code, some Windows internals, non-Python agents).
* **Field name**: the finding amount is also accepted as `amount` / `amount_minor` (ints only).
* **Scale**: hard episodes were above the brief's 20-200 documents guidance; invoice ranges were retuned (mean documents: easy ~116, medium ~155, hard ~179; about 95% of hard episodes <= 200). Stored datasets were regenerated; `FINAL_BENCHMARK.*` and `SPEC_COMPLIANCE.md` (71/71 PASS) were regenerated from real runs (same 175 issues; reconciliation reward 0.968).
* **Organizer tooling**: `pairs`, `paired-eval`, `generate-eval` commands (stored pairs in `datasets/pairs`, robustness metric, secret-derived hidden set).
* Clean-environment run (J4) repeated on this final state: install exit 0, 1366 passed, benchmark identical to `FINAL_BENCHMARK.json`, demo 4/4 (reward 0.958). Demo figures in `DEMO_SCRIPT.md` / `FINAL_ENGINEERING_REPORT.md` were refreshed from the live demo (seed 7 now yields 141 documents, 289 ledger lines, 21 inspection steps).
* **Reward sensitivity**: `python -m traceforge ablation` (`FINAL_ABLATION.json/.csv`, reproduced by `tests/test_ablations.py`): the auditor with one capability removed loses exactly what it should; explanation text changes nothing.
* **OS boundary for hostile agents**: `Dockerfile`, `runner/container.py`, `tools/container_smoke.sh` (no network, read-only, no capabilities, only the agent folder mounted read-only). Only the generated command line is tested here; **no container was run** (no Docker in this environment).
* **CI**: `.github/workflows/ci.yml` (Ubuntu/Windows/macOS x Python 3.11/3.12). Defined, but its result was not observed from this environment.
* Not verified: Linux/macOS, Python 3.11 (a local 3.11 interpreter was not usable).
