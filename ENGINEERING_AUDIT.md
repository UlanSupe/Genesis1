# ENGINEERING_AUDIT

Scope: the whole code base produced in the first stage (generator, environment, evaluator, policies,
runner, trajectories, transformations, CLI, 390 tests). Method: reading every module, then **measuring**
with probes (`tools/audit_probe.py`, `tools/probe_ledger.py`, `tools/stress_gen.py`), fuzzing, an
independent ledger reconstruction, and a mutation check of the test-suite (`tools/mutation_check.py`).
Only problems that were actually observed are listed; "Verified OK" lists what was checked and found sound.

## Findings

| # | Severity | File | Problem | Fixed by | Confirmed by |
|---|---|---|---|---|---|
| 1 | **High** | `environment/isolation.py` | A hostile agent could exhaust the **runner's** resources: stdout was read with an unbounded `readline` (a multi-GB line without newline = out-of-memory in the evaluator process), stderr went to a file with no limit (disk fill), and there was no total episode deadline. | Bounded `readline(max+1)` that kills the agent, stderr drained into an 8 KB tail buffer, per-call timeout + total deadline, POSIX `RLIMIT_AS/CPU` (best effort; not available on Windows). | `test_adversarial.py::test_16…`, `::test_17…`, `::test_18…`; mutants M29, M30 |
| 2 | Medium | `environment/isolation.py` | The "scrubbed" agent environment still carried `PYTHONPATH` = the project's `src` path, i.e. a custom agent learned where the project (and any stored `datasets/…/private`) lives. Found by the filesystem-attack test. | `PYTHONPATH` only for the built-in trusted worker; custom command agents get none. | `test_adversarial.py::test_15_agent_cannot_find_private_files_or_paths`; mutant M31 |
| 3 | Medium | `generator/compound_scenarios.py` | **Structural tell**: the cross-border scenario always paid exactly 40 % and 35 % of the invoice. Measured over 30 episodes: 38 and 35 payments at exactly those values vs ≤17 for any other value, so an agent could spot the scenario's payments without auditing. The period-end lookalikes were also always 5–9 h apart. | Random shares 20–45 %; lookalike gap 5–24 h, start 30–70 h before the end. | `test_generator_quality.py::test_cross_border_scenario_has_no_constant_allocation_fingerprint`, `::test_period_end_scenario_lookalike_gap_varies`; audit section "structural tells"; mutant M34 |
| 4 | Medium | `generator/mutations.py` | **Decoys were immune to mutation**: all 320 mutated documents in the probe were `regular` (0 decoys). "Looks like a decoy" therefore meant "safe", a free shortcut that defeats the purpose of decoys. | Only scaffold documents are reserved; decoys can carry errors (22–25 % of mutated documents are decoys now). | `test_generator_quality.py::test_decoys_can_carry_errors_so_decoy_looking_is_not_a_safe_signal`; audit "structural tells"; mutant M33 |
| 5 | Medium | `config.py`, `environment/audit_env.py` | Invalid configuration leaked raw exceptions: all-zero or negative `mutation_weights` → `ValueError`; `n_invoices="48"`, `timezone_minutes="x"`, `clean_rate="a"` → `TypeError`; `period_start="banana"` → `ValueError`; `reset(config=[…])` → `TypeError`. | Strict type/range validation, everything raised as `ConfigError`. | `test_generator_quality.py::test_bad_config_raises_config_error_never_a_raw_exception` (35 inputs), `::test_config_must_be_an_object`; mutant M32 |
| 6 | Medium | `generator/validation.py` | The validator **crashed** (`KeyError`) on referentially broken data instead of reporting a failed check — found while writing tampering tests. A validator that throws can be mistaken for "no result". | Every check runs in a guard; an unevaluable check FAILS with the reason. | `test_validator_integrity.py::test_referential_integrity_checks` |
| 7 | Medium (design gap) | `generator/validation.py` | The validator compared derived issues to truth, but had **no independent rebuild of the clean ledger, no replay of mutation manifests, no referential-integrity / outstanding-balance checks**. A generator bug that was mirrored by `derive_issues` would have gone unnoticed. | Added `expected_ledger` (separate implementation), `replay_manifests`, `referential_problems`, `outstanding_balances`, `manifest_problems`: 5 new checks (`1b`, `2b`, `4b`, `4c`, …). Measured on 300+ random episodes: **0 mismatches** (the original code was in fact consistent). | `test_validator_integrity.py` (15 tamper tests), `test_property.py`; mutants M35, M36, M47 |
| 8 | Medium | `generator/transformations.py` | The metamorphic engine was a regex substitution over serialised JSON, checked only by aggregate invariants; nothing proved that individual financial links survived, and an id-like string in free text would be rewritten. | Field-aware mapping + `verify_links` (row-by-row / document-by-document isomorphism, injectivity, no stale ids, truth/baseline isomorphism, referential integrity); 24 pairs with negative controls. | `test_metamorphic.py` (24 pairs, 10 corruption controls, field-awareness test); `transformation_report.json`; mutants M37, M38 |
| 9 | Low | `generator/financial_world.py`, `clean_cases.py` | Low diversity: every default episode had 48 invoices, 8 counterparties, the same decoy volumes and amount scale; difficulty only changed decoy counts. | Per-difficulty ranges for invoices/counterparties, amount scale per episode, payment-behaviour mix (more split/partial links on harder levels), varying decoy volumes; presets `easy/medium/hard/compound`. | `test_generator_quality.py::test_episodes_vary_…`, `::test_difficulty_grows_with_interrelations_not_with_noise`, `::test_repeated_amounts_and_order_of_documents_vary`; mutants M42, M43 |
| 10 | Low | `generator/mutations.py` | Allocation evidence accepted only the exact combined set; a finding citing the document plus **either** changed ledger line (which suffices to show the gap) was rejected. | Three allowed sets for the `partial` variant. | `test_generator_quality.py::test_allocation_evidence_accepts_any_single_changed_line_with_the_document`; mutant M46 |
| 11 | Low | `environment/audit_env.py` | `reset(seed, config)` was not Gymnasium-style (`options`, `close()`); accepted any seed type. | `reset(seed, options, *, config)` with alias handling, `close()`, seed type check. | `test_generator_quality.py::test_reset_is_gymnasium_style_and_options_alias_config`; mutant M40 |
| 12 | Low | `tests/` | Four **weak tests**: an evidence-retrievability loop that returned after 80 steps (so not every id was checked), a "policies get only JSON" test that asserted almost nothing, a pagination property test whose assertion was silently skipped for small pages, and one assertion that was always true. | Rewritten (all ids checked in several envs; static import/call analysis of agent code; exact page counts; assertions that can fail). | `test_evidence.py`, `test_policies.py`, `test_property.py` |
| 13 | Low | `runner/episode_runner.py` | (Found during phase-3 work.) An agent that died during `reset` raised an exception into the caller instead of being scored as an empty submission. | `reset` failure is contained exactly like an `act` failure. | `test_adversarial.py::test_11…`, `test_isolation.py::test_crashing_agent_is_scored_as_empty_submission_not_a_crash`; mutant M48 |

No **Critical** problem was found: no path to the private truth through observations, `info`, errors,
trajectories or the agent process was found by any probe or test.

## Answers to the 15 audit questions

| # | Question | Result | Evidence |
|---|---|---|---|
| 1 | Does the generator really work? | Yes | 300-episode stress run + 66-episode audit sweep: 0 failures, all on attempt 0 |
| 2 | Is the history consistent before mutation? | Yes | independent `expected_ledger` equals the stored baseline on every episode (check `1b`) |
| 3 | Does each mutation create the claimed discrepancy? | Yes | `derive_issues` (own implementation) == truth on every episode; check `4` |
| 4 | Are expected amounts correct? | Yes | check `3` compares type, ids, amount and currency |
| 5 | Is the evidence sufficient? | Yes (see #10 for a Low-severity strictness issue) | every evidence id fetched with agent tools (`test_evidence.py`) |
| 6 | Accidental contradictions? | None found | property tests: derived issues = declared mutations, global ledger balance, outstanding balances ∈ [0, total] |
| 7 | Are independent errors mixed? | No | one mutation per document; compound scenarios use separate journals; `4b` replay shows only declared rows changed |
| 8 | Does private truth reach observations? | No | 3000 + 2000 garbage actions with a token scanner, 15 attack tests, in-process memory scan from the agent process |
| 9 | reset/step correct? | Yes, plus #11 | `test_environment.py` (35), `test_adversarial.py` |
| 10 | Matcher per spec? | Yes | exhaustive brute-force comparison on 400 random instances, order independence on 60×8 shuffles |
| 11 | Reward exact? | Yes | exact fractions; parametrised formula tests; mutants M05, M06, M14 killed |
| 12 | Can limits be bypassed? | No | 100-step, 50-row, unknown-tool, double-submit, calculator attacks all contained |
| 13 | Do baselines really work? | Yes | benchmark: oracle F1 = 1.0; reconciliation reads data only through the tools (static test forbids evaluator imports) |
| 14 | Reproducible? | Yes | identical digests across 4 `PYTHONHASHSEED` values in separate processes; stored datasets regenerate bit-for-bit |
| 15 | Fictitious tests? | 4 weak tests found and fixed (#12); mutation check of the suite: see `FINAL_ENGINEERING_REPORT.md` | `tools/mutation_check.py` |

## Verified OK (probed, nothing to fix)

* determinism across processes and hash seeds (4 values);
* 3000 random garbage actions: 0 unhandled exceptions, 0 leaks (`tools/audit_probe.py`, audit section "environment fuzzing");
* calculator: 1500 random expressions, 0 unexpected exceptions, 0 slow evaluations (the earlier `1e99999999999` hang had been fixed in stage 1);
* money: no floats in any `*_minor` field of generated data;
* the generator never uses global `random` state, `eval`, `exec`, network or third-party modules (AST scan in `audit`).

## Accepted risks (documented, not fixed)

* `AuditEnv` is the **evaluator's** object. Handing it to untrusted in-process Python code gives that code access to the truth via introspection. Use the isolated runner (`--isolated`) for agents you do not trust.
* A malicious agent with file-system access can read `datasets/*/private/`. Mitigation: use in-memory episodes (`reset(seed, options)`) or run the agent in a container / under another OS user. `RLIMIT` protection is POSIX-only.
* The simple reconciliation baseline solves every generated episode (F1 = 1.0): this is a consequence of the validator requiring that every error is derivable from public data. Difficulty for LLM agents comes from the step budget and data volume, not from hidden information.


---

# Stage 2 addendum: audit of the test-suite itself and of the new generator logic

## Mutation check - what it really showed

The first full mutation run (28 mutants) reported all killed, but **the killer test was not recorded** for 24 of them
(a harness parsing bug), so that claim was weaker than it looked. A rerun that records the killing test showed **5 survivors**:

| Mutant | Verdict | Resolution |
|---|---|---|
| M12 greedy matching (no augmenting paths) | **Real gap (Medium)**: with canonical ordering a greedy pass is optimal on every hand-written case and on 400 random brute-force instances | new test with an instance where the canonical-order greedy provably gives TP=1 vs maximum 2; random instances now include an id that sorts first |
| M20 / M21 `list_records` / `get_record` return live references | **Real gap (Medium)**: nothing proved an agent cannot corrupt environment state through returned records | `test_returned_records_are_copies_agents_cannot_corrupt_environment_state` |
| M31 project path leaked to custom agents | **Real gap (Low)**: the path-leak test passed only because the scratch directory name did not contain "Genesis" | the test now asserts `PYTHONPATH` is absent from the agent's environment |
| M18 mark_all exceeds 30 findings | **Equivalent mutant**: `FullScanPolicy` truncates to 30 anyway | none needed |

After the fixes: M12, M20, M21, M31 are killed by dedicated tests, and the 15 mutants written for this stage's new logic
(difficulty profiles, evidence reconstruction, totals, decoy integrity, affected-id check, explanation text, recurring/split
decoys, per-level benchmark, hard compound rate, evidence cross-references) are all killed: **19 / 19** (`reports/mutation_check.json`).
Not repeated after the final code: the other 44 mutants.

## New generator findings

| # | Severity | Finding | Fix | Test |
|---|---|---|---|---|
| 14 | Medium | Difficulty changed only volume; the *kind* and subtlety of anomalies were identical at every level, so "hard" was not harder to investigate | `PROFILES` in `generator/mutations.py` (variants, magnitudes, shifts, lookalike reposts, cross-referencing evidence) | `test_realism_levels.py::test_fx_wrong_rate_magnitude_gets_subtler_with_difficulty` and 5 more; mutants M49-M51, M62, M63 |
| 15 | Low | Truth had no root cause / detailed explanation | `root_cause` + `explanation` per issue; transformed consistently (ids inside text remapped) | `test_every_issue_has_root_cause_...`, `test_transforms_preserve_links_explanations_...`; M58 |
| 16 | Low | Recurring-series decoy could be cut to < 3 invoices by the window; a split-payment decoy existed only by chance at easy difficulty (found by the new tests) | bounded cadences; explicit split-payment decoy | `test_recurring_series_decoy_exists_and_is_clean`, `test_decoy_kinds_all_present_in_every_difficulty`; M56, M57 |
| 17 | Low | One `float()` on fx rates in the validator | `Decimal` | `test_money_modules_use_no_float_calls` |
| 18 | Info | Reconstruction check proves evidence is **sufficient**, not **minimal** (a document id alone already reconstructs, because its journal is a public lookup) | documented; the matcher still requires the specific cited lines | `test_reconstruction_fails_with_evidence_of_another_issue` (negative control) |
