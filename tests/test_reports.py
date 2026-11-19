"""Benchmark runner, performance probe, audit and the new CLI commands."""
import csv
import json
from pathlib import Path

import pytest

from traceforge import cli
from traceforge.reporting import audit
from traceforge.reporting.performance import run_performance
from traceforge.runner.benchmark import COLUMNS, run_benchmark, summarize_rows, suite_episodes, write_benchmark

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bench():
    return run_benchmark(("dev", "presets"), datasets=ROOT / "datasets", seeds=2)


def test_every_policy_runs_on_identical_episodes(bench):
    by_policy = {}
    for r in bench["rows"]:
        by_policy.setdefault(r["policy"], []).append((r["suite"], r["episode"], r["issues"]))
    assert set(by_policy) == {"random", "reconciliation", "mark_all", "oracle"}
    assert all(v == by_policy["oracle"] for v in by_policy.values())
    assert len(by_policy["oracle"]) == 10 + 4 * 2


def test_rows_carry_all_requested_metrics(bench):
    for r in bench["rows"]:
        assert set(r) == set(COLUMNS)
        assert 0 <= r["precision"] <= 1 and 0 <= r["recall"] <= 1 and 0 <= r["f1"] <= 1 and -1 <= r["reward"] <= 1
        assert r["runtime_s"] >= 0 and r["findings"] >= 0 and r["agent_error"] is None
        assert r["tp"] + r["fn"] == r["issues"]


def test_oracle_row_is_perfect_and_costs_zero_inspection(bench):
    for r in (x for x in bench["rows"] if x["policy"] == "oracle"):
        assert r["f1"] == 1.0 and r["reward"] == 1.0 and r["inspection_steps"] == 0


def test_summary_is_computed_from_rows_not_hard_coded(bench):
    s = summarize_rows(bench["rows"])
    for entry in s:
        sel = [r for r in bench["rows"] if r["suite"] == entry["suite"] and r["policy"] == entry["policy"]]
        assert entry["episodes"] == len(sel) and entry["tp"] == sum(r["tp"] for r in sel)
        assert entry["mean_f1"] == pytest.approx(sum(r["f1"] for r in sel) / len(sel), abs=1e-4)


def test_benchmark_is_reproducible_apart_from_timing():
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("runtime_s", "env_elapsed_s")} for r in rows]
    a = run_benchmark(("presets",), datasets=ROOT / "datasets", seeds=2)
    b = run_benchmark(("presets",), datasets=ROOT / "datasets", seeds=2)
    assert strip(a["rows"]) == strip(b["rows"])


def test_json_and_csv_outputs(bench, tmp_path):
    j, c = write_benchmark(bench, tmp_path)
    data = json.loads(j.read_text())
    rows = list(csv.DictReader(c.open(encoding="utf-8")))
    assert len(rows) == len(data["rows"]) and list(rows[0]) == list(COLUMNS)
    assert float(rows[0]["f1"]) == data["rows"][0]["f1"]
    assert (tmp_path / "benchmark_summary.csv").exists()


def test_benchmark_includes_preset_suite_covering_all_presets():
    labels = {l.split("-")[0] for l, _, _ in suite_episodes("presets", ROOT / "datasets", 2)}
    assert labels == {"easy", "medium", "hard", "compound"}
    with pytest.raises(ValueError):
        suite_episodes("nonsense", ROOT / "datasets", 1)


def test_benchmark_does_not_assume_a_winner(bench):
    """The harness reports whatever happened; random must be allowed to lose AND the table must
    be able to represent any ordering (checked by feeding a synthetic reversed table)."""
    fake = [{**r, "f1": 1.0 - r["f1"], "tp": 0} for r in bench["rows"]]
    s = {(x["suite"], x["policy"]): x["mean_f1"] for x in summarize_rows(fake)}
    real = {(x["suite"], x["policy"]): x["mean_f1"] for x in summarize_rows(bench["rows"])}
    assert s[("dev", "random")] != real[("dev", "random")]


def test_performance_probe_measures_every_stage_at_all_sizes(tmp_path):
    rep = run_performance(tmp_path, sizes=(200, 500, 1000, 2000))
    rows = rep["rows"]
    assert [r["label"] for r in rows][-1].startswith("docs") and len(rows) == 5
    sizes = [r["ledger_rows"] for r in rows[:4]]
    assert sizes == sorted(sizes) and sizes[0] >= 200 and sizes[1] >= 500 and sizes[2] >= 1000 and sizes[3] >= 2000
    assert rows[-1]["documents"] >= 200
    for r in rows:
        for k in ("generate_s", "reset_from_dir_ms", "list_records_50_ms", "get_document_ms", "matcher_30x12_ms", "peak_python_mem_MB_generate"):
            assert r[k] > 0
        assert r["generate_s"] < 30 and r["reconciliation_f1"] == 1.0
    assert (tmp_path / "performance.json").exists()


def test_generation_time_scales_roughly_linearly(tmp_path):
    rows = run_performance(None, sizes=(500, 2000))["rows"][:2]
    ratio = (rows[1]["generate_s"] / rows[0]["generate_s"]) / (rows[1]["ledger_rows"] / rows[0]["ledger_rows"])
    assert ratio < 3                                    # not quadratic


def test_audit_passes_and_reports_every_section(tmp_path):
    rep = audit.run_audit(tmp_path, performance=False)
    assert rep["ok"], [s for s in rep["sections"] if not s["ok"]]
    names = " ".join(s["name"] for s in rep["sections"])
    for part in ("data sweep", "architecture", "tells", "reproducibility", "environment fuzzing", "calculator fuzzing"):
        assert part in names
    assert (tmp_path / "audit_report.json").exists()


def test_audit_architecture_scan_finds_a_planted_violation(tmp_path, monkeypatch):
    bad = tmp_path / "pkg"
    bad.mkdir()
    (bad / "evil.py").write_text("import random\nx = random.random()\ny = eval('1')\n", encoding="utf-8")
    monkeypatch.setattr(audit, "PKG", bad)
    ok, detail, data = audit.architecture()
    assert not ok and data["forbidden_calls"] and data["global_random"]


def test_cli_audit_and_benchmark_commands(tmp_path, capsys):
    assert cli.run(["audit", "--out", str(tmp_path), "--skip-performance"]) == 0
    assert "AUDIT: PASS" in capsys.readouterr().out
    assert cli.run(["benchmark", "--suite", "dev", "--datasets", str(ROOT / "datasets"), "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "reconciliation" in out and "oracle" in out and (tmp_path / "benchmark.csv").exists()


def test_cli_preset_flag(tmp_path, capsys):
    assert cli.run(["generate", "--seed", "4", "--preset", "compound", "--out", str(tmp_path), "--name", "c"]) == 0
    from traceforge.episode_io import load_episode
    ep = load_episode(tmp_path / "c")
    assert ep.private.provenance["compound_scenarios"]
    assert cli.run(["demo", "--preset", "easy", "--seed", "2", "--no-report"]) == 0
    assert "SUBMITTED FINDINGS" in capsys.readouterr().out


def test_runtime_is_standard_library_only_and_offline():
    import ast
    import sys
    pkg = ROOT / "src" / "traceforge"
    net = {"socket", "ssl", "http", "urllib", "ftplib", "smtplib", "xmlrpc", "requests", "httpx", "aiohttp"}
    seen = set()
    for f in pkg.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                seen |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                seen.add(node.module.split(".")[0])
    assert not (seen & net), seen & net
    assert all(m == "traceforge" or m in sys.stdlib_module_names for m in seen), sorted(m for m in seen if m not in sys.stdlib_module_names)
    assert "dependencies = []" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_spec_compliance_marks_only_what_tests_prove(tmp_path):
    from traceforge.reporting import spec_compliance as sc
    req = sc.REQUIREMENTS[0]
    all_pass = {t: [True] for _, _, ts in sc.REQUIREMENTS for t in ts}
    assert all(r["status"] == "PASS" for r in sc.evaluate(all_pass))
    one_fail = {**all_pass, req[2][0]: [True, False]}
    assert sc.evaluate(one_fail)[0]["status"] == "FAIL"
    missing = {k: v for k, v in all_pass.items() if k != req[2][1]}
    assert sc.evaluate(missing)[0]["status"] == "UNVERIFIED"
    skipped = {**all_pass, req[2][0]: [None]}
    assert sc.evaluate(skipped)[0]["status"] == "UNVERIFIED"
    md = sc.render(sc.evaluate(one_fail))
    assert "| Requirement | Implementation | Test" in md and "**FAIL**" in md


def test_every_test_cited_in_the_spec_matrix_exists():
    import re
    cited = {t for _, _, ts in __import__("traceforge.reporting.spec_compliance", fromlist=["x"]).REQUIREMENTS for t in ts}
    missing = []
    for t in cited:
        f, name = t.split("::")
        text = (ROOT / "tests" / f).read_text(encoding="utf-8")
        if not re.search(rf"def {re.escape(name)}\(", text):
            missing.append(t)
    assert not missing, missing


def test_benchmark_breaks_results_down_by_difficulty_level(bench, tmp_path):
    levels = {r["level"] for r in bench["rows"]}
    assert {"easy", "medium", "hard", "compound"} <= {l.split("+")[0] for l in levels}
    for e in bench["by_level"]:
        sel = [r for r in bench["rows"] if r["level"] == e["level"] and r["policy"] == e["policy"]]
        assert e["episodes"] == len(sel) and e["tp"] == sum(r["tp"] for r in sel) and e["issues"] == sum(r["issues"] for r in sel)
    oracle = [e for e in bench["by_level"] if e["policy"] == "oracle"]
    assert oracle and all(e["mean_f1"] == 1.0 and e["mean_inspection_steps"] == 0 for e in oracle)
    write_benchmark(bench, tmp_path)
    rows = list(csv.DictReader((tmp_path / "benchmark_by_level.csv").open(encoding="utf-8")))
    assert len(rows) == len(bench["by_level"]) and "mean_reward" in rows[0]
