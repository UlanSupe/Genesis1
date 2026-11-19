import json
import subprocess
import sys
from pathlib import Path

import pytest

from traceforge import cli
from traceforge.generator.builder import dataset_specs, generate_dataset
from traceforge.runner.episode_runner import run_split
from traceforge.evaluation.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds")
    generate_dataset(d)
    return d


def test_dataset_has_ten_train_and_ten_dev_episodes(datasets):
    for split in ("train", "dev"):
        names = json.loads((datasets / split / "index.json").read_text())["episodes"]
        assert len(names) == 10
        for n in names:
            assert (datasets / split / n / "public" / "ledger.csv").exists()
            assert (datasets / split / n / "private" / "truth.json").exists()


def test_dataset_covers_clean_each_type_both_scenarios_and_many_currencies():
    from traceforge.episode_io import load_episode
    for split in ("train", "dev"):
        eps = [load_episode(ROOT / "datasets" / split / n) for n, *_ in dataset_specs(split)]
        truth_types = [{t["type"] for t in e.private.truth} for e in eps]
        assert sum(1 for t in truth_types if not t) >= 2                          # clean
        assert {"duplicate", "allocation", "fx", "cutoff"} <= set().union(*truth_types)
        scen = {s["scenario"] for e in eps for s in e.private.provenance["compound_scenarios"]}
        assert scen == {"cross_border_partial", "period_end_duplicate"}
        assert len({e.public.manifest["base_currency"] for e in eps}) >= 4
        assert len({e.public.manifest["period_start"][:7] for e in eps}) >= 8     # different periods
        assert all(e.private.provenance["validation"]["ok"] for e in eps)


def test_run_all_policies_over_a_split(datasets):
    scores = {p: summarize(run_split(datasets, "dev", p, mode="eval")) for p in ("oracle", "reconciliation", "mark_all", "random")}
    assert scores["oracle"]["mean_f1"] == 1.0
    assert scores["reconciliation"]["mean_f1"] > scores["random"]["mean_f1"]
    assert scores["mark_all"]["mean_reward"] < 0


def test_run_split_writes_clean_trajectories(datasets, tmp_path):
    run_split(datasets, "train", "reconciliation", mode="eval", traj_dir=tmp_path)
    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == 10
    text = "".join(f.read_text(encoding="utf-8") for f in files)
    assert "ISS-" not in text and "private" not in text


def test_cli_generate_single_inspect_demo_evaluate(tmp_path, capsys):
    assert cli.run(["generate", "--seed", "3", "--out", str(tmp_path), "--name", "one", "--scenario", "period_end_duplicate"]) == 0
    assert cli.run(["inspect", str(tmp_path / "one"), "--table", "invoices", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "ledger rows" in out and "expected_amount_minor" not in out           # public view only
    assert cli.run(["inspect", str(tmp_path / "one"), "--instructor"]) == 0
    assert "INSTRUCTOR MODE" in capsys.readouterr().out
    assert cli.run(["demo", "--episode", str(tmp_path / "one"), "--no-report"]) == 0
    demo = capsys.readouterr().out
    assert "SUBMITTED FINDINGS" in demo and "expected_amount_minor" not in demo and "INSTRUCTOR" not in demo
    assert cli.run(["demo", "--seed", "5", "--instructor", "--no-report"]) == 0
    assert "INSTRUCTOR MODE" in capsys.readouterr().out


def test_cli_evaluate_prints_table_and_writes_json(datasets, tmp_path, capsys):
    rc = cli.run(["evaluate", "--datasets", str(datasets), "--split", "dev", "--policy", "all",
                  "--out", str(tmp_path), "--trajectories", str(tmp_path / "t")])
    out = capsys.readouterr().out
    assert rc == 0 and "reconciliation" in out and "oracle" in out
    data = json.loads((tmp_path / "evaluation.json").read_text())
    assert {d["policy"] for d in data} == {"random", "reconciliation", "mark_all", "oracle"}


def test_cli_runs_as_module_with_one_command():
    p = subprocess.run([sys.executable, "-m", "traceforge", "--help"], capture_output=True, text=True,
                       cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert p.returncode == 0 and all(c in p.stdout for c in ("generate", "verify", "demo", "evaluate", "inspect"))


def test_verify_command_end_to_end_without_nested_pytest(tmp_path):
    from traceforge.reporting.verification_report import run_verification
    report = run_verification(ROOT, run_tests=False, out_dir=tmp_path)
    failed = [s for s in report["sections"] if not s["ok"]]
    assert report["ok"], failed
    names = " ".join(s["name"] for s in report["sections"])
    assert "audit" in names and "benchmark" in names and "metamorphic" in names
    for f in ("verification_report.md", "transformation_report.json", "audit_report.json", "benchmark.json", "benchmark.csv", "performance.json"):
        assert (tmp_path / f).exists(), f
    pairs = json.loads((tmp_path / "transformation_report.json").read_text())
    assert pairs["summary"]["pairs"] == 24 and pairs["summary"]["failed_pairs"] == []


def test_missing_dataset_gives_clean_error(tmp_path, capsys):
    assert cli.run(["evaluate", "--datasets", str(tmp_path / "nope"), "--out", str(tmp_path / "o")]) == 2
