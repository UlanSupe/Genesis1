"""The demonstration: real execution, reproducible, leak-free, usable without colour or extras."""
import html.parser
import io
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from conftest import ALL4
from traceforge import cli
from traceforge.config import GeneratorConfig
from traceforge.environment.audit_env import AuditEnv
from traceforge.generator.builder import generate_episode
from traceforge.policies.reconciliation_policy import ReconciliationPolicy
from traceforge.reporting import html_report
from traceforge.reporting.demo_report import DEMO_CONFIG, DEMO_SEED, run_demo
from traceforge.reporting.terminal import Term, detect_color
from traceforge.runner.benchmark import run_benchmark
from traceforge.runner.episode_runner import make_agent, run_episode

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[")


def run_cli(tmp_path, capsys, *extra, color="--no-color", bench=False):
    args = ["demo", color, "--report-dir", str(tmp_path / "rep"), "--trajectories", str(tmp_path / "traj"),
            "--datasets", str(ROOT / "datasets")] + ([] if bench else ["--no-benchmark"]) + list(extra)
    assert cli.run([a for a in args if a]) == 0
    return capsys.readouterr().out


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    buf = io.StringIO()
    res = run_demo(term=Term(buf, color=False), report_dir=d / "rep", traj_dir=d / "traj", datasets=ROOT / "datasets", benchmark_seeds=2)
    return res, buf.getvalue(), d


# ------------------------------------------------------------------ execution and content
def test_demo_command_runs_with_a_fixed_seed_and_shows_every_stage(tmp_path, capsys):
    out = run_cli(tmp_path, capsys)
    for part in ("[1/6]", "[2/6]", "[3/6]", "[4/6]", "[5/6]", "[6/6]", "Generated synthetic company", "SUBMITTED FINDINGS",
                 "TRUE POSITIVE", "precision", "recall", "F1", "inspection steps", "reward =", "Reproduce", f"seed {DEMO_SEED}"):
        assert part in out, part
    for tool in ("list_records", "get_document", "get_record", "calculator", "submit"):
        assert tool in out


def test_demo_default_scenario_has_interacting_anomalies():
    ep = generate_episode(DEMO_SEED, DEMO_CONFIG)
    scen = ep.private.provenance["compound_scenarios"]
    assert {s["scenario"] for s in scen} == {"cross_border_partial", "period_end_duplicate"}
    assert len(ep.private.truth) == 4 and {t["type"] for t in ep.private.truth} == {"duplicate", "allocation", "fx", "cutoff"}


def test_demo_agent_really_uses_the_investigation_tools(demo_run):
    res, out, _ = demo_run
    assert {"list_records", "get_document", "get_record", "calculator", "submit"} <= set(res["tool_usage"])
    assert sum(res["tool_usage"].values()) == len(res["steps"]) == res["evaluation"]["inspection_steps"] + 1
    assert all(f["evidence_ids"] for f in res["findings"])


def test_demo_reports_the_evaluator_comparison(demo_run):
    res, out, _ = demo_run
    ev = res["evaluation"]
    assert ev["tp"] == sum(f["status"] == "TP" for f in res["findings"]) and ev["fp"] == sum(f["status"] == "FP" for f in res["findings"])
    assert (ev["tp"], ev["fp"], ev["fn"]) == (4, 0, 0) and ev["f1"] == 1.0
    assert "false negatives (FN)" in out and "TRUE POSITIVE" in out


# ------------------------------------------------------------------ metrics are really computed
def test_printed_metrics_equal_the_official_formula_and_an_independent_run(demo_run):
    res, out, _ = demo_run
    ev = res["evaluation"]
    tp, fp, fn = ev["tp"], ev["fp"], ev["fn"]
    f1 = Fraction(1) if 2 * tp + fp + fn == 0 else Fraction(2 * tp, 2 * tp + fp + fn)
    want = float(max(Fraction(-1), min(Fraction(1), 2 * f1 - 1 - Fraction(1, 500) * ev["inspection_steps"])))
    assert ev["reward"] == want and ev["f1"] == float(f1)
    assert f"{ev['reward']:.3f}" in out and f"{ev['f1']:.3f}" in out and str(ev["inspection_steps"]) in out
    # independent execution of the same agent on the same episode through the normal runner
    r = run_episode(AuditEnv("eval"), make_agent("reconciliation", 0, False, True), seed=DEMO_SEED, config=DEMO_CONFIG)
    assert (r["tp"], r["fp"], r["fn"], r["inspection_steps"], r["reward"]) == (tp, fp, fn, ev["inspection_steps"], ev["reward"])


def test_verification_costs_exactly_its_extra_steps_and_keeps_the_findings():
    plain = run_episode(AuditEnv("eval"), make_agent("reconciliation"), seed=DEMO_SEED, config=DEMO_CONFIG)
    ver = run_episode(AuditEnv("eval"), make_agent("reconciliation", 0, False, True), seed=DEMO_SEED, config=DEMO_CONFIG)
    assert (ver["tp"], ver["fp"], ver["fn"]) == (plain["tp"], plain["fp"], plain["fn"])
    assert ver["inspection_steps"] > plain["inspection_steps"] and ver["reward"] < plain["reward"]
    assert ver["reward"] == pytest.approx(plain["reward"] - 0.002 * (ver["inspection_steps"] - plain["inspection_steps"]))


def test_verification_drops_a_finding_the_calculator_contradicts():
    env = AuditEnv("eval")
    obs, _ = env.reset(DEMO_SEED, DEMO_CONFIG)
    pol = ReconciliationPolicy(verify=True)
    pol.reset(obs)
    sabotaged, done, action = False, False, None
    while not done:
        action = pol.act(obs)
        if pol._findings is not None and not sabotaged:
            fx = [f for f in pol._findings if f["type"] == "fx"]
            assert fx
            fx[0]["_expected_anchor"] += 1000             # pretend the finding rested on a wrong number
            sabotaged = True
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    submitted = action["findings"]["findings"]
    assert "fx" not in [f["type"] for f in submitted] and len(submitted) == 3
    assert not any(k.startswith("_") for f in submitted for k in f)          # private notes never leave the policy


def test_isolated_agent_demo_matches_in_process_demo(tmp_path):
    a = run_demo(term=Term(io.StringIO(), color=False), report_dir=None, traj_dir=None, benchmark=False)
    b = run_demo(term=Term(io.StringIO(), color=False), report_dir=None, traj_dir=None, benchmark=False, isolated=True)
    assert a["evaluation"] == b["evaluation"] and a["steps"] == b["steps"]


# ------------------------------------------------------------------ reproducibility
def _normalise(text: str) -> str:
    text = re.sub(r"(?m)^  (HTML report|JSON result|Agent trajectory) .*$\n?", "", text)   # per-run output paths
    return re.sub(r"\d+(\.\d+)?s\)", "Xs)", text)           # the printed benchmark runtime


def test_fixed_seed_demo_is_reproducible_apart_from_timing(tmp_path):
    runs = []
    for i in range(2):
        buf = io.StringIO()
        res = run_demo(term=Term(buf, color=False), report_dir=tmp_path / f"r{i}", traj_dir=tmp_path / f"t{i}",
                       datasets=ROOT / "datasets", benchmark_seeds=2)
        res.pop("runtime_seconds")
        res.pop("trajectory")                      # absolute path of this test's temp directory
        res["benchmark"].pop("seconds")
        for e in res["benchmark"]["summary"]:
            e.pop("mean_runtime_s")
        runs.append((res, _normalise(buf.getvalue())))
    assert runs[0][0] == runs[1][0] and runs[0][1] == runs[1][1]
    assert (tmp_path / "t0" / f"demo_seed{DEMO_SEED}.jsonl").read_text() == (tmp_path / "t1" / f"demo_seed{DEMO_SEED}.jsonl").read_text()


def test_different_seed_gives_a_different_episode_and_digest(demo_run):
    res, _, _ = demo_run
    other = run_demo(seed=DEMO_SEED + 1, term=Term(io.StringIO(), color=False), report_dir=None, traj_dir=None, benchmark=False)
    assert other["public_data_sha256"] != res["public_data_sha256"] and other["episode_id"] == res["episode_id"]


# ------------------------------------------------------------------ no truth leakage
def _forbidden(seed=DEMO_SEED, config=DEMO_CONFIG):
    env = AuditEnv("eval")
    env.reset(seed, config)
    out = ["ISS-", "MUT-", "root_cause", "expected_amount_minor", "allowed_evidence_sets", "causal_description", "clean_baseline"]
    for t in env.evaluator_truth():
        out += [t["root_cause"], t["explanation"][:50]]
    return out


def test_normal_demo_never_exposes_the_hidden_truth(demo_run):
    res, out, d = demo_run
    blobs = {"terminal": out, "html": (d / "rep" / "demo_report.html").read_text(encoding="utf-8"),
             "json": (d / "rep" / "demo_result.json").read_text(encoding="utf-8"),
             "trajectory": (d / "traj" / f"demo_seed{DEMO_SEED}.jsonl").read_text(encoding="utf-8")}
    for name, text in blobs.items():
        for token in _forbidden():
            assert token not in text, (name, token)
    assert "INSTRUCTOR" not in out and "instructor" not in blobs["json"].lower().replace('"instructor_included": false', "")


def test_instructor_mode_shows_truth_in_terminal_and_report_but_never_in_the_trajectory(tmp_path):
    buf = io.StringIO()
    res = run_demo(term=Term(buf, color=False), report_dir=tmp_path / "r", traj_dir=tmp_path / "t", benchmark=False, instructor=True)
    out, page = buf.getvalue(), (tmp_path / "r" / "demo_report.html").read_text(encoding="utf-8")
    assert "INSTRUCTOR MODE" in out and "journal_posted_twice" in out and "never visible to the agent" in out
    assert "Instructor section" in page and "journal_posted_twice" in page
    traj = (tmp_path / "t" / f"demo_seed{DEMO_SEED}.jsonl").read_text(encoding="utf-8")
    for token in _forbidden():
        assert token not in traj, token
    assert res["instructor_included"] is True and len(res["instructor"]) == 4


def test_demo_observations_are_leak_scanned_during_the_run(demo_run, monkeypatch):
    from traceforge.reporting import demo_report
    calls = []
    real = demo_report.assert_no_leak
    monkeypatch.setattr(demo_report, "assert_no_leak", lambda o, t: (calls.append(1), real(o, t)))
    run_demo(term=Term(io.StringIO(), color=False), report_dir=None, traj_dir=None, benchmark=False)
    assert len(calls) >= 20


# ------------------------------------------------------------------ terminal: colour support
def test_no_color_output_is_plain_ascii_and_has_no_escape_codes(tmp_path, capsys):
    out = run_cli(tmp_path, capsys)
    assert not ANSI.search(out) and out.isascii()


def test_color_is_used_only_when_requested_or_supported(tmp_path, capsys, monkeypatch):
    forced = run_cli(tmp_path, capsys, color="--color")
    assert ANSI.search(forced) and forced.isascii()
    plain_ref = run_cli(tmp_path, capsys)
    assert re.sub(r"\x1b\[[0-9;]*m", "", forced) == plain_ref        # colour changes nothing but the escape codes
    monkeypatch.delenv("NO_COLOR", raising=False)
    plain = run_cli(tmp_path, capsys, color="")                   # captured stdout is not a TTY -> no colour
    assert not ANSI.search(plain)


def test_colour_detection_rules(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self): return True
    for var in ("NO_COLOR", "FORCE_COLOR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setattr("traceforge.reporting.terminal._enable_windows_vt", lambda: True)
    assert detect_color(Tty()) is True and detect_color(io.StringIO()) is False
    monkeypatch.setenv("TERM", "dumb")
    assert detect_color(Tty()) is False
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("NO_COLOR", "1")
    assert detect_color(Tty()) is False
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert detect_color(io.StringIO()) is True
    assert detect_color(Tty(), requested=False) is False and detect_color(io.StringIO(), requested=True) is True


def test_table_renderer_aligns_truncates_and_stays_ascii():
    buf = io.StringIO()
    Term(buf, color=False).table(["name", "value"], [["a" * 80, 5], ["bé", 12345]], "lr", max_col=20)
    lines = buf.getvalue().splitlines()
    assert len({len(l) for l in lines}) == 1 and lines[0] == lines[2] == lines[-1]
    assert "..." in buf.getvalue() and buf.getvalue().isascii() and "?" in buf.getvalue()


# ------------------------------------------------------------------ clear errors
def test_clear_error_messages_and_exit_codes(tmp_path, capsys):
    assert cli.run(["demo", "--episode", str(tmp_path / "missing"), "--no-report"]) == 2
    assert "episode directory not found" in capsys.readouterr().err
    assert cli.run(["demo", "--policy", "oracle", "--isolated", "--no-report"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "oracle" in err and "Traceback" not in err
    assert cli.run(["generate", "--seed", "1", "--out", str(tmp_path), "--config", str(tmp_path / "nope.json")]) == 2


# ------------------------------------------------------------------ HTML report
class _Checker(html.parser.HTMLParser):
    VOID = {"meta", "br", "hr", "img", "input", "link", "line", "rect"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.counts = [], [], {}

    def handle_starttag(self, tag, attrs):
        self.counts[tag] = self.counts.get(tag, 0) + 1
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.counts[tag] = self.counts.get(tag, 0) + 1

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append((tag, self.stack[-3:]))
        else:
            self.stack.pop()


def test_html_report_is_generated_and_well_formed(demo_run):
    res, _, d = demo_run
    page = (d / "rep" / "demo_report.html").read_text(encoding="utf-8")
    chk = _Checker()
    chk.feed(page)
    assert not chk.errors and not chk.stack, (chk.errors[:2], chk.stack)
    assert chk.counts.get("svg", 0) >= 6 and chk.counts.get("table", 0) >= 5
    for heading in ("What this is", "Architecture", "Generated scenario", "Anomaly categories", "How the agent investigated",
                    "Evaluation of this episode", "Baseline policy comparison", "Reproducibility"):
        assert heading in page
    assert "<script src" not in page and "http://" not in page.replace("http://www.w3.org", "") and "https://" not in page   # self-contained, offline


def test_html_charts_and_tables_use_the_real_benchmark_numbers(demo_run):
    res, _, d = demo_run
    page = (d / "rep" / "demo_report.html").read_text(encoding="utf-8")
    again = run_benchmark(("dev", "presets"), datasets=ROOT / "datasets", seeds=2)
    ov = res["benchmark"]["overall"]
    for pol in ("random", "reconciliation", "mark_all", "oracle"):
        rows = [r for r in again["rows"] if r["policy"] == pol]
        assert ov[pol]["f1"] == pytest.approx(sum(r["f1"] for r in rows) / len(rows))
        assert ov[pol]["reward"] == pytest.approx(sum(r["reward"] for r in rows) / len(rows))
        assert f"{ov[pol]['f1']:.3f}" in page and f"{ov[pol]['reward']:.3f}" in page
    island = json.loads(re.search(r"<script type='application/json' id='report-data'>(.*?)</script>", page, re.S).group(1))
    assert island["benchmark"]["overall"] == ov and island["episode"]["evaluation"] == res["evaluation"]
    assert res["benchmark"]["runs"] == len(again["rows"])


def test_html_escapes_untrusted_text():
    evil = "</script><img src=x onerror=alert(1)>"
    res = {"seed": 1, "episode_id": evil, "version": "1", "python": "3", "platform": "x", "config": {"a": evil},
           "agent": {"policy": "p", "verify": False, "isolated": False},
           "manifest": {"record_count": 1, "document_count": 1, "base_currency": "EUR", "timezone": "+00:00",
                        "period_start": "2024-01-01T00:00", "period_end": "2024-02-01T00:00"},
           "facts": {"entities": 1, "invoices": 1, "payments": 0, "refunds": 0, "split_payments": 0, "fx_rows": 1, "currency_docs": {"EUR": 1}},
           "steps": [{"step": 1, "tool": evil, "summary": evil, "error": False}], "tool_usage": {"submit": 1},
           "findings": [{"type": evil, "affected_ids": [evil], "claimed_amount_minor": 1, "currency": "EUR", "evidence_ids": [evil], "status": "FP", "explanation": ""}],
           "evaluation": {"tp": 0, "fp": 1, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "reward": -1.0, "inspection_steps": 0, "valid": True},
           "trajectory": None, "reproduce": evil, "public_data_sha256": evil, "runtime_seconds": 0.1}
    page = html_report.render_html(res)
    assert evil not in page and "<img" not in page and "<\\/script" not in page
    assert page.count("</script>") == 1 and "&lt;img" in page                  # escaped in the HTML body
    island = json.loads(re.search(r"<script type='application/json' id='report-data'>(.*?)</script>", page, re.S).group(1))
    assert island["episode"]["episode_id"] == evil                              # data island round-trips exactly


def test_report_without_benchmark_still_renders(tmp_path):
    res = run_demo(term=Term(io.StringIO(), color=False), report_dir=tmp_path, traj_dir=None, benchmark=False)
    page = (tmp_path / "demo_report.html").read_text(encoding="utf-8")
    assert "Baseline policy comparison" not in page and "Reproducibility" in page and "benchmark" not in res


def test_demo_works_with_the_standard_library_only():
    code = ("import io, sys\n"
            "from traceforge.reporting.demo_report import run_demo\n"
            "from traceforge.reporting.terminal import Term\n"
            "r = run_demo(term=Term(io.StringIO(), color=False), report_dir=None, traj_dir=None, benchmark=False)\n"
            "third = sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] not in sys.stdlib_module_names "
            "and not m.startswith('_') and m.split('.')[0] != 'traceforge'})\n"
            "print('F1', r['evaluation']['f1'], 'THIRD', third)\n")
    p = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")})
    assert p.returncode == 0, p.stderr[-400:]
    assert "F1 1.0" in p.stdout and "THIRD []" in p.stdout


def test_demo_runs_as_one_command_from_the_module_entry_point(tmp_path):
    p = subprocess.run([sys.executable, "-m", "traceforge", "demo", "--no-color", "--no-benchmark", "--report-dir", str(tmp_path),
                        "--trajectories", str(tmp_path)], capture_output=True, text=True, cwd=ROOT,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert p.returncode == 0 and "reward =" in p.stdout and (tmp_path / "demo_report.html").exists()
