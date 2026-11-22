"""The audit-hook sandbox for Python agent processes. Every attack runs twice - WITHOUT the sandbox
(it must succeed, so the test is not vacuous) and WITH it (it must be blocked)."""
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from conftest import ALL4
from traceforge.environment.audit_env import AuditEnv
from traceforge.environment.isolation import SubprocessAgent
from traceforge.runner.episode_runner import run_episode
from traceforge.runner.sandbox import sandboxed_command

ROOT = Path(__file__).resolve().parents[1]

HOSTILE = textwrap.dedent('''
    import json, os, sys, tempfile
    def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()

    def attempt(fn):
        try:
            fn()
            return "ALLOWED"
        except PermissionError:
            return "BLOCKED"
        except Exception as e:                      # the attempt was permitted but failed for another reason
            return "ALLOWED(" + type(e).__name__ + ")"

    def run_attacks(cfg):
        import socket, subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        def listing(): os.listdir(cfg["outside_dir"])
        def sock():   socket.create_connection(("127.0.0.1", 9), timeout=1)
        def dns():    socket.getaddrinfo("example.com", 80)
        def proc():   subprocess.run([sys.executable, "-c", "print(1)"], capture_output=True, timeout=10)
        def shell():  os.system("exit 0")
        def native():
            try:                                    # on Windows even `import ctypes` loads kernel32 -> blocked at import
                import ctypes
                ctypes.WinDLL("kernel32") if os.name == "nt" else ctypes.CDLL(None)
            except (AttributeError, OSError) as e:
                if "kernel32" in str(e) or "dlopen" in str(e):
                    raise PermissionError(str(e))
                raise
        def mk():     os.mkdir(os.path.join(here, "newdir"))
        def rm():     os.remove(cfg["victim"])
        def tmpw():   open(os.path.join(os.environ.get("TEMP", "/tmp"), "tf_probe.txt"), "w").write("x")
        return {
            "read_private_truth": attempt(lambda: open(cfg["private_file"]).read()),
            "read_file_outside": attempt(lambda: open(cfg["outside_file"]).read()),
            "list_outside_dir": attempt(listing),
            "write_own_dir": attempt(lambda: open(os.path.join(here, "x.txt"), "w").write("x")),
            "write_tmp_file": attempt(tmpw),
            "mkdir": attempt(mk),
            "remove_file": attempt(rm),
            "network_connect": attempt(sock),
            "dns_lookup": attempt(dns),
            "subprocess": attempt(proc),
            "os_system": attempt(shell),
            "native_library": attempt(native),
            # controls that a legitimate agent needs
            "read_own_script": attempt(lambda: open(__file__).read()),
            "stdlib_imports": attempt(lambda: [__import__(m) for m in ("json", "decimal", "datetime", "random", "re", "collections")]),
        }

    cfg = json.loads(os.environ_cfg) if hasattr(os, "environ_cfg") else None
    for line in sys.stdin:
        m = json.loads(line)
        if m["type"] == "close": break
        if m["type"] == "reset":
            cfg = m["observation"]["manifest"]["__attack_cfg__"] if "__attack_cfg__" in m["observation"]["manifest"] else json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg.json")))
            send({"type": "ready"}); continue
        report = run_attacks(cfg)
        send({"type": "action", "action": {"action": "get_record", "table": "ledger", "id": "R" + json.dumps(report)}})
''')


@pytest.fixture
def lab(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    outside = tmp_path / "outside"
    (outside / "datasets" / "x" / "private").mkdir(parents=True)
    private_file = outside / "datasets" / "x" / "private" / "truth.json"
    private_file.write_text('{"expected_amount_minor": 1}', encoding="utf-8")
    (outside / "notes.txt").write_text("secret notes", encoding="utf-8")
    victim = outside / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    (agent / "agent.py").write_text(HOSTILE.replace("cfg = json.loads(os.environ_cfg) if hasattr(os, \"environ_cfg\") else None\n", "cfg = None\n"),
                                    encoding="utf-8")
    (agent / "cfg.json").write_text(json.dumps({"private_file": str(private_file), "outside_file": str(outside / "notes.txt"),
                                                "outside_dir": str(outside), "victim": str(victim)}), encoding="utf-8")
    return {"agent": agent / "agent.py", "outside": outside, "victim": victim}


def attack(lab, sandboxed: bool) -> dict:
    cmd = sandboxed_command(lab["agent"]) if sandboxed else [sys.executable, str(lab["agent"])]
    env = AuditEnv("eval")
    obs, _ = env.reset(1, ALL4)
    agent = SubprocessAgent(cmd=cmd, timeout=60)
    try:
        agent.reset(obs)
        action = agent.act(obs)
    finally:
        agent.close()
    return json.loads(action["id"][1:])


ATTACKS = ["read_private_truth", "read_file_outside", "list_outside_dir", "write_own_dir", "write_tmp_file", "mkdir", "remove_file",
           "network_connect", "dns_lookup", "subprocess", "os_system", "native_library"]


def test_without_the_sandbox_the_attacks_would_work(lab):
    report = attack(lab, sandboxed=False)
    assert all(report[a].startswith("ALLOWED") for a in ATTACKS), report         # proves the test is not vacuous
    assert not lab["victim"].exists()                                            # the agent really deleted a file


def test_with_the_sandbox_every_attack_is_blocked_and_nothing_changes(lab):
    victim_text = lab["victim"].read_text()
    report = attack(lab, sandboxed=True)
    assert {a: report[a] for a in ATTACKS} == {a: "BLOCKED" for a in ATTACKS}, report
    assert lab["victim"].read_text() == victim_text
    assert not (lab["agent"].parent / "agent" / "x.txt").exists() and not (lab["agent"].parent / "agent" / "newdir").exists()


def test_the_sandbox_does_not_break_what_a_legitimate_agent_needs(lab):
    report = attack(lab, sandboxed=True)
    assert report["read_own_script"] == "ALLOWED" and report["stdlib_imports"] == "ALLOWED"


def test_private_paths_are_denied_even_inside_an_allowed_root(tmp_path):
    agent = tmp_path / "agent"
    (agent / "private").mkdir(parents=True)
    (agent / "private" / "data.txt").write_text("x")
    (agent / "ok.txt").write_text("fine")
    script = agent / "a.py"
    script.write_text(textwrap.dedent('''
        import json, sys
        def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
        def tryread(p):
            try: open(p).read(); return "ALLOWED"
            except PermissionError: return "BLOCKED"
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        for line in sys.stdin:
            m = json.loads(line)
            if m["type"] == "close": break
            if m["type"] == "reset": send({"type": "ready"}); continue
            r = {"ok": tryread(os.path.join(here, "ok.txt")), "private_dir": tryread(os.path.join(here, "private", "data.txt")),
                 "truth_name": tryread(os.path.join(here, "truth.json"))}
            send({"type": "action", "action": {"action": "get_record", "table": "ledger", "id": "R" + json.dumps(r)}})
    '''), encoding="utf-8")
    env = AuditEnv("eval")
    obs, _ = env.reset(1, ALL4)
    agent_proc = SubprocessAgent(cmd=sandboxed_command(script), timeout=60)
    try:
        agent_proc.reset(obs)
        rep = json.loads(agent_proc.act(obs)["id"][1:])
    finally:
        agent_proc.close()
    assert rep == {"ok": "ALLOWED", "private_dir": "BLOCKED", "truth_name": "BLOCKED"}


def test_a_sandboxed_agent_still_plays_and_is_scored_normally():
    cmd = sandboxed_command(ROOT / "examples" / "custom_agent.py")
    res = run_episode(AuditEnv("eval"), lambda env: SubprocessAgent(cmd=cmd, timeout=60), seed=7, config={"clean": True})
    assert res["agent_error"] is None and res["terminated"] and res["f1"] == 1.0 and res["steps"] == 2


def test_built_in_policies_run_inside_the_sandbox_by_default():
    a = SubprocessAgent("reconciliation")
    try:
        assert "--sandbox" in a.cmd
    finally:
        a.close()
    b = SubprocessAgent("reconciliation", sandbox=False)
    try:
        assert "--sandbox" not in b.cmd
    finally:
        b.close()


def test_sandboxed_built_in_agent_result_equals_the_in_process_result():
    from traceforge.runner.episode_runner import make_agent
    a = run_episode(AuditEnv("eval"), make_agent("reconciliation"), seed=4, config=ALL4)
    b = run_episode(AuditEnv("eval"), make_agent("reconciliation", isolated=True), seed=4, config=ALL4)
    assert all(a[k] == b[k] for k in ("tp", "fp", "fn", "f1", "reward", "steps"))


def test_custom_sandboxed_agent_does_not_learn_the_project_location_from_its_environment(lab):
    script = lab["agent"].parent / "env_probe.py"
    script.write_text(textwrap.dedent('''
        import json, os, sys
        def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
        for line in sys.stdin:
            m = json.loads(line)
            if m["type"] == "close": break
            if m["type"] == "reset": send({"type": "ready"}); continue
            send({"type": "action", "action": {"action": "get_record", "table": "ledger", "id": "R" + json.dumps(sorted(os.environ))}})
    '''), encoding="utf-8")
    env = AuditEnv("eval")
    obs, _ = env.reset(1, ALL4)
    agent = SubprocessAgent(cmd=sandboxed_command(script), timeout=60)
    try:
        agent.reset(obs)
        names = json.loads(agent.act(obs)["id"][1:])
    finally:
        agent.close()
    assert not [n for n in names if n.upper() in ("TRACEFORGE_HOME", "VIRTUAL_ENV") or "DATASET" in n.upper()]
