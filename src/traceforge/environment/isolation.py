"""Process isolation and leak guards.

The agent runs in a SEPARATE OS process started with a scrubbed environment and an
EMPTY working directory; it receives JSON observations and returns JSON actions.
Private data (truth, provenance, clean baseline) exists only in the evaluator process.

Runner-side resource protection against a misbehaving/hostile agent:
  * stdout is read with a hard per-message cap (an unterminated multi-GB line cannot
    exhaust the runner's memory; the agent is killed instead);
  * stderr is drained into a bounded tail buffer (cannot fill the disk or block);
  * every call has a timeout and the whole episode has a total deadline;
  * on POSIX the agent also gets RLIMIT_AS / RLIMIT_CPU (best effort; not on Windows).

Limits (documented, not hidden): a malicious agent with arbitrary code execution can
still read files the OS user can read (e.g. a dataset's private/ folder). For adversarial
agents run the agent process in a container / under a different OS user; episodes
generated in memory (reset(seed, options)) never touch the disk."""
from __future__ import annotations

import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from ..exceptions import AgentProtocolError, LeakError
from ..runner.agent_protocol import MAX_LINE, action_from, decode, encode

SRC_DIR = str(Path(__file__).resolve().parents[2])
_KEEP_ENV = ("SYSTEMROOT", "PATH", "TEMP", "TMP", "LANG")
STDERR_TAIL = 8192


def assert_no_leak(obj, tokens) -> None:
    """Raise LeakError if any private token appears in the JSON form of `obj`."""
    text = json.dumps(obj, sort_keys=True, default=str)
    for i, tok in enumerate(tokens):
        if tok and tok in text:
            raise LeakError(f"private token #{i} found in agent-visible data")


def _posix_limits(mem_mb: int, cpu_s: int):
    def apply():                                     # runs in the child before exec
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    return apply


class SubprocessAgent:
    """Runs a policy (or any command speaking the protocol) in an isolated process."""

    def __init__(self, policy: str | None = None, seed: int = 0, cmd: list | None = None,
                 timeout: float = 60.0, total_timeout: float = 900.0, max_line: int = MAX_LINE,
                 mem_mb: int = 2048, cpu_seconds: int = 900, verify: bool = False):
        builtin = cmd is None
        if cmd is None:
            if policy is None:
                raise ValueError("policy or cmd required")
            cmd = [sys.executable, "-m", "traceforge.runner.agent_worker",
                   "--policy", policy, "--seed", str(seed)] + (["--verify"] if verify else [])
        self.cmd, self.timeout, self.max_line = cmd, timeout, max_line
        self.deadline = time.monotonic() + total_timeout
        self._tmp = tempfile.TemporaryDirectory(prefix="tf_agent_")
        self.workdir = Path(self._tmp.name) / "work"          # empty cwd for the agent
        self.workdir.mkdir()
        env = {k: os.environ[k] for k in _KEEP_ENV if k in os.environ}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        if builtin:                       # only OUR worker needs the package; foreign agents must not learn
            env["PYTHONPATH"] = SRC_DIR   # where the project (and any stored dataset) lives
        extra = {"preexec_fn": _posix_limits(mem_mb, cpu_seconds)} if os.name == "posix" else {}
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, cwd=self.workdir, env=env, **extra)
        except OSError:
            self._tmp.cleanup()
            raise
        # binary pipes wrapped explicitly: bounded readline on stdout, read1 on stderr
        self._stdin = io.TextIOWrapper(self.proc.stdin, encoding="utf-8", newline="\n", write_through=True)
        self._stdout = io.TextIOWrapper(self.proc.stdout, encoding="utf-8", errors="replace", newline="\n")
        self._q: queue.Queue = queue.Queue()
        self._err_tail: deque = deque(maxlen=STDERR_TAIL)
        self._closed = False
        threading.Thread(target=self._pump_out, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()

    # --- reader threads -------------------------------------------------------
    def _pump_out(self) -> None:
        try:
            while True:
                line = self._stdout.readline(self.max_line + 1)   # bounded read
                if not line:
                    break
                if len(line) > self.max_line:
                    self._q.put("OVERSIZE")
                    self.proc.kill()
                    return
                self._q.put(line)
        except (ValueError, OSError):
            pass
        self._q.put(None)

    def _pump_err(self) -> None:
        try:
            while True:
                chunk = self.proc.stderr.read1(1024)
                if not chunk:
                    break
                self._err_tail.extend(chunk)                         # keeps only the tail
        except (ValueError, OSError):
            pass

    def stderr_tail(self) -> str:
        return bytes(self._err_tail).decode("utf-8", errors="replace")

    # --- protocol ---------------------------------------------------------------
    def _call(self, msg: dict) -> dict:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.close()
            raise AgentProtocolError("episode deadline exceeded")
        try:
            self._stdin.write(encode(msg))
            self._stdin.flush()
            line = self._q.get(timeout=min(self.timeout, remaining))
        except (queue.Empty, BrokenPipeError, OSError):
            self.close()
            raise AgentProtocolError("agent did not answer") from None
        if line == "OVERSIZE":
            self.close()
            raise AgentProtocolError("agent message too large")
        if line is None:
            raise AgentProtocolError("agent process exited")
        return decode(line)

    def reset(self, observation: dict) -> None:
        if self._call({"type": "reset", "observation": observation}).get("type") != "ready":
            raise AgentProtocolError("agent did not acknowledge reset")

    def act(self, observation: dict) -> dict:
        return action_from(self._call({"type": "observation", "observation": observation}))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.proc.poll() is None:
                try:
                    self._stdin.write(encode({"type": "close"}))
                    self._stdin.flush()
                except OSError:
                    pass
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        finally:
            for f in (self._stdin, self._stdout, self.proc.stderr):
                try:
                    f.close()
                except OSError:
                    pass
            try:
                self._tmp.cleanup()
            except OSError:
                pass
