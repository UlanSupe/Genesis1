"""Best-effort sandbox for PYTHON agent processes (defence in depth).

Installs an irreversible `sys.addaudithook` that, for the rest of the process lifetime, blocks:
  * any file WRITE / create / delete / rename / mkdir / chmod / link (nothing on disk can be changed);
  * any file READ or directory listing outside the allowed roots (interpreter, site-packages, the
    traceforge package, the agent script's own directory, the empty working directory);
  * ANY path that contains a `private` component or is named truth.json / provenance.json /
    clean_baseline.json / transformation.json, even inside an allowed root;
  * sockets (connect, bind, send, DNS lookups) - no network at all;
  * subprocess / os.system / exec / spawn / fork - no shell, no helper programs;
  * loading native libraries through ctypes (the usual way around audit hooks).

Usage:   python -m traceforge.runner.sandbox [--allow-read DIR ...] agent_script.py [args]
The built-in agent worker installs the same hook (`--sandbox`).

HONEST LIMITS: audit hooks are a speed bump, not a security boundary. Native code and a few
interpreter internals (e.g. Windows `_winapi`) are not audited, and non-Python agents cannot be
covered at all. For adversarial agents still use a container / separate OS user; the evaluator
never relies on this module for correctness (truth is simply not in the agent process)."""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import sysconfig
from pathlib import Path

DENY_NAMES = {"truth.json", "provenance.json", "clean_baseline.json", "transformation.json"}
DENY_COMPONENTS = {"private"}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
BLOCKED_EVENTS = {
    # network
    "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo", "socket.gethostbyname",
    "socket.gethostbyaddr", "socket.getnameinfo", "socket.getservbyname", "socket.getservbyport",
    # processes
    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty", "os.startfile",
    # filesystem mutation
    "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.chmod", "os.chown", "os.link", "os.symlink", "os.truncate",
    "os.utime", "os.mkfifo", "os.mknod", "os.chflags", "os.lchown", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
    "shutil.copytree", "shutil.move", "shutil.rmtree", "shutil.make_archive", "shutil.unpack_archive",
    # native code
    "ctypes.dlopen", "ctypes.dlsym", "ctypes.cdata", "winreg.OpenKey", "winreg.CreateKey", "winreg.SetValue",
    "webbrowser.open",
}


class SandboxViolation(PermissionError):
    """Raised inside the agent process when it attempts a blocked operation."""


def _norm(p) -> str:
    return os.path.normcase(os.path.realpath(os.fsdecode(p)))


def _under(path: str, roots: list) -> bool:
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def default_roots(extra=()) -> list:
    paths = sysconfig.get_paths()
    roots = [paths[k] for k in ("stdlib", "platstdlib", "purelib", "platlib") if k in paths]
    roots += [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, os.getcwd()]
    roots += [str(Path(__file__).resolve().parents[2])]                    # src/ (the traceforge package)
    roots += list(extra)
    return sorted({_norm(r) for r in roots if r})


def install(allow_read=()) -> None:
    roots = default_roots(allow_read)
    special = {_norm(os.devnull), "nul", "con"}

    def denied_by_name(path: str) -> bool:
        parts = {c.lower() for c in Path(path).parts}
        return Path(path).name.lower() in DENY_NAMES or bool(parts & DENY_COMPONENTS)

    def check_path(path, write: bool, what: str) -> None:
        if isinstance(path, int):                       # an already-open descriptor (pipes, std streams)
            return
        p = _norm(path)
        if p in special:
            return
        if write:
            raise SandboxViolation(f"sandbox: {what} {os.fsdecode(path)!r} is not allowed (read-only agent)")
        if denied_by_name(p):
            raise SandboxViolation(f"sandbox: access to {os.fsdecode(path)!r} is not allowed")
        if not _under(p, roots):
            raise SandboxViolation(f"sandbox: {what} {os.fsdecode(path)!r} is outside the allowed locations")

    def hook(event: str, args) -> None:
        if event in BLOCKED_EVENTS:
            raise SandboxViolation(f"sandbox: {event} is not allowed")
        if event == "open":
            path, mode, flags = (list(args) + [None, None])[:3]
            write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                    (isinstance(flags, int) and flags > 0 and bool(flags & WRITE_FLAGS))
            check_path(path, write, "open")
        elif event in ("os.listdir", "os.scandir"):
            if args and args[0] is not None:
                check_path(args[0], False, "listing")

    sys.addaudithook(hook)


def sandboxed_command(script: str | Path, allow_read=()) -> list:
    """Command line that runs `script` (a Python agent speaking the runner protocol) inside the sandbox."""
    cmd = [sys.executable, "-m", "traceforge.runner.sandbox"]
    for d in allow_read:
        cmd += ["--allow-read", str(d)]
    return cmd + [str(script)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="run a Python agent script inside the traceforge audit-hook sandbox")
    ap.add_argument("--allow-read", action="append", default=[], help="extra directory the agent may read")
    ap.add_argument("script")
    ap.add_argument("script_args", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    script = Path(ns.script).resolve()
    install([str(script.parent), *ns.allow_read])
    sys.argv = [str(script), *ns.script_args]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
