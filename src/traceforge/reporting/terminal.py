"""Tiny terminal presentation helper (standard library only).

* Colour is used only when it is supported and wanted: `--color/--no-color`, the NO_COLOR and
  FORCE_COLOR environment variables, stdout being a TTY and TERM != "dumb". On Windows the
  console's VT mode is switched on when possible; otherwise colour stays off.
* Output is pure ASCII (tables use + - |), so it never fails on legacy code pages and is
  identical with and without colour apart from the escape codes."""
from __future__ import annotations

import os
import re
import sys

_ANSI = re.compile(r"\x1b\[[0-9;]*m")           # a real ESC [ ... m sequence
CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34",
         "magenta": "35", "cyan": "36"}


def _enable_windows_vt() -> bool:
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        handle = k.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if not k.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def detect_color(stream=None, requested: bool | None = None) -> bool:
    if requested is not None:                      # an explicit --color / --no-color always wins
        if requested:
            _enable_windows_vt()                   # best effort; modern terminals understand ANSI anyway
        return requested
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    if not (hasattr(stream, "isatty") and stream.isatty()) or os.environ.get("TERM") == "dumb":
        return False
    return _enable_windows_vt()


def ascii_safe(text) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


class Term:
    def __init__(self, stream=None, color: bool | None = None, width: int = 100):
        self.stream = stream or sys.stdout
        self.color = detect_color(self.stream, color)
        self.width = width

    # -- low level ------------------------------------------------------------
    def style(self, text, *names) -> str:
        text = ascii_safe(text)
        if not self.color or not names:
            return text
        return "\033[" + ";".join(CODES[n] for n in names) + "m" + text + "\033[0m"

    def write(self, text: str = "") -> None:
        self.stream.write(ascii_safe(text) + "\n")        # ANSI escapes are ASCII too
        self.stream.flush()

    # -- structure --------------------------------------------------------------
    def banner(self, title: str, subtitle: str = "") -> None:
        bar = "=" * self.width
        self.write(self.style(bar, "cyan"))
        self.write(self.style(title, "bold", "cyan"))
        if subtitle:
            self.write(self.style(subtitle, "dim"))
        self.write(self.style(bar, "cyan"))

    def section(self, n: int, total: int, title: str) -> None:
        self.write("")
        self.write(self.style(f"[{n}/{total}] {title}", "bold", "blue"))
        self.write(self.style("-" * self.width, "dim"))

    def kv(self, pairs: list, key_width: int | None = None) -> None:
        w = key_width or max(len(ascii_safe(k)) for k, _ in pairs)
        for k, v in pairs:
            self.write(f"  {self.style(ascii_safe(k).ljust(w), 'dim')}  {ascii_safe(v)}")

    def progress(self, i: int, total: int, width: int = 20) -> str:
        filled = int(width * i / total)
        return "[" + "#" * filled + "." * (width - filled) + f"] {i:>3}/{total}"

    def ok(self, text): return self.style(text, "green")
    def bad(self, text): return self.style(text, "red")
    def warn(self, text): return self.style(text, "yellow")

    def table(self, headers: list, rows: list, align: str | None = None, max_col: int = 46) -> None:
        """ASCII table. align: string of 'l'/'r' per column (default left)."""
        cells = [[self._cut(ascii_safe(c), max_col) for c in r] for r in rows]
        heads = [ascii_safe(h) for h in headers]
        align = align or "l" * len(heads)
        vis = lambda x: _ANSI.sub("", x)             # width = visible characters (cells may carry colour codes)
        widths = [max(len(heads[i]), *(len(vis(r[i])) for r in cells)) if cells else len(heads[i]) for i in range(len(heads))]
        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

        def line(vals, bold=False):
            parts = []
            for i, (v, w) in enumerate(zip(vals, widths)):
                pad = " " * (w - len(vis(v)))
                parts.append(" " + (pad + v if align[i] == "r" else v + pad) + " ")
            text = "|" + "|".join(parts) + "|"
            return self.style(text, "bold") if bold else text
        self.write(sep)
        self.write(line(heads, bold=True))
        self.write(sep)
        for r in cells:
            self.write(line(r))
        self.write(sep)

    @staticmethod
    def _cut(s: str, n: int) -> str:
        if _ANSI.search(s):                           # never cut through an escape sequence
            return s
        return s if len(s) <= n else s[: n - 3] + "..."
