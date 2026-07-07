"""Colored, per-stage CLI logging: banners, ✓/✗, timings, final summary table.

Colors degrade gracefully when stdout is not a TTY (e.g. piped to a file or run
by the evaluator's harness). No third-party dependency.
"""
from __future__ import annotations

import os
import sys


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        # Best-effort enable of ANSI VT processing on modern Windows terminals.
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


class Logger:
    def __init__(self):
        # Best-effort: force UTF-8 stdout so banners/glyphs render on Windows.
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        enc = (getattr(sys.stdout, "encoding", "") or "").lower()
        self.unicode = "utf" in enc
        self.color = _supports_color()
        # Symbols with ASCII fallbacks for non-UTF terminals.
        self.S_RUN = "▶" if self.unicode else ">"
        self.S_OK = "✓" if self.unicode else "[OK]"
        self.S_FAIL = "✗" if self.unicode else "[X]"
        self.S_WARN = "⚠" if self.unicode else "[!]"
        self.HBAR = "─" if self.unicode else "-"

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    # --- stage lifecycle -------------------------------------------------
    def stage_banner(self, name: str, index: int, note: str = "") -> None:
        bar = "=" * 60
        suffix = f"  ({note})" if note else ""
        print(self._c("36", bar))
        print(self._c("1;36", f"[{index:02d}] {name}{suffix}"))
        print(self._c("36", bar))

    def stage_start(self, name: str, index: int) -> None:
        print(self._c("1;34", f"\n{self.S_RUN} [{index:02d}] {name} ..."))

    def stage_done(self, name: str, elapsed_ms: float) -> None:
        mark = self._c("1;32", self.S_OK)
        print(f"  {mark} {name} completed in {elapsed_ms:.1f} ms")

    def stage_fail(self, name: str, exc: Exception) -> None:
        mark = self._c("1;31", self.S_FAIL)
        print(f"  {mark} {name} FAILED: {exc}")

    # --- generic ---------------------------------------------------------
    def info(self, msg: str) -> None:
        print(f"    {msg}")

    def warn(self, msg: str) -> None:
        print(f"  {self._c('1;33', self.S_WARN)} {msg}")

    def ok(self, msg: str) -> None:
        print(f"  {self._c('1;32', self.S_OK)} {msg}")

    def error(self, msg: str) -> None:
        print(f"  {self._c('1;31', self.S_FAIL)} {msg}")

    # --- summary table ---------------------------------------------------
    def summary_table(self, timings: dict[str, float], winner: str, metric: str,
                      score: float) -> None:
        bar = self.HBAR * 60
        print("\n" + self._c("1;36", bar))
        print(self._c("1;36", " PIPELINE SUMMARY"))
        print(self._c("1;36", bar))
        for name, ms in timings.items():
            print(f"  {name:<28} {ms:>8.1f} ms")
        total = sum(timings.values())
        print(self._c("2", f"  {'TOTAL':<28} {total:>8.1f} ms"))
        print(self._c("1;32", f"\n  WINNER: {winner}  ({metric}={score:.4f})"))
        print(self._c("1;36", bar))
