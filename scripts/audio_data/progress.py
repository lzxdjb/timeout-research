"""Small stderr-only progress displays for the standalone data preparation tools."""

from __future__ import annotations

import math
import sys
import time


class Progress:
    """TTY bars in auto mode; explicit on mode also emits periodic log lines.

    Unknown totals report counts and throughput without inventing a percentage.
    Initial counts support resumed work without counting old work in throughput.
    """

    def __init__(self, label, *, total=None, unit="records", mode="auto", interval=2.0, initial=0):
        if mode not in {"auto", "on", "off"}:
            raise ValueError("progress mode must be auto, on, or off")
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("progress interval must be finite and nonnegative")
        if initial < 0 or (total is not None and total < 0):
            raise ValueError("progress counts must be nonnegative")
        self.label = str(label)
        self.total = total
        self.unit = unit
        self.current = initial
        self.initial = initial
        self.interval = interval
        self.stream = sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = mode == "on" or (mode == "auto" and self.tty)
        self.started = time.monotonic()
        self.last_render = self.started
        self.closed = False
        self._render(force=True)

    def update(self, amount=1):
        if amount < 0:
            raise ValueError("progress increments must be nonnegative")
        self.current += amount
        self._render()

    def set_total(self, total):
        if total is not None and total < 0:
            raise ValueError("progress total must be nonnegative")
        self.total = total
        self._render(force=True)

    def refresh(self, *, force=False):
        """Emit a heartbeat when work is waiting without a counter change."""
        self._render(force=force)

    def _count(self, value):
        if self.unit == "bytes":
            for divisor, suffix in ((2**30, "GiB"), (2**20, "MiB"), (2**10, "KiB")):
                if value >= divisor:
                    return f"{value / divisor:.2f} {suffix}"
            return f"{value:,.0f} B"
        return f"{value:,.0f}"

    def _render(self, *, force=False):
        if not self.enabled or self.closed:
            return
        now = time.monotonic()
        if not force and now - self.last_render < self.interval:
            return
        elapsed = max(0.0, now - self.started)
        rate = (self.current - self.initial) / elapsed if elapsed > 0 else 0.0
        count = self._count(self.current)
        if self.total is not None:
            fraction = min(1.0, self.current / self.total) if self.total else 1.0
            count += f"/{self._count(self.total)} {fraction:.1%}"
            if self.tty:
                filled = round(20 * fraction)
                count = "[" + "=" * filled + " " * (20 - filled) + "] " + count
        units = "" if self.unit == "bytes" else f" {self.unit}"
        rate_units = "" if self.unit == "bytes" else f" {self.unit}"
        line = f"{self.label}: {count}{units} | {self._count(rate)}{rate_units}/s | {elapsed:.1f}s elapsed"
        self.stream.write(("\r\033[K" if self.tty else "") + line + ("" if self.tty else "\n"))
        self.stream.flush()
        self.last_render = now

    def close(self):
        if self.closed:
            return
        self._render(force=True)
        if self.enabled and self.tty:
            self.stream.write("\n")
            self.stream.flush()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
