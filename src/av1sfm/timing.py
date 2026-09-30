"""Wall-clock and CPU accounting for pipeline stages."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


def _cpu_seconds() -> float:
    """User + system CPU of this process and its waited-for children (e.g. ffmpeg)."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


@dataclass
class Stage:
    name: str
    wall_s: float = 0.0
    cpu_s: float = 0.0

    @property
    def cpu_percent(self) -> float:
        """Average CPU utilisation, 100 % = one core fully busy (like `top`)."""
        return 100.0 * self.cpu_s / self.wall_s if self.wall_s > 0 else 0.0


@dataclass
class Timer:
    stages: list[Stage] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str):
        st = Stage(name)
        w0, c0 = time.perf_counter(), _cpu_seconds()
        try:
            yield st
        finally:
            st.wall_s = time.perf_counter() - w0
            st.cpu_s = _cpu_seconds() - c0
            self.stages.append(st)

    def total(self, names: list[str] | None = None) -> Stage:
        sel = [s for s in self.stages if names is None or s.name in names]
        return Stage("+".join(s.name for s in sel), sum(s.wall_s for s in sel),
                     sum(s.cpu_s for s in sel))  # fmt: skip

    def asdict(self) -> dict:
        return {s.name: {"wall_s": round(s.wall_s, 3), "cpu_s": round(s.cpu_s, 3),
                         "cpu_percent": round(s.cpu_percent, 1)} for s in self.stages}  # fmt: skip
