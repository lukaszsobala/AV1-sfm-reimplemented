"""Console progress for long runs: one line per stage instead of the libraries' logs.

    ⠼ [5/7] Verifying image pairs (RANSAC)  ━━━━━━━━━━━━━╸───────────  400/780 pairs · 0:21

On a terminal the current stage is redrawn in place and replaced by a line
with its time when it finishes; otherwise (a file, a pipe, `--verbose`) only
the finished lines are printed.

`Progress.capture` sends everything written to file descriptors 1 and 2
(COLMAP's log, FFmpeg's and dav1d's messages, Python warnings) to a log
file. Stages with a `parse` function read their counts from those lines, e.g.
COLMAP's "Processed file [12/40]". If a stage fails, the end of the log is
printed with the error. The log is written by a small child process, which
also prints the end of the log if av1sfm dies without a Python exception
(e.g. a failed CHECK in COLMAP aborts the process).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

# A log-line parser returns the fields to update (done, total, detail) or None.
Parser = Callable[[str], dict | None]

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SYNC = "\x00av1sfm-progress-sync\n"  # marks a stage's end in the captured output
_END = "\x00av1sfm-progress-end\n"  # the capture ended normally

# The log writer: stdin -> log file, every line forwarded to the parent; if
# the input ends without _END (the parent died), the log's end goes to the
# terminal. Arguments: log path, terminal fd, forwarding fd.
_TEE = r"""
import collections, os, sys
log_path, term, fwd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sync, end = b"\x00av1sfm-progress-sync\n", b"\x00av1sfm-progress-end\n"
tail, clean, forward = collections.deque(maxlen=25), False, True
with open(log_path, "wb") as log:
    for line in sys.stdin.buffer:
        if line == end:
            clean = True
            continue
        text = line.removesuffix(sync)
        if text:
            log.write(text)
            log.flush()
            tail.append(text)
        if forward:
            try:
                os.write(fwd, line)
            except OSError:
                forward = False
if not clean and tail:
    msg = b"\nav1sfm stopped; the last lines of " + log_path.encode() + b":\n"
    os.write(term, msg + b"".join(tail))
"""


@dataclass
class Step:
    text: str
    total: int | None = None
    unit: str = ""
    parse: Parser | None = None
    done: int = 0
    detail: str = ""
    result: str = ""  # shown on the finished line, e.g. "77,980 points"
    start: float = field(default_factory=time.perf_counter)
    _owner: Progress | None = field(default=None, repr=False)

    def update(
        self, done: int | None = None, total: int | None = None, detail: str | None = None
    ) -> None:
        owner = self._owner
        with owner._lock if owner else _NO_LOCK:
            if total is not None:
                self.total = total
            if done is not None:
                self.done = done
            if detail is not None:
                self.detail = detail

    def advance(self, n: int = 1) -> None:
        self.update(done=self.done + n)


class _NoLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc) -> None:
        return None


_NO_LOCK = _NoLock()


class Progress:
    """Numbered stages with a live bar; `Progress(enabled=False)` does nothing."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        live: bool | None = None,
        enabled: bool = True,
        total_steps: int | None = None,
    ) -> None:
        self.enabled = enabled
        self.total_steps = total_steps
        self.steps_done = 0
        self._stream = stream or sys.stderr
        self._live = live
        self._lock = threading.Lock()
        self._step: Step | None = None
        self._tail: deque[str] = deque(maxlen=25)
        self._capturing = False
        self._synced = threading.Condition(self._lock)
        self._sync_sent = self._sync_seen = 0

    # -- stages -----------------------------------------------------------------

    @contextmanager
    def step(
        self, text: str, *, total: int | None = None, unit: str = "", parse: Parser | None = None
    ) -> Iterator[Step]:
        """A stage; on success its line shows the elapsed time and `Step.result`."""
        st = Step(text, total, unit, parse, _owner=self if self.enabled else None)
        if not self.enabled:
            yield st
            return
        number = self.steps_done + 1
        with self._lock:
            self._step = st
        ticker = self._start_ticker(st, number) if self._is_live() else None
        ok = False
        try:
            yield st
            ok = True
        finally:
            if ticker is not None:
                ticker.set()
            self._sync()  # the stage's last log lines still count for it
            with self._lock:
                self._step = None
                self.steps_done = number
                self._finish_line(st, number, ok)

    # -- capturing the libraries' output -----------------------------------------

    @contextmanager
    def capture(self, log_path: str | Path | None) -> Iterator[None]:
        """Redirect fds 1 and 2 into `log_path` while the block runs (None: no-op)."""
        if log_path is None or not self.enabled:
            yield
            return
        sys.stdout.flush()
        sys.stderr.flush()
        saved = {fd: os.dup(fd) for fd in (1, 2)}
        stream = self._stream
        if stream is sys.stderr:  # keep drawing on the real terminal
            self._live = self._is_live()
            self._stream = os.fdopen(os.dup(saved[2]), "w", encoding=_encoding(sys.stderr))
        r, w = os.pipe()
        fr, fw = os.pipe()
        tee = subprocess.Popen(
            [sys.executable, "-I", "-c", _TEE, str(log_path), str(saved[2]), str(fw)],
            stdin=r,
            pass_fds=(saved[2], fw),
        )
        os.close(r)
        os.close(fw)
        reader = threading.Thread(target=self._read_log, args=(fr,), daemon=True)
        reader.start()
        os.dup2(w, 1)
        os.dup2(w, 2)
        os.close(w)
        self._capturing = True
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            self._capturing = False
            os.write(2, _END.encode())
            for fd in (1, 2):
                os.dup2(saved[fd], fd)
                os.close(saved[fd])
            try:
                tee.wait(timeout=10)  # EOF once the pipe's last writer is closed
            except subprocess.TimeoutExpired:
                tee.kill()  # a leftover child process still holds the pipe
            reader.join(timeout=5)
            if failed and self._tail:
                self._write(f"Last lines of {log_path}:\n" + "".join(self._tail))
            if self._stream is not stream:
                self._stream.close()
                self._stream = stream

    def _read_log(self, fd: int) -> None:
        """Lines forwarded by the log writer: counts for the stage, the log's tail."""
        with os.fdopen(fd, "r", errors="replace") as pipe:
            for line in pipe:
                if line.endswith(_SYNC):  # possibly after another writer's partial line
                    with self._synced:
                        self._sync_seen += 1
                        self._synced.notify_all()
                    line = line.removesuffix(_SYNC)
                    if not line:
                        continue
                with self._lock:
                    self._tail.append(line)
                    st = self._step
                if st is not None and st.parse is not None:
                    fields = st.parse(line)
                    if fields:
                        st.update(**fields)

    def _sync(self) -> None:
        """Wait until the log reader has seen everything written so far."""
        if not self._capturing:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        with self._synced:
            self._sync_sent += 1
            target = self._sync_sent
        os.write(2, _SYNC.encode())
        with self._synced:
            self._synced.wait_for(lambda: self._sync_seen >= target, timeout=2)

    # -- drawing ----------------------------------------------------------------

    def _is_live(self) -> bool:
        if self._live is None:
            self._live = hasattr(self._stream, "isatty") and self._stream.isatty()
        return self._live

    def _start_ticker(self, st: Step, number: int) -> threading.Event:
        stop = threading.Event()

        def run() -> None:
            k = 0
            while not stop.wait(0.1):
                with self._lock:
                    if self._step is st:
                        spin = SPINNER[k % len(SPINNER)]
                        self._draw(self._live_line(st, number, spin, _width(self._stream)))
                k += 1

        threading.Thread(target=run, daemon=True).start()
        return stop

    def _counter(self, number: int) -> str:
        total = max(self.total_steps or 0, number)
        return f"[{number}/{total}]" if self.total_steps else f"[{number}]"

    def _live_line(self, st: Step, number: int, spin: str, width: int) -> str:
        head = f"{spin} {self._counter(number)} {st.text}"
        tail = []
        if st.total:
            tail.append(f"{min(st.done, st.total):,}/{st.total:,} {st.unit}".rstrip())
        elif st.done:
            tail.append(f"{st.done:,} {st.unit}".rstrip())
        if st.detail:
            tail.append(st.detail)
        tail.append(_clock(time.perf_counter() - st.start))
        right = "  " + " · ".join(tail)
        room = width - len(head) - len(right) - 3
        if st.total and room >= 10:
            bar = "  " + _bar(st.done / st.total, min(room, 30))
        else:
            bar = ""
        return _fit(head + bar + right, width - 1)

    def _finish_line(self, st: Step, number: int, ok: bool) -> None:
        secs = time.perf_counter() - st.start
        mark = ("✓" if ok else "✗") if self._unicode() else ("ok" if ok else "FAILED")
        parts = [f"{mark} {self._counter(number)} {st.text}"]
        if st.result:
            parts.append(st.result)
        elif st.total and st.unit:
            parts.append(f"{min(st.done, st.total):,} {st.unit}")
        line = "  ".join(parts)
        clock = f"{secs:.1f} s" if ok else f"failed after {secs:.1f} s"
        if self._live:
            pad = _width(self._stream) - 1 - len(line) - len(clock)
            line = line + " " * max(2, pad) + clock
            self._draw(line, end="\n")
        else:
            self._write(f"{line}  {clock}\n")

    def _draw(self, line: str, end: str = "") -> None:
        self._write("\r\x1b[2K" + line + end)

    def _write(self, text: str) -> None:
        if not self._unicode():
            text = text.encode("ascii", "replace").decode()
        try:
            self._stream.write(text)
            self._stream.flush()
        except OSError, ValueError:
            pass

    def _unicode(self) -> bool:
        return "utf" in _encoding(self._stream).lower()

    def message(self, text: str) -> None:
        """A line outside any stage (results, hints)."""
        if self.enabled:
            with self._lock:
                self._write(text + "\n")


def _encoding(stream) -> str:
    return getattr(stream, "encoding", None) or "utf-8"  # io.StringIO: None


def _width(stream) -> int:
    try:
        return os.get_terminal_size(stream.fileno()).columns
    except OSError, ValueError, AttributeError:
        return shutil.get_terminal_size().columns


def _bar(frac: float, width: int) -> str:
    frac = min(max(frac, 0.0), 1.0)
    full = int(frac * width)
    half = "╸" if full < width and frac * width - full >= 0.5 else ""
    return "━" * full + half + "\x1b[2m" + "─" * (width - full - len(half)) + "\x1b[0m"


def _fit(line: str, width: int) -> str:
    """Truncate to `width` visible characters (escape sequences take no room)."""
    visible = re.sub(r"\x1b\[[0-9;]*m", "", line)
    if len(visible) <= width:
        return line
    return visible[: max(width - 1, 0)] + "…"


def _clock(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


# -- log-line parsers -------------------------------------------------------------


def counter(pattern: str, *, offset: int = 0) -> Parser:
    """Parser for "...[i/N]...": done = i + offset, total = N."""
    rx = re.compile(pattern)

    def parse(line: str) -> dict | None:
        m = rx.search(line)
        if m is None:
            return None
        return {"done": int(m.group(1)) + offset, "total": int(m.group(2))}

    return parse


# COLMAP 4.2 log lines
SIFT_EXTRACTION = counter(r"Processed file \[(\d+)/(\d+)\]")
SEQUENTIAL_MATCHING = counter(r"Processing image \[(\d+)/(\d+)\]", offset=-1)


def exhaustive_matching(line: str) -> dict | None:
    m = re.search(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]", line)
    if m is None:
        return None
    i, n, j, k = (int(g) for g in m.groups())
    return {"done": (i - 1) * k + j - 1, "total": n * k}


def verification_batches(batch_size: int, num_pairs: int) -> Parser:
    """COLMAP's "Processing batch [i/N]" of geometric verification, as pairs."""
    rx = re.compile(r"Processing batch \[(\d+)/(\d+)\]")

    def parse(line: str) -> dict | None:
        m = rx.search(line)
        return {"done": min((int(m.group(1)) - 1) * batch_size, num_pairs)} if m else None

    return parse


def incremental_mapper(line: str) -> dict | None:
    if m := re.search(r"Registering image #(\d+) \(num_reg_frames=(\d+)\)", line):
        return {"done": int(m.group(2)), "detail": f"adding image #{m.group(1)}"}
    if m := re.search(r"Registering initial image pair #(\d+) and #(\d+)", line):
        return {"done": 0, "detail": f"initial pair #{m.group(1)} and #{m.group(2)}"}
    if "Finding good initial image pair" in line:
        return {"done": 0, "detail": "choosing an initial image pair"}
    if "Global bundle adjustment" in line:
        return {"detail": "bundle adjustment"}
    return None


def global_mapper(line: str) -> dict | None:
    if m := re.search(r"=== Running (.+?) ===", line):
        return {"detail": m.group(1)}
    if m := re.search(r"Global bundle adjustment iteration (\d+) / (\d+)", line):
        return {"detail": f"bundle adjustment {m.group(1)}/{m.group(2)}"}
    return None
