import io
import os
import re
import subprocess
import sys

import pytest

from av1sfm.progress import (
    SIFT_EXTRACTION,
    Progress,
    exhaustive_matching,
    incremental_mapper,
    verification_batches,
)


def test_finished_lines():
    out = io.StringIO()
    p = Progress(out, live=False, total_steps=2)
    with p.step("Encoding", total=4, unit="frames") as st:
        st.update(3)
        st.advance()
    with p.step("Tracks") as st:
        st.result = "1,234 tracks"
    lines = out.getvalue().splitlines()
    assert re.fullmatch(r"✓ \[1/2\] Encoding  4 frames  \d+\.\d s", lines[0]), lines
    assert re.fullmatch(r"✓ \[2/2\] Tracks  1,234 tracks  \d+\.\d s", lines[1]), lines
    assert p.steps_done == 2


def test_failed_step_is_marked_and_raises():
    out = io.StringIO()
    p = Progress(out, live=False)
    with pytest.raises(RuntimeError), p.step("Mapping"):
        raise RuntimeError("boom")
    assert re.fullmatch(r"✗ \[1\] Mapping  failed after \d+\.\d s\n", out.getvalue())


def test_disabled_does_nothing():
    out = io.StringIO()
    p = Progress(out, enabled=False)
    with p.step("x", total=3) as st:
        st.update(2)
    assert out.getvalue() == "" and p.steps_done == 0


def test_live_line_fits_the_width():
    out = io.StringIO()
    p = Progress(out, live=True, total_steps=7)
    st = p.step("Verifying image pairs (RANSAC)", total=780, unit="pairs").__enter__()
    st.update(400, detail="batch 3")
    for width in (40, 80, 120):
        line = p._live_line(st, 5, "⠋", width)
        visible = re.sub(r"\x1b\[[0-9;]*m", "", line)
        assert len(visible) <= width - 1, (width, visible)
        assert visible.startswith("⠋ [5/7] Verifying")
    assert "400/780 pairs · batch 3" in re.sub(r"\x1b\[[0-9;]*m", "", p._live_line(st, 5, "⠋", 120))
    assert "━" in p._live_line(st, 5, "⠋", 120)  # the bar, when there is room


def test_capture_logs_and_parses(tmp_path):
    out = io.StringIO()
    p = Progress(out, live=False)
    log = tmp_path / "run.log"
    with p.capture(log), p.step("Extracting", unit="images", parse=SIFT_EXTRACTION) as st:
        os.write(2, b"I1008 feature_extraction.cc:270] Processed file [3/12]\n")
        subprocess.run([sys.executable, "-c", "print('from a child')"], check=True)
        os.write(2, b"I1008 feature_extraction.cc:270] Processed file [7/12]\n")
    assert (st.done, st.total) == (7, 12)
    text = log.read_text()
    assert "Processed file [3/12]" in text and "from a child" in text
    assert out.getvalue().startswith("✓ [1] Extracting  7 images")


def test_capture_shows_the_log_tail_on_failure(tmp_path):
    out = io.StringIO()
    p = Progress(out, live=False)
    with pytest.raises(ValueError), p.capture(tmp_path / "run.log"):
        os.write(2, b"E1008 something went wrong\n")
        raise ValueError
    assert "something went wrong" in out.getvalue()


def test_colmap_parsers():
    assert incremental_mapper(
        "I1008 incremental_pipeline.cc:620] Registering image #17 (num_reg_frames=5)"
    ) == {"done": 5, "detail": "adding image #17"}
    assert incremental_mapper("Retriangulation and Global bundle adjustment") == {
        "detail": "bundle adjustment"
    }
    assert incremental_mapper("unrelated") is None
    assert exhaustive_matching("pairing.cc:214] Processing block [2/3, 1/3]") == {
        "done": 3,
        "total": 9,
    }
    parse = verification_batches(200, 450)
    assert parse("pairing.cc:996] Processing batch [3/3]") == {"done": 400}


def test_capture_reports_a_crash(tmp_path):
    # A C++ abort skips Python's cleanup; the log writer prints the log's end.
    code = (
        "import os, sys\n"
        "from av1sfm.progress import Progress\n"
        "p = Progress(sys.stderr, live=False)\n"
        f"with p.capture({str(tmp_path / 'run.log')!r}), p.step('Mapping'):\n"
        "    os.write(2, b'F1008 check failed: camera is bogus\\n')\n"
        "    os.abort()\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode != 0
    assert "av1sfm stopped" in r.stderr and "camera is bogus" in r.stderr, r.stderr
    assert "camera is bogus" in (tmp_path / "run.log").read_text()
