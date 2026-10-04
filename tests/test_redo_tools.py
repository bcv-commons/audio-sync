"""Regression tests for the redo tooling's failure modes found on 2026-10-03/04:
a healthy worker killed by the hang watcher, no-op groups stalling the redo,
and the supervisor's handling of a hung (SIGUSR1-killed) worker."""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import redo_hang_watch as watch
import run_gpu_redo as redo
from align_verse_words import ALIGNMENT_METHOD

REPO = Path(__file__).resolve().parent.parent


# ── hang watcher ─────────────────────────────────────────────────────────────

def test_progress_age_uses_newest_of_marker_heartbeat_and_start(tmp_path):
    marker = tmp_path / "m.inflight.json"
    hb = tmp_path / "m.inflight.json.hb"
    now = 100_000.0
    # nothing written yet: aged from the process start
    assert watch.progress_age_minutes(marker, now - 600, now) == 10
    marker.write_text("{}")
    os.utime(marker, (now - 1800, now - 1800))
    assert watch.progress_age_minutes(marker, now - 7200, now) == 30
    # the heartbeat keeps a worker that is only skipping chapters alive
    hb.write_text("")
    os.utime(hb, (now - 60, now - 60))
    assert watch.progress_age_minutes(marker, now - 7200, now) == 1


# ── "is this chapter already done" ──────────────────────────────────────────

def _quality(path: Path, method: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": {"method": method}}))


def test_chapter_is_current_looks_at_every_recording(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = Path("export/timing-data/nt/abc/ABCXYZ/MAT")
    standard = out / "MAT_001_ABCXYZN1DA_timing.json"
    drama = out / "MAT_001_ABCXYZN2DA_timing.json"
    _quality(out / "MAT_001_ABCXYZN1DA_words_quality.json", ALIGNMENT_METHOD)
    _quality(out / "MAT_001_ABCXYZN2DA_words_quality.json", "anchored-star-v1")
    # a list entry for the dramatized copy must not be redone when the
    # standard recording is already current
    assert redo.chapter_is_current(drama)
    assert redo.chapter_is_current(standard)
    _quality(out / "MAT_002_ABCXYZN1DA_words_quality.json", "anchored-star-v1")
    assert not redo.chapter_is_current(out / "MAT_002_ABCXYZN1DA_timing.json")
    assert not redo.chapter_is_current(out / "MAT_003_ABCXYZN1DA_timing.json")


# ── supervisor: a hung worker is quarantined and restarted ───────────────────

def test_supervisor_quarantines_a_hung_worker_and_restarts(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").symlink_to(sys.executable)
    (tmp_path / "tools/run_gpu_redo_supervisor.py").write_text(
        (REPO / "tools/run_gpu_redo_supervisor.py").read_text())
    (tmp_path / "tools/run_gpu_redo.py").write_text(textwrap.dedent("""
        import json, os, signal, sys
        marker = sys.argv[sys.argv.index("--inflight-marker") + 1]
        flag = marker + ".second"
        if os.path.exists(flag):
            sys.exit(0)                                   # second attempt: done
        open(flag, "w").close()
        open(marker, "w").write(json.dumps({"timing_path": "x/CH_1_timing.json", "iso": "x",
                                            "distinct_id": "X", "book": "B", "chapter": 1}))
        os.kill(os.getpid(), signal.SIGUSR1)              # what redo_hang_watch.py sends
        """))
    (tmp_path / "report.json").write_text(json.dumps({"chapters": []}))
    (tmp_path / "q.json").write_text(json.dumps({"chapters": []}))
    run = subprocess.run(
        [sys.executable, "tools/run_gpu_redo_supervisor.py", "--report", "report.json",
         "--quarantine", "q.json"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    quarantined = json.loads((tmp_path / "q.json").read_text())["chapters"]
    assert quarantined == [{"path": "x/CH_1_timing.json", "reason": "HANG"}]
    assert "Attempt 2" in run.stdout + run.stderr
