"""Unit tests for chapter_state.py -- the classifier that replaces the 7
separate "is this chapter done" implementations the 2026-10-02 survey
found (see single-pipeline-plan-2026-10-02.md). The one case none of the
originals handled on their own: DEFERRED (a defer_to_dbt redirect).
"""
import json
import time
from pathlib import Path

import pytest

from chapter_state import ChapterStatus, chapter_fully_done, chapter_state, has_real_timing, needs_run


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


class TestChapterState:
    def test_missing_when_no_file(self, tmp_path):
        assert chapter_state(tmp_path / "nope_timing.json") == ChapterStatus.MISSING

    def test_missing_when_none(self):
        assert chapter_state(None) == ChapterStatus.MISSING

    def test_done_when_real_timing_exists(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, {"id": "MAT 1", "pos": [1.0, 2.0]})
        assert chapter_state(p) == ChapterStatus.DONE

    def test_deferred_for_defer_to_dbt_redirect(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, {"status": "defer_to_dbt", "reason": "pending MMS redo"})
        assert chapter_state(p) == ChapterStatus.DEFERRED

    def test_deferred_chapter_is_not_done(self, tmp_path):
        # The whole point: a normal "is it done" check must never treat
        # a redirect as finished.
        p = tmp_path / "x_timing.json"
        _write(p, {"status": "defer_to_dbt"})
        assert chapter_state(p) != ChapterStatus.DONE

    def test_missing_for_legacy_list_format(self, tmp_path):
        # Pre-Aug-12-redesign shape: a bare list of verse dicts, not our
        # {"pos": [...]} shape. Treated as absent, not a crash.
        p = tmp_path / "x_timing.json"
        _write(p, [{"verse_start": "1", "timestamp": 1.0}])
        assert chapter_state(p) == ChapterStatus.MISSING

    def test_missing_for_corrupt_json(self, tmp_path):
        p = tmp_path / "x_timing.json"
        p.write_text("{not valid json")
        assert chapter_state(p) == ChapterStatus.MISSING

    def test_stale_when_input_newer_than_output(self, tmp_path):
        out = tmp_path / "x_timing.json"
        _write(out, {"pos": [1.0]})
        inp = tmp_path / "input.json"
        inp.write_text("x")
        now = time.time()
        import os
        os.utime(out, (now - 10, now - 10))
        os.utime(inp, (now, now))
        assert chapter_state(out, inp) == ChapterStatus.STALE

    def test_done_when_no_input_is_newer(self, tmp_path):
        out = tmp_path / "x_timing.json"
        _write(out, {"pos": [1.0]})
        inp = tmp_path / "input.json"
        inp.write_text("x")
        now = time.time()
        import os
        os.utime(inp, (now - 10, now - 10))
        os.utime(out, (now, now))
        assert chapter_state(out, inp) == ChapterStatus.DONE

    def test_force_always_means_stale(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, {"pos": [1.0]})
        assert chapter_state(p, force=True) == ChapterStatus.STALE

    def test_missing_input_path_is_ignored(self, tmp_path):
        out = tmp_path / "x_timing.json"
        _write(out, {"pos": [1.0]})
        assert chapter_state(out, tmp_path / "ghost.json") == ChapterStatus.DONE


class TestHasRealTiming:
    def test_true_for_real_pos(self, tmp_path):
        p = tmp_path / "x.json"
        _write(p, {"pos": [1.0]})
        assert has_real_timing(p) is True

    def test_false_for_defer_to_dbt(self, tmp_path):
        p = tmp_path / "x.json"
        _write(p, {"status": "defer_to_dbt"})
        assert has_real_timing(p) is False

    def test_false_for_missing_file(self, tmp_path):
        assert has_real_timing(tmp_path / "nope.json") is False


class TestNeedsRun:
    """Drop-in replacement for align_pipeline.py's old needs_run() --
    same boolean contract, verified here against its own former cases."""

    def test_true_when_missing(self, tmp_path):
        assert needs_run(tmp_path / "nope.json") is True

    def test_false_when_done(self, tmp_path):
        p = tmp_path / "x.json"
        _write(p, {"pos": [1.0]})
        assert needs_run(p) is False

    def test_true_when_deferred(self, tmp_path):
        # A deferred chapter still "needs (a real) run" -- it's not done.
        p = tmp_path / "x.json"
        _write(p, {"status": "defer_to_dbt"})
        assert needs_run(p) is True

    def test_true_when_forced(self, tmp_path):
        p = tmp_path / "x.json"
        _write(p, {"pos": [1.0]})
        assert needs_run(p, force=True) is True


class TestChapterFullyDone:
    def test_false_when_book_dir_missing(self, tmp_path):
        assert chapter_fully_done(tmp_path, "nt", "iso", "DID", "MAT", 1) is False

    def test_true_when_real_output_exists_for_some_fileset(self, tmp_path):
        book_dir = tmp_path / "nt" / "iso" / "DID" / "MAT"
        book_dir.mkdir(parents=True)
        _write(book_dir / "MAT_001_FOOBAR_timing.json", {"pos": [1.0]})
        (book_dir / "MAT_001_FOOBAR_words.json").write_text("{}")
        assert chapter_fully_done(tmp_path, "nt", "iso", "DID", "MAT", 1) is True

    def test_false_when_only_a_defer_to_dbt_redirect_exists(self, tmp_path):
        # The exact bug this module exists to prevent: a redirect record
        # must never be counted as "fully done" (e.g. for purge or skip
        # decisions) just because a file is sitting at the expected path.
        book_dir = tmp_path / "nt" / "iso" / "DID" / "MAT"
        book_dir.mkdir(parents=True)
        _write(book_dir / "MAT_001_FOOBAR_timing.json", {"status": "defer_to_dbt"})
        (book_dir / "MAT_001_FOOBAR_words.json").write_text('{"status": "defer_to_dbt"}')
        assert chapter_fully_done(tmp_path, "nt", "iso", "DID", "MAT", 1) is False

    def test_false_when_words_json_missing(self, tmp_path):
        book_dir = tmp_path / "nt" / "iso" / "DID" / "MAT"
        book_dir.mkdir(parents=True)
        _write(book_dir / "MAT_001_FOOBAR_timing.json", {"pos": [1.0]})
        assert chapter_fully_done(tmp_path, "nt", "iso", "DID", "MAT", 1) is False
