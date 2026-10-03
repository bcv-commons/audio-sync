"""Shared "is this chapter done?" classifier.

Consolidates align_pipeline.py's needs_run() and _chapter_output_exists(),
pipeline/purge_aligned_audio.py's _has_real_timing(), and the equivalent ad
hoc checks once duplicated across whisper_transcribe.py, mms_align_words.py
and align_words.py (the latter two's discover_work_items() were removed
2026-10-03 with the step-only CLIs that used them) -- all of them
were independently answering "does this chapter already have real output"
with slightly different rules (see single-pipeline-plan-2026-10-02.md
for the full inventory of 7 separate implementations this replaces).

The one addition none of those had: DEFERRED. A defer_to_dbt redirect
record (apply_arbiter_corrections.py's whole-chapter-redirect mechanism,
confirmed 2026-10-01: 16,846 chapters) is a real _timing.json that
exists on disk, but it is NOT "done" -- it's a placeholder pending a real
re-alignment, and must never be purge-eligible or silently treated as
finished. Before this module existed, that distinction had to be
re-implemented at every call site that cared (purge already had it;
nothing else did).
"""
import json
from enum import Enum
from pathlib import Path


class ChapterStatus(Enum):
    DONE = "done"          # real pos[] timing exists, inputs aren't newer
    MISSING = "missing"    # no real output at all (absent, corrupt, or an
                            # unrecognized/legacy shape -- treated the same
                            # as absent: there's nothing to trust here)
    STALE = "stale"        # real output exists, but an input is newer
    DEFERRED = "deferred"  # defer_to_dbt redirect -- pending a real redo,
                            # never purge-eligible, never silently "done"


def has_real_timing(timing_path: Path) -> bool:
    """True only if timing_path holds actual pos[] alignment data -- not
    a defer_to_dbt redirect, not a legacy list-format file (pre-Aug-12
    redesign), not anything else. Moved here from
    pipeline/purge_aligned_audio.py's own _has_real_timing() (2026-10-01) --
    same logic, now the one shared source of truth.
    """
    try:
        data = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "pos" in data


def chapter_state(
    timing_path: Path | None, *input_paths: Path | None, force: bool = False,
) -> ChapterStatus:
    """Classify one chapter's timing output against its own content and
    (optionally) its inputs' mtimes.

    Callers that only ever checked existence (the old needs_run() calls
    with no input_paths, e.g. Whisper/MMS steps) get the same answer as
    before: MISSING/DEFERRED need a run, DONE doesn't, and STALE can only
    ever be reached when at least one input_path is given.
    """
    if timing_path is None or not timing_path.exists():
        return ChapterStatus.MISSING
    if force:
        return ChapterStatus.STALE
    try:
        data = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ChapterStatus.MISSING
    if isinstance(data, dict) and data.get("status") == "defer_to_dbt":
        return ChapterStatus.DEFERRED
    if not (isinstance(data, dict) and "pos" in data):
        return ChapterStatus.MISSING
    out_mtime = timing_path.stat().st_mtime
    if any(p and p.exists() and p.stat().st_mtime > out_mtime for p in input_paths):
        return ChapterStatus.STALE
    return ChapterStatus.DONE


def needs_run(output_path: Path | None, *input_paths: Path | None, force: bool = False) -> bool:
    """Drop-in replacement for align_pipeline.py's needs_run() -- same
    signature and return contract (True = should run), now expressed in
    terms of chapter_state() so DEFERRED is handled identically to
    MISSING/STALE here (a normal run should still fill in a never-aligned
    chapter even if some OTHER chapter nearby happens to be deferred --
    this function only ever looks at the one output_path it's given).
    """
    return chapter_state(output_path, *input_paths, force=force) != ChapterStatus.DONE


def chapter_fully_done(
    output_dir: Path, canon: str, iso: str, distinct_id: str, book: str, chapter: int,
) -> bool:
    """Glob-based "is this chapter done, for some fileset" check -- same
    approach align_pipeline.py's old _chapter_output_exists() used (the
    audio_fileset suffix varies per edition and isn't known here without
    having fetched the chapter at least once), but now a defer_to_dbt
    redirect correctly does NOT count as done.
    """
    book_dir = output_dir / canon / iso / distinct_id / book
    if not book_dir.is_dir():
        return False
    prefix = f"{book}_{chapter:03d}_"
    for timing_path in book_dir.glob(f"{prefix}*_timing.json"):
        words_path = timing_path.with_name(timing_path.name.replace("_timing.json", "_words.json"))
        if words_path.exists() and chapter_state(timing_path) == ChapterStatus.DONE:
            return True
    return False


def alignment_method(quality_path: Path) -> str | None:
    """The alignment-method tag a chapter's local quality file carries
    (summary.method), or None for output written before tags existed.
    Lets a redo tell old-method output from current output without
    re-deriving it from scores."""
    try:
        data = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = data.get("summary") if isinstance(data, dict) else None
    return summary.get("method") if isinstance(summary, dict) else None
