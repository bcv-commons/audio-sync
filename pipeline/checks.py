"""Per-chapter checks over finished timing output -- the one place these
live. Used by the publish gate (tools/pre_publish_check.py), the per-language
reports (tools/check_timing_quality.py, tools/check_verse_only_fallback.py)
and align_pipeline.py's end-of-language checks.

All functions are pure reads of one chapter's files and never raise on a
missing or malformed file.
"""
import json
from pathlib import Path

from timing_files import (
    _parse_timing_path,
    find_all_downloaded_timecode,
    find_pipeline_timing_files,
    find_quality_files,
    load_quality,
)


def load_verse_starts(timing_path) -> list[tuple[str, float]] | None:
    """(verse number, start time) pairs from a chapter timing file, verse 0
    excluded, in file order.

    Understands both live shapes: our compact {"pos": [...]} output (pos[i]
    is verse i+1) and DBT's downloaded list of {"verse_start", "timestamp"}
    dicts. Returns None for anything else -- a defer_to_dbt redirect, OBS
    story timing (also a list, but of segment dicts), unreadable files.
    """
    try:
        with open(timing_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict) and "pos" in data:
        return [(str(i + 1), t) for i, t in enumerate(data["pos"]) if t is not None]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "verse_start" in data[0]:
        return [(str(e["verse_start"]), e["timestamp"]) for e in data if str(e.get("verse_start")) != "0"]
    return None


def has_backwards_jump(timing_path: Path) -> bool:
    """True if any verse's timestamp precedes the previous verse's -- never
    legitimate, always a mis-alignment."""
    verses = load_verse_starts(timing_path)
    if not verses:
        return False
    return any(verses[i][1] < verses[i - 1][1] for i in range(1, len(verses)))


def analyze_chapter_timing(path):
    """Check a timing.json for verse-level issues.

    Returns dict with {verses, dupes, backwards, tiny, gaps, dupe_verses} or
    None. dupe_verses is the list of verse_start strings that are the LATER
    verse in each zero-delta transition — lets a caller cross-reference
    with per-word confidence scores to tell apart two genuinely different
    causes that both show up as "consecutive verses share a timestamp":
    a real alignment failure (0.0-score fallback, several verses in a row)
    vs. two verses genuinely spoken back-to-back with no perceptible gap
    (high-confidence, an isolated single pair) — confirmed as distinct,
    real cases 2026-08-10 (see internal-docs/gpu-wedge-forensics.md-
    adjacent session history around the mms_align_words.py CTC-infeasible
    fallback fix). See check_language()'s DUPES vs DUPES-OK vs DUPES-EMPTY
    split — the latter (a third, even more common cause, confirmed
    2026-09-02) is a dupe verse that's empty after cleaning (e.g. a lone
    leftover punctuation mark on its own reference-text line), which
    check_language() detects separately via each dupe verse's word count.
    """
    verses = load_verse_starts(path)
    if verses is None or len(verses) < 2:
        return None

    dupes = backwards = tiny = gaps = 0
    dupe_verses = []
    for i in range(1, len(verses)):
        delta = verses[i][1] - verses[i - 1][1]
        if delta == 0:
            dupes += 1
            dupe_verses.append(verses[i][0])
        elif delta < 0:
            backwards += 1
        elif delta < 0.1:
            tiny += 1
        elif delta > 120:
            gaps += 1

    return {
        "verses": len(verses),
        "dupes": dupes,
        "backwards": backwards,
        "tiny": tiny,
        "gaps": gaps,
        "dupe_verses": dupe_verses,
    }


# A chapter whose word-quality summary is this null-heavy or this
# low-scoring essentially never happens from real alignment — even a
# genuinely hard chapter has SOME well-scored words. These thresholds are
# deliberately conservative (real corruption is null_count == total_words,
# avg_score == 0.0 exactly) to leave headroom above legitimately rough
# chapters without false-flagging them.
_FALLBACK_NULL_FRACTION = 0.9
_FALLBACK_AVG_SCORE_MAX = 0.02


def has_fallback_corruption(quality_path: Path) -> bool:
    """True if a chapter's alignment has collapsed to near-total fallback/
    failure — nearly every word has a null timestamp and/or zero
    confidence score.

    This is the on-disk fingerprint of a poisoned CUDA context: each
    individual word/verse failure looks like an isolated, plausible case
    (a verse's audio genuinely too short for its text, say), but a whole
    CHAPTER with this shape essentially never happens from real alignment
    — even a genuinely hard chapter has some well-scored words. Unlike a
    backwards jump, this failure mode doesn't corrupt monotonicity or
    produce an error — the pipeline logs it as a normal success, which is
    exactly how 4,926 chapters were silently corrupted in a single
    2026-08-12 run before this check existed (confirmed: every one had
    summary.avg_score == 0.0 and null_count == total_words).

    Path-agnostic: applies equally to the fusion pipeline's
    *_words_quality.json (source values "mms"/"whisper"/"mms_gap_fill"/
    "mms_drift_fix") and verse-only mode's (source
    "local"/"fallback"/"interpolated") — both write the same
    {"summary": {"total_words", "null_count", "avg_score", ...}} shape
    (align_words.py / align_verse_words.py), and a poisoned context
    corrupts either pipeline identically.
    """
    try:
        with open(quality_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    summary = data.get("summary")
    if not summary:
        return False
    total = summary.get("total_words", 0)
    if not total:
        return False
    null_fraction = summary.get("null_count", 0) / total
    avg_score = summary.get("avg_score", 1.0)
    return null_fraction >= _FALLBACK_NULL_FRACTION or avg_score <= _FALLBACK_AVG_SCORE_MAX


def is_legacy_format(timing_path: Path) -> bool:
    """True if this *_timing.json is still the old per-verse-dict list
    shape instead of the compact {"pos": [...]} dict.

    Excludes export/timing-data/obs/ — OBS's story/segment timing files
    are also a JSON list, but that's their own unrelated Contract-B shape
    (see pipeline/align_obs_words.py), not this pipeline's legacy format.
    """
    if "/obs/" in timing_path.as_posix():
        return False
    try:
        with open(timing_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(data, list)


def is_held_back(quality_path: Path) -> bool:
    """True if the verse-only chapter gate held this chapter back (too many
    low-score verses and no usable DBT timing to defer to) -- see
    align_verse_words.py's GATE_LOW_SCORE / GATE_MAX_LOW_SHARE."""
    try:
        with open(quality_path) as f:
            return (json.load(f).get("summary") or {}).get("gate") == "held_back"
    except (json.JSONDecodeError, OSError):
        return False


def check_chapter_fallback(quality_path: Path) -> dict | None:
    """Fallback stats for one chapter's _words_quality.json.

    Returns {verses, fallback_verses, fallback_rate, fallback_verse_nums}
    or None if this isn't verse_only_mode output (no "source" field on
    its words — the fusion-mode pipeline's quality files have per-word
    scores but no per-word "source": "fallback"/"local"/"interpolated"
    tag).

    "interpolated" (added 2026-09-24, see align_verse_words.py's
    _interpolate_fallback_runs()) is a verse that still has no real
    per-word alignment — same underlying failure as "fallback" — but
    whose verse-level start/end was corrected to interpolate between the
    chapter's own nearest real-aligned neighbors instead of being left at
    the raw whole-chapter pace estimate. Counted the same as "fallback"
    here: this tool reports how often a real alignment wasn't found at
    all, regardless of how good the resulting position estimate is.
    """
    try:
        with open(quality_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    verses = data.get("verses", {})
    if not verses:
        return None

    total = 0
    fallback_verses = []
    saw_source_tag = False
    for vnum, words in verses.items():
        if not words:
            continue
        total += 1
        # verse_only_mode's vocabulary is {"local", "fallback",
        # "interpolated"} — fusion-mode quality files use a disjoint
        # vocabulary ({"whisper"} only, or no "source" key at all for the
        # common MMS-wins case), so this positively identifies
        # verse_only_mode output rather than just detecting *some*
        # "source" key (which fusion-mode files can also carry, and did
        # wrongly match here at first).
        if any(w.get("source") in ("local", "fallback", "interpolated") for w in words):
            saw_source_tag = True
        if words and all(w.get("source") in ("fallback", "interpolated") for w in words):
            fallback_verses.append(vnum)

    if not saw_source_tag or total == 0:
        # Fusion-mode output (or an empty/malformed file) — not what
        # this tool is for, see check_timing_quality.py instead.
        return None

    return {
        "verses": total,
        "fallback_verses": len(fallback_verses),
        "fallback_rate": len(fallback_verses) / total,
        "fallback_verse_nums": fallback_verses,
    }


# ── Per-language checks (align_pipeline.py's end-of-language step; the
# CLIs in tools/check_timing_quality.py and tools/check_verse_only_fallback.py) ──

def analyze_chapter_words(words_path):
    """Check a words.json for null timestamps and duplicate word times.

    Returns dict with {nulls, word_dupes, total_words} or None.
    """
    if not words_path.exists():
        return None

    with open(words_path) as f:
        data = json.load(f)

    verses = data.get("beg", {})
    nulls = word_dupes = total_words = 0

    for vnum, timestamps in verses.items():
        if not timestamps:
            continue
        prev = None
        for ts in timestamps:
            total_words += 1
            if ts is None:
                nulls += 1
            else:
                if prev is not None and ts == prev:
                    word_dupes += 1
                prev = ts

    return {"nulls": nulls, "word_dupes": word_dupes, "total_words": total_words}


def check_language(iso, testament=None):
    """Run all quality checks for one language.

    Returns dict with aggregate stats and per-chapter details, or None.
    """
    pipeline_files = find_pipeline_timing_files(iso, testament)
    downloaded_tc = find_all_downloaded_timecode(iso, testament)
    quality_files = find_quality_files(iso, testament)

    if not pipeline_files:
        return None

    # Build quality lookup: (canon, distinct_id, book, chapter_str) -> quality data
    quality_by_key = {}
    for canon, qf in quality_files:
        data = load_quality(qf)
        distinct_id = qf.parent.parent.name
        book = data["book"]
        parts = qf.stem.replace("_words_quality", "").split("_", 2)
        ch = parts[1] if len(parts) >= 2 else None
        if ch:
            quality_by_key[(canon, distinct_id, book, ch)] = data

    chapters = []
    for canon, tf in pipeline_files:
        distinct_id, book, chapter_str = _parse_timing_path(tf)
        if not chapter_str:
            continue

        key = (canon, distinct_id, book, chapter_str)

        # Verse-level analysis (generated)
        gen_timing = analyze_chapter_timing(tf)
        if gen_timing is None:
            continue

        # Word-level analysis
        words_path = tf.parent / tf.name.replace("_timing.json", "_words.json")
        word_stats = analyze_chapter_words(words_path)

        # Quality score
        q_data = quality_by_key.get(key)
        avg_score = q_data["summary"]["avg_score"] if q_data else None
        low_q = q_data["summary"]["low_quality_count"] if q_data else 0

        # Compare with original if available
        orig_better = False
        if key in downloaded_tc:
            dl_timing = analyze_chapter_timing(downloaded_tc[key])
            if dl_timing:
                gen_issues = gen_timing["dupes"] + gen_timing["backwards"]
                dl_issues = dl_timing["dupes"] + dl_timing["backwards"]
                if dl_issues < gen_issues and gen_issues > 0:
                    orig_better = True

        # Build flags
        flags = []
        if gen_timing["dupes"] > 0:
            # A dupe verse with zero words in words.json's "beg" list is a
            # verse that cleaned down to nothing (e.g. a lone leftover
            # punctuation mark on its own reference-text line — confirmed
            # for real 2026-09-02 across 673 sampled DUPES chapters in 28
            # languages: 100% were this exact case, not an alignment
            # failure). align_words.py deliberately reuses the previous
            # verse's timestamp for these (nothing to time), so this is
            # expected, unfixable-at-the-alignment-level output, not an
            # issue — distinct from DUPES-OK's "two verses genuinely
            # spoken back-to-back" case below, but equally benign.
            empty_dupe_verses = set()
            if word_stats is not None:
                try:
                    with open(words_path, encoding="utf-8") as wf:
                        beg_by_verse = json.load(wf).get("beg", {})
                    empty_dupe_verses = {
                        vnum for vnum in gen_timing["dupe_verses"]
                        if not beg_by_verse.get(vnum)
                    }
                except (OSError, json.JSONDecodeError):
                    pass

            if empty_dupe_verses and len(empty_dupe_verses) == len(gen_timing["dupe_verses"]):
                flags.append("DUPES-EMPTY")
            else:
                # Cross-reference the specific dupe verses' word scores to
                # tell a real alignment failure (0.0-score fallback) apart
                # from two verses genuinely spoken back-to-back with no gap
                # (high-confidence). Only the former is an actual issue —
                # see analyze_chapter_timing()'s docstring for how this was
                # found. Empty-verse dupes are excluded from this average
                # (they have no words/score by definition, which would
                # otherwise drag dupe_avg down or leave it undefined).
                dupe_scores = []
                if q_data:
                    for vnum in gen_timing["dupe_verses"]:
                        if vnum in empty_dupe_verses:
                            continue
                        for w in q_data["verses"].get(vnum, []):
                            dupe_scores.append(w["score"])
                dupe_avg = sum(dupe_scores) / len(dupe_scores) if dupe_scores else None
                if dupe_avg is not None and dupe_avg >= 0.5:
                    flags.append("DUPES-OK")
                else:
                    flags.append("DUPES")
        if gen_timing["backwards"] > 0:
            flags.append("BACKWARDS")
        if gen_timing["tiny"] >= 3:
            flags.append("TINY-STEPS")
        if gen_timing["gaps"] > 0:
            flags.append("GAPS")
        if word_stats and word_stats["nulls"] >= 3:
            flags.append("NULLS")
        if avg_score is not None and avg_score < 0.5:
            flags.append("LOW-SCORE")
        if orig_better:
            flags.append("ORIG-BETTER")

        chapters.append({
            "canon": canon,
            "distinct_id": distinct_id,
            "book": book,
            "chapter": chapter_str,
            "dupes": gen_timing["dupes"],
            "backwards": gen_timing["backwards"],
            "tiny": gen_timing["tiny"],
            "gaps": gen_timing["gaps"],
            "nulls": word_stats["nulls"] if word_stats else 0,
            "low_q": low_q,
            "avg_score": avg_score,
            "orig_better": orig_better,
            "flags": flags,
        })

    if not chapters:
        return None

    # DUPES-OK and DUPES-EMPTY are both informational, not an issue (see
    # check_language()'s DUPES/DUPES-OK/DUPES-EMPTY split above) — a
    # chapter flagged with ONLY those shouldn't count toward has_issues,
    # and their dupe counts shouldn't inflate total_dupes (which "how many
    # chapters actually need attention" tooling, e.g.
    # tools/requeue_dupes_chapters.py, reads).
    BENIGN_DUPE_FLAGS = ("DUPES-OK", "DUPES-EMPTY")
    has_issues = sum(
        1 for c in chapters
        if any(f not in BENIGN_DUPE_FLAGS for f in c["flags"])
    )
    return {
        "iso": iso,
        "chapters": len(chapters),
        "has_issues": has_issues,
        "total_dupes": sum(c["dupes"] for c in chapters if "DUPES" in c["flags"]),
        "total_dupes_ok": sum(c["dupes"] for c in chapters if "DUPES-OK" in c["flags"]),
        "total_dupes_empty": sum(c["dupes"] for c in chapters if "DUPES-EMPTY" in c["flags"]),
        "total_backwards": sum(c["backwards"] for c in chapters),
        "total_nulls": sum(c["nulls"] for c in chapters),
        "total_low_q": sum(c["low_q"] for c in chapters),
        "orig_better_count": sum(1 for c in chapters if c["orig_better"]),
        "chapter_details": chapters,
    }



DEFAULT_THRESHOLD = 0.2


def check_language_fallback(iso: str, testament: str | None = None, threshold: float = DEFAULT_THRESHOLD) -> dict | None:
    """Fallback-rate summary for one language across all its chapters.

    Returns None if this language has no verse_only_mode-shaped quality
    output at all (either it hasn't been aligned yet, or every chapter
    found is fusion-mode output).
    """
    chapters = []
    for canon, qf in find_quality_files(iso, testament):
        stats = check_chapter_fallback(qf)
        if stats is None:
            continue
        distinct_id, book, chapter_str = _parse_timing_path(qf)
        chapters.append({
            "canon": canon, "distinct_id": distinct_id, "book": book, "chapter": chapter_str,
            **stats,
            "flagged": stats["fallback_rate"] >= threshold,
        })

    if not chapters:
        return None

    flagged = [c for c in chapters if c["flagged"]]
    total_verses = sum(c["verses"] for c in chapters)
    total_fallback = sum(c["fallback_verses"] for c in chapters)

    return {
        "iso": iso,
        "chapters": len(chapters),
        "flagged_chapters": len(flagged),
        "total_verses": total_verses,
        "total_fallback_verses": total_fallback,
        "overall_fallback_rate": total_fallback / total_verses if total_verses else 0.0,
        "chapter_details": chapters,
    }

