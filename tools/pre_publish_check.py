#!/usr/bin/env python3
"""
Pre-publish plausibility gate for scripts/publish-align.sh.

Scans the exact tree about to be published for three independent,
unrelated defect shapes:

  1. Backwards timestamp jumps (*_timing.json) — a verse whose timestamp
     is earlier than the previous verse's. Unlike dupes/gaps/tiny-steps
     (which check_timing_quality.py also tracks but which can be
     legitimate — two verses spoken back-to-back, a genuinely long intro),
     a backwards jump is never correct: it means the alignment placed a
     verse at the wrong point in the audio entirely (see the ENGBSB/ZLMAVB
     ISA 51 case that motivated this check — verse 1 landed at 118.7s,
     verse 2 at 45.24s).

  2. Fallback/failure collapse (*_words_quality.json) — a chapter whose
     alignment is almost entirely null timestamps / zero-confidence
     scores. See has_fallback_corruption()'s docstring — this is the
     on-disk fingerprint of a poisoned CUDA context (pipeline/gpu_health.py),
     confirmed real 2026-08-12 across 4,926 chapters that "succeeded" with
     no error while producing unusable output.

  3. Legacy pre-compact-format *_timing.json (the old per-verse-dict list
     shape, superseded by the Aug 12 {"pos": [...]} redesign — see
     is_legacy_format()). Not a quality defect, just a shape no current or
     downstream reader should be fed. Confirmed 2026-09-01: the only
     source left is dramatized/alt ("N2DA"/"O2DA") audio tracks, which
     pipeline/whisper_transcribe.py's discover_chapter_files() has since
     deliberately stopped aligning whenever a standard track exists (a
     quality decision — dramatized audio measurably hurts alignment), so
     this set can never grow from the current pipeline; it only quarantines
     the ~453 spa/SPABDA+SPAWTC files left over from before that decision.

Both checks need no prior/before state to compare against — each is a
standalone judgment about the file's own content, unlike
tools/fix_timing_gaps.py's snapshot/revert mechanism, which only catches a
regression relative to a specific run's own before/after pair.

Rather than blocking the whole publish over one bad chapter (this repo's
existing philosophy — see ruff.toml's BLE001 comment: one failure must
never take down the rest of a run), flagged chapters are quarantined:
written to an rclone --exclude-from file so publish-align.sh can skip
just those files and publish everything else.

Usage:
    python tools/pre_publish_check.py [--out PATH]

Exit code 0 always (advisory — publish-align.sh decides what to do with
the exclude file); prints a one-line summary either way.
"""

import argparse
import json
import sys
from pathlib import Path

TIMING_DIR = Path("export/timing-data")
DEFAULT_OUT = Path("_runs/pre_publish_quarantine.txt")

# A chapter whose word-quality summary is this null-heavy or this
# low-scoring essentially never happens from real alignment — even a
# genuinely hard chapter has SOME well-scored words. These thresholds are
# deliberately conservative (real corruption is null_count == total_words,
# avg_score == 0.0 exactly) to leave headroom above legitimately rough
# chapters without false-flagging them.
_FALLBACK_NULL_FRACTION = 0.9
_FALLBACK_AVG_SCORE_MAX = 0.02


def has_backwards_jump(timing_path: Path) -> bool:
    """True if any verse's timestamp precedes the previous verse's.

    Handles the new compact format ({"pos": [...]}), the old verbose
    per-verse-dict list this whole corpus is still in until reprocessed,
    and safely no-ops on shapes that are neither (e.g. OBS's story/segment
    timing files, which also live under export/timing-data/ but aren't
    Bible-chapter timing at all).
    """
    try:
        with open(timing_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if isinstance(data, dict) and "pos" in data:
        verses = [t for t in data["pos"] if t is not None]
    elif isinstance(data, list) and data and isinstance(data[0], dict) and "verse_start" in data[0]:
        verses = [e["timestamp"] for e in data if str(e.get("verse_start")) != "0"]
    else:
        return False

    return any(verses[i] < verses[i - 1] for i in range(1, len(verses)))


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
    "mms_drift_fix") and verse-only mode's (source "local"/"fallback") —
    both write the same {"summary": {"total_words", "null_count",
    "avg_score", ...}} shape (align_words.py / align_verse_words.py), and
    a poisoned context corrupts either pipeline identically.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                         help=f"Where to write the rclone exclude list (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    if not TIMING_DIR.exists():
        print(f"[pre-publish-check] {TIMING_DIR} not found, nothing to check")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("")
        return

    backwards_stems = [
        tf.stem.replace("_timing", "")
        for tf in sorted(TIMING_DIR.rglob("*_timing.json"))
        if has_backwards_jump(tf)
    ]
    fallback_stems = [
        qf.stem.replace("_words_quality", "")
        for qf in sorted(TIMING_DIR.rglob("*_words_quality.json"))
        if has_fallback_corruption(qf)
    ]
    legacy_stems = [
        tf.stem.replace("_timing", "")
        for tf in sorted(TIMING_DIR.rglob("*_timing.json"))
        if is_legacy_format(tf)
    ]

    # Union — a chapter can in principle trip more than one check; the
    # exclude list only needs each stem once regardless of how many reasons.
    all_stems = sorted(set(backwards_stems) | set(fallback_stems) | set(legacy_stems))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # One glob pattern per chapter, matches all its output files
    # (_timing.json, _words.json, _words_quality.json) at any depth.
    args.out.write_text("".join(f"**/{stem}_*\n" for stem in all_stems))

    def _print_flagged(label: str, stems: list[str]):
        print(f"[pre-publish-check] {len(stems)} chapter(s) with {label} — quarantined, not publishing:")
        for stem in stems[:20]:
            print(f"  {stem}")
        if len(stems) > 20:
            print(f"  ... and {len(stems) - 20} more")

    if backwards_stems:
        _print_flagged("a backwards timestamp jump", backwards_stems)
    if fallback_stems:
        _print_flagged("fallback/failure collapse (near-total null/zero-score alignment)", fallback_stems)
    if legacy_stems:
        _print_flagged("legacy pre-compact-format timing.json", legacy_stems)

    if all_stems:
        print(f"[pre-publish-check] {len(all_stems)} chapter(s) total quarantined — full list: {args.out}")
    else:
        print("[pre-publish-check] no problem chapters found — nothing quarantined")


if __name__ == "__main__":
    sys.exit(main())
