#!/usr/bin/env python3
"""
Requeue chapters flagged with DUPES (duplicate verse-start timestamps) for
re-alignment, after the 2026-08-08 fix to mms_align_words.py/align_words.py
(CTC-infeasible whole-chapter fallback was using a start=0.0/end=0.0
sentinel that downstream code treats as a real timestamp, not "unaligned"
— see internal-docs/gpu-wedge-forensics.md and the session history around
that date for the full story).

Deliberately scoped to DUPES only, not every flagged chapter — BACKWARDS,
LOW-SCORE, ORIG-BETTER, and GAPS are separate, pre-existing quality
concerns unrelated to this specific bug; re-running those wouldn't change
anything (same audio, same fundamental mismatch) and would just burn GPU
time. Confirmed by checking the flag breakdown before writing this: arb
and rus have zero DUPES chapters (this bug apparently never hit them),
so they're excluded even though they showed up in an earlier flagged-
chapter count.

Deletes only the per-chapter output that needs to be regenerated
(_mms_words.json, _timing.json, _words.json, _words_quality.json) —
leaves _whisper_words.json alone, since transcription itself was never
the problem and re-running it is wasted GPU time. A normal
align_pipeline.py run afterwards will naturally redo exactly these
chapters via its existing skip-if-exists logic and leave everything else
untouched.

This script only deletes files — it does NOT run any GPU work itself.
Actually re-aligning the requeued chapters is a separate, deliberate step
(see the printed instructions at the end), so this is safe to run any
time regardless of what else is using the GPU.

Usage:
    # Preview only (default) — prints what would be deleted, deletes nothing
    python tools/requeue_dupes_chapters.py --iso-list cmn,ind,spa,por

    # Actually delete (still doesn't touch the GPU)
    python tools/requeue_dupes_chapters.py --iso-list cmn,ind,spa,por --apply
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from check_timing_quality import check_language
from quality_report import TIMING_DIR

WORD_TIMING_DIR = Path("word-timing-data")


def find_chapter_files(canon: str, iso: str, distinct_id: str, book: str, chapter_str: str):
    """Locate this chapter's actual on-disk files (audio_fileset varies per
    edition, so glob rather than reconstruct the filename)."""
    timing_dir = TIMING_DIR / canon / iso / distinct_id / book
    word_dir = WORD_TIMING_DIR / canon / iso / distinct_id / book

    timing_matches = list(timing_dir.glob(f"{book}_{chapter_str}_*_timing.json"))
    if not timing_matches:
        return None

    timing_path = timing_matches[0]
    audio_fileset = timing_path.stem.replace(f"{book}_{chapter_str}_", "").replace("_timing", "")

    return {
        "timing": timing_path,
        "words": timing_dir / f"{book}_{chapter_str}_{audio_fileset}_words.json",
        "quality": timing_dir / f"{book}_{chapter_str}_{audio_fileset}_words_quality.json",
        "mms_words": word_dir / f"{book}_{chapter_str}_{audio_fileset}_mms_words.json",
        # whisper_words.json deliberately NOT included — kept as-is
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso-list", type=str, required=True, help="Comma-separated ISO codes to check")
    parser.add_argument("--apply", action="store_true", help="Actually delete files (default: dry-run preview only)")
    args = parser.parse_args()

    isos = [i.strip() for i in args.iso_list.split(",") if i.strip()]
    total_requeued = 0
    total_missing = 0

    for iso in isos:
        result = check_language(iso)
        if not result:
            print(f"{iso}: no pipeline data found, skipping")
            continue

        dupes_chapters = [c for c in result["chapter_details"] if "DUPES" in c["flags"]]
        if not dupes_chapters:
            print(f"{iso}: 0 DUPES chapters, nothing to requeue")
            continue

        print(f"{iso}: {len(dupes_chapters)} DUPES chapter(s)")
        for c in dupes_chapters:
            files = find_chapter_files(c["canon"], iso, c["distinct_id"], c["book"], c["chapter"])
            if not files:
                print(f"  {c['canon']}/{c['distinct_id']}/{c['book']} {c['chapter']}: "
                      f"could not locate on-disk files, skipping")
                total_missing += 1
                continue

            for label, path in files.items():
                if not path.exists():
                    continue
                if args.apply:
                    path.unlink()
                    print(f"  deleted ({label}): {path}")
                else:
                    print(f"  would delete ({label}): {path}")
            total_requeued += 1

    print()
    if not args.apply:
        print(f"DRY RUN — {total_requeued} chapter(s) would be requeued across {len(isos)} language(s). "
              f"Re-run with --apply to actually delete.")
    else:
        print(f"Requeued {total_requeued} chapter(s) across {len(isos)} language(s) "
              f"({total_missing} could not be located on disk).")
        print()
        print("Next step (actually re-aligns them — needs the GPU, run when it's free):")
        print(f"  python pipeline/align_pipeline.py --iso-list {','.join(isos)} --books ALL --no-download")


if __name__ == "__main__":
    main()
