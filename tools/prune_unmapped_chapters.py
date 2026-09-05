#!/usr/bin/env python3
"""
Delete alignment output for text chapters a chapter_map leaves unmapped.

When a language config declares a chapter_map (an audio/text versification
mismatch — see map_audio_chapter_to_text() in pipeline/text_processing.py),
some text chapters end up with no audio mapped to them at all. Two ways
that happens:

  * a MERGE — one audio chapter holds two text chapters, so only the first
    gets the audio (bul PSA: audio 9 covers texts 9+10, so text 10 is
    orphaned);
  * a SKIP — a SPLIT, where two audio chapters each hold half of one text
    chapter, which no offset can express (bul PSA: audio 114+115 both hold
    parts of text 116).

Those chapters usually already have output on disk from before the
chapter_map existed, produced by aligning them against MISMATCHED audio.
That output is wrong, and it is wrong in the worst way: it looks fine.
Re-running with the map in place does not overwrite it, because nothing
writes to those chapters any more — so it would silently survive as
plausible-looking, confidently-timed nonsense. This prunes it.

Dry-run by default; pass --apply to actually delete.

    python tools/prune_unmapped_chapters.py --iso bul
    python tools/prune_unmapped_chapters.py --iso bul --apply
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from text_processing import load_language_config, map_audio_chapter_to_text  # noqa: E402

TIMING_DIR = Path("export/timing-data")
WORD_TIMING_DIR = Path("word-timing-data")
MAX_CHAPTER = 150


def mapped_text_chapters(config, book: str, audio_fileset: str) -> set[int]:
    """Every text chapter some audio chapter maps onto, for this book."""
    covered = set()
    for audio_ch in range(1, MAX_CHAPTER + 1):
        t = map_audio_chapter_to_text(config, book, audio_fileset, audio_ch)
        if t is not None:
            covered.add(t)
    return covered


def find_chapter_files(iso: str, book: str, chapter: int) -> list[Path]:
    """Every output artefact for one (book, chapter) across both trees."""
    found = []
    pattern = f"{book}_{chapter:03d}_*"
    for base in (TIMING_DIR, WORD_TIMING_DIR):
        for canon_dir in base.glob("*"):
            book_dir = canon_dir / iso
            if not book_dir.exists():
                continue
            for did_dir in book_dir.glob("*"):
                d = did_dir / book
                if d.exists():
                    found.extend(sorted(d.glob(pattern)))
    return found


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", required=True, help="Language ISO 639-3 code")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete (default: dry-run, just report)")
    args = parser.parse_args()

    iso = args.iso.lower()
    config = load_language_config(iso)
    if not config.chapter_map:
        print(f"{iso}: no chapter_map configured — nothing to prune.")
        return

    total_files = 0
    for rule in config.chapter_map:
        book = rule.get("book")
        audio_fileset = rule.get("audio_fileset")
        if not book:
            continue
        covered = mapped_text_chapters(config, book, audio_fileset)

        # Only consider chapters that actually exist on disk as output —
        # "unmapped" is meaningless for a chapter this edition never had.
        orphans = []
        for ch in range(1, MAX_CHAPTER + 1):
            if ch in covered:
                continue
            files = find_chapter_files(iso, book, ch)
            if files:
                orphans.append((ch, files))

        if not orphans:
            print(f"{iso} {book}: no unmapped chapters with output on disk.")
            continue

        print(f"\n{iso} {book} ({audio_fileset}): "
              f"{len(orphans)} unmapped chapter(s) with stale output")
        for ch, files in orphans:
            print(f"  text chapter {ch}: {len(files)} file(s)")
            for f in files:
                print(f"      {f}")
                total_files += 1
                if args.apply:
                    try:
                        f.unlink()
                    except OSError as e:
                        print(f"      !! failed to delete: {e}")

    print()
    if args.apply:
        print(f"Deleted {total_files} file(s).")
    else:
        print(f"DRY RUN — {total_files} file(s) would be deleted. "
              f"Re-run with --apply to actually remove them.")


if __name__ == "__main__":
    main()
