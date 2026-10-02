#!/usr/bin/env python3
"""Run fusion (Step 2) for chapters whose whisper_words.json is newer than
their existing _timing.json/_words.json -- without requiring the source
mp3 to still be on disk.

Why this exists: align_pipeline.py's chapter discovery (discover_chapter_files())
enumerates chapters by globbing *.mp3 files. For the Whisper-backfill
population specifically, the mp3 was already purged back when the chapter
was first (MMS-only) aligned, long before whisper_backfill.py added the
missing Whisper transcript -- so align_pipeline.py --skip-whisper --skip-mms
finds ZERO chapters for these isos, regardless of --force-fusion. Confirmed
2026-10-01 on abq (0 local mp3s, 28+18 chapters with real pre-existing
MMS+now-Whisper data, align_pipeline.py reported "all already processed").

Fusion itself never needs the audio for this case -- with mms_components=None
(we're not re-running MMS), align_words.py's gap-fill re-alignment path
(the only place that touches audio_path) is unreachable (see
`if audio_path and mms_components:` in align_words.py). Text is also
unaffected by the audio purge -- only audio gets purged, so the reference
.txt is still sitting right next to where the mp3 used to be.

So this discovers chapters directly from existing _timing.json files
(export/timing-data/) instead, matches each to its whisper_words.json
(word-timing-data/), and calls align_pipeline.py's own run_fusion_chapter()
directly -- same function, same fusion code, just reached without the
audio-dependent discovery step.

Usage:
    python tools/fusion_only_redo.py --iso abq
    python tools/fusion_only_redo.py --iso abq,aca,ach
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from align_pipeline import needs_run, run_fusion_chapter, WORD_TIMING_DIR  # noqa: E402
from text_processing import load_language_config  # noqa: E402

OUTPUT_DIR = Path("export/timing-data")


def log(msg):
    import time
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def find_text_path(canon: str, iso: str, distinct_id: str, book: str, chapter_str: str) -> Path | None:
    """Reference text lives next to where the (possibly now-purged) audio
    was -- only audio gets purged, text does not. Same base-dir resolution
    whisper_transcribe.py's discover_chapter_files() uses."""
    from whisper_transcribe import _find_base_dir
    base_dir = _find_base_dir(canon, iso, distinct_id)
    if not base_dir:
        return None
    book_dir = base_dir / book
    if not book_dir.exists():
        return None
    candidates = sorted(book_dir.glob(f"{book}_{chapter_str}_*.txt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if "_ET" in p.stem else 1, p.name))
    return candidates[0]


def fuse_iso(iso: str) -> dict:
    config = load_language_config(iso)
    stats = {"fused": 0, "skipped_fresh": 0, "skipped_no_whisper": 0,
             "skipped_no_text": 0, "failed": 0}

    for canon_dir in sorted(OUTPUT_DIR.iterdir()):
        canon = canon_dir.name
        iso_dir = canon_dir / iso
        if not iso_dir.exists():
            continue
        for distinct_dir in sorted(iso_dir.iterdir()):
            distinct_id = distinct_dir.name
            if not distinct_dir.is_dir():
                continue
            for book_dir in sorted(distinct_dir.iterdir()):
                if not book_dir.is_dir():
                    continue
                book = book_dir.name
                for timing_path in sorted(book_dir.glob(f"{book}_*_timing.json")):
                    stem = timing_path.name.replace("_timing.json", "")
                    parts = stem.split("_", 2)
                    if len(parts) < 3:
                        continue
                    chapter_str = parts[1]
                    audio_fileset = parts[2]
                    try:
                        chapter_num = int(chapter_str)
                    except ValueError:
                        continue

                    word_book_dir = WORD_TIMING_DIR / canon / iso / distinct_id / book
                    mms_path = word_book_dir / f"{book}_{chapter_str}_{audio_fileset}_mms_words.json"
                    whisper_path = word_book_dir / f"{book}_{chapter_str}_{audio_fileset}_whisper_words.json"

                    if not whisper_path.exists():
                        stats["skipped_no_whisper"] += 1
                        continue

                    if not needs_run(timing_path, mms_path if mms_path.exists() else None, whisper_path):
                        stats["skipped_fresh"] += 1
                        continue

                    text_path = find_text_path(canon, iso, distinct_id, book, chapter_str)
                    if not text_path:
                        log(f"  {iso}/{distinct_id} {book} {chapter_num}: no text file found, skipping")
                        stats["skipped_no_text"] += 1
                        continue

                    words_path = Path(str(timing_path).replace("_timing.json", "_words.json"))
                    item = {
                        "mms_path": mms_path if mms_path.exists() else None,
                        "whisper_path": whisper_path,
                        "ref_text_path": text_path,
                        "timing_path": timing_path,
                        "words_path": words_path,
                        "preserve_existing_timing": False,
                        "book": book,
                        "chapter": chapter_num,
                        "chapter_str": chapter_str,
                        "canon": canon,
                        "iso": iso,
                        "distinct_id": distinct_id,
                        "audio_fileset": audio_fileset,
                    }
                    try:
                        result = run_fusion_chapter(item, config, mms_components=None)
                    except Exception as e:
                        log(f"  {iso}/{distinct_id} {book} {chapter_num}: EXCEPTION {e}")
                        stats["failed"] += 1
                        continue
                    if "error" in result:
                        log(f"  {iso}/{distinct_id} {book} {chapter_num}: ERROR {result['error']}")
                        stats["failed"] += 1
                        continue
                    stats["fused"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", type=str, required=True, help="Comma-separated iso codes")
    args = parser.parse_args()

    isos = [c.strip() for c in args.iso.split(",")]
    for iso in isos:
        log(f"{iso}: scanning for stale fusion output ...")
        stats = fuse_iso(iso)
        log(f"{iso}: fused={stats['fused']} skipped_fresh={stats['skipped_fresh']} "
            f"skipped_no_whisper={stats['skipped_no_whisper']} "
            f"skipped_no_text={stats['skipped_no_text']} failed={stats['failed']}")


if __name__ == "__main__":
    main()
