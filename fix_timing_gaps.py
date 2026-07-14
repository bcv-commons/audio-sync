#!/usr/bin/env python3
"""Fix timing gaps in alignment output by re-running fusion with gap-fill.

Scans all chapters in export/timing-data/ for verse-1 gaps > threshold,
then re-runs the fusion step with MMS segment re-alignment to fix them.

Usage:
    python fix_timing_gaps.py                    # scan and report only
    python fix_timing_gaps.py --fix              # scan and fix
    python fix_timing_gaps.py --fix --iso fra    # fix specific language
    python fix_timing_gaps.py --threshold 2.0    # custom gap threshold
"""

import argparse
import json
import sys
import time
from pathlib import Path

TIMING_DIR = Path("export/timing-data")
WORD_TIMING_DIR = Path("word-timing-data")
DOWNLOADS_DIR = Path("downloads/BB")
GAP_THRESHOLD_DEFAULT = 3.0


def find_gaps(threshold=GAP_THRESHOLD_DEFAULT, iso_filter=None):
    """Scan all chapters and find those with verse-1 gaps above threshold."""
    gaps = []
    for words_file in sorted(TIMING_DIR.rglob("*_words.json")):
        if "_quality" in words_file.name:
            continue
        parts = words_file.relative_to(TIMING_DIR).parts
        if len(parts) < 5:
            continue
        canon, iso, did, book = parts[0], parts[1], parts[2], parts[3]

        if iso_filter and iso != iso_filter:
            continue

        try:
            d = json.load(open(words_file))
        except (json.JSONDecodeError, IOError):
            continue

        verses = d.get("verses", {})
        # Check all verses, not just verse 1
        for v_num_str, v_words in verses.items():
            if len(v_words) < 2:
                continue
            for wi in range(len(v_words) - 1):
                if v_words[wi] is None or v_words[wi + 1] is None:
                    continue
                gap = v_words[wi + 1] - v_words[wi]
                if gap > threshold:
                    ch = words_file.stem.split("_")[1]
                    fileset = words_file.stem.split("_")[2] if len(words_file.stem.split("_")) > 2 else ""
                    gaps.append({
                        "canon": canon,
                        "iso": iso,
                        "did": did,
                        "book": book,
                        "chapter": ch,
                        "fileset": fileset,
                        "verse": v_num_str,
                        "word_idx": wi,
                        "word1_time": v_words[wi],
                        "word2_time": v_words[wi + 1],
                        "gap": gap,
                        "words_file": words_file,
                    })
                    break  # only first gap per verse
    return gaps


def fix_gaps(gaps):
    """Fix detected gaps by re-running fusion with MMS gap-fill."""
    if not gaps:
        print("No gaps to fix.")
        return

    # Group by (canon, iso, did) to load MMS model once
    from collections import defaultdict
    by_fileset = defaultdict(list)
    for g in gaps:
        key = (g["canon"], g["iso"], g["did"])
        by_fileset[key].append(g)

    # Load MMS model
    print("[INFO] Loading MMS model...")
    from mms_align_words import load_mms_model
    mms_loaded = load_mms_model()
    print("[INFO] MMS model loaded.")

    fixed_count = 0
    failed_count = 0

    for (canon, iso, did), chapter_gaps in sorted(by_fileset.items()):
        # Group by chapter
        by_chapter = defaultdict(list)
        for g in chapter_gaps:
            by_chapter[g["chapter"]].append(g)

        for ch, ch_gaps in sorted(by_chapter.items()):
            g = ch_gaps[0]  # use first gap for chapter info
            book = g["book"]
            fileset = g["fileset"]

            # Find audio file
            audio_dir = DOWNLOADS_DIR / canon / iso / did / book
            audio_files = list(audio_dir.glob(f"{book}_{ch}_*.mp3")) if audio_dir.exists() else []
            if not audio_files:
                print(f"  [SKIP] {canon}/{iso}/{did}/{book} ch{ch}: no audio file")
                failed_count += 1
                continue

            # Find text, MMS, Whisper files
            text_files = list(audio_dir.glob(f"{book}_{ch}_*.txt")) if audio_dir.exists() else []
            mms_dir = WORD_TIMING_DIR / canon / iso / did / book
            mms_files = list(mms_dir.glob(f"{book}_{ch}_*_mms_words.json")) if mms_dir.exists() else []
            whisper_files = list(mms_dir.glob(f"{book}_{ch}_*_whisper_words.json")) if mms_dir.exists() else []

            if not mms_files:
                print(f"  [SKIP] {canon}/{iso}/{did}/{book} ch{ch}: no MMS data")
                failed_count += 1
                continue

            # Build fusion item
            timing_file = TIMING_DIR / canon / iso / did / book / f"{book}_{ch}_{fileset}_timing.json"
            words_file = g["words_file"]

            from whisper_transcribe import load_language_config
            config = load_language_config(iso)

            item = {
                "book": book,
                "chapter_str": ch,
                "ref_text_path": text_files[0] if text_files else None,
                "timing_path": timing_file,
                "words_path": words_file,
                "mms_path": mms_files[0],
                "whisper_path": whisper_files[0] if whisper_files else None,
                "audio_path": audio_files[0],
            }

            print(f"  [FIX] {canon}/{iso}/{did}/{book} ch{ch} "
                  f"(v{g['verse']} gap={g['gap']:.1f}s)...", end=" ")
            t0 = time.time()

            from align_words import process_chapter
            stats = process_chapter(item, config, mms_components=mms_loaded)

            elapsed = time.time() - t0
            if "error" in stats:
                print(f"ERROR: {stats['error']}")
                failed_count += 1
            else:
                # Check if gap was actually fixed
                try:
                    d = json.load(open(words_file))
                    v_words = d["verses"].get(g["verse"], [])
                    wi = g["word_idx"]
                    if wi < len(v_words) - 1 and v_words[wi] and v_words[wi + 1]:
                        new_gap = v_words[wi + 1] - v_words[wi]
                        if new_gap < g["gap"]:
                            print(f"fixed ({g['gap']:.1f}s → {new_gap:.1f}s) [{elapsed:.1f}s]")
                            fixed_count += 1
                        else:
                            print(f"unchanged ({new_gap:.1f}s) [{elapsed:.1f}s]")
                            failed_count += 1
                    else:
                        print(f"done [{elapsed:.1f}s]")
                        fixed_count += 1
                except Exception:
                    print(f"done [{elapsed:.1f}s]")
                    fixed_count += 1

    print(f"\n[DONE] Fixed: {fixed_count}, Failed/Unchanged: {failed_count}")


def main():
    parser = argparse.ArgumentParser(description="Fix timing gaps in alignment output")
    parser.add_argument("--fix", action="store_true", help="Actually fix gaps (default: report only)")
    parser.add_argument("--iso", help="Filter to specific language ISO code")
    parser.add_argument("--threshold", type=float, default=GAP_THRESHOLD_DEFAULT,
                        help=f"Gap threshold in seconds (default: {GAP_THRESHOLD_DEFAULT})")
    args = parser.parse_args()

    print(f"Scanning for gaps > {args.threshold}s...")
    gaps = find_gaps(threshold=args.threshold, iso_filter=args.iso)

    # Deduplicate: one entry per chapter (take the worst gap)
    by_chapter = {}
    for g in gaps:
        key = (g["canon"], g["iso"], g["did"], g["book"], g["chapter"])
        if key not in by_chapter or g["gap"] > by_chapter[key]["gap"]:
            by_chapter[key] = g
    gaps = sorted(by_chapter.values(), key=lambda g: (g["iso"], g["book"], g["chapter"]))

    print(f"Found {len(gaps)} chapters with gaps across {len(set(g['iso'] for g in gaps))} languages\n")

    if not gaps:
        return

    # Report
    for g in gaps:
        print(f"  {g['canon']}/{g['iso']}/{g['did']}/{g['book']} ch{g['chapter']}: "
              f"v{g['verse']} word{g['word_idx']+1} {g['word1_time']:.2f}→{g['word2_time']:.2f} "
              f"({g['gap']:.1f}s)")

    if not args.fix:
        print(f"\nRun with --fix to repair these gaps.")
        return

    print(f"\nFixing {len(gaps)} chapters...")
    fix_gaps(gaps)


if __name__ == "__main__":
    main()
