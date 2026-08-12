#!/usr/bin/env python3
"""Fix timing gaps in alignment output by re-running fusion with gap-fill.

Scans all chapters in export/timing-data/ for verse-1 gaps > threshold,
then re-runs the fusion step with MMS segment re-alignment to fix them.

Every --fix run snapshots each chapter before touching it and re-measures
after, reverting if the fix made overall quality worse rather than better
(see fix_gaps()'s docstring) — added for the 2026-08-12 full-corpus
gap-fix pass (6,644 chapters); remove the snapshot/revert/report
machinery once that run is confirmed clean, it isn't meant to be
permanent.

Usage:
    python tools/fix_timing_gaps.py                    # scan and report only
    python tools/fix_timing_gaps.py --fix              # scan and fix
    python tools/fix_timing_gaps.py --fix --iso fra    # fix specific language
    python tools/fix_timing_gaps.py --threshold 2.0    # custom gap threshold
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

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

        verses = d.get("beg", {})
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


def _quality_path(words_file: Path) -> Path:
    return Path(str(words_file).replace("_words.json", "_words_quality.json"))


def _snapshot_chapter(timing_file: Path, words_file: Path, quality_file: Path) -> dict:
    """Read current on-disk state + derived quality metrics for a chapter,
    for later before/after regression comparison and revert. Returns raw
    bytes (for revert) alongside the metrics (for comparison) so this is
    the single source of truth for both.
    """
    from check_timing_quality import analyze_chapter_timing, analyze_chapter_words

    snap = {
        "timing_bytes": timing_file.read_bytes() if timing_file.exists() else None,
        "words_bytes": words_file.read_bytes() if words_file.exists() else None,
        "quality_bytes": quality_file.read_bytes() if quality_file.exists() else None,
        "timing_metrics": analyze_chapter_timing(timing_file) if timing_file.exists() else None,
        "word_metrics": analyze_chapter_words(words_file) if words_file.exists() else None,
        "avg_score": None,
    }
    if quality_file.exists():
        try:
            with open(quality_file) as f:
                snap["avg_score"] = json.load(f).get("summary", {}).get("avg_score")
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return snap


def _restore_chapter(timing_file: Path, words_file: Path, quality_file: Path, snap: dict) -> None:
    """Write a snapshot's bytes back to disk, undoing whatever the fix attempt wrote."""
    if snap["timing_bytes"] is not None:
        timing_file.write_bytes(snap["timing_bytes"])
    if snap["words_bytes"] is not None:
        words_file.write_bytes(snap["words_bytes"])
    if snap["quality_bytes"] is not None:
        quality_file.write_bytes(snap["quality_bytes"])


# Tolerance below which an avg_score dip is treated as noise, not a real
# regression — CTC/gap-fill scores aren't perfectly deterministic to the
# 3rd decimal across re-runs even when nothing meaningfully changed.
_SCORE_REGRESSION_EPSILON = 0.01


def _compare_snapshots(before: dict, after: dict) -> tuple[bool, list[str]]:
    """Compare before/after snapshots. Returns (regressed, reasons)."""
    reasons = []
    if (before["avg_score"] is not None and after["avg_score"] is not None
            and after["avg_score"] < before["avg_score"] - _SCORE_REGRESSION_EPSILON):
        reasons.append(f"avg_score dropped {before['avg_score']:.3f} -> {after['avg_score']:.3f}")
    if before["timing_metrics"] and after["timing_metrics"]:
        if after["timing_metrics"]["backwards"] > before["timing_metrics"]["backwards"]:
            reasons.append(f"backwards jumps increased "
                            f"{before['timing_metrics']['backwards']} -> {after['timing_metrics']['backwards']}")
        if after["timing_metrics"]["dupes"] > before["timing_metrics"]["dupes"]:
            reasons.append(f"dupes increased "
                            f"{before['timing_metrics']['dupes']} -> {after['timing_metrics']['dupes']}")
    if before["word_metrics"] and after["word_metrics"]:
        if after["word_metrics"]["nulls"] > before["word_metrics"]["nulls"]:
            reasons.append(f"null words increased "
                            f"{before['word_metrics']['nulls']} -> {after['word_metrics']['nulls']}")
    return bool(reasons), reasons


def fix_gaps(gaps, report_path: Path | None = None):
    """Fix detected gaps by re-running fusion with MMS gap-fill.

    Sanity check (temporary, added for the 2026-08-12 full-corpus gap-fix
    pass — remove once that run is confirmed clean): every chapter is
    snapshotted before the fix attempt and re-measured after. If the
    fix made overall chapter quality worse (avg score dropped beyond
    noise tolerance, backwards jumps increased, dupes increased, or null
    words increased) — not just "did the one targeted gap shrink" — the
    original files are restored and the chapter is left untouched rather
    than published in a worse state. Every chapter's before/after numbers
    are written to report_path (JSONL) regardless of verdict, so the
    whole run can be audited afterward, not just the reverted cases.
    """
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
    reverted_count = 0

    report_f = open(report_path, "a") if report_path else None

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
            quality_file = _quality_path(words_file)

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

            chapter_key = f"{canon}/{iso}/{did}/{book} ch{ch}"
            print(f"  [FIX] {chapter_key} "
                  f"(v{g['verse']} gap={g['gap']:.1f}s)...", end=" ")
            t0 = time.time()

            before = _snapshot_chapter(timing_file, words_file, quality_file)

            from align_words import process_chapter
            stats = process_chapter(item, config, mms_components=mms_loaded)

            elapsed = time.time() - t0
            if "error" in stats:
                print(f"ERROR: {stats['error']}")
                failed_count += 1
                if report_f:
                    report_f.write(json.dumps({
                        "chapter": chapter_key, "verdict": "error", "error": stats["error"],
                    }) + "\n")
                continue

            after = _snapshot_chapter(timing_file, words_file, quality_file)
            regressed, reasons = _compare_snapshots(before, after)

            # DEBUG (temporary, for this run): always show the before/after
            # numbers, not just on regression, so the sanity check itself is
            # visible in the log while this pass is being watched.
            print(f"[before avg={before['avg_score']}, backwards="
                  f"{before['timing_metrics']['backwards'] if before['timing_metrics'] else '?'}, "
                  f"nulls={before['word_metrics']['nulls'] if before['word_metrics'] else '?'} | "
                  f"after avg={after['avg_score']}, backwards="
                  f"{after['timing_metrics']['backwards'] if after['timing_metrics'] else '?'}, "
                  f"nulls={after['word_metrics']['nulls'] if after['word_metrics'] else '?'}]", end=" ")

            if regressed:
                _restore_chapter(timing_file, words_file, quality_file, before)
                reverted_count += 1
                print(f"REVERTED — quality regressed ({'; '.join(reasons)}) [{elapsed:.1f}s]")
                verdict = "reverted"
            else:
                # Check if the specific triggering gap actually shrank —
                # still tracked for the existing fixed/unchanged distinction,
                # now on top of (not instead of) the holistic check above.
                try:
                    d = json.loads(after["words_bytes"])
                    v_words = d["beg"].get(g["verse"], [])
                    wi = g["word_idx"]
                    if wi < len(v_words) - 1 and v_words[wi] and v_words[wi + 1]:
                        new_gap = v_words[wi + 1] - v_words[wi]
                        if new_gap < g["gap"]:
                            print(f"fixed ({g['gap']:.1f}s -> {new_gap:.1f}s) [{elapsed:.1f}s]")
                            fixed_count += 1
                            verdict = "fixed"
                        else:
                            print(f"unchanged ({new_gap:.1f}s), no regression [{elapsed:.1f}s]")
                            failed_count += 1
                            verdict = "unchanged"
                    else:
                        print(f"done, no regression [{elapsed:.1f}s]")
                        fixed_count += 1
                        verdict = "done"
                except Exception:
                    print(f"done, no regression [{elapsed:.1f}s]")
                    fixed_count += 1
                    verdict = "done"

            if report_f:
                report_f.write(json.dumps({
                    "chapter": chapter_key,
                    "verdict": verdict,
                    "regression_reasons": reasons,
                    "before": {"avg_score": before["avg_score"], "timing": before["timing_metrics"],
                               "words": before["word_metrics"]},
                    "after": {"avg_score": after["avg_score"], "timing": after["timing_metrics"],
                              "words": after["word_metrics"]},
                }) + "\n")
                report_f.flush()

    if report_f:
        report_f.close()

    print(f"\n[DONE] Fixed: {fixed_count}, Failed/Unchanged: {failed_count}, "
          f"Reverted (quality regression): {reverted_count}")


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
    report_path = Path("_runs") / f"gapfix_sanity_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Per-chapter before/after report: {report_path}\n")
    fix_gaps(gaps, report_path=report_path)


if __name__ == "__main__":
    main()
