#!/usr/bin/env python3
"""
Compare pipeline timing vs downloaded timecode across multiple languages.

Finds chapters where both pipeline-generated and original downloaded timing
exist, compares verse timestamps, and reports quality per language.

Usage:
    # Auto-discover all languages and show summary
    python compare_timing.py

    # Single language with per-chapter detail
    python compare_timing.py --iso fra

    # Multiple specific languages
    python compare_timing.py --iso-list fra,swe,por

    # Filter to NT only
    python compare_timing.py --testament nt
"""

import argparse
import sys
from pathlib import Path

from quality_report import (
    TIMING_DIR,
    DOWNLOADS_DIR,
    TIMECODE_CATEGORIES,
    find_all_downloaded_timecode,
    find_pipeline_timing_files,
    load_timing_verses,
    compare_verse_timings,
    _parse_timing_path,
    _get_canons,
)


def discover_languages(testament=None):
    """Auto-discover ISO codes with both pipeline timing and downloaded timecode."""
    canons = _get_canons(testament)

    # Collect ISOs with pipeline timing data
    pipeline_isos = set()
    for canon in canons:
        canon_dir = TIMING_DIR / canon
        if canon_dir.exists():
            for iso_dir in canon_dir.iterdir():
                if iso_dir.is_dir():
                    pipeline_isos.add(iso_dir.name)

    # Filter to those that also have downloaded timecode
    result = []
    for iso in sorted(pipeline_isos):
        found = False
        for canon in canons:
            if found:
                break
            for cat in TIMECODE_CATEGORIES:
                cat_dir = DOWNLOADS_DIR / canon / cat / iso
                if cat_dir.exists() and any(cat_dir.iterdir()):
                    result.append(iso)
                    found = True
                    break

    return result


def compare_language(iso, testament=None):
    """Compare pipeline vs downloaded timing for a single language.

    Returns dict with aggregate stats and per-chapter details, or None if
    no common chapters exist.
    """
    downloaded_tc = find_all_downloaded_timecode(iso, testament)
    pipeline_files = find_pipeline_timing_files(iso, testament)

    # Build lookup for pipeline timing by key
    pipeline_by_key = {}
    for canon, tf in pipeline_files:
        distinct_id, book, chapter_str = _parse_timing_path(tf)
        if chapter_str:
            key = (canon, distinct_id, book, chapter_str)
            pipeline_by_key[key] = tf

    # Find common chapters
    common_keys = set(downloaded_tc.keys()) & set(pipeline_by_key.keys())
    if not common_keys:
        return None

    good = drift = bad = 0
    all_mean_deltas = []
    overall_max_delta = 0.0
    chapter_details = []

    for key in sorted(common_keys):
        canon, distinct_id, book, chapter_str = key
        dl_verses = load_timing_verses(downloaded_tc[key])
        pl_verses = load_timing_verses(pipeline_by_key[key])
        cmp = compare_verse_timings(dl_verses, pl_verses)

        if cmp is None:
            continue

        if cmp["status"] == "GOOD":
            good += 1
        elif cmp["status"] == "DRIFT":
            drift += 1
        else:
            bad += 1

        all_mean_deltas.append(cmp["mean_delta"])
        overall_max_delta = max(overall_max_delta, cmp["max_delta"])

        chapter_details.append({
            "canon": canon,
            "distinct_id": distinct_id,
            "book": book,
            "chapter": chapter_str,
            "status": cmp["status"],
            "mean_delta": cmp["mean_delta"],
            "max_delta": cmp["max_delta"],
            "max_verse": cmp["max_verse"],
            "common_verses": cmp["common_verses"],
        })

    chapters_compared = good + drift + bad
    if chapters_compared == 0:
        return None

    return {
        "iso": iso,
        "chapters": chapters_compared,
        "good": good,
        "drift": drift,
        "bad": bad,
        "mean_delta": sum(all_mean_deltas) / len(all_mean_deltas),
        "max_delta": overall_max_delta,
        "chapter_details": chapter_details,
    }


WORD_TIMING_DIR = Path("word-timing-data")


def clean_bad_chapters(results):
    """Delete all pipeline output files for BAD chapters."""
    bad_chapters = []
    for r in results:
        for d in r["chapter_details"]:
            if d["status"] == "BAD":
                bad_chapters.append((r["iso"], d))

    if not bad_chapters:
        print("No BAD chapters to clean.")
        return

    deleted = 0
    for iso, d in bad_chapters:
        canon, distinct_id, book, chapter_str = d["canon"], d["distinct_id"], d["book"], d["chapter"]

        # Find the pipeline timing file to extract the fileset from its filename
        timing_dir = TIMING_DIR / canon / iso / distinct_id / book
        timing_matches = list(timing_dir.glob(f"{book}_{chapter_str}_*_timing.json")) if timing_dir.exists() else []

        if not timing_matches:
            continue

        # Extract fileset from filename: BOOK_CCC_FILESET_timing.json
        stem = timing_matches[0].stem.replace("_timing", "")
        parts = stem.split("_", 2)
        if len(parts) < 3:
            continue
        fileset = parts[2]

        # Delete fusion outputs (export/timing-data)
        for suffix in ["_timing.json", "_words.json", "_words_quality.json", ".srt"]:
            f = timing_dir / f"{book}_{chapter_str}_{fileset}{suffix}"
            if f.exists():
                f.unlink()
                deleted += 1

        # Delete intermediate files (word-timing-data)
        word_dir = WORD_TIMING_DIR / canon / iso / distinct_id / book
        for suffix in ["_whisper_words.json", "_mms_words.json"]:
            f = word_dir / f"{book}_{chapter_str}_{fileset}{suffix}"
            if f.exists():
                f.unlink()
                deleted += 1

    print(f"Cleaned {len(bad_chapters)} BAD chapters ({deleted} files deleted)")


def print_summary_table(results):
    """Print multi-language summary table."""
    print("\nTiming Comparison: pipeline vs downloaded timecode\n")

    fmt = "  {:<5} {:>8}  {:>4}  {:>5}  {:>3}  {:>6}  {:>6}"
    print(fmt.format("ISO", "Chapters", "GOOD", "DRIFT", "BAD", "Mean-d", "Max-d"))
    print("  " + "-" * 47)

    t_ch = t_good = t_drift = t_bad = 0
    t_max = 0.0
    t_mean_deltas = []

    for r in results:
        print(fmt.format(
            r["iso"], r["chapters"], r["good"], r["drift"], r["bad"],
            f"{r['mean_delta']:.2f}s", f"{r['max_delta']:.1f}s",
        ))
        t_ch += r["chapters"]
        t_good += r["good"]
        t_drift += r["drift"]
        t_bad += r["bad"]
        t_max = max(t_max, r["max_delta"])
        t_mean_deltas.append(r["mean_delta"])

    print("  " + "-" * 47)
    overall_mean = sum(t_mean_deltas) / len(t_mean_deltas) if t_mean_deltas else 0
    print(fmt.format("ALL", t_ch, t_good, t_drift, t_bad,
                      f"{overall_mean:.2f}s", f"{t_max:.1f}s"))
    print()


def print_detail_table(result):
    """Print per-chapter detail for a single language."""
    details = result["chapter_details"]
    if not details:
        return

    print(f"  Detail: {result['iso']} ({result['chapters']} chapters compared)\n")

    fmt = "    {:<5} {:<10} {:<4} {:>3}  {:<5}  {:>6}  {:>6}  {:>4}  {:>6}"
    print(fmt.format("Canon", "Fileset", "Book", "Ch", "Status", "Mean-d", "Max-d", "MaxV", "Verses"))
    print("    " + "-" * 62)

    for d in details:
        print(fmt.format(
            d["canon"].upper(), d["distinct_id"], d["book"], d["chapter"],
            d["status"], f"{d['mean_delta']:.2f}s", f"{d['max_delta']:.1f}s",
            d["max_verse"], d["common_verses"],
        ))

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare pipeline timing vs downloaded timecode across languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--iso", type=str, help="Single language ISO 639-3 code")
    parser.add_argument("--iso-list", type=str, help="Comma-separated ISO codes")
    parser.add_argument("--testament", type=str, choices=["nt", "ot", "both"],
                        default=None, help="Filter to NT or OT")
    parser.add_argument("--clean-bad", action="store_true",
                        help="Delete all pipeline output for BAD chapters")

    args = parser.parse_args()

    # Determine language list
    show_detail = False
    if args.iso:
        iso_codes = [args.iso.lower()]
        show_detail = True
    elif args.iso_list:
        iso_codes = [c.strip().lower() for c in args.iso_list.split(",")]
        show_detail = True
    else:
        iso_codes = discover_languages(args.testament)
        if not iso_codes:
            print("No languages found with both pipeline timing and downloaded timecode.")
            sys.exit(0)

    # Run comparisons
    results = []
    for iso in iso_codes:
        result = compare_language(iso, args.testament)
        if result:
            results.append(result)
        else:
            print(f"  {iso}: no common chapters found (skipped)")

    if not results:
        print("No chapters to compare.")
        sys.exit(0)

    # Print summary table (always)
    print_summary_table(results)

    # Print per-chapter detail (only when specific languages requested)
    if show_detail:
        for result in results:
            print_detail_table(result)

    # Clean BAD chapters if requested
    if args.clean_bad:
        clean_bad_chapters(results)


if __name__ == "__main__":
    main()
