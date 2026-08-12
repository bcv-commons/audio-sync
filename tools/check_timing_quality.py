#!/usr/bin/env python3
"""
Check timing quality across pipeline-generated data.

Detects verse-level issues (duplicates, backwards jumps, tiny steps, large gaps),
word-level issues (null timestamps), and compares against original downloaded
timecodes to flag chapters where the original is better.

Usage:
    # All languages with pipeline data
    python tools/check_timing_quality.py

    # Single language with per-chapter detail
    python tools/check_timing_quality.py --iso fra

    # Multiple languages
    python tools/check_timing_quality.py --iso-list fra,swe,por

    # Filter to NT only
    python tools/check_timing_quality.py --testament nt
"""

import argparse
import json
import sys
from pathlib import Path

from quality_report import (
    TIMING_DIR,
    DOWNLOADS_DIR,
    TIMECODE_CATEGORIES,
    find_all_downloaded_timecode,
    find_pipeline_timing_files,
    find_quality_files,
    load_quality,
    _parse_timing_path,
    _get_canons,
)


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
    fallback fix). See check_language()'s DUPES vs DUPES-OK split.
    """
    with open(path) as f:
        data = json.load(f)

    # Two live formats: our own pipeline output (compact {"pos": [...]})
    # and DBT's original downloaded timecodes under downloads/BB/ (old
    # verbose list-of-verse-dicts, an external source this tool doesn't
    # control and which will never move to the compact format).
    if isinstance(data, dict) and "pos" in data:
        # pos[i] is verse (i+1)'s timestamp (no verse-0 slot — see
        # align_words.py's write_timing_json() docstring) — the +1 keeps
        # verse-number strings real, matching the quality file's "verses"
        # keys (real verse numbers) that dupe_verses gets cross-referenced
        # against below.
        verses = [(str(i + 1), t) for i, t in enumerate(data["pos"]) if t is not None]
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        verses = [(str(e["verse_start"]), e["timestamp"]) for e in data
                  if str(e["verse_start"]) != "0"]
    else:
        return None

    if len(verses) < 2:
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


def discover_pipeline_languages(testament=None):
    """Find all ISO codes with pipeline timing data."""
    canons = _get_canons(testament)
    isos = set()
    for canon in canons:
        canon_dir = TIMING_DIR / canon
        if canon_dir.exists():
            for d in canon_dir.iterdir():
                if d.is_dir():
                    isos.add(d.name)
    return sorted(isos)


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
            # Cross-reference the specific dupe verses' word scores to tell
            # a real alignment failure (0.0-score fallback) apart from two
            # verses genuinely spoken back-to-back with no gap (high-
            # confidence). Only the former is an actual issue — see
            # analyze_chapter_timing()'s docstring for how this was found.
            dupe_scores = []
            if q_data:
                for vnum in gen_timing["dupe_verses"]:
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

    # DUPES-OK is informational, not an issue (see check_language()'s
    # DUPES/DUPES-OK split above) — a chapter flagged with ONLY that
    # shouldn't count toward has_issues, and its dupe count shouldn't
    # inflate total_dupes (which "how many chapters actually need
    # attention" tooling, e.g. tools/requeue_dupes_chapters.py, reads).
    has_issues = sum(
        1 for c in chapters
        if any(f != "DUPES-OK" for f in c["flags"])
    )
    return {
        "iso": iso,
        "chapters": len(chapters),
        "has_issues": has_issues,
        "total_dupes": sum(c["dupes"] for c in chapters if "DUPES" in c["flags"]),
        "total_dupes_ok": sum(c["dupes"] for c in chapters if "DUPES-OK" in c["flags"]),
        "total_backwards": sum(c["backwards"] for c in chapters),
        "total_nulls": sum(c["nulls"] for c in chapters),
        "total_low_q": sum(c["low_q"] for c in chapters),
        "orig_better_count": sum(1 for c in chapters if c["orig_better"]),
        "chapter_details": chapters,
    }


def print_summary(results):
    """Print multi-language summary table."""
    print("\nTiming Quality Check\n")

    fmt = "  {:<5} {:>8}  {:>6}  {:>5}  {:>4}  {:>5}  {:>5}  {:>11}"
    print(fmt.format("ISO", "Chapters", "Issues", "Dupes", "Back", "Nulls", "LowQ", "Orig-Better"))
    print("  " + "-" * 62)

    t = {k: 0 for k in ["ch", "issues", "dupes", "back", "nulls", "lowq", "orig"]}

    for r in results:
        print(fmt.format(
            r["iso"], r["chapters"], r["has_issues"],
            r["total_dupes"] or "", r["total_backwards"] or "",
            r["total_nulls"] or "", r["total_low_q"] or "",
            r["orig_better_count"] or "",
        ))
        t["ch"] += r["chapters"]
        t["issues"] += r["has_issues"]
        t["dupes"] += r["total_dupes"]
        t["back"] += r["total_backwards"]
        t["nulls"] += r["total_nulls"]
        t["lowq"] += r["total_low_q"]
        t["orig"] += r["orig_better_count"]

    print("  " + "-" * 62)
    print(fmt.format("ALL", t["ch"], t["issues"],
                      t["dupes"] or "", t["back"] or "",
                      t["nulls"] or "", t["lowq"] or "",
                      t["orig"] or ""))
    print()


def print_detail(result):
    """Print per-chapter detail for a single language."""
    details = result["chapter_details"]
    if not details:
        return

    # Show all chapters or only flagged ones. Listing includes DUPES-OK
    # chapters too (still informative to see), but the "with issues" count
    # matches has_issues (DUPES-OK isn't a real issue — see check_language()).
    flagged = [d for d in details if d["flags"]]
    clean = len(details) - len(flagged)

    print(f"  Detail: {result['iso']} ({result['chapters']} chapters, "
          f"{result['has_issues']} with issues)\n")

    if not flagged:
        print("    All chapters clean.\n")
        return

    fmt = "    {:<5} {:<10} {:<4} {:>3}  {:>5}  {:>4}  {:>4}  {:>4}  {:>5}  {:>4}  {:>5}  {}"
    print(fmt.format("Canon", "Fileset", "Book", "Ch", "Dupes", "Back", "Tiny",
                      "Gaps", "Nulls", "LowQ", "Score", "Flags"))
    print("    " + "-" * 80)

    for d in flagged:
        score_str = f"{d['avg_score']:.2f}" if d["avg_score"] is not None else ""
        print(fmt.format(
            d["canon"].upper(), d["distinct_id"], d["book"], d["chapter"],
            d["dupes"] or "", d["backwards"] or "", d["tiny"] or "",
            d["gaps"] or "", d["nulls"] or "", d["low_q"] or "",
            score_str, ",".join(d["flags"]),
        ))

    if clean > 0:
        print(f"\n    ({clean} clean chapters not shown)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Check timing quality across pipeline-generated data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--iso", type=str, help="Single language ISO 639-3 code")
    parser.add_argument("--iso-list", type=str, help="Comma-separated ISO codes")
    parser.add_argument("--testament", type=str, choices=["nt", "ot", "both"],
                        default=None, help="Filter to NT or OT")

    args = parser.parse_args()

    show_detail = False
    if args.iso:
        iso_codes = [args.iso.lower()]
        show_detail = True
    elif args.iso_list:
        iso_codes = [c.strip().lower() for c in args.iso_list.split(",")]
        show_detail = True
    else:
        iso_codes = discover_pipeline_languages(args.testament)
        if not iso_codes:
            print("No pipeline timing data found.")
            sys.exit(0)

    results = []
    for iso in iso_codes:
        result = check_language(iso, args.testament)
        if result:
            results.append(result)
        else:
            print(f"  {iso}: no pipeline timing data (skipped)")

    if not results:
        print("No data to check.")
        sys.exit(0)

    print_summary(results)

    if show_detail:
        for result in results:
            print_detail(result)


if __name__ == "__main__":
    main()
