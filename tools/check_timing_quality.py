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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from checks import check_language  # noqa: E402
from timing_files import TIMING_DIR, _get_canons  # noqa: E402


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
