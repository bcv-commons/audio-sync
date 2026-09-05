#!/usr/bin/env python3
"""
Fallback-rate diagnostic for verse_only_mode languages.

verse_only_mode (config.verse_only_mode = true, see
pipeline/config/languages/<iso>.toml) skips Whisper entirely and aligns
each verse independently within a pace-anchored window (see
pipeline/align_verse_words.py's module docstring). When a verse's local
MMS confidence is too low to trust, that verse falls back to a pure
pace estimate: every word gets score=0.0, source="fallback", and
start==end (zero duration) — see align_verse_words.py's
verse_anchored_align().

This is a distinct failure mode from a normal low MMS score, and NOT
the same thing tools/check_timing_quality.py's LOW-SCORE flag tells
you: a chapter can have a perfectly reasonable-looking avg_score while
still having whole verses that are pure guesses, because the other
verses' higher scores dilute the average. Nor is it the same as DUPES
(a fallback verse's timestamp can legitimately differ from its
neighbors' even though it was never really aligned).

This reads each chapter's _words_quality.json (align_verse_words.py
writes the same on-disk shape align_words.py's fusion path does, via
write_quality_json — see that module's docstring) and reports the
FALLBACK RATE: the fraction of a chapter's non-empty verses where every
word was a pace-estimate rather than a real local alignment.

Usage:
    python tools/check_verse_only_fallback.py                    # every verse_only_mode language
    python tools/check_verse_only_fallback.py --iso hin
    python tools/check_verse_only_fallback.py --iso-list hin,bod
    python tools/check_verse_only_fallback.py --testament ot
    python tools/check_verse_only_fallback.py --threshold 0.3    # flag chapters at/above 30% fallback (default 0.2)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from text_processing import load_language_config  # noqa: E402
from quality_report import find_quality_files, _parse_timing_path  # noqa: E402

CONFIG_DIR = Path(__file__).parent.parent / "pipeline" / "config" / "languages"
DEFAULT_THRESHOLD = 0.2


def discover_verse_only_languages() -> list[str]:
    """ISO codes with verse_only_mode = true in their config file."""
    isos = []
    for f in sorted(CONFIG_DIR.glob("*.toml")):
        if f.stem == "default":
            continue
        try:
            if load_language_config(f.stem).verse_only_mode:
                isos.append(f.stem)
        except Exception:
            continue
    return isos


def check_chapter_fallback(quality_path: Path) -> dict | None:
    """Fallback stats for one chapter's _words_quality.json.

    Returns {verses, fallback_verses, fallback_rate, fallback_verse_nums}
    or None if this isn't verse_only_mode output (no "source" field on
    its words — the fusion-mode pipeline's quality files have per-word
    scores but no per-word "source": "fallback"/"local" tag).
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
        # verse_only_mode's vocabulary is exactly {"local", "fallback"} —
        # fusion-mode quality files use a disjoint vocabulary ({"whisper"}
        # only, or no "source" key at all for the common MMS-wins case),
        # so this positively identifies verse_only_mode output rather
        # than just detecting *some* "source" key (which fusion-mode
        # files can also carry, and did wrongly match here at first).
        if any(w.get("source") in ("local", "fallback") for w in words):
            saw_source_tag = True
        if words and all(w.get("source") == "fallback" for w in words):
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


def print_summary(results: list[dict]):
    print("\nVerse-Only-Mode Fallback Check\n")
    fmt = "  {:<5} {:>8}  {:>8}  {:>10}  {:>9}"
    print(fmt.format("ISO", "Chapters", "Flagged", "Verses", "Fallback%"))
    print("  " + "-" * 48)
    for r in results:
        print(fmt.format(
            r["iso"], r["chapters"], r["flagged_chapters"],
            r["total_verses"], f"{r['overall_fallback_rate']:.1%}",
        ))
    print()


def print_detail(result: dict, threshold: float):
    flagged = [c for c in result["chapter_details"] if c["flagged"]]
    print(f"  Detail: {result['iso']} ({result['chapters']} chapter(s), "
          f"{len(flagged)} at/above {threshold:.0%} fallback)\n")
    if not flagged:
        print("    No chapters at/above threshold.\n")
        return

    fmt = "    {:<5} {:<10} {:<4} {:>3}  {:>7}  {:>9}  {}"
    print(fmt.format("Canon", "Fileset", "Book", "Ch", "Verses", "Fallback%", "Fallback verses"))
    print("    " + "-" * 80)
    for c in sorted(flagged, key=lambda c: -c["fallback_rate"]):
        nums = ",".join(c["fallback_verse_nums"][:8])
        if len(c["fallback_verse_nums"]) > 8:
            nums += f",+{len(c['fallback_verse_nums']) - 8} more"
        print(fmt.format(
            c["canon"].upper(), c["distinct_id"], c["book"], c["chapter"],
            c["verses"], f"{c['fallback_rate']:.0%}", nums,
        ))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    lang_group = parser.add_mutually_exclusive_group()
    lang_group.add_argument("--iso", type=str, help="Single language ISO code")
    lang_group.add_argument("--iso-list", type=str, help="Comma-separated ISO codes")
    parser.add_argument("--testament", type=str, choices=["nt", "ot", "both"], default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help=f"Flag a chapter when its fallback rate is at/above this fraction (default {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    if args.iso:
        isos = [args.iso.lower()]
    elif args.iso_list:
        isos = [c.strip().lower() for c in args.iso_list.split(",") if c.strip()]
    else:
        isos = discover_verse_only_languages()

    results = []
    for iso in isos:
        r = check_language_fallback(iso, args.testament, args.threshold)
        if r:
            results.append(r)

    if not results:
        print("No verse_only_mode-shaped output found.")
        return

    print_summary(results)
    for r in results:
        if r["flagged_chapters"]:
            print_detail(r, args.threshold)


if __name__ == "__main__":
    main()
