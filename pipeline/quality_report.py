#!/usr/bin/env python3
"""
Quality report for alignment pipeline output.

Reads *_words_quality.json files and displays per-chapter and per-verse
quality summaries. Automatically compares pipeline timing against downloaded
timecode when available.

Usage:
    # Overview for a language (sorted worst-first)
    python quality_report.py --iso heb

    # Only NT or OT
    python quality_report.py --iso heb --testament ot

    # Drill down into a specific chapter
    python quality_report.py --iso heb --chapter GEN_017

    # Show all verses (not just low-quality ones) when drilling down
    python quality_report.py --iso heb --chapter GEN_017 --all-verses

    # Filter to low-quality chapters only
    python quality_report.py --iso heb --min-low-q 1
"""

import argparse
import json
import sys
from pathlib import Path

TIMING_DIR = Path("export/timing-data")
DOWNLOADS_DIR = Path("downloads/BB")
TIMECODE_CATEGORIES = ["with-timecode", "audio-with-timecode"]


def find_quality_files(iso: str, testament: str = None) -> list:
    """Find all *_words_quality.json files for a language."""
    files = []
    canons = _get_canons(testament)

    for canon in canons:
        canon_dir = TIMING_DIR / canon / iso
        if not canon_dir.exists():
            continue
        for qf in sorted(canon_dir.rglob("*_words_quality.json")):
            files.append((canon, qf))
    return files


def find_pipeline_timing_files(iso: str, testament: str = None) -> list:
    """Find all *_timing.json files in pipeline output for a language."""
    files = []
    canons = _get_canons(testament)

    for canon in canons:
        canon_dir = TIMING_DIR / canon / iso
        if not canon_dir.exists():
            continue
        for tf in sorted(canon_dir.rglob("*_timing.json")):
            files.append((canon, tf))
    return files


def find_downloaded_timecode(canon: str, iso: str, distinct_id: str, book: str, chapter_str: str) -> Path | None:
    """Find downloaded timecode file for a specific chapter."""
    for cat in TIMECODE_CATEGORIES:
        book_dir = DOWNLOADS_DIR / canon / cat / iso / distinct_id / book
        if not book_dir.exists():
            continue
        matches = list(book_dir.glob(f"{book}_{chapter_str}_*_timing.json"))
        if matches:
            return matches[0]
    return None


def find_all_downloaded_timecode(iso: str, testament: str = None) -> dict:
    """Find all downloaded timecode files for a language.

    Returns: {(canon, distinct_id, book, chapter_str): Path}
    """
    result = {}
    canons = _get_canons(testament)

    for canon in canons:
        # Search in category subdirs (with-timecode, audio-with-timecode)
        search_dirs = []
        for cat in TIMECODE_CATEGORIES:
            cat_dir = DOWNLOADS_DIR / canon / cat / iso
            if cat_dir.exists():
                search_dirs.append(cat_dir)
        # Also search direct language dir (downloads/BB/{canon}/{iso}/)
        direct_dir = DOWNLOADS_DIR / canon / iso
        if direct_dir.exists():
            search_dirs.append(direct_dir)

        for search_dir in search_dirs:
            for tf in search_dir.rglob("*_timing.json"):
                distinct_id = tf.parent.parent.name
                book = tf.parent.name
                # Parse chapter from filename: BOOK_CCC_FILESET_timing.json
                parts = tf.stem.replace("_timing", "").split("_", 2)
                if len(parts) >= 2:
                    chapter_str = parts[1]
                    key = (canon, distinct_id, book, chapter_str)
                    result[key] = tf
    return result


def load_timing_verses(path: Path) -> dict:
    """Load a timing.json file and return {verse_start: timestamp}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(entry["verse_start"]): entry["timestamp"] for entry in data}


def compare_verse_timings(downloaded: dict, pipeline: dict) -> dict:
    """Compare two verse timing dicts. Returns comparison stats."""
    # Only compare non-zero verses that exist in both
    common_verses = []
    for v in sorted(downloaded.keys(), key=lambda x: int(x)):
        if v == "0":
            continue
        if v in pipeline:
            common_verses.append(v)

    if not common_verses:
        return None

    deltas = []
    verse_deltas = {}
    for v in common_verses:
        delta = abs(pipeline[v] - downloaded[v])
        deltas.append(delta)
        verse_deltas[v] = pipeline[v] - downloaded[v]

    mean_delta = sum(deltas) / len(deltas)
    max_delta = max(deltas)
    max_verse = common_verses[deltas.index(max_delta)]

    if mean_delta < 0.5:
        status = "GOOD"
    elif mean_delta < 2.0:
        status = "DRIFT"
    else:
        status = "BAD"

    return {
        "common_verses": len(common_verses),
        "mean_delta": mean_delta,
        "max_delta": max_delta,
        "max_verse": max_verse,
        "status": status,
        "verse_deltas": verse_deltas,
        "downloaded": downloaded,
        "pipeline": pipeline,
    }


def load_quality(path: Path) -> dict:
    """Load a quality JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_canons(testament: str = None) -> list:
    canons = []
    if testament in (None, "nt", "both"):
        canons.append("nt")
    if testament in (None, "ot", "both"):
        canons.append("ot")
    return canons


def _parse_timing_path(timing_path: Path) -> tuple:
    """Extract (distinct_id, book, chapter_str) from a pipeline timing path."""
    distinct_id = timing_path.parent.parent.name
    book = timing_path.parent.name
    parts = timing_path.stem.replace("_timing", "").split("_", 2)
    chapter_str = parts[1] if len(parts) >= 2 else None
    return distinct_id, book, chapter_str


def print_overview(iso: str, testament: str, min_low_q: int, sort_by: str, fileset: str = None):
    """Print per-chapter quality overview."""
    files = find_quality_files(iso, testament)
    downloaded_tc = find_all_downloaded_timecode(iso, testament)
    pipeline_timing_files = find_pipeline_timing_files(iso, testament)

    # Apply fileset filter
    if fileset:
        fileset_upper = fileset.upper()
        files = [(c, qf) for c, qf in files if qf.parent.parent.name.upper() == fileset_upper]
        downloaded_tc = {k: v for k, v in downloaded_tc.items() if k[1].upper() == fileset_upper}
        pipeline_timing_files = [(c, tf) for c, tf in pipeline_timing_files
                                  if _parse_timing_path(tf)[0].upper() == fileset_upper]

    # Build set of pipeline chapters: (canon, distinct_id, book, chapter_str)
    pipeline_chapters = set()
    pipeline_timing_by_key = {}
    for canon, tf in pipeline_timing_files:
        distinct_id, book, chapter_str = _parse_timing_path(tf)
        if chapter_str:
            key = (canon, distinct_id, book, chapter_str)
            pipeline_chapters.add(key)
            pipeline_timing_by_key[key] = tf

    # Build set of quality chapters
    quality_chapters = set()
    for canon, qf in files:
        data = load_quality(qf)
        distinct_id = qf.parent.parent.name
        book = data["book"]
        # Extract chapter from filename
        parts = qf.stem.replace("_words_quality", "").split("_", 2)
        chapter_str = parts[1] if len(parts) >= 2 else None
        if chapter_str:
            quality_chapters.add((canon, distinct_id, book, chapter_str))

    if not files and not downloaded_tc:
        print(f"No quality files or downloaded timecode found for {iso}")
        print(f"  Searched: {TIMING_DIR}/*//{iso}/")
        print(f"  Run fusion first: python3 align_words.py --iso {iso} --force")
        sys.exit(1)

    # --- Main quality table (existing behavior + timecode comparison) ---
    if files:
        rows = []
        total_words = 0
        total_low_q = 0
        total_null = 0
        all_scores = []
        has_any_tc = False

        for canon, qf in files:
            data = load_quality(qf)
            s = data["summary"]
            book = data["book"]
            chapter = data["chapter"]
            distinct_id = qf.parent.parent.name

            if min_low_q > 0 and s["low_quality_count"] < min_low_q:
                continue

            whisper_pct = round(100 * s["from_whisper"] / s["total_words"]) if s["total_words"] > 0 else 0
            low_q_verses = ",".join(s["low_quality_verses"][:8])
            if len(s["low_quality_verses"]) > 8:
                low_q_verses += "..."

            # Timecode comparison
            parts = qf.stem.replace("_words_quality", "").split("_", 2)
            chapter_file_str = parts[1] if len(parts) >= 2 else None
            tc_comparison = None
            if chapter_file_str:
                tc_key = (canon, distinct_id, book, chapter_file_str)
                if tc_key in downloaded_tc and tc_key in pipeline_timing_by_key:
                    dl_verses = load_timing_verses(downloaded_tc[tc_key])
                    pl_verses = load_timing_verses(pipeline_timing_by_key[tc_key])
                    tc_comparison = compare_verse_timings(dl_verses, pl_verses)
                    if tc_comparison:
                        has_any_tc = True

            rows.append({
                "canon": canon.upper(),
                "book": book,
                "chapter": chapter,
                "distinct_id": distinct_id,
                "words": s["total_words"],
                "avg_score": s["avg_score"],
                "low_q": s["low_quality_count"],
                "null": s["null_count"],
                "whisper_pct": whisper_pct,
                "low_q_verses": low_q_verses,
                "tc": tc_comparison,
            })

            total_words += s["total_words"]
            total_low_q += s["low_quality_count"]
            total_null += s["null_count"]
            all_scores.extend([s["avg_score"]] * s["total_words"])

        if rows:
            # Sort
            if sort_by == "score":
                rows.sort(key=lambda r: r["avg_score"])
            elif sort_by == "low-q":
                rows.sort(key=lambda r: -r["low_q"])
            else:
                rows.sort(key=lambda r: (r["canon"], r["book"], r["chapter"]))

            overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
            print(f"\nQuality Report: {iso} ({len(rows)} chapters, {total_words} words)")
            print(f"Overall avg score: {overall_avg:.3f}, low-quality words: {total_low_q}, null timestamps: {total_null}")
            print()

            if has_any_tc:
                fmt = "  {:<6} {:<4} {:>3}  {:>5}  {:>5}  {:>5}  {:>5}  {:>8}  {:>6}  {:>6}  {:>5}  {}"
                print(fmt.format("Canon", "Book", "Ch", "Words", "Score", "Low-Q", "Null", "Whisper%", "dMean", "dMax", "TCmp", "Low-Q Verses"))
                print("  " + "-" * 100)
            else:
                fmt = "  {:<6} {:<4} {:>3}  {:>5}  {:>5}  {:>5}  {:>5}  {:>8}  {}"
                print(fmt.format("Canon", "Book", "Ch", "Words", "Score", "Low-Q", "Null", "Whisper%", "Low-Q Verses"))
                print("  " + "-" * 80)

            for r in rows:
                score_str = f"{r['avg_score']:.3f}"
                low_q_str = str(r["low_q"]) if r["low_q"] > 0 else ""
                null_str = str(r["null"]) if r["null"] > 0 else ""
                if has_any_tc:
                    if r["tc"]:
                        d_mean = f"{r['tc']['mean_delta']:.1f}s"
                        d_max = f"{r['tc']['max_delta']:.1f}s"
                        tc_status = r["tc"]["status"]
                    else:
                        d_mean = ""
                        d_max = ""
                        tc_status = ""
                    print(fmt.format(
                        r["canon"], r["book"], r["chapter"],
                        r["words"], score_str, low_q_str, null_str,
                        f"{r['whisper_pct']}%", d_mean, d_max, tc_status,
                        r["low_q_verses"],
                    ))
                else:
                    print(fmt.format(
                        r["canon"], r["book"], r["chapter"],
                        r["words"], score_str, low_q_str, null_str,
                        f"{r['whisper_pct']}%", r["low_q_verses"],
                    ))

            if has_any_tc:
                print("  " + "-" * 100)
                print(fmt.format("", "", "", total_words, f"{overall_avg:.3f}", total_low_q, total_null, "", "", "", "", ""))
            else:
                print("  " + "-" * 80)
                print(fmt.format("", "", "", total_words, f"{overall_avg:.3f}", total_low_q, total_null, "", ""))
            print()
        elif min_low_q > 0:
            print(f"No chapters match the filter (min_low_q={min_low_q})")

    # --- Coverage gaps: downloaded timecode exists but no pipeline quality data ---
    gaps = []
    for tc_key, tc_path in sorted(downloaded_tc.items()):
        canon, distinct_id, book, chapter_str = tc_key
        if tc_key not in quality_chapters:
            # Check if pipeline timing exists (partial pipeline run)
            has_pipeline = tc_key in pipeline_chapters
            gaps.append({
                "canon": canon.upper(),
                "distinct_id": distinct_id,
                "book": book,
                "chapter": chapter_str,
                "has_pipeline": has_pipeline,
            })

    if gaps:
        # Group by fileset for compact display
        by_fileset = {}
        for g in gaps:
            fkey = (g["canon"], g["distinct_id"])
            if fkey not in by_fileset:
                by_fileset[fkey] = {"books": {}, "has_pipeline_any": False}
            book = g["book"]
            if book not in by_fileset[fkey]["books"]:
                by_fileset[fkey]["books"][book] = {"chapters": 0, "has_pipeline": 0}
            by_fileset[fkey]["books"][book]["chapters"] += 1
            if g["has_pipeline"]:
                by_fileset[fkey]["books"][book]["has_pipeline"] += 1
                by_fileset[fkey]["has_pipeline_any"] = True

        print(f"\n=== Coverage Gaps: downloaded timecode without pipeline quality data ===")
        print(f"  (These have original timecode but alignment pipeline was not run or incomplete)\n")

        for (canon, distinct_id), info in sorted(by_fileset.items()):
            total_ch = sum(b["chapters"] for b in info["books"].values())
            total_pipeline = sum(b["has_pipeline"] for b in info["books"].values())
            books_list = ", ".join(
                f"{b}({info['books'][b]['chapters']}ch)" for b in sorted(info["books"].keys())
            )
            pipeline_note = ""
            if total_pipeline > 0:
                pipeline_note = f" — {total_pipeline}/{total_ch} have pipeline timing (no quality)"
            print(f"  {canon} {distinct_id}: {total_ch} chapters — {books_list}{pipeline_note}")

        print()

    # --- Pipeline-only: pipeline timing but no downloaded timecode ---
    pipeline_only = []
    for p_key in sorted(pipeline_chapters):
        canon, distinct_id, book, chapter_str = p_key
        if p_key not in downloaded_tc:
            pipeline_only.append({
                "canon": canon.upper(),
                "distinct_id": distinct_id,
                "book": book,
                "chapter": chapter_str,
            })

    if pipeline_only:
        by_fileset = {}
        for p in pipeline_only:
            fkey = (p["canon"], p["distinct_id"])
            if fkey not in by_fileset:
                by_fileset[fkey] = {}
            book = p["book"]
            if book not in by_fileset[fkey]:
                by_fileset[fkey][book] = 0
            by_fileset[fkey][book] += 1

        print(f"=== Pipeline-only filesets (no downloaded timecode to compare) ===\n")
        for (canon, distinct_id), books in sorted(by_fileset.items()):
            total_ch = sum(books.values())
            books_list = ", ".join(f"{b}({c}ch)" for b, c in sorted(books.items()))
            print(f"  {canon} {distinct_id}: {total_ch} chapters — {books_list}")
        print()


def print_chapter_detail(iso: str, chapter_spec: str, show_all: bool, testament: str, fileset: str = None):
    """Print per-verse detail for a specific chapter."""
    # Parse chapter_spec: "GEN_017" or "GEN 17"
    parts = chapter_spec.replace(" ", "_").split("_", 1)
    if len(parts) != 2:
        print(f"Invalid chapter spec: {chapter_spec}")
        print("Expected format: GEN_017 or GEN_17 or 'GEN 17'")
        sys.exit(1)
    book = parts[0].upper()
    chapter_str = parts[1].zfill(3)

    # Find all matching quality files (could be multiple filesets)
    files = find_quality_files(iso, testament)
    matches = []
    for canon, qf in files:
        data = load_quality(qf)
        distinct_id = qf.parent.parent.name
        if fileset and distinct_id.upper() != fileset.upper():
            continue
        if data["book"] == book and (data["chapter"] == chapter_str or
                data["chapter"].lstrip("0") == chapter_str.lstrip("0")):
            matches.append((canon, qf, data))

    # Also find filesets with downloaded timecode but no quality data
    tc_only_filesets = []
    canons = _get_canons(testament)
    for canon in canons:
        for cat in TIMECODE_CATEGORIES:
            cat_dir = DOWNLOADS_DIR / canon / cat / iso
            if not cat_dir.exists():
                continue
            for fileset_dir in sorted(cat_dir.iterdir()):
                if not fileset_dir.is_dir():
                    continue
                distinct_id = fileset_dir.name
                if fileset and distinct_id.upper() != fileset.upper():
                    continue
                # Skip if we already have quality data for this fileset
                if any(qf.parent.parent.name == distinct_id for _, qf, _ in matches):
                    continue
                book_dir = fileset_dir / book
                if not book_dir.exists():
                    continue
                tc_matches = list(book_dir.glob(f"{book}_{chapter_str}_*_timing.json"))
                if tc_matches:
                    tc_only_filesets.append((canon, distinct_id, tc_matches[0]))

    if not matches and not tc_only_filesets:
        print(f"No data found for {book} {chapter_str} ({iso})")
        if fileset:
            print(f"  (filtered to fileset: {fileset})")
        sys.exit(1)

    # Print quality detail for each matching fileset
    for canon, qf, data in matches:
        s = data["summary"]
        distinct_id = qf.parent.parent.name
        whisper_pct = round(100 * s["from_whisper"] / s["total_words"]) if s["total_words"] > 0 else 0

        print(f"\nQuality Detail: {iso} — {book} {data['chapter']} — {distinct_id} ({canon.upper()})")
        print(f"  Words: {s['total_words']}, Avg score: {s['avg_score']}, "
              f"Low-quality: {s['low_quality_count']}, Null: {s['null_count']}, "
              f"Whisper: {whisper_pct}%")
        print()

        # --- Timecode comparison ---
        qf_parts = qf.stem.replace("_words_quality", "").split("_", 2)
        chapter_file_str = qf_parts[1] if len(qf_parts) >= 2 else None
        if chapter_file_str:
            _print_timecode_comparison(canon, iso, distinct_id, book, chapter_file_str)

        # Load reference text for word display
        ref_text = _find_ref_text(canon, iso, qf)

        low_q_threshold = 0.3
        verses = data["verses"]
        for vnum in sorted(verses.keys(), key=lambda v: int(v)):
            words = verses[vnum]
            if not words:
                continue

            has_low_q = any(w["score"] < low_q_threshold for w in words)
            if not show_all and not has_low_q:
                continue

            # Get reference words for this verse
            ref_words = _get_verse_words(ref_text, int(vnum)) if ref_text else None

            avg = sum(w["score"] for w in words) / len(words)
            from_w = sum(1 for w in words if w.get("source") == "whisper")
            marker = " *** LOW" if has_low_q else ""
            print(f"  Verse {vnum} ({len(words)} words, avg={avg:.3f}, "
                  f"{from_w}/{len(words)} whisper){marker}")

            # Show individual words
            for i, w in enumerate(words):
                if not show_all and w["score"] >= low_q_threshold:
                    continue
                ref_word = ref_words[i] if ref_words and i < len(ref_words) else "?"
                flag = " <--" if w["score"] < low_q_threshold else ""
                source = w.get("source", "mms")
                print(f"    [{i:3d}] {w['score']:.3f} {source:>7}  {ref_word}{flag}")

            print()

        if not show_all:
            total_verses = sum(1 for v in verses.values() if v)
            shown = sum(1 for v in verses.values() if v and any(w["score"] < low_q_threshold for w in v))
            if shown < total_verses:
                print(f"  ({shown}/{total_verses} verses shown — use --all-verses to see all)")
                print()

    # Print timecode-only detail for filesets without quality data
    for canon, distinct_id, tc_path in tc_only_filesets:
        print(f"\nTimecode Only: {iso} — {book} {chapter_str} — {distinct_id} ({canon.upper()})")
        print(f"  (No pipeline quality data — showing timecode comparison only)")
        print()

        # Find pipeline timing for this fileset
        pl_dir = TIMING_DIR / canon / iso / distinct_id / book
        pl_matches = list(pl_dir.glob(f"{book}_{chapter_str}_*_timing.json")) if pl_dir.exists() else []

        if pl_matches:
            dl_verses = load_timing_verses(tc_path)
            pl_verses = load_timing_verses(pl_matches[0])
            cmp = compare_verse_timings(dl_verses, pl_verses)
            if cmp:
                if cmp["status"] == "GOOD":
                    status_label = "GOOD AGREEMENT"
                elif cmp["status"] == "DRIFT":
                    status_label = "MINOR DRIFT"
                else:
                    status_label = "UNRELIABLE — prefer downloaded timecode"

                print(f"  === Verse Timing: pipeline vs downloaded timecode ===")
                print(f"  Status: {status_label} (mean delta={cmp['mean_delta']:.2f}s, max={cmp['max_delta']:.2f}s at v{cmp['max_verse']})")
                print()

                fmt = "    {:>5}  {:>10}  {:>10}  {:>10}"
                print(fmt.format("Verse", "Downloaded", "Pipeline", "Delta"))
                print("    " + "-" * 45)

                for v in sorted(cmp["verse_deltas"].keys(), key=lambda x: int(x)):
                    dl_ts = cmp["downloaded"].get(v, 0)
                    pl_ts = cmp["pipeline"].get(v, 0)
                    delta = cmp["verse_deltas"][v]
                    flag = "  ***" if abs(delta) > 2.0 else ""
                    print(fmt.format(v, f"{dl_ts:.2f}s", f"{pl_ts:.2f}s", f"{delta:+.2f}s{flag}"))
                print()
            else:
                print(f"  No common verses to compare")
                print()
        else:
            print(f"  Downloaded timecode available but no pipeline timing to compare")
            print()


def _print_timecode_comparison(canon: str, iso: str, distinct_id: str, book: str, chapter_str: str):
    """Print verse timing comparison between downloaded and pipeline."""
    dl_path = find_downloaded_timecode(canon, iso, distinct_id, book, chapter_str)
    if not dl_path:
        return

    pl_path = TIMING_DIR / canon / iso / distinct_id / book
    pl_matches = list(pl_path.glob(f"{book}_{chapter_str}_*_timing.json"))
    if not pl_matches:
        return

    dl_verses = load_timing_verses(dl_path)
    pl_verses = load_timing_verses(pl_matches[0])
    cmp = compare_verse_timings(dl_verses, pl_verses)
    if not cmp:
        return

    if cmp["status"] == "GOOD":
        status_label = "GOOD AGREEMENT"
    elif cmp["status"] == "DRIFT":
        status_label = "MINOR DRIFT"
    else:
        status_label = "UNRELIABLE — prefer downloaded timecode"

    print(f"  === Verse Timing: pipeline vs downloaded timecode ===")
    print(f"  Status: {status_label} (mean delta={cmp['mean_delta']:.2f}s, max={cmp['max_delta']:.2f}s at v{cmp['max_verse']})")
    print()

    fmt = "    {:>5}  {:>10}  {:>10}  {:>10}"
    print(fmt.format("Verse", "Downloaded", "Pipeline", "Delta"))
    print("    " + "-" * 45)

    for v in sorted(cmp["verse_deltas"].keys(), key=lambda x: int(x)):
        dl_ts = cmp["downloaded"].get(v, 0)
        pl_ts = cmp["pipeline"].get(v, 0)
        delta = cmp["verse_deltas"][v]
        flag = "  ***" if abs(delta) > 2.0 else ""
        print(fmt.format(v, f"{dl_ts:.2f}s", f"{pl_ts:.2f}s", f"{delta:+.2f}s{flag}"))

    print()



def _find_ref_text(canon: str, iso: str, quality_path: Path):
    """Try to find the reference text file for a chapter."""
    distinct_id = quality_path.parent.parent.name
    book = quality_path.parent.name

    # Extract chapter from filename: BOOK_CCC_FILESET_words_quality.json
    stem = quality_path.stem.replace("_words_quality", "")
    parts = stem.split("_", 2)
    if len(parts) < 2:
        return None
    chapter_str = parts[1]

    categories = ["syncable", "with-timecode", "audio-with-timecode"]
    for cat in categories:
        book_dir = DOWNLOADS_DIR / canon / cat / iso / distinct_id / book
        if not book_dir.exists():
            continue
        txts = list(book_dir.glob(f"{parts[0]}_{chapter_str}_*.txt"))
        if txts:
            with open(txts[0], "r", encoding="utf-8") as f:
                return f.readlines()
    return None


def _get_verse_words(ref_lines: list, verse_num: int) -> list:
    """Get words for a verse from reference text lines."""
    if verse_num < 1 or verse_num > len(ref_lines):
        return None
    line = ref_lines[verse_num - 1].strip()
    if not line:
        return None
    return line.split()


def main():
    parser = argparse.ArgumentParser(
        description="Quality report for alignment pipeline output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--iso", type=str, required=True, help="ISO 639-3 language code")
    parser.add_argument("--testament", type=str, choices=["nt", "ot", "both"], default=None,
                        help="Filter to NT or OT")
    parser.add_argument("--chapter", type=str, default=None,
                        help="Drill down to a chapter (e.g. GEN_017, JHN_7)")
    parser.add_argument("--all-verses", action="store_true",
                        help="Show all verses when drilling down (not just low-quality)")
    parser.add_argument("--min-low-q", type=int, default=0,
                        help="Only show chapters with at least N low-quality words")
    parser.add_argument("--fileset", type=str, default=None,
                        help="Filter to a specific fileset (e.g. SWESFV, HEBM95)")
    parser.add_argument("--sort", type=str, choices=["score", "low-q", "name"], default="score",
                        help="Sort order for overview (default: score, worst first)")

    args = parser.parse_args()

    if args.chapter:
        print_chapter_detail(args.iso, args.chapter, args.all_verses, args.testament, args.fileset)
    else:
        print_overview(args.iso, args.testament, args.min_low_q, args.sort, args.fileset)


if __name__ == "__main__":
    main()
