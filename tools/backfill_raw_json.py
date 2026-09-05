#!/usr/bin/env python3
"""
Backfill .raw.json sidecars next to already-downloaded .txt chapter files.

Retroactive companion to the forward-write path added in
download_language_content.py (download_text / _fetch_helloao_chapter):
that path only writes .raw.json for chapters downloaded *after* the change.
This walks the existing downloads/BB tree (~2,155 edition/book folders as of
2026-08) and backfills what it safely can, in two independent tiers per
chapter:

  1. Provenance ("source"/"fileset_id") — always determinable, no network.
     Primary signal is the book dir's existing source.json (written by
     _write_source_json at download time); when that's missing or doesn't
     cover this exact fileset_id, falls back to resolve_preferred_text_source()
     (catalog-only, no network) to get the source type, and the fileset_id is
     always read straight from the .txt filename itself either way.
  2. Verse numbering (verse_start/verse_end) — only written when the .txt
     file's line count exactly matches the canonical expected verse count
     for that book/chapter (pipeline/canonical_verse_counts.json, derived
     from lexeme-aligner's config/canonical_index/whole_bible.json). A
     mismatch means either bridged verses or non-standard versification —
     guessing would silently mislabel verses, so those chapters are just
     skipped (no .raw.json written), same as if this pass never ran. They
     self-heal later via lexeme-aligner's own live-fetch fallback + write-back
     (a separate, not-yet-implemented follow-up), not via this script.

Read-only against .txt files — only ever adds a new .raw.json sibling.
Never touches or re-downloads anything. Safe to re-run (skips chapters that
already have a .raw.json).

Usage:
    python tools/backfill_raw_json.py                # full tree, report only summary
    python tools/backfill_raw_json.py --iso fra       # single language
    python tools/backfill_raw_json.py --dry-run       # count what would be written, write nothing
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from download_language_content import OUTPUT_DIR, resolve_preferred_text_source  # noqa: E402

CANONICAL_COUNTS_PATH = Path(__file__).resolve().parent.parent / "pipeline" / "canonical_verse_counts.json"


def load_canonical_counts() -> dict:
    with open(CANONICAL_COUNTS_PATH) as f:
        return json.load(f)


def parse_txt_filename(txt_path: Path) -> tuple[str, int, str] | None:
    """BOOK_chapter_filesetid.txt -> (book, chapter, fileset_id). fileset_id
    itself commonly contains underscores (e.g. "ENGNASN_ET"), hence maxsplit=2."""
    parts = txt_path.stem.split("_", 2)
    if len(parts) != 3:
        return None
    book, chapter_str, fileset_id = parts
    try:
        chapter = int(chapter_str)
    except ValueError:
        return None
    return book, chapter, fileset_id


def determine_source(book_dir: Path, fileset_id: str, iso: str, canon: str, distinct_id: str,
                      _cache: dict = {}) -> str:
    """"dbt" or "helloao" for this fileset_id. source.json is the cheap,
    accurate signal (tracks exactly what download_chapter used); falls back
    to a catalog-only re-resolution when source.json is missing or stale."""
    source_json = book_dir / "source.json"
    if source_json.exists():
        try:
            data = json.loads(source_json.read_text())
            text_src = data.get("text", "")
            src_type, _, src_ref = text_src.partition(":")
            if src_type in ("dbt", "helloao"):
                return src_type
        except (OSError, json.JSONDecodeError):
            pass

    key = (iso, canon, distinct_id)
    if key not in _cache:
        src_type, _ = resolve_preferred_text_source(iso, canon, distinct_id)
        _cache[key] = src_type
    return _cache[key]


def backfill(iso_filter: str | None, dry_run: bool) -> None:
    counts = load_canonical_counts()

    stats = {"written": 0, "skipped_mismatch": 0, "skipped_no_canonical": 0, "already_present": 0}

    for canon_dir in sorted(OUTPUT_DIR.iterdir()):
        if not canon_dir.is_dir():
            continue
        canon = canon_dir.name.upper()

        for iso_dir in sorted(canon_dir.iterdir()):
            if not iso_dir.is_dir():
                continue
            iso = iso_dir.name
            if iso_filter and iso != iso_filter:
                continue

            for edition_dir in sorted(iso_dir.iterdir()):
                if not edition_dir.is_dir():
                    continue
                distinct_id = edition_dir.name

                for book_dir in sorted(edition_dir.iterdir()):
                    if not book_dir.is_dir():
                        continue

                    for txt_path in sorted(book_dir.glob("*.txt")):
                        raw_path = txt_path.with_suffix(".raw.json")
                        if raw_path.exists():
                            stats["already_present"] += 1
                            continue

                        parsed = parse_txt_filename(txt_path)
                        if not parsed:
                            continue
                        book, chapter, fileset_id = parsed

                        expected = counts.get(book, {}).get(str(chapter))
                        if expected is None:
                            stats["skipped_no_canonical"] += 1
                            continue

                        lines = [ln for ln in txt_path.read_text(encoding="utf-8", errors="replace").split("\n") if ln.strip()]
                        if len(lines) != expected:
                            stats["skipped_mismatch"] += 1
                            continue

                        source = determine_source(book_dir, fileset_id, iso, canon, distinct_id)
                        verses = [
                            {"verse_start": i + 1, "verse_end": None, "verse_text": line}
                            for i, line in enumerate(lines)
                        ]

                        if not dry_run:
                            with open(raw_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    {"source": source, "fileset_id": fileset_id, "verses": verses},
                                    f, ensure_ascii=False, indent=0,
                                )
                        stats["written"] += 1

    print("Backfill complete" + (" (dry run)" if dry_run else "") + ":")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", help="Restrict to a single language ISO code")
    parser.add_argument("--dry-run", action="store_true", help="Count what would be written, write nothing")
    args = parser.parse_args()
    backfill(args.iso, args.dry_run)
