#!/usr/bin/env python3
"""
Prepare cross-source alignment by downloading helloAO text for DBT audio-only filesets.

Detects languages that have:
  - audio-only in downloads/BB/ (DBT audio, no text)
  - a matching helloAO translation

Downloads the helloAO text via API and converts it to ET-format .txt files
placed alongside the DBT audio in downloads/BB/, enabling the alignment pipeline
to process them as normal.

This is the proactive/batch counterpart to the reactive per-chapter fallback
already in download_language_content.py (_find_helloao_id() / _fetch_helloao_chapter()
/ _get_external_text_source(), used by download_chapter() when a single chapter
turns out to have no DBT text). Reuses those functions directly rather than
reimplementing the matching a second way.

Matching (2026-07-29) comes from the `bibles` repo's catalog-overlap.json on
CDN — a real text comparison between DBT and helloAO translations, not the
old {iso}_{version} naming-convention guess. No local cache file or MONO
tooling is required; catalog-overlap.json is fetched (and cached locally
under api-cache/dbt-catalog/) the same way as the other DBT catalogs. See
_find_helloao_id()'s docstring in download_language_content.py.

Usage:
    python prepare_cross_source.py --dry-run        # Preview what would be downloaded
    python prepare_cross_source.py --iso gaz         # Prepare one language
    python prepare_cross_source.py --all             # Prepare all detected pairs
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from download_language_content import (
    HELLOAO_API,
    _fetch_helloao_chapter,
    _load_dbt_catalog,
)

DOWNLOADS_BB = Path("downloads/BB")


def _overlap_helloao_match(overlap_entries, iso, canon, dbt_id):
    """Find dbt_id's verified helloAO match in catalog-overlap.json's entries.

    Tries the full canon key (nt/ot) first, then the portion variant
    (ntp/otp) — same fallback as download_language_content.py's
    _catalog_entries().
    """
    for key in (f"{iso}:{canon}", f"{iso}:{canon}p"):
        for group in overlap_entries.get(key, []):
            ids = group.get("ids", [])
            if f"d:{dbt_id}" not in ids:
                continue
            for i in ids:
                if i.startswith("h:"):
                    return i[2:]
    return None


def find_cross_source_pairs(iso_filter=None):
    """
    Find DBT audio-only filesets that have a matching helloAO translation.

    Returns list of dicts:
      {iso, canon, dbt_id, helloao_id, dbt_dir, books}
    """
    overlap_entries = _load_dbt_catalog("catalog-overlap").get("entries", {})
    pairs = []

    for canon in ("nt", "ot"):
        canon_dir = DOWNLOADS_BB / canon
        if not canon_dir.is_dir():
            continue

        for iso_dir in sorted(canon_dir.iterdir()):
            if not iso_dir.is_dir():
                continue
            iso = iso_dir.name
            if iso_filter and iso != iso_filter:
                continue

            for fileset_dir in sorted(iso_dir.iterdir()):
                if not fileset_dir.is_dir():
                    continue
                dbt_id = fileset_dir.name

                # Skip if it already has text files
                has_text = any(fileset_dir.rglob("*.txt"))
                if has_text:
                    continue

                # Skip if not uppercase (not a DBT ID)
                if not dbt_id.isupper() or "_" in dbt_id:
                    continue

                # Look up a *verified* (text-compared) helloAO match —
                # catalog-overlap.json, not a naming-convention guess.
                helloao_id = _overlap_helloao_match(overlap_entries, iso, canon, dbt_id)
                if not helloao_id:
                    continue

                # List books that have audio
                books = sorted(
                    d.name for d in fileset_dir.iterdir()
                    if d.is_dir() and any(d.glob("*.mp3"))
                )
                if not books:
                    continue

                pairs.append({
                    "iso": iso,
                    "canon": canon,
                    "dbt_id": dbt_id,
                    "helloao_id": helloao_id,
                    "dbt_dir": fileset_dir,
                    "books": books,
                })

    return pairs


def prepare_pair(pair, dry_run=False):
    """
    Download helloAO text and create ET-format .txt files alongside DBT audio.
    """
    iso = pair["iso"]
    canon = pair["canon"]
    dbt_id = pair["dbt_id"]
    helloao_id = pair["helloao_id"]
    dbt_dir = pair["dbt_dir"]
    books = pair["books"]

    print(f"\n  {canon}/{iso}/{dbt_id} ← text from helloAO:{helloao_id}")

    # First, get the book list from helloAO to know chapter counts. This is
    # the one piece genuinely specific to batch prep — the reactive fallback
    # doesn't need it since it's only ever asked for one known chapter.
    books_url = f"{HELLOAO_API}/{helloao_id}/books.json"
    try:
        with urllib.request.urlopen(books_url, timeout=30) as resp:
            books_data = json.loads(resp.read())
    except Exception as e:
        print(f"    Failed to fetch book list: {e}")
        return 0

    helloao_books = {}
    for b in books_data.get("books", []):
        helloao_books[b["id"]] = b["numberOfChapters"]

    total_written = 0
    for book in books:
        if book not in helloao_books:
            print(f"    {book}: not available in helloAO")
            continue

        num_chapters = helloao_books[book]
        book_dir = dbt_dir / book

        # Get list of chapters that have audio
        audio_chapters = set()
        for mp3 in book_dir.glob("*.mp3"):
            parts = mp3.stem.split("_", 2)
            if len(parts) >= 2:
                try:
                    audio_chapters.add(int(parts[1]))
                except ValueError:
                    pass

        for ch in sorted(audio_chapters):
            if ch > num_chapters:
                continue

            chapter_str = f"{ch:03d}"
            # Same filename convention as download_chapter()'s external-text
            # fallback: helloAO ID (without underscores, uppercase) as the
            # text fileset marker.
            txt_filename = f"{book}_{chapter_str}_{helloao_id.replace('_', '').upper()}_ET.txt"
            txt_path = book_dir / txt_filename

            if dry_run:
                if not txt_path.exists():
                    print(f"    Would fetch: {book} {ch}")
                    total_written += 1
                continue

            if _fetch_helloao_chapter(helloao_id, book, ch, txt_path):
                total_written += 1

        if not dry_run:
            written_for_book = len(list(book_dir.glob(f"{book}_*_ET.txt")))
            print(f"    {book}: {written_for_book} chapters with text ({len(audio_chapters)} audio chapters)")

    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="Prepare cross-source alignment (helloAO text + DBT audio)",
    )
    parser.add_argument("--iso", type=str, help="Process one language")
    parser.add_argument("--all", action="store_true", help="Process all detected pairs")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    if not args.iso and not args.all:
        parser.error("Specify --iso <ISO> or --all")

    pairs = find_cross_source_pairs(iso_filter=args.iso)

    if not pairs:
        print("No cross-source pairs found.")
        sys.exit(0)

    print(f"Found {len(pairs)} cross-source pairs:")
    for p in pairs:
        print(f"  {p['canon']}/{p['iso']}/{p['dbt_id']} ← helloAO:{p['helloao_id']} ({len(p['books'])} books)")

    if args.dry_run:
        print("\n[DRY RUN]")

    total = 0
    for pair in pairs:
        total += prepare_pair(pair, dry_run=args.dry_run)

    action = "Would prepare" if args.dry_run else "Prepared"
    print(f"\n{action} {total} chapter text files across {len(pairs)} filesets.")


if __name__ == "__main__":
    main()
