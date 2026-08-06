#!/usr/bin/env python3
"""
Import contributed Bible audio+text into the pipeline format.

Reads from contrib/{iso}_{distinct_id}/ and copies files into
downloads/contrib/{canon}/{iso}/{distinct_id}/{BOOK}/ with the
naming convention the alignment pipeline expects.

Two contribution layouts are supported:

LEGACY layout (e.g., contrib/nor_NBS/):
    {BOOK}_{chapter}.mp3
    {BOOK}_{chapter}.txt   (or {DISTINCT_ID}_{BOOK}_{chapter}.txt)
    source.toml

EXTERNAL layout (e.g., contrib/deu/):
    audio.json              -- describes remote audio source; clients fetch directly
    text/{BOOK}_{chapter}.txt
    timecodes/{BOOK}_{chapter}.csv  -- one timestamp per verse, in seconds
    source.toml             -- with `distinct_id = "..."` if dirname has no underscore

For EXTERNAL contributions:
  - Audio files are NOT downloaded locally; clients fetch them directly via the
    URL pattern in audio.json. The audio.json is copied to the version dir so
    the export can ship it to clients.
  - Pre-aligned CSV timecodes are converted into _timing.json (DBT format), so
    the alignment pipeline (whisper/MMS) does not need to run for those chapters.

Usage:
    python import_contrib.py              # Import all contributions
    python import_contrib.py --iso deu    # Import one language
    python import_contrib.py --dry-run    # Preview only
"""

import argparse
import json
import re
import shutil
import sys
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

from pathlib import Path

CONTRIB_DIR = Path("contrib")
DOWNLOADS_DIR = Path("downloads/contrib")


def load_source_toml(contrib_path):
    """Load and validate source.toml from a contribution directory.

    iso and distinct_id are derived from the directory name ({iso}_{distinct_id}).
    If the directory name has no underscore, distinct_id must be supplied via
    `distinct_id` in source.toml. Only `canon` is otherwise required.
    """
    toml_path = contrib_path / "source.toml"
    if not toml_path.exists():
        print(f"  SKIP: {contrib_path.name} — no source.toml")
        return None

    if tomllib is None:
        # Fallback: simple key=value parser for basic TOML
        data = {}
        for line in toml_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            data[key.strip()] = val.strip().strip('"').strip("'")
    else:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

    # Derive iso and distinct_id from directory name, with TOML fallback
    dir_name = contrib_path.name
    parts = dir_name.split("_", 1)
    if len(parts) == 2:
        iso, distinct_id = parts[0], parts[1]
    else:
        iso = dir_name
        distinct_id = data.get("distinct_id")
        if not distinct_id:
            print(f"  SKIP: {dir_name} — directory name has no underscore and "
                  f"source.toml has no `distinct_id`")
            return None

    if "canon" not in data:
        print(f"  SKIP: {dir_name} — missing 'canon' in source.toml")
        return None

    data["iso"] = iso
    data["distinct_id"] = distinct_id
    return data


def load_audio_json(contrib_path):
    """Load audio.json if present (describes remote audio source)."""
    p = contrib_path / "audio.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def strip_chapter_header(line, chapter_num):
    """Strip a leading chapter header from verse 1, e.g.,
    'Johannes Kapitel 1 Am Anfang ...' -> 'Am Anfang ...'.

    Looks for `Kapitel {chapter_num}` or `Capitulo {chapter_num}` etc.
    Currently handles German `Kapitel`. Add patterns as needed.
    """
    pat = re.compile(rf"^.*?\bKapitel\s+{chapter_num}\b\s*", re.IGNORECASE)
    return pat.sub("", line, count=1)


def find_chapter_files_legacy(contrib_path):
    """Find audio+text pairs in the LEGACY layout (root-level files).

    Expects: {BOOK}_{chapter}.mp3 and {BOOK}_{chapter}.txt
    Chapter numbers can be padded or unpadded (e.g., JHN_1 or JHN_001).
    Returns list of (book, chapter_num, mp3_path, txt_path)
    """
    pairs = []
    mp3_files = sorted(contrib_path.glob("*.mp3"))

    for mp3 in mp3_files:
        stem = mp3.stem  # e.g., "JHN_1" or "JHN_001"
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        book, chapter_str = parts
        try:
            chapter_num = int(chapter_str)
        except ValueError:
            continue

        # Find matching text file — try multiple naming patterns
        txt = None
        candidate = contrib_path / f"{stem}.txt"
        if candidate.exists():
            txt = candidate
        if not txt:
            alt_str = str(chapter_num) if len(chapter_str) > 1 and chapter_str[0] == "0" else f"{chapter_num:03d}"
            candidate = contrib_path / f"{book}_{alt_str}.txt"
            if candidate.exists():
                txt = candidate
        if not txt:
            for pattern in (f"*_{book}_{chapter_str}.txt", f"*_{book}_{chapter_num}.txt"):
                txt_candidates = list(contrib_path.glob(pattern))
                if txt_candidates:
                    txt = txt_candidates[0]
                    break
        if not txt:
            print(f"    WARN: No text for {book} {chapter_num}")
            continue

        pairs.append((book, chapter_num, mp3, txt, None))  # no timecode for legacy

    return pairs


def find_chapter_files_external(contrib_path, audio_meta):
    """Find chapter files in the EXTERNAL layout.

    Looks in text/, timecodes/ subfolders. Audio is referenced remotely via audio_meta.
    Returns list of (book, chapter_num, None, txt_path, csv_path) — mp3 is None
    because it gets downloaded later from the remote URL.
    """
    text_dir = contrib_path / "text"
    timecodes_dir = contrib_path / "timecodes"
    if not text_dir.is_dir():
        return []

    pairs = []
    for txt in sorted(text_dir.glob("*.txt")):
        stem = txt.stem  # e.g., "JHN_1"
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        book, chapter_str = parts
        try:
            chapter_num = int(chapter_str)
        except ValueError:
            continue

        # Verify book is in audio.json (skip otherwise)
        if audio_meta and book not in audio_meta.get("books", {}):
            continue

        # Look for matching CSV (pre-aligned timecode)
        csv = timecodes_dir / f"{stem}.csv" if timecodes_dir.is_dir() else None
        if csv and not csv.exists():
            csv = None

        pairs.append((book, chapter_num, None, txt, csv))

    return pairs


def csv_to_timing_json(csv_path, book, chapter_num):
    """Convert a per-verse-timestamp CSV to the pipeline's _timing.json format.

    CSV format (sermon-online convention): one float per line (seconds), N
    entries for an N-verse chapter, where each value is the END of verse N
    (== START of verse N+1 for non-final verses, or END of the chapter for
    the final verse).

    So:
      v1 starts at 0 (chapter beginning, before any title/intro)
      v_i (i >= 2) starts at CSV[i-2]
      v_N ends at CSV[N-1]   (recorded as a sentinel "verse N+1" entry so the
                              export pipeline can compute v_N's end boundary)

    Output: list of dicts matching DBT timing format.
    """
    chapter_str = f"{chapter_num:03d}"
    timestamps = []
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                timestamps.append(float(line))
            except ValueError:
                continue

    if not timestamps:
        return None

    n = len(timestamps)

    def _entry(verse: int, ts: float) -> dict:
        return {
            "book": book,
            "chapter": chapter_str,
            "verse_start": str(verse),
            "verse_start_alt": str(verse),
            "timestamp": round(ts, 2),
        }

    entries = [_entry(0, 0)]  # chapter anchor
    entries.append(_entry(1, 0))  # v1 starts at chapter start (covers any title/intro)
    for i in range(2, n + 1):
        entries.append(_entry(i, timestamps[i - 2]))
    # Sentinel for end of the final verse: timestamp the export uses as v_N's end
    entries.append(_entry(n + 1, timestamps[n - 1]))
    return entries


def import_contribution(contrib_path, meta, dry_run=False):
    """Import one contribution into downloads/contrib/."""
    iso = meta["iso"]
    distinct_id = meta["distinct_id"]
    canon = meta["canon"]

    audio_meta = load_audio_json(contrib_path)
    if audio_meta:
        pairs = find_chapter_files_external(contrib_path, audio_meta)
        layout = "external"
    else:
        pairs = find_chapter_files_legacy(contrib_path)
        layout = "legacy"

    if not pairs:
        print(f"  No chapter files found in {contrib_path.name}")
        return 0

    audio_suffix = "N2DA" if canon == "nt" else "O2DA"
    audio_fileset = f"{distinct_id}{audio_suffix}"
    text_fileset = f"{distinct_id}_{canon.upper()}_ET"

    version_dir = DOWNLOADS_DIR / canon / iso / distinct_id

    # Copy audio.json to the version dir so the export pipeline can ship it
    if audio_meta and not dry_run:
        version_dir.mkdir(parents=True, exist_ok=True)
        with open(version_dir / "audio.json", "w", encoding="utf-8") as f:
            json.dump(audio_meta, f, indent=2, ensure_ascii=False)

    imported = 0
    aligned = 0
    for book, chapter_num, mp3, txt, csv in pairs:
        chapter_str = f"{chapter_num:03d}"
        book_dir = version_dir / book

        mp3_target = book_dir / f"{book}_{chapter_str}_{audio_fileset}.mp3"
        txt_target = book_dir / f"{book}_{chapter_str}_{text_fileset}.txt"
        timing_target = book_dir / f"{book}_{chapter_str}_{audio_fileset}_timing.json"

        if dry_run:
            new_items = []
            if mp3 is not None and not mp3_target.exists():
                new_items.append(f"copy {mp3.name}")
            if not txt_target.exists():
                new_items.append(f"copy text {txt.name}")
            if csv and not timing_target.exists():
                new_items.append(f"convert timecodes {csv.name}")
            if new_items:
                print(f"    {book} {chapter_num}: " + ", ".join(new_items))
                imported += 1
            continue

        book_dir.mkdir(parents=True, exist_ok=True)

        # Audio: only copy if a local mp3 was supplied (legacy layout).
        # External layout points clients directly at remote URLs via audio.json.
        if mp3 is not None and not mp3_target.exists():
            shutil.copy2(mp3, mp3_target)

        # Text — strip chapter header from verse 1 if present
        if not txt_target.exists():
            with open(txt, encoding="utf-8") as f:
                lines = f.readlines()
            if lines and layout == "external":
                lines[0] = strip_chapter_header(lines[0], chapter_num)
            with open(txt_target, "w", encoding="utf-8") as f:
                f.writelines(lines)

        # Pre-aligned timecodes → _timing.json
        if csv and not timing_target.exists():
            entries = csv_to_timing_json(csv, book, chapter_num)
            if entries:
                with open(timing_target, "w", encoding="utf-8") as f:
                    json.dump(entries, f, indent=2, ensure_ascii=False)
                aligned += 1

        imported += 1

    if aligned and not dry_run:
        print(f"    Pre-aligned {aligned} chapters from contributed timecodes")

    return imported


def main():
    parser = argparse.ArgumentParser(
        description="Import contributed Bible audio+text into pipeline format",
    )
    parser.add_argument("--iso", type=str, help="Import one language only")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    if not CONTRIB_DIR.is_dir():
        print(f"No {CONTRIB_DIR}/ directory found")
        sys.exit(1)

    total = 0
    for contrib_path in sorted(CONTRIB_DIR.iterdir()):
        if not contrib_path.is_dir() or contrib_path.name.startswith("."):
            continue

        meta = load_source_toml(contrib_path)
        if not meta:
            continue

        if args.iso and meta["iso"] != args.iso:
            continue

        print(f"\n  {contrib_path.name} → {meta['canon']}/{meta['iso']}/{meta['distinct_id']}")
        count = import_contribution(contrib_path, meta, dry_run=args.dry_run)
        total += count
        if count:
            print(f"    {'Would import' if args.dry_run else 'Imported'} {count} chapters")

    action = "Would import" if args.dry_run else "Imported"
    print(f"\n{action} {total} chapters total.")


if __name__ == "__main__":
    main()
