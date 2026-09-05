#!/usr/bin/env python3
"""
Purge cached source audio (downloads/BB/**/*.mp3) for chapters that are
already fully aligned — i.e. both _timing.json and _words.json exist in
export/timing-data/. Nothing else is touched: word-timing-data/ (raw
Whisper/MMS output), export/timing-data/ (final fused output), and every
non-.mp3 file under downloads/ (text, per-chapter .json sidecars) all
stay.

Why this is safe: downloads/BB/ is a fetch-and-cache layer, not primary
output (see download_language_content.py) — ensure_chapter_ready()/
download_chapter() already re-fetch on demand whenever a chapter's audio
isn't on disk. Once a chapter's alignment is complete, its mp3 is only
needed again for two edge cases, both already re-fetch-tolerant:
  - --force / --force-fusion re-running MMS gap-fill re-alignment on that
    chapter (audio_path is read if present; a missing file there would
    need re-fetching, not crash — align_pipeline.py already handles a
    chapter with no local audio via has_downloads/ensure_chapter_ready).
  - tools/fix_timing_gaps.py --fix re-processing a specific chapter later.

Scoped to downloads/BB/ only (the DBT source tree) — confirmed 2026-09-02
this is ~336GB, by far the dominant consumer of this repo's disk (vs.
~200MB in downloads/contrib/ and ~20MB in downloads/helloao/, neither of
which caused the incident this tool was built for). downloads/obs/ is a
separate, much smaller (~6GB) tree with its own retention question, not
addressed here.

Built after a real incident (2026-09-02): downloads/BB/ grew to 336GB and
filled the disk to 0 bytes free, stalling both the DBT and OBS batches for
hours. A one-time manual cleanup recovered ~250GB; this module is that
same logic, wired into pipeline/shard_align.py so it runs automatically
after each language finishes, instead of needing another manual sweep.

Usage:
    python tools/purge_aligned_audio.py --iso fra
    python tools/purge_aligned_audio.py --iso-list fra,deu,spa
    python tools/purge_aligned_audio.py --iso fra --dry-run
"""

import argparse
from datetime import datetime
from pathlib import Path

DOWNLOADS_ROOT = Path("downloads/BB")
EXPORT_ROOT = Path("export/timing-data")
CANONS = ("nt", "ot")


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def purge_iso_audio(iso: str, dry_run: bool = False) -> tuple[int, int]:
    """Delete downloads/BB/{nt,ot}/{iso}/**/*.mp3 for chapters whose
    _timing.json and _words.json both already exist.

    Returns (files_deleted, bytes_freed). Never raises on an individual
    file's OSError (e.g. a concurrent worker still has it open) — logs a
    warning and continues, since one unremovable file must never abort
    the rest of the sweep.
    """
    deleted = 0
    freed = 0
    for canon in CANONS:
        iso_dir = DOWNLOADS_ROOT / canon / iso
        if not iso_dir.is_dir():
            continue
        for mp3 in iso_dir.rglob("*.mp3"):
            rel = mp3.relative_to(DOWNLOADS_ROOT / canon)
            parts = rel.parts  # {iso}/{distinct_id}/{book}/{stem}.mp3
            if len(parts) != 4:
                continue
            _iso, distinct_id, book, fname = parts
            stem = fname[:-4]
            out_dir = EXPORT_ROOT / canon / _iso / distinct_id / book
            if not ((out_dir / f"{stem}_timing.json").exists()
                    and (out_dir / f"{stem}_words.json").exists()):
                continue
            sz = mp3.stat().st_size
            if dry_run:
                deleted += 1
                freed += sz
                continue
            try:
                mp3.unlink()
                deleted += 1
                freed += sz
            except OSError as e:
                log(f"  Could not delete {mp3}: {e}", "WARN")
    return deleted, freed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    lang_group = parser.add_mutually_exclusive_group(required=True)
    lang_group.add_argument("--iso", type=str, help="Single language ISO code")
    lang_group.add_argument("--iso-list", type=str, help="Comma-separated ISO codes")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted, delete nothing")
    args = parser.parse_args()

    isos = [args.iso] if args.iso else [c.strip() for c in args.iso_list.split(",") if c.strip()]

    total_files = 0
    total_bytes = 0
    for iso in isos:
        deleted, freed = purge_iso_audio(iso, dry_run=args.dry_run)
        verb = "would delete" if args.dry_run else "deleted"
        log(f"[{iso}] {verb} {deleted} file(s), {freed / 1e9:.2f} GB")
        total_files += deleted
        total_bytes += freed

    verb = "Would free" if args.dry_run else "Freed"
    log(f"{verb} {total_bytes / 1e9:.2f} GB across {total_files} file(s), {len(isos)} language(s)")


if __name__ == "__main__":
    main()
