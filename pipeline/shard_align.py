#!/usr/bin/env python3
"""
Shard-and-launch parallel wrapper around align_pipeline.py.

Splits a single language's requested --books/--template scope into N
groups, balanced by total chapter count (not book count — PSA alone is
150 chapters, OBA is 1), and launches one align_pipeline.py subprocess
per group concurrently against the same GPU/CPU.

Why book-group subprocesses instead of a real chapter-level work queue:
today's align_pipeline.py loads its own Whisper+MMS models per process
and has no shared queue mechanism, so true chapter-level parallelism
(N persistent workers each loading models once, pulling chapters from a
shared queue) needs real changes inside the pipeline. This script is the
practical interim step — N independent processes, balanced so no worker
sits idle disproportionately. compute_balanced_shards() below is written
to be reusable as-is if/when that real queue gets built: same input
(book -> chapter-count), same greedy balancing, just handed to queue
workers instead of subprocess.Popen.

Worker count is hardware-dependent (VRAM/CPU headroom varies per box —
see the OOM lesson in conf/hw.local.json.example's mms_chunk_minutes
note), so it is NEVER hardcoded here. It comes from hw.local.json's
"parallel_workers" field, default 1 — i.e. no parallelism, identical to
running align_pipeline.py directly. That default means every machine
(Mac, CPU-only box, a shared server) behaves exactly as before unless
its own hw.local.json explicitly opts in after testing on real chapters
(same rule as mms_chunk_minutes — don't extrapolate from headroom,
verify empirically). --workers on the CLI overrides it for one run, same
override pattern as every other hw.local.json-backed flag in this repo.

Usage:
    python pipeline/shard_align.py --iso fra --books ALL
    python pipeline/shard_align.py --iso fra --books NT --workers 4
    python pipeline/shard_align.py --iso fra --books ALL --publish --force
    python pipeline/shard_align.py --iso fra --books ALL --dry-run

Any flag not recognized by this script's own parser (e.g. --publish,
--force, --no-download, --model, --whisper-cpu, --mms-cpu,
--mms-chunk-minutes, --device, --skip-whisper/--skip-mms/--skip-fusion)
is forwarded unchanged to every shard's align_pipeline.py subprocess —
this script deliberately does not duplicate that flag list, so it can't
drift out of sync with align_pipeline.py's own argparse setup.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from align_pipeline import build_refs_from_books
from whisper_transcribe import NT_BOOKS, OT_BOOKS, load_all_template_refs
from hw_config import load_hw_config

ALL_BOOKS = {**OT_BOOKS, **NT_BOOKS}
THIS_DIR = Path(__file__).resolve().parent


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [shard_align] {message}", flush=True)


def compute_balanced_shards(book_chapters: Dict[str, Set[int]], n: int) -> List[List[str]]:
    """Split books into n groups, balanced by summed chapter count.

    Greedy longest-processing-time-first (LPT) heuristic: place the
    biggest remaining book into whichever group currently has the
    smallest total. Not optimal but simple, deterministic, and within a
    few percent of optimal for this kind of input (few dozen items).

    book_chapters: {book: set_of_chapter_numbers}. Returns n lists of
    book codes (some may be empty if n exceeds the number of books).
    """
    groups: List[List[str]] = [[] for _ in range(n)]
    totals = [0] * n
    by_size = sorted(book_chapters.items(), key=lambda kv: -len(kv[1]))
    for book, chapters in by_size:
        i = totals.index(min(totals))
        groups[i].append(book)
        totals[i] += len(chapters)
    return groups


def format_book_chapters(book: str, chapters: Set[int]) -> str:
    """Render a book + chapter set as an align_pipeline.py --books token.

    Full-book coverage collapses to the bare book code ("JHN"); a partial
    set becomes "BOOK:1-3,17" — the exact syntax build_refs_from_books()
    parses, so this is a straight inverse of that function.
    """
    full = ALL_BOOKS.get(book)
    if full is not None and chapters == set(range(1, full + 1)):
        return book

    ranges: List[str] = []
    ordered = sorted(chapters)
    start = prev = ordered[0]
    for c in ordered[1:]:
        if c == prev + 1:
            prev = c
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = c
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return f"{book}:{','.join(ranges)}"


def shard_one_language(iso: str, refs: Dict[str, set], workers: int, passthrough: list) -> bool:
    """Shard one language's book/chapter refs across N workers, launch, wait.

    Returns True if every worker finished OK, False if any failed.
    """
    total_chapters = sum(len(chs) for chs in refs.values())
    n = min(workers, len(refs))  # never more workers than books to hand out
    if n < workers:
        log(f"[{iso}] Only {len(refs)} book(s) in scope — capping workers at {n} (requested {workers})")

    shards = [g for g in compute_balanced_shards(refs, n) if g]
    log(f"[{iso}] Sharding {total_chapters} chapters across {len(shards)} book(s)/{len(refs)} total into "
        f"{len(shards)} worker(s):")
    for i, group in enumerate(shards, 1):
        count = sum(len(refs[b]) for b in group)
        log(f"  worker {i}: {len(group)} book(s), {count} chapter(s) — {', '.join(sorted(group))}")

    if passthrough:
        log(f"[{iso}] Forwarding extra flags to every worker: {' '.join(passthrough)}")

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir = Path("_runs") / f"shard-{iso}-{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for i, group in enumerate(shards, 1):
        books_arg = ",".join(format_book_chapters(b, refs[b]) for b in sorted(group))
        log_path = log_dir / f"worker-{i}.log"
        cmd = [
            sys.executable, str(THIS_DIR / "align_pipeline.py"),
            "--iso", iso, "--books", books_arg,
            *passthrough,
        ]
        log(f"[{iso}] worker {i} log: {log_path}")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((i, proc, log_file, log_path))

    log(f"[{iso}] {len(procs)} worker(s) launched — waiting for completion "
        f"(tail -f {log_dir}/worker-N.log to watch progress live)")

    failed = []
    while procs:
        time.sleep(2)
        still_running = []
        for i, proc, log_file, log_path in procs:
            code = proc.poll()
            if code is None:
                still_running.append((i, proc, log_file, log_path))
                continue
            log_file.close()
            if code == 0:
                log(f"[{iso}] worker {i} finished OK")
            else:
                log(f"[{iso}] worker {i} FAILED (exit {code}) — see {log_path}")
                failed.append(i)
        procs = still_running

    if failed:
        log(f"[{iso}] Done — {len(failed)}/{len(shards)} worker(s) failed: {failed}. "
            f"Check {log_dir}/worker-N.log for the failed shard(s); rerun just "
            f"that shard's books once the cause is fixed (already-processed "
            f"chapters are skipped automatically, so this is safe to retry).")
        return False

    log(f"[{iso}] Done — all {len(shards)} worker(s) finished OK. Logs in {log_dir}/")
    return True


def main():
    hw = load_hw_config()

    parser = argparse.ArgumentParser(
        description="Shard one or more languages' alignment work across N parallel align_pipeline.py workers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    lang_group = parser.add_mutually_exclusive_group(required=True)
    lang_group.add_argument("--iso", type=str, help="Single language ISO 639-3 code (e.g., fra)")
    lang_group.add_argument(
        "--iso-list", type=str,
        help="Comma-separated ISO 639-3 codes (e.g., fra,arb,hin) — languages are sharded and "
             "processed one at a time, in order; each language's N workers run concurrently, "
             "but the next language doesn't start until the current one finishes.",
    )
    parser.add_argument(
        "--books", type=str, default=None,
        help="Same syntax as align_pipeline.py --books (e.g. NT, OT, ALL, JHN,MAT, GEN:1-3,17)",
    )
    parser.add_argument("--template", type=str, default=None, help="Same as align_pipeline.py --template")
    parser.add_argument(
        "--testament", type=str, choices=["nt", "ot", "both"], default="both",
        help="Which testament to shard (default: both)",
    )
    parser.add_argument(
        "--workers", type=int, default=hw["parallel_workers"],
        help=f"Number of concurrent align_pipeline.py processes per language (default: {hw['parallel_workers']}, "
             "from conf/hw.local.json's parallel_workers)",
    )
    args, passthrough = parser.parse_known_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if args.books:
        base_refs = build_refs_from_books(args.books)
    else:
        base_refs = load_all_template_refs(args.template)
    if not base_refs:
        log("No books/chapters resolved — nothing to shard.")
        sys.exit(1)

    if args.testament == "nt":
        base_refs = {b: chs for b, chs in base_refs.items() if b in NT_BOOKS}
    elif args.testament == "ot":
        base_refs = {b: chs for b, chs in base_refs.items() if b in OT_BOOKS}

    isos = [args.iso] if args.iso else [c.strip().lower() for c in args.iso_list.split(",") if c.strip()]

    failed_isos = []
    for iso in isos:
        log("")
        log(f"===== Language: {iso} =====")
        ok = shard_one_language(iso, dict(base_refs), args.workers, passthrough)
        if not ok:
            failed_isos.append(iso)

    log("")
    if failed_isos:
        log(f"Done — {len(failed_isos)}/{len(isos)} language(s) had at least one failed worker: "
            f"{failed_isos}. Rerun the same command to retry (already-done chapters are skipped).")
        sys.exit(1)

    log(f"Done — all {len(isos)} language(s) finished OK.")


if __name__ == "__main__":
    main()
