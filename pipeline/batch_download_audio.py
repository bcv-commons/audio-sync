#!/usr/bin/env python3
"""
Standalone batch audio+text downloader — races ahead of the alignment
pipeline through the same edition queue (tools/build_edition_queue.py),
prefetching so the aligner finds chapters already on disk when it gets to
them.

Safe to run at the same time as align_pipeline.py's own on-demand
per-chapter fetch — align_pipeline.py needs NO changes for this, it keeps
behaving exactly as it does today. Both this script and align_pipeline.py
go through ensure_chapter_ready() -> download_job() -> download_chapter(),
which now has a claim-file collision guard (see download_language_content.py's
_fetch_with_claim() / _acquire_download_claim()) — whichever process gets
to a given chapter first does the real fetch, the other just waits for it
rather than duplicating the work or racing on the same file write.

I/O-bound, so this runs a thread pool far wider than the GPU-bound
aligner's single-worker-per-shard model (default 6, --workers to adjust).

Resumable for free: ensure_chapter_ready()'s existing skip-if-exists check
means killing and restarting this script just continues where it left off
— no separate progress state to track or corrupt.

Usage:
    python pipeline/batch_download_audio.py
    python pipeline/batch_download_audio.py --workers 8
    python pipeline/batch_download_audio.py --phase 1
    python pipeline/batch_download_audio.py --iso azb --limit 5
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent))

import download_language_content as dlc  # noqa: E402
from download_language_content import (  # noqa: E402
    BIBLE_API_KEY,
    ensure_chapter_ready,
    get_dbt_book_coverage,
    log,
)
from whisper_transcribe import NT_BOOKS, OT_BOOKS  # noqa: E402

DEFAULT_QUEUE = Path("_runs/edition_queue.json")


def _canon_books(canon: str) -> dict[str, int]:
    return NT_BOOKS if canon.startswith("nt") else OT_BOOKS


def _edition_chapters(edition: dict) -> list[tuple[str, int]]:
    """Real (book, chapter) list for one edition — DBT's own per-book
    coverage when available (avoids requesting chapters that don't exist,
    see get_dbt_book_coverage's docstring), the full standard testament
    otherwise (fail-open, same convention align_pipeline.py's own
    resolution uses).
    """
    books = _canon_books(edition["canon"])
    wanted = [(b, c) for b, chs in books.items() for c in range(1, chs + 1)]

    coverage = get_dbt_book_coverage(edition["distinct_id"])
    if coverage is not None:
        wanted = [(b, c) for b, c in wanted if c in coverage.get(b, [])]
    return wanted


def download_edition(edition: dict, force: bool) -> dict:
    """Fetch every chapter of one edition. Returns a small per-edition
    stats dict for progress reporting — never raises for an individual
    chapter failure (ensure_chapter_ready already reports False rather
    than throwing), so one bad edition can't take down the whole run.
    """
    chapters = _edition_chapters(edition)
    ok = 0
    failed = 0
    for book, chapter in chapters:
        success = ensure_chapter_ready(
            edition["iso"], edition["canon"], edition["distinct_id"],
            book, chapter, force=force,
        )
        if success:
            ok += 1
        else:
            failed += 1
    return {**edition, "chapters": len(chapters), "ok": ok, "failed": failed}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--queue", type=Path, default=DEFAULT_QUEUE,
        help=f"Edition queue JSON from tools/build_edition_queue.py (default: {DEFAULT_QUEUE})",
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Concurrent download workers (I/O-bound, default: 6)",
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=None,
        help="Only process this phase (default: phase 1 then phase 2, in queue order)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N editions (testing)",
    )
    parser.add_argument(
        "--iso", type=str, default=None,
        help="Only process this one ISO code (testing)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--rate-delay", type=float, default=0.0,
        help="Seconds between API calls, applied per worker thread (default: 0)",
    )
    args = parser.parse_args()

    if not BIBLE_API_KEY:
        log("BIBLE_API_KEY not set in .env file", "ERROR")
        sys.exit(1)

    if args.rate_delay > 0:
        dlc.API_RATE_DELAY = args.rate_delay
        log(f"Rate limiting: {args.rate_delay}s delay between API calls (per worker)", "INFO")

    if not args.queue.exists():
        log(f"{args.queue} not found — run tools/build_edition_queue.py first", "ERROR")
        sys.exit(1)
    with open(args.queue) as f:
        queue = json.load(f)

    editions = queue["editions"]
    if args.phase is not None:
        editions = [e for e in editions if e["phase"] == args.phase]
    if args.iso:
        editions = [e for e in editions if e["iso"] == args.iso]
    if args.limit:
        editions = editions[: args.limit]

    if not editions:
        log("No editions matched the given filters — nothing to do")
        return

    log(f"Batch downloader: {len(editions)} edition(s) queued, {args.workers} worker(s)")

    totals = {"editions_done": 0, "chapters_ok": 0, "chapters_failed": 0}
    totals_lock = Lock()
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_edition, e, args.force): e for e in editions}
        for future in as_completed(futures):
            edition = futures[future]
            try:
                result = future.result()
            except Exception as e:
                log(
                    f"  {edition['iso']}/{edition['canon']}/{edition['distinct_id']}: "
                    f"unhandled error: {e}", "ERROR",
                )
                continue
            with totals_lock:
                totals["editions_done"] += 1
                totals["chapters_ok"] += result["ok"]
                totals["chapters_failed"] += result["failed"]
                done = totals["editions_done"]
            extra = f", {result['failed']} failed" if result["failed"] else ""
            log(
                f"[{done}/{len(editions)}] {result['iso']}/{result['canon']}/{result['distinct_id']} "
                f"(phase {result['phase']}): {result['ok']}/{result['chapters']} chapters ready{extra}",
            )

    elapsed = time.time() - start
    log("")
    log("=" * 70)
    log("Batch download summary")
    log("=" * 70)
    log(f"  Editions processed:  {totals['editions_done']}/{len(editions)}")
    log(f"  Chapters ready:      {totals['chapters_ok']}")
    log(f"  Chapters failed:     {totals['chapters_failed']}")
    log(f"  Elapsed:             {elapsed:.0f}s")


if __name__ == "__main__":
    main()
