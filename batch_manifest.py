"""
Batch manifest reader for audio-sync.

This module replaces the old export_templates imports (get_template_ids,
load_template_book_chapters, load_template_refs). In the Phase 1 design,
core (bible-story-builder) emits /_batches/<id>.json to CDN and audio-sync
reads it here to know what to align. Core also handles version-exclusion
filtering before jobs reach the manifest, so audio-sync never imports
version_exclude.

Usage:
    BATCH_ID=<id> python whisper_transcribe.py
    or set --batch-id CLI arg and pass it through to load_batch().

Expected manifest shape at _batches/<id>.json (or data/_batches/<id>.json),
or at https://cdn.bibel.wiki/_batches/queue/{high,normal,low}/<id>.json if
not found locally:
{
  "id": "<batch_id>",
  "template": "<template_id>",
  "jobs": [
    {
      "iso": "eng",
      "language": "English",
      "canon": "NT",
      "distinct_id": "ENGKJV",
      "has_downloads": true,
      "chapters": {"MAT": [1, 2, 3], "MRK": [1]}
    }
  ]
}

Status (2026-07-15): Core now publishes under the Contract C tiered queue
(internal-docs/audio-sync-interface.md §7 in MONO) — high/normal/low
priority directories, not the old flat _batches/<id>.json path. This module
only does discovery/fetch (checks local dirs, then tries each CDN tier);
the priority-queue worker loop itself (§7.2 cooperative preemption between
chapters) is future work, not built yet.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

_BATCH_DIRS = [Path("_batches"), Path("data/_batches")]

BATCH_CDN_QUEUE_BASE = "https://cdn.bibel.wiki/_batches/queue/"
_QUEUE_TIERS = ("high", "normal", "low")


def load_batch(batch_id: Optional[str] = None) -> dict:
    """Load a batch manifest by ID (env BATCH_ID or explicit arg).

    Checks local dirs first (for hand-crafted test manifests), then falls
    back to fetching from CDN — tries each priority tier (high, normal,
    low) in turn — and caches the result to _batches/<id>.json.
    """
    bid = batch_id or os.environ.get("BATCH_ID")
    if not bid:
        raise RuntimeError(
            "No batch ID. Set BATCH_ID env var or pass --batch-id.\n"
            "Core (bible-story-builder) should emit _batches/<id>.json via "
            "'make batch'. See internal-docs/audio-sync-interface.md in MONO."
        )
    for d in _BATCH_DIRS:
        path = d / f"{bid}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)

    # Not found locally — try each CDN priority tier, cache what we get
    for tier in _QUEUE_TIERS:
        try:
            req = urllib.request.Request(
                f"{BATCH_CDN_QUEUE_BASE}{tier}/{bid}.json",
                headers={"User-Agent": "audio-sync"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read()
        except Exception:
            continue
        cache_path = _BATCH_DIRS[0] / f"{bid}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        return json.loads(data)

    searched = ", ".join(str(d / f"{bid}.json") for d in _BATCH_DIRS)
    searched += ", " + ", ".join(f"{BATCH_CDN_QUEUE_BASE}{t}/{bid}.json" for t in _QUEUE_TIERS)
    raise FileNotFoundError(
        f"Batch manifest not found for id={bid!r} locally or on CDN.\nSearched: {searched}"
    )


def get_book_chapters(
    batch: dict, template_filter: Optional[str] = None
) -> dict[str, set[int]]:
    """
    Return {BOOK: {chapter_ints}} merged across all jobs in the batch.

    Replaces get_template_ids + load_template_book_chapters from export_templates.
    template_filter is accepted for API compatibility but the batch is already
    scoped by core at emit time.
    """
    merged: dict[str, set] = {}
    for job in batch.get("jobs", []):
        for book, chapters in job.get("chapters", {}).items():
            merged.setdefault(book, set()).update(chapters)
    return merged


def get_template_chapters_from_batch(
    batch: dict, template_ids: Optional[list[str]] = None
) -> set[tuple[str, int]]:
    """
    Return set of (BOOK, chapter_int) pairs covered by the batch.

    Replaces load_template_refs + the hand-rolled loop from export_templates.
    template_ids filter is accepted for API compatibility; the batch from core
    is already scoped to the relevant templates.
    """
    chapters: set[tuple[str, int]] = set()
    for job in batch.get("jobs", []):
        for book, ch_list in job.get("chapters", {}).items():
            for ch in ch_list:
                chapters.add((book, ch))
    return chapters


def get_jobs(batch: dict) -> list[dict]:
    """Return the job list from the batch."""
    return batch.get("jobs", [])
