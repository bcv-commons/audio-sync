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

Expected manifest shape at _batches/<id>.json (or data/_batches/<id>.json):
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

Status (2026-07-14): STUB. Core's batch emitter (Phase 3 of the repo split)
is not yet implemented. Until then, place a hand-crafted _batches/<id>.json
and set BATCH_ID. See internal-docs/audio-sync-interface.md in MONO for
the full contract specification.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

_BATCH_DIRS = [Path("_batches"), Path("data/_batches")]


def load_batch(batch_id: Optional[str] = None) -> dict:
    """Load a batch manifest by ID (env BATCH_ID or explicit arg)."""
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
    searched = ", ".join(str(d / f"{bid}.json") for d in _BATCH_DIRS)
    raise FileNotFoundError(
        f"Batch manifest not found for id={bid!r}.\nSearched: {searched}"
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
