"""
OBS narration batch manifest reader for audio-sync.

Parallel to batch_manifest.py, deliberately not folded into it: OBS
narration jobs are story-shaped (one audio clip per story, segment
markers instead of verses), not book/chapter/verse-shaped like Contract
A's DBT batches — forcing the two through one reader would mean fake
book/chapter fields, so this is its own small module instead. See
internal-docs/obs-narration-import-spec.md in bible-story-builder for the
full spec (MONO-side; this file just consumes what it publishes).

Content is not re-hosted here — same principle applied to audio (the
manifest's audio_url points straight at door43's release asset). The
manifest gives content_base_url; story text is fetched directly from
door43 per story as needed (fetch_story_text below), not staged as a
local copy of content we don't own.

Usage:
    from obs_batch_manifest import load_obs_batch, fetch_story_text
    batch = load_obs_batch("ahr")
    for story_id, story in batch["stories"].items():
        raw_md = fetch_story_text(batch, story_id)
        ...

Expected manifest shape at _obs_batches/<iso>.json (or
data/_obs_batches/<iso>.json), or at
https://cdn.bibel.wiki/_obs_batches/<iso>.json if not found locally:
{
  "iso": "ahr",
  "source": "OBS-TLF",
  "source_repo": "OBS-TLF/ahr_obs",
  "license": "CC BY-SA 4.0",
  "checking_level": "1",
  "content_base_url": "https://git.door43.org/OBS-TLF/ahr_obs/raw/branch/master/content",
  "stories": {
    "01": {
      "audio_url": "https://git.door43.org/OBS-TLF/ahr_obs/releases/download/v1/ahr_obs_v1_01_128kbps.m4a",
      "segment_count": 16
    }
  }
}

Status (2026-08-05): live — 1 of 92 languages staged so far (ahr), rest
pending MONO's fetch/staging script. See internal-docs/obs-narration-
import-spec.md's revision history for the full "why" behind the shape.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

_OBS_BATCH_DIRS = [Path("_obs_batches"), Path("data/_obs_batches")]

OBS_BATCH_CDN_BASE = "https://cdn.bibel.wiki/_obs_batches/"


def load_obs_batch(iso: str) -> dict:
    """Load an OBS narration batch manifest by ISO code.

    Checks local dirs first (for hand-crafted test manifests), then falls
    back to fetching from CDN and caches the result to _obs_batches/<iso>.json.
    """
    for d in _OBS_BATCH_DIRS:
        path = d / f"{iso}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)

    try:
        req = urllib.request.Request(
            f"{OBS_BATCH_CDN_BASE}{iso}.json", headers={"User-Agent": "audio-sync"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
    except Exception as e:
        searched = ", ".join(str(d / f"{iso}.json") for d in _OBS_BATCH_DIRS)
        raise FileNotFoundError(
            f"OBS batch manifest not found for iso={iso!r} locally or on CDN "
            f"({OBS_BATCH_CDN_BASE}{iso}.json: {e}).\nSearched: {searched}"
        ) from e

    cache_path = _OBS_BATCH_DIRS[0] / f"{iso}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return json.loads(data)


def get_stories(batch: dict) -> dict:
    """Return the {story_id: {audio_url, segment_count}} dict from a batch."""
    return batch.get("stories", {})


def fetch_story_text(batch: dict, story_id: str, *, timeout: int = 30) -> str:
    """Fetch one story's raw .md text directly from door43 (not re-hosted).

    Always a live fetch — story text isn't cached locally the way audio
    is, since it's small and re-fetching is cheap. Callers that need to
    avoid repeat fetches (e.g. iterating stories) should cache the result
    themselves.
    """
    url = f"{batch['content_base_url']}/{story_id}.md"
    req = urllib.request.Request(url, headers={"User-Agent": "audio-sync"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")
