#!/usr/bin/env python3
"""
Cache every OBS (Open Bible Stories) reference illustration as a small,
comparison-ready grayscale array, for frame-change detection against
captured video frames — see internal-docs (or ask) for the bigger picture:
video-only OBS sources (e.g. OBS-OBS4All's Godwari/gdx) have no native-
language text to align against at all, but each story is narrated over a
fixed sequence of still images. Detecting exactly when the displayed
image changes in the video gives real segment-boundary timestamps without
needing any text alignment — this is step one: get every reference image
cached in a form fast to compare against.

The illustrations are a SINGLE GLOBAL ART SET shared by every language's
OBS translation, not localized per language — confirmed directly
2026-09-08: ahr's (Angika) staged manifest's content_base_url points at
totally different story text, but its story 1 markdown references the
exact same "obs-en-01-01.jpg" etc. URLs as English's own story 1. So this
only ever needs to be built once, from any one story-text source (English
is used here simply because it's the canonical/original), and the result
is valid for matching video frames in ANY language's OBS video.

Source of the ordered image-URL list per story: unfoldingWord/en_obs's
own story markdown (git.door43.org), the same canonical repo
docs/message-to-bibles-versification-reply-style investigation already
established as trustworthy this session — each image marker
"![OBS Image](url)" in reading order corresponds to that story's segments
1, 2, 3, ... in order (same structural assumption pipeline/align_obs_words.
py's IMAGE_RE-based text splitter already relies on).

Output (under api-cache/obs-images/):
    raw/{story}_{segment}.jpg   — cached original downloads (skip re-fetch
                                  on a later run; also useful to inspect
                                  by eye)
    manifest.json               — {"01_01": {"story","segment","url",
                                  "width","height"}, ...}
    reference_frames.npz        — one array per key, dtype uint8, shape
                                  (FRAME_SIZE, FRAME_SIZE), grayscale.
                                  Small and resize/compression-invariant
                                  on purpose: a captured video frame will
                                  never pixel-match a reference JPEG
                                  exactly (different codec, scaling,
                                  compositing), so comparison should be a
                                  coarse structural match (e.g. MSE or
                                  normalized cross-correlation on this
                                  downsampled grayscale array), not exact
                                  equality. FRAME_SIZE is deliberately
                                  small (64) — a fingerprint, not a
                                  thumbnail meant for viewing.

Usage:
    python tools/cache_obs_images.py               # build/update the full cache
    python tools/cache_obs_images.py --story 01     # just one story (debugging)
    python tools/cache_obs_images.py --force        # re-download even if cached
"""

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

EN_OBS_CONTENT_BASE = "https://git.door43.org/unfoldingWord/en_obs/raw/branch/master/content"
IMAGE_RE = re.compile(r"!\[OBS Image\]\((?P<url>[^)]+)\)")

CACHE_DIR = Path("api-cache/obs-images")
RAW_DIR = CACHE_DIR / "raw"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
NPZ_PATH = CACHE_DIR / "reference_frames.npz"

STORY_COUNT = 50
FRAME_SIZE = 64  # fingerprint side length in pixels — see module docstring


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def fetch_story_image_urls(story_id: str) -> list[str]:
    """Ordered list of this story's reference image URLs, from English's
    own story markdown (see module docstring for why English specifically
    is fine to use as the source for every language)."""
    url = f"{EN_OBS_CONTENT_BASE}/{story_id}.md"
    req = urllib.request.Request(url, headers={"User-Agent": "audio-sync-cache-obs-images"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read().decode("utf-8")
    return [m.group("url") for m in IMAGE_RE.finditer(raw)]


def ensure_raw_image(key: str, url: str, force: bool = False) -> Path:
    """Download (and cache) one reference image's original bytes."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{key}.jpg"
    if dest.exists() and not force:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "audio-sync-cache-obs-images"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def to_fingerprint(image_path: Path) -> np.ndarray:
    """Grayscale, FRAME_SIZE x FRAME_SIZE uint8 array — see module
    docstring for why this size/shape rather than a full-resolution copy."""
    with Image.open(image_path) as img:
        img = img.convert("L").resize((FRAME_SIZE, FRAME_SIZE), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--story", type=str, default=None, help="Only this story id (e.g. 01)")
    parser.add_argument("--force", action="store_true", help="Re-download even if already cached")
    args = parser.parse_args()

    story_ids = [args.story] if args.story else [f"{i:02d}" for i in range(1, STORY_COUNT + 1)]

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    arrays: dict[str, np.ndarray] = {}
    if NPZ_PATH.exists():
        with np.load(NPZ_PATH) as existing:
            arrays = {k: existing[k] for k in existing.files}

    fetched, cached_hit, failed = 0, 0, 0
    for story_id in story_ids:
        try:
            urls = fetch_story_image_urls(story_id)
        except Exception as e:
            log(f"story {story_id}: failed to fetch markdown ({e})", "ERROR")
            failed += 1
            continue

        if not urls:
            log(f"story {story_id}: no image markers found", "WARN")
            continue

        for segment_idx, url in enumerate(urls, start=1):
            key = f"{story_id}_{segment_idx:02d}"
            if key in arrays and not args.force:
                cached_hit += 1
                continue
            try:
                raw_path = ensure_raw_image(key, url, force=args.force)
                arr = to_fingerprint(raw_path)
                with Image.open(raw_path) as img:
                    width, height = img.size
            except Exception as e:
                log(f"{key}: failed ({url}): {e}", "ERROR")
                failed += 1
                continue
            arrays[key] = arr
            manifest[key] = {
                "story": story_id,
                "segment": segment_idx,
                "url": url,
                "width": width,
                "height": height,
            }
            fetched += 1

        log(f"story {story_id}: {len(urls)} image(s)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ_PATH, **arrays)
    manifest["_generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["_frame_size"] = FRAME_SIZE
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    log(f"Done — {fetched} fetched, {cached_hit} already cached, {failed} failed. "
        f"{len(arrays)} total reference frame(s) in {NPZ_PATH}")


if __name__ == "__main__":
    main()
