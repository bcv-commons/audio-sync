#!/usr/bin/env python3
"""
Download the source YouTube video for every story of an OBS-OBS4All
language, for frame-change detection (see tools/cache_obs_images.py — this
is the video-side counterpart: door43 only hosts the audio extracted from
these videos, plus a link back to the original YouTube playlist; the
actual video frames only exist on YouTube itself).

Lowest usable video-only resolution is fetched deliberately (~144p) —
frame-change detection against the 64x64 reference fingerprints doesn't
need HD, and 12 languages x 50 stories at full resolution would be a
needless ~25GB+ vs. ~2-3GB this way. Video-only (no audio track) since we
already have each story's audio separately via the .m4a door43 already
serves.

Story number comes from parsing the video title (e.g. "50.⁠ ⁠यीशु पासो आवे
है" -> 50), NOT from playlist position — confirmed 2026-09-08 the OBS4All
playlists are NOT in story order (gdx's playlist lists story 50 first).

Usage:
    python tools/fetch_obs_videos.py --iso gdx
    python tools/fetch_obs_videos.py --iso-list gdx,awa,gju
    python tools/fetch_obs_videos.py --iso gdx --force
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import yt_dlp

OUTPUT_ROOT = Path("downloads/obs-video")
OBS_BATCH_DIR = Path("_obs_batches")
# Confirmed 2026-09-10 (gju stories 03, 23): a plain "^\s*(\d{1,2})\b" missed
# titles where the story number has leading RTL punctuation before it
# ("۔23 جناب...") or no space after it ("3سلاب" — Python's \b treats an
# Urdu letter as a \w character same as a digit, so no boundary exists
# between them). Skipping up to 3 leading non-digit characters (covers
# whitespace AND stray punctuation) and dropping the trailing \b (at most
# 2 digits is already an unambiguous story number, 1-50) fixes both.
TITLE_STORY_RE = re.compile(r"^\D{0,3}(\d{1,2})")

# Prefer smallest usable video-only stream; fall back gracefully if a
# playlist doesn't have exactly this format id (confirmed 2026-09-08:
# YouTube's itag 160 = 256x144 mp4 video-only, but format availability
# can vary per-video).
FORMAT_SELECTOR = "bestvideo[height<=144]/worstvideo"

# Confirmed 2026-09-09/10: an 11-language, ~450-video marathon with zero
# pacing between downloads triggered YouTube's "Sign in to confirm you're
# not a bot" block partway through (killed und-x-chopali entirely, 40/50
# of und-x-bhagiliyani, 30/50 of und-x-choudhancy) — it eased off again a
# couple hours later (the next two languages fetched cleanly), consistent
# with an IP-level rate flag rather than a per-video problem. Two
# mitigations: (1) per-video jitm so a normal run looks less like a
# scraping burst, (2) on hitting the bot-check specifically, back off much
# longer and retry once — a short-lived flag might clear inside the pause;
# a longer-lived one won't, and we just move on rather than hammering it.
SLEEP_MIN_SEC = 2.0
SLEEP_MAX_SEC = 6.0
BOT_CHECK_MARKER = "Sign in to confirm you"
BOT_CHECK_BACKOFF_SEC = 90.0


def log(message: str, level: str = "INFO"):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}")


def find_playlist_url(iso: str) -> str | None:
    """Look up this language's source YouTube playlist from its staged
    OBS batch manifest's door43 repo releases (the "YouTube - <Language>
    Playlist" asset — see gdx_obs's own README for why this exists: it's
    the acknowledged source the .m4a audio was extracted from)."""
    manifest_path = OBS_BATCH_DIR / f"{iso}.json"
    if not manifest_path.exists():
        log(f"{iso}: no staged manifest at {manifest_path}", "ERROR")
        return None
    source_repo = json.loads(manifest_path.read_text())["source_repo"]
    url = f"https://git.door43.org/api/v1/repos/{source_repo}/releases"
    req = urllib.request.Request(url, headers={"User-Agent": "audio-sync-fetch-obs-videos"})
    with urllib.request.urlopen(req, timeout=20) as r:
        releases = json.load(r)
    for release in releases:
        for asset in release.get("assets", []):
            if "youtube.com" in (asset.get("browser_download_url") or ""):
                return asset["browser_download_url"]
    return None


def fetch_language(iso: str, force: bool = False) -> tuple[int, int]:
    playlist_url = find_playlist_url(iso)
    if not playlist_url:
        log(f"{iso}: no YouTube playlist asset found", "ERROR")
        return 0, 0

    out_dir = OUTPUT_ROOT / iso
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"{iso}: listing playlist {playlist_url}")
    with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    entries = info.get("entries", [])
    log(f"{iso}: {len(entries)} video(s) in playlist")

    fetched, skipped = 0, 0
    for entry in entries:
        title = entry.get("title", "")
        m = TITLE_STORY_RE.match(title)
        if not m:
            log(f"{iso}: could not parse story number from title {title!r} — skipping", "WARN")
            continue
        story_id = f"{int(m.group(1)):02d}"
        dest = out_dir / f"{story_id}.mp4"
        if dest.exists() and not force:
            skipped += 1
            continue

        video_url = entry.get("url") or f"https://www.youtube.com/watch?v={entry['id']}"
        ydl_opts = {
            "format": FORMAT_SELECTOR,
            "outtmpl": str(dest),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        time.sleep(random.uniform(SLEEP_MIN_SEC, SLEEP_MAX_SEC))
        for attempt in (1, 2):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                fetched += 1
                log(f"{iso} story {story_id}: downloaded ({title!r})")
                break
            except Exception as e:
                if BOT_CHECK_MARKER in str(e) and attempt == 1:
                    log(f"{iso} story {story_id}: bot-check hit — backing off {BOT_CHECK_BACKOFF_SEC:.0f}s and retrying once", "WARN")
                    time.sleep(BOT_CHECK_BACKOFF_SEC)
                    continue
                log(f"{iso} story {story_id}: download failed ({e})", "ERROR")
                break

    return fetched, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", type=str, default=None)
    parser.add_argument("--iso-list", type=str, default=None, help="Comma-separated iso codes")
    parser.add_argument("--force", action="store_true", help="Re-download even if already cached")
    args = parser.parse_args()

    if args.iso_list:
        isos = [s.strip() for s in args.iso_list.split(",") if s.strip()]
    elif args.iso:
        isos = [args.iso]
    else:
        parser.error("--iso or --iso-list required")

    total_fetched, total_skipped = 0, 0
    for iso in isos:
        fetched, skipped = fetch_language(iso, force=args.force)
        total_fetched += fetched
        total_skipped += skipped

    log(f"Done — {total_fetched} downloaded, {total_skipped} already cached")


if __name__ == "__main__":
    main()
