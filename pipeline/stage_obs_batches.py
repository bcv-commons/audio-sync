#!/usr/bin/env python3
"""
Stage _obs_batches/<iso>.json manifests from Door43's catalog API.

Interim stand-in for MONO's not-yet-built fetch/staging script (see
obs_batch_manifest.py's docstring and internal-docs/obs-narration-import-
spec.md in bible-story-builder). Queries Door43's catalog search directly
— confirmed authoritative and pre-filtered (hasAudio=true), not a guess:
`GET /api/v1/catalog/search?subject=Open Bible Stories&hasAudio=true`
returns exactly the repos MONO's own stats-ext endpoint reports (98
repos / 92 languages as of 2026-09-01), so no separate org-scraping or
release-asset heuristics are needed to find the candidate set.

92 languages, not 98 repos: 6 languages (bfy, bhd, bho, dgo, mai, srb)
have a repo in both OBS-TLF and translationCore-Create-BCS. Resolved by
preferring translationCore-Create-BCS on conflict, matching the existing
hand-staged mai.json (source_repo "BCS/mai_obs" — BCS is
translationCore-Create-BCS's old short org name; see obs_batch_manifest.py
docstring's source_repo examples).

content_base_url must use the release's target_commitish (the actual git
branch, e.g. "master") — NOT branch_or_tag_name (e.g. "v1"): Gitea's
/raw/branch/<ref>/ only resolves real branches, confirmed directly
(raw/branch/v1/... 404s, raw/branch/master/... 200s on the same repo).

segment_count is intentionally omitted from generated manifests — it's
informational only (align_obs_words.py derives real segments by parsing
each story's .md at alignment time; a manifest's segment_count is never
read for anything but a log line, defaulting to "?" when absent) and
fetching every story's .md just to count image markers would mean ~4,600
extra requests (92 langs x ~50 stories) for no functional gain.

Usage:
    python pipeline/stage_obs_batches.py              # stage every new language
    python pipeline/stage_obs_batches.py --iso ahr     # stage/re-check just one
    python pipeline/stage_obs_batches.py --force       # overwrite existing manifests
    python pipeline/stage_obs_batches.py --dry-run      # show what would be staged
"""

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

CATALOG_SEARCH_URL = (
    "https://git.door43.org/api/v1/catalog/search"
    "?subject=Open%20Bible%20Stories&hasAudio=true&limit=100"
)
METADATA_URL_TMPL = "https://git.door43.org/api/v1/catalog/metadata/{full_name}/{tag}"
RELEASES_URL_TMPL = "https://git.door43.org/api/v1/repos/{full_name}/releases"
OBS_BATCH_DIR = Path("_obs_batches")

# Preferred org on a language collision (see module docstring).
PREFERRED_OWNER = "translationCore-Create-BCS"

STORY_AUDIO_RE = re.compile(r"_(\d{2})_[^_]*kbps\.(?:m4a|mp3)$", re.IGNORECASE)


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def _fetch_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "audio-sync-stage-obs"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch_catalog_entries() -> list[dict]:
    """Fetch every OBS-with-audio catalog entry, deduped to one per language."""
    data = _fetch_json(CATALOG_SEARCH_URL)
    entries = data["data"]

    by_lang: dict[str, dict] = {}
    for e in entries:
        iso = e["language"]
        existing = by_lang.get(iso)
        if existing is None or (
            existing["owner"] != PREFERRED_OWNER and e["owner"] == PREFERRED_OWNER
        ):
            by_lang[iso] = e
    return [by_lang[k] for k in sorted(by_lang)]


def extract_stories(release: dict) -> dict:
    """Build the {story_id: {audio_url}} dict from a release's assets —
    only real per-story audio (.m4a or .mp3 — STORY_AUDIO_RE accepts
    both), skipping non-audio assets (APKs, PDFs, whole-book zips, etc. —
    confirmed some repos publish audio-less releases with e.g. only an
    Android app asset, or bundle every story into one zip instead of
    per-story files, which this deliberately does NOT unpack)."""
    stories = {}
    for asset in release.get("assets", []):
        m = STORY_AUDIO_RE.search(asset["name"])
        if not m:
            continue
        story_id = m.group(1)
        stories[story_id] = {"audio_url": asset["browser_download_url"]}
    return stories


def find_stories_across_releases(full_name: str) -> tuple[dict, dict] | None:
    """Fallback for a repo whose catalog-reported "latest" release has no
    usable per-story audio (e.g. unfoldingWord/en_obs: the catalog search
    API's release is v9, a docs-only update with just a PDF asset — the
    real per-story .mp3s sit in the older v8 release, which the catalog
    entry never surfaces at all). Scans every release newest-first via the
    plain repo releases API (not the catalog search endpoint, which only
    ever reports one release per repo) and returns the first one with any
    matching story audio.

    Generic on purpose, not en-specific — any repo where a newer
    non-audio release (a docs/PDF/app update) shadows an older audio one
    hits this same shape, and there is no way to tell which repos those
    are ahead of time other than trying.

    Returns (release, stories) or None if no release has any usable audio.
    """
    try:
        releases = _fetch_json(RELEASES_URL_TMPL.format(full_name=full_name))
    except Exception as e:
        log(f"  {full_name}: releases fetch failed ({e})", "WARN")
        return None
    for release in releases:
        stories = extract_stories(release)
        if stories:
            return release, stories
    return None


def build_manifest(entry: dict) -> dict | None:
    iso = entry["language"]
    full_name = entry["full_name"]
    release = entry.get("release", {})
    stories = extract_stories(release)

    if not stories:
        log(f"{iso} ({full_name}): no story audio in catalog's reported release "
            f"({release.get('tag_name', '?')}) — checking older releases too", "INFO")
        found = find_stories_across_releases(full_name)
        if found is None:
            log(f"{iso} ({full_name}): no story audio in ANY release — skipping", "WARN")
            return None
        release, stories = found
        log(f"  found {len(stories)} stor(y/ies) in release {release.get('tag_name', '?')} instead", "INFO")

    target_commitish = release.get("target_commitish") or "master"

    try:
        meta = _fetch_json(METADATA_URL_TMPL.format(full_name=full_name, tag=entry["branch_or_tag_name"]))
        license_ = meta.get("dublin_core", {}).get("rights", "")
        checking_level = meta.get("checking", {}).get("checking_level", "")
    except Exception as e:
        log(f"{iso}: metadata fetch failed ({e}) — leaving license/checking_level blank", "WARN")
        license_ = ""
        checking_level = ""

    return {
        "iso": iso,
        "source": entry["owner"],
        "source_repo": full_name,
        "license": license_,
        "checking_level": checking_level,
        "content_base_url": f"https://git.door43.org/{full_name}/raw/branch/{target_commitish}/content",
        "stories": stories,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iso", type=str, default=None, help="Stage only this language")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be staged, write nothing")
    args = parser.parse_args()

    log("Fetching Door43 OBS catalog (subject=Open Bible Stories, hasAudio=true)...")
    entries = fetch_catalog_entries()
    log(f"{len(entries)} language(s) found in catalog")

    if args.iso:
        entries = [e for e in entries if e["language"] == args.iso]
        if not entries:
            log(f"No catalog entry for iso={args.iso!r}", "ERROR")
            sys.exit(1)

    OBS_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    staged, skipped, failed = 0, 0, 0

    for entry in entries:
        iso = entry["language"]
        out_path = OBS_BATCH_DIR / f"{iso}.json"
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        manifest = build_manifest(entry)
        if manifest is None:
            failed += 1
            continue

        if args.dry_run:
            log(f"[dry-run] would stage {iso} ({entry['full_name']}): {len(manifest['stories'])} stories")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            log(f"Staged {iso} ({entry['full_name']}): {len(manifest['stories'])} stories -> {out_path}")
        staged += 1

    log(f"Done — {staged} staged, {skipped} already existed (use --force to overwrite), {failed} had no usable audio")


if __name__ == "__main__":
    main()
