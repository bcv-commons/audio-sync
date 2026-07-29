#!/usr/bin/env python3
"""
Fetch Bible audio/text/timing content from DBT (+ helloAO fallback) for a
single resolved batch_manifest.py job.

This is the fetch half of what used to be a combined catalog+fetch script.
The "what to align" decision (which language/version/chapters) is made
upstream by core (bible-story-builder) and handed to this repo as a batch
manifest job — see batch_manifest.py. This file no longer does any of its
own template/language/exclusion scanning; it only knows how to pull DBT
filesets (and helloAO external text) down to disk for a job it's given.

Usage:
    # Fetch every job in a batch manifest (BATCH_ID env var or --batch-id)
    BATCH_ID=<id> python download_language_content.py
    python download_language_content.py --batch-id <id> --content-types audio,text

    # Call the fetch primitive directly from another script (preferred):
    from download_language_content import download_job
    download_job({"iso": "eng", "canon": "NT", "distinct_id": "ENGKJV",
                  "chapters": {"MAT": [1, 2, 3]}})

Prerequisites:
    1. Set BIBLE_API_KEY in .env file

Fileset resolution for unenriched jobs (no audio_fileset/text_fileset given)
comes from the `bibles` repo's published DBT catalogs on CDN — see
get_best_fileset_from_catalog() and its module comment. This replaced the
old local sorted/BB scan (2026-07-29): sorted/BB required running MONO's
sort_cache_data.py locally and was never populated in this repo; the CDN
catalogs need no local generation step and are validated against the live
DBT API at publish time.

Output:
    downloads/BB/{canon}/{iso}/{distinct_id}/{BOOK}/
        {BOOK}_{CHAPTER:03d}_{FILESET_ID}.{ext}
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print("Error: Required packages not installed.")
    print("Please run: pip install -r requirements.txt")
    print(f"Missing module: {e.name}")
    sys.exit(1)

from batch_manifest import load_batch, get_jobs

# Load environment variables
load_dotenv()

# API Configuration
BIBLE_API_KEY = os.getenv("BIBLE_API_KEY", "")
BIBLE_API_BASE_URL = "https://4.dbt.io/api"
API_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60
API_RATE_DELAY = 0.0  # seconds between API calls, set via --rate-delay

# Directories
OUTPUT_DIR = Path("downloads/BB")
ERROR_LOG_DIR = Path("download_log")
# CROSSREF_PATH / _load_crossref() below is MONO's own naming-convention
# ebible crossref — same unverified-guessing category as the old helloAO ID
# matching (see _find_helloao_id's history). It's only used as a last-resort
# ebible_id lookup in _get_external_text_source() and has no catalog-overlap
# equivalent yet (catalog-overlap.json only covers dbt/helloao/pkf, no
# ebible ids) — still a real gap, not superseded.
CROSSREF_PATH = Path("data/version-crossref.json")
HELLOAO_API = "https://bible.helloao.org/api"

# `bibles` repo's published DBT catalogs (2026-07-29) — replace the old
# local sorted/BB fileset scan and the old {iso}_{version} naming-guess for
# helloAO matching. See get_best_fileset_from_catalog() / _find_helloao_id().
DBT_CATALOG_CDN_BASE = "https://cdn.bibel.wiki/dbt/_app/"
DBT_CATALOG_CACHE_DIR = Path("api-cache/dbt-catalog")
_dbt_catalog_cache: Dict[str, dict] = {}

_crossref_cache = None

def _load_crossref():
    """Load version crossref data (cached)."""
    global _crossref_cache
    if _crossref_cache is None:
        if CROSSREF_PATH.exists():
            with open(CROSSREF_PATH) as f:
                _crossref_cache = json.load(f)
        else:
            _crossref_cache = {}
    return _crossref_cache

_versions_cache = {}
VERSIONS_DIR = Path("export/versions-data")

def _load_versions_data(iso):
    """Load versions.json for a language (cached)."""
    if iso not in _versions_cache:
        vf = VERSIONS_DIR / iso / "versions.json"
        if vf.exists():
            with open(vf) as f:
                _versions_cache[iso] = json.load(f)
        else:
            _versions_cache[iso] = {}
    return _versions_cache[iso]

def _extract_version_id(iso, distinct_id):
    """Extract version ID from distinct_id by stripping ISO prefix.
    E.g., FRALSN -> LSN, TPIPNG -> PNG, ENGKJV -> KJV"""
    iso_upper = iso.upper()
    # Try 3-char ISO prefix (standard)
    if distinct_id.upper().startswith(iso_upper) and len(distinct_id) > len(iso_upper):
        return distinct_id[len(iso_upper):]
    # Try FRN-style prefix (French uses FRN not FRA for some filesets)
    # Check versions.json keys directly
    versions = _load_versions_data(iso)
    for vid in versions:
        if vid.upper() == distinct_id.upper() or distinct_id.upper().endswith(vid.upper()):
            return vid
    # Fallback: return as-is
    return distinct_id


def _load_dbt_catalog(name: str) -> dict:
    """Fetch+cache one of the `bibles` repo's published DBT catalogs.

    name is one of "catalog-text", "catalog-audio", "catalog-overlap"
    (cdn.bibel.wiki/dbt/_app/<name>.json). Checks the on-disk cache first,
    then fetches from CDN and caches the raw response — same
    fetch-then-cache pattern as batch_manifest.py's queue tiers. Returns
    {} (not an exception) on any fetch/parse failure so callers degrade to
    "no match found" rather than crashing a whole batch over one lookup.
    """
    if name in _dbt_catalog_cache:
        return _dbt_catalog_cache[name]

    cache_path = DBT_CATALOG_CACHE_DIR / f"{name}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            _dbt_catalog_cache[name] = data
            return data
        except (json.JSONDecodeError, IOError):
            pass

    import urllib.request

    url = f"{DBT_CATALOG_CDN_BASE}{name}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "audio-sync"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        data = json.loads(raw)
    except Exception as e:
        log(f"Failed to fetch {url}: {e}", "WARNING")
        data = {}
        _dbt_catalog_cache[name] = data
        return data

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    _dbt_catalog_cache[name] = data
    return data


def _catalog_entries(iso: str, canon: str, distinct_id: str, catalog_name: str) -> list:
    """Look up one distinct_id's entries in catalog-text.json/catalog-audio.json.

    Tries the full canon key (nt/ot) first, then the portion variant
    (ntp/otp) for languages whose only DBT coverage is a partial testament.
    """
    catalog = _load_dbt_catalog(catalog_name).get("entries", {})
    canon_l = canon.lower()
    for key in (f"{iso}:{canon_l}", f"{iso}:{canon_l}p"):
        by_distinct_id = catalog.get(key, {})
        if distinct_id in by_distinct_id:
            return by_distinct_id[distinct_id]
    return []


def get_best_fileset_from_catalog(iso: str, canon: str, distinct_id: str) -> Optional[Dict]:
    """
    Resolve the best audio/text fileset for (iso, canon, distinct_id) from
    the `bibles` repo's catalog-text.json/catalog-audio.json — the CDN
    replacement for the old local sorted/BB scan.

    Each catalog entry stores only the *suffix* after distinct_id (e.g.
    "id": "a:N2DA" under distinct_id "ENGKJV" reconstructs to fileset id
    "ENGKJVN2DA"); the bibles team validated this reconstruction against
    the live DBT API at publish time, and it round-tripped independently
    against a couple of samples used here too (e.g. AAIWBT, AAAMLTN_ET).

    Canon-level only — catalog entries don't carry DBT's per-book size/
    coverage field the way sorted/BB metadata did, so a fileset picked
    here is assumed to cover every book in canon. That matches how batch
    jobs are already scoped (core resolves iso/canon/distinct_id, not
    individual books, before chapters are handed to this job).

    Returns the same shape as the old get_best_fileset_for_book(), minus
    per-book text_fileset variation.
    """
    audio_entries = _catalog_entries(iso, canon, distinct_id, "catalog-audio")
    text_entries = _catalog_entries(iso, canon, distinct_id, "catalog-text")

    if not audio_entries and not text_entries:
        return None

    # Audio priority (lower tuple = preferred), same ordering as the old
    # sorted/BB logic: non-dramatized > dramatized, non-streaming >
    # streaming, mp3 > opus.
    audio_fileset = None
    alt_audio_fileset = None
    audio_candidates = []
    for e in audio_entries:
        suffix = e["id"][2:]  # strip "a:" prefix
        fid = f"{distinct_id}{suffix}"
        is_dramatized = "2DA" in fid or "2SA" in fid
        is_stream = fid.endswith("SA")
        is_opus = e.get("c") == "opus"
        priority = (int(is_dramatized), int(is_stream), int(is_opus), fid)
        audio_candidates.append((priority, fid, is_dramatized))
    if audio_candidates:
        audio_candidates.sort()
        audio_fileset = audio_candidates[0][1]
        primary_is_drama = audio_candidates[0][2]
        for _, fid, is_drama in audio_candidates[1:]:
            if is_drama != primary_is_drama and "-opus" not in fid:
                alt_audio_fileset = fid
                break

    # Text priority: plain > json > other. USX dropped entirely — same
    # reasoning as the old code: DBT's USX endpoint often 404s even when
    # the fileset is listed.
    text_candidates = []
    for e in text_entries:
        fmt = e.get("fmt", [])
        if fmt == ["u"]:
            continue
        suffix = e["id"][2:]  # strip "t:" prefix
        fid = f"{distinct_id}{suffix}"
        base = 0 if "pl" in fmt else 1 if "j" in fmt else 2
        text_candidates.append(((base, fid), fid))
    text_candidates.sort()
    text_fileset_candidates = [fid for _, fid in text_candidates]
    text_fileset = text_fileset_candidates[0] if text_fileset_candidates else None

    if not audio_fileset and not text_fileset:
        return None

    result = {
        "distinct_id": distinct_id,
        "canon": canon,
        "audio_fileset": audio_fileset,
        "text_fileset": text_fileset,
        "text_fileset_candidates": text_fileset_candidates,
        # Not carried by catalog-{text,audio}.json — timing existence is
        # discovered per-chapter by download_timing()'s own 404 handling,
        # same as the enriched-manifest path already does.
        "timing_available": False,
    }
    if alt_audio_fileset:
        result["alt_audio_fileset"] = alt_audio_fileset
    return result


def _find_helloao_id(iso: str, canon: str, distinct_id: str) -> Optional[str]:
    """Find a *verified* helloAO translation match via catalog-overlap.json.

    This is real text comparison done by the `bibles` repo, not naming-
    convention guessing — replaces the old {iso}_{version_id} pattern
    match, which the bibles team flagged as unreliable and dropped from
    their own equivalent tooling for the same reason.
    """
    catalog = _load_dbt_catalog("catalog-overlap").get("entries", {})
    canon_l = canon.lower()
    for key in (f"{iso}:{canon_l}", f"{iso}:{canon_l}p"):
        for group in catalog.get(key, []):
            ids = group.get("ids", [])
            if f"d:{distinct_id}" not in ids:
                continue
            for i in ids:
                if i.startswith("h:"):
                    return i[2:]
    return None


def _fetch_helloao_chapter(helloao_id, book, chapter_num, dest_path):
    """Fetch a chapter's text from helloAO API and write as ET-format .txt file."""
    import urllib.request

    if dest_path.exists():
        return True

    url = f"{HELLOAO_API}/{helloao_id}/{book}/{chapter_num}.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        log(f"  helloAO fetch failed {book} {chapter_num}: {e}", "WARN")
        return False

    content = data.get("chapter", {}).get("content", [])
    verses = []
    for item in content:
        if item.get("type") == "verse":
            text_parts = []
            for part in item.get("content", []):
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            verses.append(" ".join(text_parts))

    if not verses:
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text("\n".join(verses) + "\n", encoding="utf-8")
    return True


def _get_external_text_source(iso, distinct_id):
    """Check versions.json for external text source when DBT has no text.
    Returns (source_type, source_id) or (None, None)."""
    vid = _extract_version_id(iso, distinct_id)
    versions = _load_versions_data(iso)
    v_entry = versions.get(vid, {})

    for tkey in ("nt", "ot"):
        sources = v_entry.get(tkey, {})
        dbt = sources.get("dbt", "")
        if "a" in dbt and "t" not in dbt:
            if sources.get("helloao") == "t":
                hao_id = _find_helloao_id(iso, tkey, distinct_id)
                if hao_id:
                    return "helloao", hao_id
            if sources.get("ebible") == "t":
                ebible_id = sources.get("ebible_id") or v_entry.get("ebible_id")
                if not ebible_id:
                    # Derive from crossref
                    crossref = _load_crossref()
                    lang_xref = crossref.get(iso, {})
                    for xvid, info in lang_xref.items():
                        if info.get("ebible"):
                            ebible_id = info["ebible"]
                            break
                if ebible_id:
                    return "ebible", ebible_id
    return None, None


def _write_source_json(base_dir, audio_source, text_source):
    """Write source.json tracking where audio and text came from."""
    source_path = base_dir / "source.json"
    source_data = {}
    if source_path.exists():
        try:
            with open(source_path) as f:
                source_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    if audio_source:
        source_data["audio"] = audio_source
    if text_source:
        source_data["text"] = text_source

    with open(source_path, "w") as f:
        json.dump(source_data, f, indent=2)
        f.write("\n")

# Book mappings
OT_BOOKS = {
    "GEN": 50,
    "EXO": 40,
    "LEV": 27,
    "NUM": 36,
    "DEU": 34,
    "JOS": 24,
    "JDG": 21,
    "RUT": 4,
    "1SA": 31,
    "2SA": 24,
    "1KI": 22,
    "2KI": 25,
    "1CH": 29,
    "2CH": 36,
    "EZR": 10,
    "NEH": 13,
    "EST": 10,
    "JOB": 42,
    "PSA": 150,
    "PRO": 31,
    "ECC": 12,
    "SNG": 8,
    "ISA": 66,
    "JER": 52,
    "LAM": 5,
    "EZK": 48,
    "DAN": 12,
    "HOS": 14,
    "JOL": 3,
    "AMO": 9,
    "OBA": 1,
    "JON": 4,
    "MIC": 7,
    "NAM": 3,
    "HAB": 3,
    "ZEP": 3,
    "HAG": 2,
    "ZEC": 14,
    "MAL": 4,
}

NT_BOOKS = {
    "MAT": 28,
    "MRK": 16,
    "LUK": 24,
    "JHN": 21,
    "ACT": 28,
    "ROM": 16,
    "1CO": 16,
    "2CO": 13,
    "GAL": 6,
    "EPH": 6,
    "PHP": 4,
    "COL": 4,
    "1TH": 5,
    "2TH": 3,
    "1TI": 6,
    "2TI": 4,
    "TIT": 3,
    "PHM": 1,
    "HEB": 13,
    "JAS": 5,
    "1PE": 5,
    "2PE": 3,
    "1JN": 5,
    "2JN": 1,
    "3JN": 1,
    "JUD": 1,
    "REV": 22,
}

# Statistics tracking
class DownloadStats:
    def __init__(self):
        self.downloaded_from_api = 0
        self.already_exists = 0
        self.failed = 0

    def report(self):
        total = self.downloaded_from_api + self.already_exists
        print("\nDownload Statistics:")
        print(f"  Already exists:      {self.already_exists}")
        print(f"  Downloaded from API: {self.downloaded_from_api}")
        print(f"  Failed:              {self.failed}")
        print(f"  Total processed:     {total}")


stats = DownloadStats()


# Error logging
class ErrorLogger:
    def __init__(self):
        # Structure: {iso: {canon: {(book, chapter): {errors}}}}
        self.errors_by_language = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {"audio_errors": [], "text_errors": [], "timing_errors": []}
                )
            )
        )

    def log_error(
        self,
        iso: str,
        canon: str,
        book: str,
        chapter: int,
        error_type: str,
        content_type: str,
        fileset: str,
        distinct_id: str,
        details: str,
    ):
        """Log an error for a specific download attempt."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "fileset": fileset,
            "distinct_id": distinct_id,
            "details": details,
        }

        chapter_key = (book, chapter)
        error_list_key = f"{content_type}_errors"
        self.errors_by_language[iso][canon][chapter_key][error_list_key].append(
            error_entry
        )

    def save_logs(self):
        """Save error logs to JSON files organized by canon."""
        if not self.errors_by_language:
            return

        for iso, canons in self.errors_by_language.items():
            for canon, chapters in canons.items():
                # Create directory: download_log/{canon}/{iso}/
                log_dir = ERROR_LOG_DIR / canon.lower() / iso
                log_dir.mkdir(parents=True, exist_ok=True)

                # File: {canon}-{iso}-error.json
                log_file = log_dir / f"{canon.lower()}-{iso}-error.json"

                # Load existing errors if file exists
                existing_data = {"language": iso, "canon": canon, "errors": []}
                if log_file.exists():
                    try:
                        with open(log_file, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                    except json.JSONDecodeError:
                        pass

                # Merge new errors
                for (book, chapter), errors in chapters.items():
                    # Check if this book/chapter already has errors
                    existing_entry = None
                    for entry in existing_data["errors"]:
                        if (
                            entry.get("book") == book
                            and entry.get("chapter") == chapter
                        ):
                            existing_entry = entry
                            break

                    if existing_entry:
                        # Append to existing errors
                        existing_entry["audio_errors"].extend(errors["audio_errors"])
                        existing_entry["text_errors"].extend(errors["text_errors"])
                        existing_entry["timing_errors"].extend(errors["timing_errors"])
                        existing_entry["timestamp"] = datetime.now().isoformat()
                    else:
                        # Add new entry
                        existing_data["errors"].append(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "book": book,
                                "chapter": chapter,
                                "audio_errors": errors["audio_errors"],
                                "text_errors": errors["text_errors"],
                                "timing_errors": errors["timing_errors"],
                            }
                        )

                # Update last_updated timestamp
                existing_data["last_updated"] = datetime.now().isoformat()

                # Save to file
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)


error_logger = ErrorLogger()


def log(message: str, level: str = "INFO"):
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)




def determine_book_canon(book: str) -> str:
    """Determine which canon a book belongs to."""
    if book in OT_BOOKS:
        return "OT"
    elif book in NT_BOOKS:
        return "NT"
    return "UNKNOWN"


# Module-level record of the most recent API failure, exposed so callers
# (like download_text) can classify "no data" results into 403 vs 404 vs
# empty-response instead of lumping them all into "no_text_available".
_LAST_API_ERROR: Optional[Dict[str, object]] = None


def _classify_api_failure() -> str:
    """Return a short error_type tag for the most recent API failure.

    "empty_data"        — HTTP 200 but data was empty / unexpected shape
    "http_403_forbidden" — DBT returned 403 (usually access-restricted bible)
    "http_404_not_found" — DBT returned 404 (chapter genuinely absent)
    "http_error"        — any other HTTP/network error
    """
    err = _LAST_API_ERROR
    if not err:
        return "empty_data"
    code = err.get("status_code")
    if code == 403:
        return "http_403_forbidden"
    if code == 404:
        return "http_404_not_found"
    return "http_error"


def make_api_request(
    endpoint: str, params: Optional[Dict] = None, use_key_param: bool = False
) -> Optional[Dict]:
    """Make API request with error handling.

    Args:
        endpoint: API endpoint path
        params: Query parameters
        use_key_param: If True, use 'key' query param instead of Bearer token (for timing endpoint)
    """
    if not BIBLE_API_KEY:
        log("BIBLE_API_KEY not set in .env file", "ERROR")
        return None

    url = f"{BIBLE_API_BASE_URL}/{endpoint}"
    request_params = params or {}

    if use_key_param:
        # Some endpoints (like timestamps) require key as query param, not Bearer token
        request_params["key"] = BIBLE_API_KEY
        request_params["v"] = "4"
        headers = {}
    else:
        # Most endpoints use Bearer token
        headers = {"Authorization": f"Bearer {BIBLE_API_KEY}"}

    global _LAST_API_ERROR
    _LAST_API_ERROR = None
    try:
        response = requests.get(
            url, headers=headers, params=request_params, timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        _LAST_API_ERROR = {"status_code": status, "url": url, "message": str(e)}
        log(f"API request failed: {e}", "ERROR")
        return None
    except requests.RequestException as e:
        _LAST_API_ERROR = {"status_code": None, "url": url, "message": str(e)}
        log(f"API request failed: {e}", "ERROR")
        return None


def get_audio_path(fileset_id: str, book: str, chapter: int) -> Optional[str]:
    """Get audio file path from API."""
    # Use the correct endpoint format: /bibles/filesets/{fileset_id}/{book}/{chapter}
    endpoint = f"bibles/filesets/{fileset_id}/{book}/{chapter}"

    # This endpoint requires key as query param, not Bearer token
    data = make_api_request(endpoint, use_key_param=True)

    if not data or "data" not in data or not data["data"]:
        return None

    return data["data"][0].get("path")


def get_text_content(fileset_id: str, book: str, chapter: int) -> Optional[dict]:
    """Get text content from API.

    Returns dict with either:
    - {'type': 'path', 'data': url} for JSON/USX filesets with downloadable files
    - {'type': 'verses', 'data': [verse_data]} for plain text filesets with inline verses
    """
    # Use the correct endpoint format: /bibles/filesets/{fileset_id}/{book}/{chapter}
    endpoint = f"bibles/filesets/{fileset_id}/{book}/{chapter}"

    # This endpoint requires key as query param, not Bearer token
    data = make_api_request(endpoint, use_key_param=True)

    if not data or "data" not in data or not data["data"]:
        return None

    first_item = data["data"][0]

    # Check if this is a downloadable file (JSON/USX format)
    if "path" in first_item:
        return {"type": "path", "data": first_item["path"]}

    # Check if this is inline verse data (plain text format)
    elif "verse_text" in first_item:
        return {"type": "verses", "data": data["data"]}

    return None


def get_timing_data(fileset_id: str, book: str, chapter: int) -> Optional[Dict]:
    """Get timing data from API for a specific chapter."""
    # Normalize fileset ID - timing API doesn't work with suffixes like -opus16
    base_fileset_id = normalize_fileset_id(fileset_id)
    endpoint = f"timestamps/{base_fileset_id}/{book}/{chapter}"

    if API_RATE_DELAY > 0:
        time.sleep(API_RATE_DELAY)

    # Timing endpoint requires key as query param, not Bearer token
    data = make_api_request(endpoint, use_key_param=True)
    if not data:
        return None

    if "error" in data:
        return None

    if "data" in data:
        timing_data = data["data"]
        # Check if data array is not empty
        if timing_data and len(timing_data) > 0:
            return timing_data
        return None

    return None


def normalize_fileset_id(fileset_id: str) -> str:
    """Remove format suffixes for API calls.

    This is used for API calls that don't accept format suffixes.

    Examples:
        AAAMLTN1DA-opus16 -> AAAMLTN1DA
        ENGESV_ET-json -> ENGESV_ET
    """
    # Remove audio format suffixes
    audio_suffixes = ["-opus16", "-opus32", "-mp3-64", "-mp3-128", "-mp3"]
    for suffix in audio_suffixes:
        if fileset_id.endswith(suffix):
            return fileset_id[: -len(suffix)]

    # Remove text format suffixes
    text_suffixes = ["-json", "-usx", "-html"]
    for suffix in text_suffixes:
        if fileset_id.endswith(suffix):
            return fileset_id[: -len(suffix)]

    return fileset_id


def download_audio(
    fileset_id: str,
    book: str,
    chapter: int,
    output_path: Path,
    iso: str,
    distinct_id: str,
    stats: DownloadStats,
    error_logger: ErrorLogger,
) -> bool:
    """Download audio file."""
    audio_path = get_audio_path(fileset_id, book, chapter)
    if not audio_path:
        # Determine canon from book
        canon = determine_book_canon(book)
        error_type = _classify_api_failure().replace("empty_data", "no_audio_available")
        err = _LAST_API_ERROR or {}
        status = err.get("status_code")
        cause = f" (HTTP {status})" if status else ""
        error_logger.log_error(
            iso,
            canon,
            book,
            chapter,
            error_type=error_type,
            content_type="audio",
            fileset=fileset_id,
            distinct_id=distinct_id,
            details=(
                f"No audio path for fileset_id={fileset_id} "
                f"(distinct_id={distinct_id}, book={book}, chapter={chapter}){cause}"
            ),
        )
        stats.failed += 1
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # HLS streams (.m3u8) need ffmpeg to download and convert to MP3
    if ".m3u8" in audio_path:
        # Append API key to the m3u8 URL for authentication
        sep = "&" if "?" in audio_path else "?"
        hls_url = f"{audio_path}{sep}key={BIBLE_API_KEY}&v=4"
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", hls_url, "-c:a", "libmp3lame", "-q:a", "2",
                 str(output_path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                log(f"  ✗ ffmpeg failed for HLS stream: {result.stderr[-200:]}", "ERROR")
                canon = determine_book_canon(book)
                error_logger.log_error(
                    iso, canon, book, chapter,
                    error_type="hls_download_failed",
                    content_type="audio",
                    fileset=fileset_id,
                    distinct_id=distinct_id,
                    details=f"ffmpeg HLS download failed for fileset_id={fileset_id}: {result.stderr[-500:]}",
                )
                stats.failed += 1
                return False
            log(f"  ✓ Downloaded (HLS): {output_path.name}", "INFO")
            stats.downloaded_from_api += 1
            return True
        except FileNotFoundError:
            log("  ✗ ffmpeg not found. Install with: brew install ffmpeg", "ERROR")
            stats.failed += 1
            return False
        except subprocess.TimeoutExpired:
            log(f"  ✗ ffmpeg timed out downloading HLS stream", "ERROR")
            stats.failed += 1
            return False

    try:
        response = requests.get(audio_path, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        log(f"  ✓ Downloaded: {output_path.name}", "INFO")
        stats.downloaded_from_api += 1
        return True
    except requests.RequestException as e:
        log(f"  ✗ Failed to download audio: {e}", "ERROR")
        canon = determine_book_canon(book)
        error_logger.log_error(
            iso,
            canon,
            book,
            chapter,
            error_type="download_failed",
            content_type="audio",
            fileset=fileset_id,
            distinct_id=distinct_id,
            details=f"Audio download failed for fileset_id={fileset_id}: {str(e)}",
        )
        stats.failed += 1
        return False


def download_text(
    fileset_id: str,
    book: str,
    chapter: int,
    output_path: Path,
    iso: str,
    distinct_id: str,
    stats: DownloadStats,
    error_logger: ErrorLogger,
) -> bool:
    """Download text file."""
    text_content = get_text_content(fileset_id, book, chapter)
    if not text_content:
        canon = determine_book_canon(book)
        error_type = _classify_api_failure()
        err = _LAST_API_ERROR or {}
        status = err.get("status_code")
        cause = f" (HTTP {status})" if status else ""
        error_logger.log_error(
            iso,
            canon,
            book,
            chapter,
            error_type=error_type,
            content_type="text",
            fileset=fileset_id,
            distinct_id=distinct_id,
            details=(
                f"No text content for fileset_id={fileset_id} "
                f"(distinct_id={distinct_id}, book={book}, chapter={chapter}){cause}"
            ),
        )
        stats.failed += 1
        return False

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if text_content["type"] == "path":
            # Download file from path (JSON/USX format)
            response = requests.get(text_content["data"], timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)

        elif text_content["type"] == "verses":
            # Extract verse text from inline data (plain text format)
            verses = []
            for verse_data in text_content["data"]:
                verse_text = verse_data.get("verse_text", "")
                if verse_text:
                    verses.append(verse_text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(verses))

        log(f"  ✓ Downloaded: {output_path.name}", "INFO")
        stats.downloaded_from_api += 1
        return True
    except requests.RequestException as e:
        log(f"  ✗ Failed to download text: {e}", "ERROR")
        canon = determine_book_canon(book)
        error_logger.log_error(
            iso,
            canon,
            book,
            chapter,
            error_type="download_failed",
            content_type="text",
            fileset=fileset_id,
            distinct_id=distinct_id,
            details=f"Text download failed for fileset_id={fileset_id}: {str(e)}",
        )
        stats.failed += 1
        return False
    except Exception as e:
        log(f"  ✗ Failed to save text: {e}", "ERROR")
        canon = determine_book_canon(book)
        error_logger.log_error(
            iso,
            canon,
            book,
            chapter,
            error_type="save_failed",
            content_type="text",
            fileset=fileset_id,
            distinct_id=distinct_id,
            details=f"Text save failed for fileset_id={fileset_id}: {str(e)}",
        )
        stats.failed += 1
        return False


def download_timing(
    fileset_id: str,
    book: str,
    chapter: int,
    output_path: Path,
    iso: str,
    distinct_id: str,
    stats: DownloadStats,
    error_logger: ErrorLogger,
) -> bool:
    """Download timing file."""
    timing_data = get_timing_data(fileset_id, book, chapter)
    if not timing_data:
        canon = determine_book_canon(book)
        error_type = _classify_api_failure().replace("empty_data", "no_timing_available")
        err = _LAST_API_ERROR or {}
        status = err.get("status_code")
        cause = f" (HTTP {status})" if status else ""
        error_logger.log_error(
            iso,
            canon,
            book,
            chapter,
            error_type=error_type,
            content_type="timing",
            fileset=fileset_id,
            distinct_id=distinct_id,
            details=(
                f"No timing data for fileset_id={fileset_id} "
                f"(distinct_id={distinct_id}, book={book}, chapter={chapter}){cause}"
            ),
        )
        stats.failed += 1
        return False

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(timing_data, f, indent=2)

        log(f"  ✓ Downloaded: {output_path.name}", "INFO")
        stats.downloaded_from_api += 1
        return True
    except Exception as e:
        log(f"  ✗ Failed to save timing: {e}", "ERROR")
        canon = determine_book_canon(book)
        error_logger.log_error(
            iso,
            canon,
            book,
            chapter,
            error_type="save_failed",
            content_type="timing",
            fileset=fileset_id,
            distinct_id=distinct_id,
            details=f"Failed to save timing data: {str(e)}",
        )
        stats.failed += 1
        return False


def download_chapter(
    iso: str,
    distinct_id: str,
    canon: str,
    book: str,
    chapter: int,
    audio_fileset: Optional[str],
    text_fileset: Optional[str],
    timing_available: bool,
    force: bool = False,
    content_types: Optional[List[str]] = None,
    alt_audio_fileset: Optional[str] = None,
    text_fileset_candidates: Optional[List[str]] = None,
    text_source_override: Optional[str] = None,
) -> bool:
    """
    Download content for a specific chapter based on requested content types.

    Args:
        content_types: List of content types to download ('audio', 'text', 'timing').
                      If None, downloads all available content types (default behavior).
        text_source_override: "helloao:<id>" or "ebible:<id>" — when text_fileset
            is None, use this instead of calling _get_external_text_source()
            (which needs local version-crossref.json/versions-data catalog
            data). Lets a caller that already knows the resolved external
            text source (e.g. an enriched batch_manifest.py job) skip that
            catalog lookup entirely.

    Returns True if all required downloads succeeded or already exist, False otherwise.
    """
    # Default to all content types if not specified (backward compatibility)
    if content_types is None:
        content_types = ["audio", "text", "timing"]
    # Output structure: downloads/BB/{canon}/{iso}/{distinct_id}/{book}/
    base_dir = OUTPUT_DIR / canon.lower() / iso / distinct_id / book
    base_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # Download audio (if requested)
    if audio_fileset and "audio" in content_types:
        audio_file = base_dir / f"{book}_{chapter:03d}_{audio_fileset}.mp3"
        if audio_file.exists() and not force:
            log(f"  ⊙ Already exists: {audio_file.name}", "INFO")
            stats.already_exists += 1
        else:
            if not download_audio(
                audio_fileset,
                book,
                chapter,
                audio_file,
                iso,
                distinct_id,
                stats,
                error_logger,
            ):
                success = False

    # Download alt audio (if available and requested)
    if alt_audio_fileset and "audio" in content_types:
        alt_audio_file = base_dir / f"{book}_{chapter:03d}_{alt_audio_fileset}.mp3"
        if alt_audio_file.exists() and not force:
            log(f"  ⊙ Already exists: {alt_audio_file.name} (alt)", "INFO")
            stats.already_exists += 1
        else:
            if not download_audio(
                alt_audio_fileset,
                book,
                chapter,
                alt_audio_file,
                iso,
                distinct_id,
                stats,
                error_logger,
            ):
                pass  # Alt audio failure is not critical

    # Download text (if requested) — try candidates in priority order so that
    # an upstream 404 on one variant falls back to the next (e.g. text_format
    # when text_plain is missing).
    text_source_tag = None
    if text_fileset and "text" in content_types:
        # Build the ordered candidate list. text_fileset is always tried first;
        # additional candidates fall through if the first one fails.
        candidates = list(text_fileset_candidates) if text_fileset_candidates else [text_fileset]
        if text_fileset not in candidates:
            candidates = [text_fileset] + candidates

        # Skip work entirely if any candidate is already on disk
        existing = next(
            (c for c in candidates if (base_dir / f"{book}_{chapter:03d}_{c}.txt").exists()),
            None,
        )
        if existing and not force:
            log(f"  ⊙ Already exists: {book}_{chapter:03d}_{existing}.txt", "INFO")
            stats.already_exists += 1
            text_source_tag = f"dbt:{existing}"
        else:
            text_downloaded = False
            for cand in candidates:
                text_file = base_dir / f"{book}_{chapter:03d}_{cand}.txt"
                if download_text(
                    cand, book, chapter, text_file,
                    iso, distinct_id, stats, error_logger,
                ):
                    if cand != text_fileset:
                        log(f"  Used fallback text fileset: {cand}", "INFO")
                    text_source_tag = f"dbt:{cand}"
                    text_downloaded = True
                    break
            if not text_downloaded:
                success = False
    elif not text_fileset and audio_fileset and "text" in content_types:
        # No DBT text — use the caller-supplied external source if given
        # (skips the version-crossref.json/versions-data catalog lookup
        # entirely), otherwise fall back to resolving it ourselves.
        if text_source_override:
            ext_type, _, ext_id = text_source_override.partition(":")
        else:
            ext_type, ext_id = _get_external_text_source(iso, distinct_id)

        if ext_type == "helloao" and ext_id:
            # Generate filename using helloAO convention
            hao_fid = ext_id.replace("_", "").upper()
            text_file = base_dir / f"{book}_{chapter:03d}_{hao_fid}_ET.txt"
            if text_file.exists() and not force:
                log(f"  ⊙ Already exists: {text_file.name} (helloAO)", "INFO")
                stats.already_exists += 1
                text_source_tag = f"helloao:{ext_id}"
            else:
                if _fetch_helloao_chapter(ext_id, book, chapter, text_file):
                    log(f"  ✓ Downloaded: {text_file.name} (helloAO)", "INFO")
                    stats.downloaded_from_api += 1
                    text_source_tag = f"helloao:{ext_id}"
                else:
                    success = False
        elif ext_type == "ebible" and ext_id:
            # No eBible fetcher exists in this codebase yet — fail loudly
            # rather than silently reporting success with no text fetched.
            log(f"  eBible text fetch not implemented (id={ext_id}) — "
                f"{book} {chapter} has no text", "ERROR")
            success = False

    # Download timing (if requested and available)
    if timing_available and audio_fileset and "timing" in content_types:
        timing_file = base_dir / f"{book}_{chapter:03d}_{audio_fileset}_timing.json"
        if timing_file.exists() and not force:
            log(f"  ⊙ Already exists: {timing_file.name}", "INFO")
            stats.already_exists += 1
        else:
            if not download_timing(
                audio_fileset,
                book,
                chapter,
                timing_file,
                iso,
                distinct_id,
                stats,
                error_logger,
            ):
                success = False

    # Write source.json if we have source info
    audio_source_tag = f"dbt:{audio_fileset}" if audio_fileset else None
    if audio_source_tag or text_source_tag:
        _write_source_json(base_dir, audio_source_tag, text_source_tag)

    return success


def download_job(
    job: Dict,
    content_types: Optional[List[str]] = None,
    force: bool = False,
) -> bool:
    """
    Fetch audio/text/timing for one resolved batch_manifest.py job.

    job shape (see batch_manifest.py), since 2026-07-15 enriched with the
    exact resolved fileset(s) — core resolves these against its own DBT
    catalog, so a job carrying them needs no local sorted/BB catalog data:
        {"iso": "eng", "canon": "NT", "distinct_id": "ENGKJV",
         "chapters": {"MAT": [1, 2, 3]},
         "audio_fileset": "ENGKJVN2DA",       # optional, enriched manifests
         "text_fileset": "ENGKJVN_ET",        # optional; null if DBT has no text
         "text_source": null}                 # optional; "helloao:<id>" / "ebible:<id>"
                                               # when text_fileset is null

    No selection or exclusion logic runs here — core has already resolved
    which language/version/chapters belong in the job before it reached
    this queue.

    Returns True if every requested chapter downloaded (or already existed)
    successfully, False if any chapter failed.
    """
    iso = job["iso"]
    canon = job["canon"].upper()
    distinct_id = job["distinct_id"]
    chapters_by_book = job.get("chapters", {})

    enriched_audio_fileset = job.get("audio_fileset")
    enriched_text_fileset = job.get("text_fileset")
    enriched_text_source = job.get("text_source")

    if enriched_audio_fileset:
        # Enriched manifest — core already resolved the exact fileset(s) for
        # this job, so no catalog resolution (sorted/BB) is needed at all.
        overall_success = True
        for book, chapters in chapters_by_book.items():
            for chapter in chapters:
                chapter_ok = download_chapter(
                    iso,
                    distinct_id,
                    canon,
                    book,
                    chapter,
                    enriched_audio_fileset,
                    enriched_text_fileset,
                    timing_available=False,  # not carried by the enriched
                    # manifest; harmless — the only current caller
                    # (download_audio_for_chapters) never requests
                    # content_types including "timing".
                    force=force,
                    content_types=content_types,
                    text_source_override=enriched_text_source,
                )
                if not chapter_ok:
                    overall_success = False
        return overall_success

    # Fallback: DBT catalog-based resolution (cdn.bibel.wiki/dbt/_app/
    # catalog-{text,audio}.json), for manifests missing per-job fileset
    # enrichment. Canon-level, so resolved once per job rather than per book
    # (see get_best_fileset_from_catalog's docstring).
    fileset_info = get_best_fileset_from_catalog(iso, canon, distinct_id)
    if not fileset_info:
        log(f"No fileset found for {iso}/{distinct_id}/{canon} in DBT catalog", "WARNING")
        return False

    overall_success = True
    for book, chapters in chapters_by_book.items():
        for chapter in chapters:
            chapter_ok = download_chapter(
                iso,
                distinct_id,
                canon,
                book,
                chapter,
                fileset_info["audio_fileset"],
                fileset_info["text_fileset"],
                fileset_info["timing_available"],
                force,
                content_types,
                alt_audio_fileset=fileset_info.get("alt_audio_fileset"),
                text_fileset_candidates=fileset_info.get("text_fileset_candidates"),
            )
            if not chapter_ok:
                overall_success = False

    return overall_success


def main():
    parser = argparse.ArgumentParser(
        description="Fetch audio/text/timing for every job in a batch_manifest.py batch"
    )
    parser.add_argument(
        "--batch-id",
        help="Batch ID to fetch (or set BATCH_ID env var)",
    )
    parser.add_argument(
        "--content-types",
        help="Content types to download: audio, text, timing (comma-separated). Default: all types",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--rate-delay",
        type=float,
        default=0.0,
        help="Seconds to wait between API calls (e.g., 0.5 or 1.0). Default: 0 (no delay)",
    )
    args = parser.parse_args()

    global API_RATE_DELAY
    if args.rate_delay > 0:
        API_RATE_DELAY = args.rate_delay
        log(f"Rate limiting: {API_RATE_DELAY}s delay between API calls", "INFO")

    content_types: Optional[List[str]] = None
    if args.content_types:
        content_types = [ct.strip().lower() for ct in args.content_types.split(",")]
        valid_types = {"audio", "text", "timing"}
        invalid_types = [ct for ct in content_types if ct not in valid_types]
        if invalid_types:
            log(f"Error: Invalid content types: {', '.join(invalid_types)}", "ERROR")
            log("Valid types: audio, text, timing", "ERROR")
            sys.exit(1)

    if not BIBLE_API_KEY:
        log("Error: BIBLE_API_KEY not set in .env file", "ERROR")
        log("Please add BIBLE_API_KEY=your_key_here to .env", "ERROR")
        sys.exit(1)

    batch = load_batch(args.batch_id)
    jobs = get_jobs(batch)
    log(f"Batch {batch.get('id')}: {len(jobs)} job(s) to fetch", "INFO")

    for i, job in enumerate(jobs, 1):
        log(
            f"[{i}/{len(jobs)}] {job['iso']}/{job['canon']}/{job['distinct_id']}",
            "INFO",
        )
        download_job(job, content_types=content_types, force=args.force)

    if error_logger.errors_by_language:
        error_logger.save_logs()
        log("Error logs saved to download_log/", "INFO")
    else:
        log("No errors to log", "INFO")

    log("=" * 70, "INFO")
    stats.report()
    log("=" * 70, "INFO")


if __name__ == "__main__":
    main()
