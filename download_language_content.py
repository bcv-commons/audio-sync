#!/usr/bin/env python3
"""
Download Bible content using canonical structure (NT/OT/PARTIAL).

This script uses the new canonical metadata from sorted/BB/ which organizes
content by canon (NT, OT, PARTIAL).

Usage:
    # Download specific language and books (default: all content types)
    python download_language_content.py eng --books Test
    python download_language_content.py spa --books GEN:1-3,MAT:1-5

    # Download with story sets
    python download_language_content.py eng --books "OBS Intro OT+NT"

    # Download specific content types
    python download_language_content.py eng --books Test --content-types audio
    python download_language_content.py eng --books Test --content-types text,timing
    python download_language_content.py eng --books Test --content-types audio,text,timing

Prerequisites:
    1. Run sort_cache_data.py first to generate sorted/BB/
    2. Set BIBLE_API_KEY in .env file

Output:
    downloads/BB/{canon}/{iso}/{distinct_id}/{BOOK}/
        {BOOK}_{CHAPTER:03d}_{FILESET_ID}.{ext}
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print("Error: Required packages not installed.")
    print("Please run: pip install -r requirements.txt")
    print(f"Missing module: {e.name}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# API Configuration
BIBLE_API_KEY = os.getenv("BIBLE_API_KEY", "")
BIBLE_API_BASE_URL = "https://4.dbt.io/api"
API_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60
API_RATE_DELAY = 0.0  # seconds between API calls, set via --rate-delay

# Directories
SORTED_DIR = Path("sorted/BB")
OUTPUT_DIR = Path("downloads/BB")
CONFIG_DIR = Path("config")
STORY_SET_CONFIG = CONFIG_DIR / "story-set.conf"
TEMPLATE_DIR = Path("templates")
ERROR_LOG_DIR = Path("download_log")
CROSSREF_PATH = Path("data/version-crossref.json")
HELLOAO_API = "https://bible.helloao.org/api"
HELLOAO_CACHE = Path("api-cache/helloao")

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


def _find_helloao_id(iso, version_id):
    """Find the helloAO translation ID for a language/version.
    Convention: {iso}_{vid} in lowercase, e.g., gaz_bib."""
    # Check helloAO cache for matching translation
    cache_file = HELLOAO_CACHE / "available_translations.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    vid_lower = version_id.lower()
    for t in data.get("translations", []):
        if not isinstance(t, dict):
            continue
        tid = t.get("id", "")
        lang = t.get("language", "")
        if lang == iso:
            # Match by version ID suffix
            tid_suffix = tid.replace(f"{iso}_", "").replace(f"{iso}", "")
            if tid_suffix.lower() == vid_lower or tid_suffix.upper() == version_id:
                return tid
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
                hao_id = _find_helloao_id(iso, vid)
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

ALL_BOOKS = {**OT_BOOKS, **NT_BOOKS}


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


def load_story_sets() -> Dict[str, List[Tuple[str, List[int]]]]:
    """Load story sets from config file."""
    story_sets = {}

    if not STORY_SET_CONFIG.exists():
        return story_sets

    with open(STORY_SET_CONFIG) as f:
        current_set = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                # Story set name
                current_set = line
                story_sets[current_set] = []
            elif current_set:
                # Parse comma-separated book:chapter specs
                for spec in line.split(","):
                    spec = spec.strip()
                    if ":" in spec:
                        book, chapter_spec = spec.split(":", 1)
                        chapters = parse_chapter_spec(chapter_spec.strip())
                        story_sets[current_set].append((book.strip(), chapters))

    return story_sets


def load_template_references(template_id: str) -> List[Tuple[str, List[int]]]:
    """
    Load Bible references from all .md files in templates/<template_id>/.

    Supports both flat and hierarchical template structures:
    - Flat: templates/OBS/01.md, templates/OBS/02.md
    - Hierarchical: templates/OBS/01/01.md, templates/OBS/02/04.md

    Only processes numbered .md files (story content), ignores index.md (navigation).

    Returns list of (book, chapters) tuples similar to story sets.
    """
    template_path = TEMPLATE_DIR / template_id

    if not template_path.exists() or not template_path.is_dir():
        log(f"Error: Template directory not found: {template_path}", "ERROR")
        return []

    # Pattern to match: [[ref:BOOK CHAPTER:VERSES]]
    # Examples: [[ref:GEN 1:1-2]], [[ref:LUK 1:5-7]], [[ref:GEN 1:10,22]]
    ref_pattern = re.compile(r"\[\[ref:\s*([A-Z0-9]+)\s+(\d+):[^\]]+\]\]")

    # Dictionary to collect chapters per book
    book_chapters: Dict[str, set] = defaultdict(set)

    # Find all .md files in template directory (recursive to support hierarchical structure)
    md_files = sorted(template_path.rglob("*.md"))

    if not md_files:
        log(f"Warning: No .md files found in {template_path}", "WARN")
        return []

    log(f"Scanning {len(md_files)} template files in {template_path}", "INFO")

    # Process each markdown file
    for md_file in md_files:
        # Only process numbered files (story content), skip index.md and other non-story files
        if not md_file.stem[0:1].isdigit():
            continue

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Find all references
            matches = ref_pattern.findall(content)
            for book, chapter in matches:
                book_chapters[book.upper()].add(int(chapter))

        except Exception as e:
            log(f"Warning: Error reading {md_file}: {e}", "WARN")
            continue

    # Convert to list of tuples with sorted chapters
    result = []
    for book in sorted(book_chapters.keys()):
        chapters = sorted(list(book_chapters[book]))
        result.append((book, chapters))

    # Log what was found
    if result:
        log(f"Found references in template '{template_id}':", "INFO")
        for book, chapters in result:
            chapter_str = ",".join(map(str, chapters))
            log(f"  {book}: {chapter_str}", "INFO")
    else:
        log(f"Warning: No Bible references found in template '{template_id}'", "WARN")

    return result


def parse_chapter_spec(spec: str) -> List[int]:
    """Parse chapter specification like '1', '1-5', '1,3,5' into list of chapter numbers."""
    chapters = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            chapters.extend(range(int(start), int(end) + 1))
        else:
            chapters.append(int(part))
    return sorted(set(chapters))


def expand_book_spec(book_spec: str) -> List[Tuple[str, List[int]]]:
    """
    Expand book specification into list of (book, chapters).

    Examples:
        'GEN' -> [('GEN', [1..50])]
        'GEN:1-3' -> [('GEN', [1,2,3])]
        'Test' -> [('PSA', [117]), ('REV', [15])]
    """
    story_sets = load_story_sets()

    # Check if it's a story set
    if book_spec in story_sets:
        return story_sets[book_spec]

    # Parse individual book spec
    if ":" in book_spec:
        book, chapter_spec = book_spec.split(":", 1)
        book = book.strip().upper()
        chapters = parse_chapter_spec(chapter_spec)
    else:
        book = book_spec.strip().upper()
        if book not in ALL_BOOKS:
            log(f"Unknown book: {book}", "ERROR")
            return []
        chapters = list(range(1, ALL_BOOKS[book] + 1))

    return [(book, chapters)]


def determine_book_canon(book: str) -> str:
    """Determine which canon a book belongs to."""
    if book in OT_BOOKS:
        return "OT"
    elif book in NT_BOOKS:
        return "NT"
    return "UNKNOWN"


def load_language_metadata(iso: str, canon: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load metadata for a language from sorted/BB/{iso}/.

    Args:
        iso: Language ISO code
        canon: Optional canon filter (NT, OT, PARTIAL)

    Returns:
        Dictionary of metadata by fileset_id
    """
    metadata_by_fileset = {}

    lang_dir = SORTED_DIR / iso
    if not lang_dir.exists():
        return metadata_by_fileset

    for fileset_dir in lang_dir.iterdir():
        if not fileset_dir.is_dir():
            continue

        metadata_file = fileset_dir / "metadata.json"
        if not metadata_file.exists():
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Filter by canon if specified
        if canon:
            metadata_canon = metadata.get("canon", "")
            if metadata_canon != canon:
                continue

        fileset_id = metadata.get("fileset", {}).get("id", "")
        if fileset_id:
            metadata_by_fileset[fileset_id] = metadata

    return metadata_by_fileset


def get_distinct_id_from_metadata(metadata: dict) -> str:
    """Extract distinct ID (Bible abbreviation) from metadata."""
    bible_abbr = metadata.get("bible", {}).get("abbr", "")
    if bible_abbr:
        return bible_abbr.upper()

    # Fallback to extracting from fileset_id if bible.abbr is missing
    fileset_id = metadata.get("fileset", {}).get("id", "")
    if len(fileset_id) >= 6:
        return fileset_id[:6].upper()
    return fileset_id.upper()


def fileset_contains_book(metadata: Dict, book: str, canon: str) -> bool:
    """Check if a fileset contains a specific book based on size field.

    DBT size codes seen in the wild:
      NT      — full New Testament
      OT      — full Old Testament
      C       — Complete (NT + OT)
      NTP     — NT Portion (partial NT)
      OTP     — OT Portion (partial OT)
      NTOTP   — NT (full) + OT Portion (partial OT)   ← common for IBT bibles
      NTPOTP  — NT Portion + OT Portion (both partial)
      OTNTP   — OT (full) + NT Portion (legacy spelling)
      P / S / PARTIAL — generic partial/story content
    """
    fileset_size = metadata.get("fileset", {}).get("size", "")

    # Full NT coverage
    if fileset_size in ("NT", "C", "NTOTP"):
        if book in NT_BOOKS:
            return True
    # Full OT coverage
    if fileset_size in ("OT", "C", "OTNTP"):
        if book in OT_BOOKS:
            return True
    # Partial NT (may or may not contain this specific NT book — let API decide)
    if fileset_size in ("NTP", "NTTP", "NTPOTP", "OTNTP") and book in NT_BOOKS:
        return True
    # Partial OT (may or may not contain this specific OT book — let API decide)
    if fileset_size in ("OTP", "OTTP", "NTPOTP", "NTOTP") and book in OT_BOOKS:
        return True
    # Generic partial / story
    if fileset_size in ("P", "S", "PARTIAL"):
        return True

    return False


def get_best_fileset_for_book(
    metadata_by_fileset: Dict[str, Dict], book: str
) -> Optional[Dict]:
    """
    Get the best fileset for a specific book.

    Returns a dict with:
        - distinct_id: Bible version abbreviation
        - canon: NT/OT/PARTIAL
        - audio_fileset: Audio fileset ID (if available)
        - text_fileset: Text fileset ID (if available)
        - timing_available: Whether timing data exists
    """
    if not metadata_by_fileset:
        return None

    # Use the first available metadata to get common info
    # (all filesets for same version should have same distinct_id/canon)
    first_metadata = next(iter(metadata_by_fileset.values()))

    distinct_id = get_distinct_id_from_metadata(first_metadata)
    canon = first_metadata.get("canon", "")

    # Find best audio and text filesets
    audio_fileset = None
    alt_audio_fileset = None  # Secondary audio (drama if primary is standard)
    text_fileset = None
    timing_available = False

    # Collect all usable audio filesets with priorities
    audio_candidates = []

    for fileset_id, metadata in metadata_by_fileset.items():
        fileset_type = metadata.get("fileset", {}).get("type", "")
        fileset_size = metadata.get("fileset", {}).get("size", "")

        # Check if this fileset contains the book
        if not fileset_contains_book(metadata, book, canon):
            continue

        # Audio priority (lower tuple = preferred):
        # 1. Plain MP3, non-dramatized, DA (N1DA or O1DA)
        # 2. Opus, non-dramatized, DA
        # 3. Plain MP3, dramatized, DA (N2DA or O2DA)
        # 4. Opus, dramatized, DA
        # 5-8. Same order but SA (streaming) variants
        if "audio" in fileset_type:
            is_dramatized = "2DA" in fileset_id or "2SA" in fileset_id
            is_opus = "-opus" in fileset_id
            is_stream = fileset_id.endswith("SA") or "stream" in fileset_type

            priority = (
                int(is_dramatized),
                int(is_stream),
                int(is_opus),
                fileset_id,
            )
            audio_candidates.append((priority, fileset_id, is_dramatized))

    # Sort by priority and select primary + alt
    if audio_candidates:
        audio_candidates.sort()
        audio_fileset = audio_candidates[0][1]
        # Find alt: if primary is standard, alt is the best drama (and vice versa)
        primary_is_drama = audio_candidates[0][2]
        for _, fid, is_drama in audio_candidates[1:]:
            if is_drama != primary_is_drama and "-opus" not in fid:
                alt_audio_fileset = fid
                break

    # Text priority — USX is intentionally skipped because the DBT USX
    # endpoint often 404s while the same content's plain/JSON/format variant
    # works fine.
    #
    # Order:
    #   0: plain text (_ET suffix or text_plain type)
    #   1: JSON (-json suffix or text_json type)
    #   2: other text (text_format, etc.)
    # Canon-specific (size != "C") variants get +4 within their bucket so
    # Complete bibles outrank partial ones at the same format.
    text_candidates = []  # list of (priority_tuple, fileset_id)
    for fileset_id, metadata in metadata_by_fileset.items():
        fileset_type = metadata.get("fileset", {}).get("type", "")
        fileset_size = metadata.get("fileset", {}).get("size", "")

        if "text" not in fileset_type:
            continue
        if not fileset_contains_book(metadata, book, canon):
            continue
        # Skip USX entirely — see comment above
        if fileset_id.endswith("-usx") or fileset_type == "text_usx":
            continue

        is_complete = fileset_size == "C"
        if (fileset_id.endswith("_ET") and "-" not in fileset_id) or fileset_type == "text_plain":
            base = 0
        elif fileset_id.endswith("-json") or fileset_type == "text_json":
            base = 1
        else:
            base = 2
        score = base if is_complete else base + 4
        text_candidates.append(((score, fileset_id), fileset_id))

        # Check for timing while we're iterating text filesets
        timing_info = metadata.get("download_ready", {})
        if timing_info.get("timing_available"):
            timing_available = True

    # Also pick up timing flag from audio filesets
    for fileset_id, metadata in metadata_by_fileset.items():
        fileset_type = metadata.get("fileset", {}).get("type", "")
        if "audio" in fileset_type:
            timing_info = metadata.get("download_ready", {})
            if timing_info.get("timing_available"):
                timing_available = True

    text_candidates.sort()
    text_fileset_candidates = [fid for _, fid in text_candidates]
    if text_fileset_candidates:
        text_fileset = text_fileset_candidates[0]

    if not audio_fileset and not text_fileset:
        return None

    result = {
        "distinct_id": distinct_id,
        "canon": canon,
        "audio_fileset": audio_fileset,
        "text_fileset": text_fileset,
        "text_fileset_candidates": text_fileset_candidates,
        "timing_available": timing_available,
    }
    if alt_audio_fileset:
        result["alt_audio_fileset"] = alt_audio_fileset
    return result


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
) -> bool:
    """
    Download content for a specific chapter based on requested content types.

    Args:
        content_types: List of content types to download ('audio', 'text', 'timing').
                      If None, downloads all available content types (default behavior).

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
        # No DBT text — check for external text source
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


def download_language(
    iso: str,
    books_spec: str,
    force: bool = False,
    force_partial: bool = False,
    required_category: Optional[str] = None,
    required_canon: Optional[str] = None,
    content_types: Optional[List[str]] = None,
):
    """
    Download content for a language.

    Args:
        content_types: List of content types to download ('audio', 'text', 'timing').
                      If None, downloads all available content types (default behavior).
    """
    log(f"Processing language: {iso}", "INFO")

    # Expand book specification
    # Split on spaces to handle template format (e.g., "GEN:1,2,3 LUK:1,2")
    # Commas within a spec like "JHN:1,2,3" are chapter separators, not book separators
    book_chapters = []
    specs = books_spec.split()

    for spec in specs:
        book_chapters.extend(expand_book_spec(spec.strip()))

    if not book_chapters:
        log("No valid books specified", "ERROR")
        return

    log(f"Books to download: {len(book_chapters)}", "INFO")

    # Group books by canon
    books_by_canon = defaultdict(list)
    for book, chapters in book_chapters:
        canon = determine_book_canon(book)
        if canon == "UNKNOWN":
            log(f"Cannot determine canon for {book}", "WARNING")
            continue

        # Filter by required canon if specified (for book-set filters like TIMING_OT, SYNC_NT)
        if required_canon and canon != required_canon:
            log(f"Skipping {book} (canon={canon}, required={required_canon})", "INFO")
            continue

        books_by_canon[canon].append((book, chapters))

    # Process each canon separately
    for canon, books in books_by_canon.items():
        # Load metadata for this canon
        metadata_by_fileset = load_language_metadata(iso, canon)
        if not metadata_by_fileset:
            # Only warn if not timing-only (timing-only commonly has canons without data)
            if content_types != ["timing"]:
                log(f"No metadata found for {iso} in {canon} canon", "WARNING")
            continue

        log(f"Processing {canon} canon ({len(books)} books)", "INFO")

        # Filter by required category if specified
        if required_category:
            filtered_metadata = {}
            for fid, meta in metadata_by_fileset.items():
                category = meta.get("aggregate_category")
                # For timing categories, accept both with-timecode and audio-with-timecode
                if required_category == "with-timecode":
                    if category in ("with-timecode", "audio-with-timecode"):
                        filtered_metadata[fid] = meta
                elif category == required_category:
                    filtered_metadata[fid] = meta

            if not filtered_metadata:
                if content_types != ["timing"]:
                    log(
                        f"No {required_category} versions found for {iso}/{canon}",
                        "WARNING",
                    )
                continue

            metadata_by_fileset = filtered_metadata
            log(
                f"Filtered to {len(metadata_by_fileset)} {required_category} filesets",
                "INFO",
            )

        # Check if this is partial content and skip unless forced
        first_metadata = next(iter(metadata_by_fileset.values()))
        category = first_metadata.get("aggregate_category", "")

        if category == "partial" and not force_partial:
            log(
                f"Skipping {iso}/{canon} (partial content - use --force-partial to download)",
                "INFO",
            )
            continue

        log(f"Found {len(metadata_by_fileset)} filesets for {iso}/{canon}", "INFO")

        # Process each book
        for book, chapters in books:
            log(
                f"Processing {book} (chapters: {min(chapters)}-{max(chapters)})", "INFO"
            )

            # Get all distinct_ids that have this book in this canon
            distinct_ids_to_try = {}
            for fileset_id, metadata in metadata_by_fileset.items():
                if not fileset_contains_book(metadata, book, canon):
                    continue

                distinct_id = get_distinct_id_from_metadata(metadata)
                if distinct_id not in distinct_ids_to_try:
                    distinct_ids_to_try[distinct_id] = []
                distinct_ids_to_try[distinct_id].append(metadata)

            if not distinct_ids_to_try:
                log(f"No filesets available for {book}", "WARNING")
                continue

            # Cross-source merge: if one version has audio-only and another
            # has text-only, merge the text filesets into the audio version
            # so both get downloaded to the same directory.
            audio_only_ids = []
            text_only_ids = []
            for did, metas in distinct_ids_to_try.items():
                version_dict = {m.get("fileset", {}).get("id", ""): m for m in metas}
                info = get_best_fileset_for_book(version_dict, book)
                if info:
                    has_audio = info["audio_fileset"] is not None
                    has_text = info["text_fileset"] is not None
                    if has_audio and not has_text:
                        audio_only_ids.append((did, info))
                    elif has_text and not has_audio:
                        text_only_ids.append((did, info))

            if audio_only_ids and text_only_ids:
                # Cross-source merge: only merge if confirmed by versions-data
                # versions.json shows which versions have DBT audio + external text
                versions_data = _load_versions_data(iso)

                for audio_did, audio_info in audio_only_ids:
                    merged = False
                    vid = _extract_version_id(iso, audio_did)
                    v_entry = versions_data.get(vid, {})

                    # Check if versions.json confirms external text for this audio
                    for tkey in ("nt", "ot"):
                        sources = v_entry.get(tkey, {})
                        dbt = sources.get("dbt", "")
                        has_external_text = (
                            sources.get("helloao") == "t"
                            or sources.get("ebible") == "t"
                        )
                        if "a" in dbt and "t" not in dbt and has_external_text:
                            # Confirmed cross-source pairing
                            text_did = text_only_ids[0][0]
                            for meta in distinct_ids_to_try[text_did]:
                                ftype = meta.get("fileset", {}).get("type", "")
                                if "text" in ftype:
                                    distinct_ids_to_try[audio_did].append(meta)
                            log(f"Cross-source merge (confirmed): {audio_did} audio + {text_did} text for {book}", "INFO")
                            merged = True
                            break

                    if not merged and text_only_ids:
                        text_did = text_only_ids[0][0]
                        log(f"Cross-source skipped (unconfirmed): {audio_did} audio + {text_did} text for {book}", "INFO")

            distinct_ids_list = list(distinct_ids_to_try.keys())
            log(
                f"Found {len(distinct_ids_to_try)} version(s) to try for {book}: {', '.join(distinct_ids_list)}",
                "INFO",
            )

            # Try each distinct_id
            # If required_category is set: stop at first success (book-set mode)
            # If required_category is None: download all versions (single language mode)

            # Sort distinct_ids by category priority when filtering by timing
            # Prefer with-timecode over audio-with-timecode
            if required_category == "with-timecode":

                def get_category_priority(distinct_id):
                    # Get category from first metadata entry for this distinct_id
                    first_meta = distinct_ids_to_try[distinct_id][0]
                    category = first_meta.get("aggregate_category", "")
                    # with-timecode (0) before audio-with-timecode (1)
                    return (0 if category == "with-timecode" else 1, distinct_id)

                distinct_ids_items = sorted(
                    distinct_ids_to_try.items(),
                    key=lambda x: get_category_priority(x[0]),
                )
            else:
                distinct_ids_items = distinct_ids_to_try.items()

            for distinct_id, version_metadata in distinct_ids_items:
                # Get best fileset info for this distinct_id
                version_dict = {
                    m.get("fileset", {}).get("id", ""): m for m in version_metadata
                }
                fileset_info = get_best_fileset_for_book(version_dict, book)

                if not fileset_info:
                    continue

                log(f"Trying {distinct_id}", "INFO")

                # Log which filesets were selected
                if fileset_info["audio_fileset"]:
                    log(f"  Audio fileset: {fileset_info['audio_fileset']}", "INFO")
                if fileset_info["text_fileset"]:
                    cands = fileset_info.get("text_fileset_candidates") or []
                    if len(cands) > 1:
                        log(f"  Text fileset: {fileset_info['text_fileset']} "
                            f"(fallbacks: {', '.join(cands[1:])})", "INFO")
                    else:
                        log(f"  Text fileset: {fileset_info['text_fileset']}", "INFO")
                elif content_types and "text" in content_types:
                    log(f"  No text fileset selected for {distinct_id}/{book} "
                        f"(no plain/JSON/format variant available)", "WARN")

                # Download each chapter for this version
                success = True
                for chapter in chapters:
                    chapter_success = download_chapter(
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
                    if not chapter_success:
                        success = False

                if success:
                    log(f"✓ Successfully downloaded {distinct_id}", "INFO")
                    # Continue to download all distinct_ids (versions) regardless of mode
                else:
                    remaining = [
                        d
                        for d in distinct_ids_list
                        if d != distinct_id
                        and distinct_ids_list.index(d)
                        > distinct_ids_list.index(distinct_id)
                    ]
                    log(
                        f"✗ {distinct_id} had failures"
                        + (
                            f", trying next version... (remaining: {', '.join(remaining)})"
                            if required_category and remaining
                            else ""
                        ),
                        "WARNING",
                    )


def get_languages_by_book_set(book_set: str) -> List[str]:
    """
    Get list of language ISO codes filtered by book-set category.

    Categories:
    - ALL: All available languages (excludes PARTIAL)
    - TIMING_ALL: Languages with timing data for NT or OT (both canons)
    - TIMING_NT: Languages with timing data for New Testament
    - TIMING_OT: Languages with timing data for Old Testament
    - SYNC_NT: Languages with syncable New Testament
    - SYNC_OT: Languages with syncable Old Testament
    - PARTIAL: Languages with only partial content

    Args:
        book_set: Category name

    Returns:
        List of language ISO codes
    """
    languages = []
    languages_to_check = []

    # Get list of languages from sorted/BB
    if SORTED_DIR.exists():
        for lang_dir in SORTED_DIR.iterdir():
            if lang_dir.is_dir() and len(lang_dir.name) == 3:
                languages_to_check.append(lang_dir.name)

    if not languages_to_check:
        log("No languages found in sorted/BB directory.", "ERROR")
        log("Please run: python sort_cache_data.py", "ERROR")
        sys.exit(1)

    # Check each language against the book-set criteria
    for iso in sorted(languages_to_check):
        if book_set == "ALL":
            # ALL excludes PARTIAL - only include languages with NT or OT content
            nt_metadata = load_language_metadata(iso, "NT")
            ot_metadata = load_language_metadata(iso, "OT")
            if nt_metadata or ot_metadata:
                languages.append(iso)
            continue

        # Load metadata for this language
        elif book_set == "TIMING_ALL":
            # Has timing data for NT or OT (either canon)
            for canon in ["NT", "OT"]:
                metadata_dict = load_language_metadata(iso, canon)
                for metadata in metadata_dict.values():
                    category = metadata.get("aggregate_category")
                    if category in ("with-timecode", "audio-with-timecode"):
                        languages.append(iso)
                        break
                else:
                    continue
                break  # Found timing in this canon, move to next language

        elif book_set == "TIMING_NT":
            # Has timing data for NT
            metadata_dict = load_language_metadata(iso, "NT")
            for metadata in metadata_dict.values():
                category = metadata.get("aggregate_category")
                if category in ("with-timecode", "audio-with-timecode"):
                    languages.append(iso)
                    break

        elif book_set == "TIMING_OT":
            # Has timing data for OT
            metadata_dict = load_language_metadata(iso, "OT")
            for metadata in metadata_dict.values():
                category = metadata.get("aggregate_category")
                if category in ("with-timecode", "audio-with-timecode"):
                    languages.append(iso)
                    break

        elif book_set == "SYNC_NT":
            # Has syncable content for NT
            metadata_dict = load_language_metadata(iso, "NT")
            for metadata in metadata_dict.values():
                if metadata.get("aggregate_category") == "syncable":
                    languages.append(iso)
                    break

        elif book_set == "SYNC_OT":
            # Has syncable content for OT
            metadata_dict = load_language_metadata(iso, "OT")
            for metadata in metadata_dict.values():
                if metadata.get("aggregate_category") == "syncable":
                    languages.append(iso)
                    break

        elif book_set == "PARTIAL":
            # Has only partial content
            metadata_dict = load_language_metadata(iso, "PARTIAL")
            if metadata_dict:
                languages.append(iso)

    return languages


def main():
    parser = argparse.ArgumentParser(
        description="Download Bible content using canonical structure"
    )
    parser.add_argument(
        "iso",
        nargs="?",
        help="Language ISO code (e.g., eng, spa). Not needed with --book-set",
    )
    parser.add_argument(
        "--books",
        help="Books to download (e.g., 'GEN', 'GEN:1-3', 'Test', 'OBS Intro OT+NT')",
    )
    parser.add_argument(
        "--template",
        help="Template ID to load references from (e.g., 'OBS'). Reads from templates/<template_id>/*.md",
    )
    parser.add_argument(
        "--book-set",
        help="Book-set filter: ALL, TIMING_ALL, TIMING_NT, TIMING_OT, SYNC_NT, SYNC_OT, PARTIAL",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--force-partial",
        action="store_true",
        help="Include partial content (single books, incomplete sets)",
    )
    parser.add_argument(
        "--content-types",
        help="Content types to download: audio, text, timing (comma-separated, e.g., 'audio,text'). Default: all types",
    )
    parser.add_argument(
        "--rate-delay",
        type=float,
        default=0.0,
        help="Seconds to wait between API calls (e.g., 0.5 or 1.0). Default: 0 (no delay)",
    )

    args = parser.parse_args()

    # Initialize variables
    required_category: Optional[str] = None
    required_canon: Optional[str] = None
    template_books: Optional[str] = None
    content_types: Optional[List[str]] = None

    # Set rate limiting
    global API_RATE_DELAY
    if args.rate_delay > 0:
        API_RATE_DELAY = args.rate_delay
        log(f"Rate limiting: {API_RATE_DELAY}s delay between API calls", "INFO")

    # Parse content types
    if args.content_types:
        content_types = [ct.strip().lower() for ct in args.content_types.split(",")]
        valid_types = {"audio", "text", "timing"}
        invalid_types = [ct for ct in content_types if ct not in valid_types]
        if invalid_types:
            log(f"Error: Invalid content types: {', '.join(invalid_types)}", "ERROR")
            log("Valid types: audio, text, timing", "ERROR")
            sys.exit(1)
        log(f"Content types to download: {', '.join(content_types)}", "INFO")

    # Handle --template argument
    if args.template:
        # Load references from template
        template_refs = load_template_references(args.template)

        if not template_refs:
            log(
                f"Error: No valid references found in template '{args.template}'",
                "ERROR",
            )
            sys.exit(1)

        # Convert template references to book spec format
        # Format: BOOK:CHAPTER,CHAPTER,... with spaces between books
        # Note: Commas within chapter lists, spaces between different books
        book_specs = []
        for book, chapters in template_refs:
            chapter_list = ",".join(map(str, chapters))
            book_specs.append(f"{book}:{chapter_list}")
        template_books = " ".join(book_specs)

        # Check for competing --books argument
        if args.books:
            log("=" * 70, "WARN")
            log("WARNING: Both --template and --books specified", "WARN")
            log(f"  --template '{args.template}' takes precedence", "WARN")
            log(f"  Ignoring --books argument: '{args.books}'", "WARN")
            log("=" * 70, "WARN")

        # Override books with template references
        args.books = template_books

    # Validate arguments
    if args.book_set:
        # Batch mode - download multiple languages filtered by book-set
        if not args.books:
            log("Error: --books argument is required", "ERROR")
            parser.print_help()
            sys.exit(1)

        valid_book_sets = [
            "ALL",
            "TIMING_ALL",
            "TIMING_NT",
            "TIMING_OT",
            "SYNC_NT",
            "SYNC_OT",
            "PARTIAL",
        ]
        if args.book_set not in valid_book_sets:
            log(f"Error: Invalid book-set '{args.book_set}'", "ERROR")
            log(f"Valid options: {', '.join(valid_book_sets)}", "ERROR")
            sys.exit(1)

        languages = get_languages_by_book_set(args.book_set)

        # Determine required category and canon based on book-set
        if args.book_set == "TIMING_ALL":
            required_category = "with-timecode"
            required_canon = None  # Allow both NT and OT
        elif args.book_set in ["TIMING_NT", "TIMING_OT"]:
            required_category = "with-timecode"
            required_canon = "NT" if args.book_set == "TIMING_NT" else "OT"
        elif args.book_set in ["SYNC_NT", "SYNC_OT"]:
            required_category = "syncable"
            required_canon = "NT" if args.book_set == "SYNC_NT" else "OT"
        elif args.book_set == "PARTIAL":
            args.force_partial = True
            required_category = "partial"
            required_canon = "PARTIAL"

        log(
            f"Book-set '{args.book_set}' matched {len(languages)} languages",
            "INFO",
        )

        if not languages:
            log("No languages found matching book-set criteria", "ERROR")
            sys.exit(1)
    else:
        # Single language mode
        if not args.iso:
            log("Error: language ISO code is required (or use --book-set)", "ERROR")
            parser.print_help()
            sys.exit(1)

        if not args.books:
            log("Error: --books argument is required", "ERROR")
            parser.print_help()
            sys.exit(1)

        if args.iso.upper() == "ALL":
            # Loop through all languages in sorted cache
            languages = sorted(
                d.name for d in SORTED_DIR.iterdir()
                if d.is_dir() and len(d.name) == 3
            )
            log(f"Processing ALL languages: {len(languages)}", "INFO")
        else:
            languages = [args.iso]

    # Verify API key
    if not BIBLE_API_KEY:
        log("Error: BIBLE_API_KEY not set in .env file", "ERROR")
        log("Please add BIBLE_API_KEY=your_key_here to .env", "ERROR")
        sys.exit(1)

    # Verify sorted directory exists
    if not SORTED_DIR.exists():
        log(f"Error: Sorted directory not found: {SORTED_DIR}", "ERROR")
        log("Please run: python sort_cache_data.py", "ERROR")
        sys.exit(1)

    # Start download
    log("=" * 70, "INFO")
    log("Bible Content Download Script (canonical structure)", "INFO")
    log("=" * 70, "INFO")

    if args.book_set:
        log(f"Batch mode: {len(languages)} languages to process", "INFO")

    # Download each language
    for i, iso in enumerate(languages, 1):
        if len(languages) > 1:
            log(f"\n[{i}/{len(languages)}] Language: {iso}", "INFO")
            log("-" * 70, "INFO")

        download_language(
            iso,
            args.books,
            args.force,
            args.force_partial,
            required_category if args.book_set else None,
            required_canon if args.book_set else None,
            content_types,
        )

    # Save error logs
    if error_logger.errors_by_language:
        error_logger.save_logs()
        log("\n✓ Error logs saved to download_log/", "INFO")
    else:
        log("\n✓ No errors to log", "INFO")

    # Report statistics
    log("=" * 70, "INFO")
    stats.report()
    log("=" * 70, "INFO")


if __name__ == "__main__":
    main()
