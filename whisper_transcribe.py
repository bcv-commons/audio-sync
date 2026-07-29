#!/usr/bin/env python3
"""
Whisper-based Bible audio transcription pipeline (Step 1 of 2).

Transcribes audio using Whisper and saves the raw word-level timeline.
Verse alignment is done separately by language-specific scripts (Step 2).

For each chapter, two output files are generated:
  - _whisper_words.json : raw word timeline in word-timing-data/ (for alignment)
  - .srt                : SRT subtitle file in export/timing-data/

By default, the script automatically:
1. Scans all templates under templates/ to determine which chapters are needed
2. Downloads any missing audio+text files for those chapters
3. Transcribes and generates all timing data, skipping anything already done

Usage:
    # Process a single language (auto-download + transcribe for all templates)
    python whisper_transcribe.py --iso nld

    # Process only chapters from a specific template
    python whisper_transcribe.py --iso nld --template John

    # Process multiple languages
    python whisper_transcribe.py --iso-list nld,pol,ell

    # Process all tier 1 languages
    python whisper_transcribe.py --tier 1

    # Process all priority languages
    python whisper_transcribe.py --all

    # Only NT or OT
    python whisper_transcribe.py --iso nld --testament nt

    # Check what would be done without doing it
    python whisper_transcribe.py --iso nld --dry-run

    # Skip auto-download (only process already-downloaded files)
    python whisper_transcribe.py --iso nld --no-download

Prerequisites:
    1. pip install -r requirements-whisper.txt
    2. Apple Silicon Mac (for mlx-whisper) OR NVIDIA GPU (for faster-whisper)
       On non-Mac systems with a CUDA GPU, faster-whisper is used automatically.
    3. BIBLE_API_KEY in .env (for auto-download)
"""

import argparse
import difflib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from batch_manifest import load_batch, get_book_chapters
from download_language_content import download_job
from text_processing import LanguageConfig, load_language_config, normalize_text

# ─── Constants ──────────────────────────────────────────────────────────────

import platform
import sys as _sys

PRIORITY_LANGUAGES_FILE = Path("whisper-priority-languages.json")
DOWNLOADS_DIR = Path("downloads/BB")
HELLOAO_DOWNLOADS = Path("downloads/helloao")
HELLOAO_API = "https://bible.helloao.org/api"
DEFAULT_OUTPUT_DIR = Path("export/timing-data")
WORD_TIMING_DIR = Path("word-timing-data")
TEMPLATE_DIR = Path("templates")

# Detect runtime backend:
#   - Apple Silicon Mac  → mlx-whisper (Metal GPU)
#   - CUDA available     → faster-whisper (NVIDIA GPU)
#   - Fallback           → faster-whisper on CPU
_IS_APPLE_SILICON = (platform.system() == "Darwin" and platform.machine() == "arm64")

def _cuda_available() -> bool:
    """Return True if PyTorch can see at least one CUDA device."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

_USE_MLX = _IS_APPLE_SILICON
_USE_CUDA = (not _IS_APPLE_SILICON) and _cuda_available()

# Set to True via set_whisper_cpu(True) or --whisper-cpu to force CPU even when CUDA is available.
_FORCE_WHISPER_CPU: bool = False


def set_whisper_cpu(force: bool) -> None:
    """Force the faster-whisper backend to run on CPU regardless of CUDA availability.

    Call this before load_whisper_model() / transcribe_audio() to override the
    automatic device selection.  Has no effect on the mlx-whisper backend (Apple
    Silicon), which always uses Metal.
    """
    global _FORCE_WHISPER_CPU
    _FORCE_WHISPER_CPU = force


# Minimum free VRAM (in bytes) required to load each Whisper model in float16.
# Used by _enough_vram_for_whisper() to decide whether to use CUDA or fall back
# to CPU (int8) automatically when VRAM is tight (e.g. MMS already loaded).
_WHISPER_VRAM_REQUIRED: Dict[str, int] = {
    "tiny":               500 * 1024 * 1024,
    "base":               500 * 1024 * 1024,
    "small":            1_100 * 1024 * 1024,
    "medium":           2_600 * 1024 * 1024,
    "large":            3_500 * 1024 * 1024,
    "large-v2":         3_500 * 1024 * 1024,
    "large-v3":         3_500 * 1024 * 1024,
    "large-v3-turbo":   1_600 * 1024 * 1024,
    "distil-large-v3":  1_600 * 1024 * 1024,
}
_DEFAULT_VRAM_REQUIRED = 3_500 * 1024 * 1024  # conservative default for unknown models


def _enough_vram_for_whisper(model_name: str) -> bool:
    """Return True if there is enough free VRAM to load the Whisper model on GPU.

    Returns False immediately if _FORCE_WHISPER_CPU is set or CUDA is unavailable.

    Should be called AFTER any other models (e.g. MMS) have already been loaded
    onto the GPU so the measurement reflects actual available memory.

    Matches on the last path component of model_name so HuggingFace repo names
    like "openai/whisper-large-v3" are handled correctly.
    """
    if _FORCE_WHISPER_CPU or not _USE_CUDA:
        return False
    try:
        import torch
        key = model_name.split("/")[-1].lower()
        required = _WHISPER_VRAM_REQUIRED.get(key, _DEFAULT_VRAM_REQUIRED)
        free, _total = torch.cuda.mem_get_info(0)
        return free >= required
    except Exception:
        return False


# Default models per backend
DEFAULT_MODEL_MLX = "mlx-community/whisper-large-v3-mlx"
DEFAULT_MODEL_FASTER = "large-v3"
DEFAULT_MODEL = DEFAULT_MODEL_MLX if _USE_MLX else DEFAULT_MODEL_FASTER
DEFAULT_MATCH_THRESHOLD = 0.5
ANCHOR_WORD_COUNT = 8

# Categories that may contain audio+text pairs (searched in order of priority)

# ISO 639-3 (used by DBP) -> ISO 639-1 (used by Whisper)
ISO639_3_TO_WHISPER = {
    "nld": "nl",
    "pol": "pl",
    "ell": "el",
    "hun": "hu",
    "fra": "fr",
    "rus": "ru",
    "kor": "ko",
    "por": "pt",
    "swe": "sv",
    "pes": "fa",
    "arb": "ar",
    "heb": "he",
    "ind": "id",
    "bul": "bg",
    "vie": "vi",
    "bos": "bs",
    "amh": "am",
    "kaz": "kk",
    "som": "so",
    "sun": "su",
    "ydd": "yi",
    "hau": "ha",
    "ibo": None,  # Not in Whisper's language list; use auto-detect
    "kan": "kn",
    "yor": "yo",
    "lin": "ln",
    "zsm": "ms",
    "eng": "en",
    "spa": "es",
    "cmn": "zh",
    "tur": "tr",
    "ron": "ro",
    "urd": "ur",
    "mar": "mr",
    "npi": "ne",
    "tat": "tt",
    "yue": "yue",
}

# Book definitions (from download_language_content.py)
OT_BOOKS = {
    "GEN": 50, "EXO": 40, "LEV": 27, "NUM": 36, "DEU": 34,
    "JOS": 24, "JDG": 21, "RUT": 4, "1SA": 31, "2SA": 24,
    "1KI": 22, "2KI": 25, "1CH": 29, "2CH": 36, "EZR": 10,
    "NEH": 13, "EST": 10, "JOB": 42, "PSA": 150, "PRO": 31,
    "ECC": 12, "SNG": 8, "ISA": 66, "JER": 52, "LAM": 5,
    "EZK": 48, "DAN": 12, "HOS": 14, "JOL": 3, "AMO": 9,
    "OBA": 1, "JON": 4, "MIC": 7, "NAM": 3, "HAB": 3,
    "ZEP": 3, "HAG": 2, "ZEC": 14, "MAL": 4,
}

NT_BOOKS = {
    "MAT": 28, "MRK": 16, "LUK": 24, "JHN": 21, "ACT": 28,
    "ROM": 16, "1CO": 16, "2CO": 13, "GAL": 6, "EPH": 6,
    "PHP": 4, "COL": 4, "1TH": 5, "2TH": 3, "1TI": 6,
    "2TI": 4, "TIT": 3, "PHM": 1, "HEB": 13, "JAS": 5,
    "1PE": 5, "2PE": 3, "1JN": 5, "2JN": 1, "3JN": 1,
    "JUD": 1, "REV": 22,
}


# ─── Logging ────────────────────────────────────────────────────────────────

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ─── Template Scanning ─────────────────────────────────────────────────────

def load_all_template_refs(template_filter: Optional[str] = None) -> Dict[str, Set[int]]:
    """
    Load Bible book/chapter scope from the current batch manifest.
    Returns {BOOK: {chapter_numbers}}.
    """
    batch = load_batch()
    return {book: set(chapters) for book, chapters in get_book_chapters(batch, template_filter).items()}


def classify_template_refs(refs: Dict[str, Set[int]]) -> Dict[str, Dict[str, Set[int]]]:
    """
    Split template references into NT and OT groups.
    Returns {"nt": {BOOK: {chapters}}, "ot": {BOOK: {chapters}}}.
    """
    result = {"nt": {}, "ot": {}}
    for book, chapters in refs.items():
        if book in NT_BOOKS:
            result["nt"][book] = chapters
        elif book in OT_BOOKS:
            result["ot"][book] = chapters
    return result


# ─── Language Resolution ────────────────────────────────────────────────────

def load_priority_languages() -> List[dict]:
    """Load the priority languages JSON file.

    Returns an empty list if the file does not exist, allowing --iso / --iso-list
    to proceed via the _build_custom_language_entry() fallback in resolve_languages().
    The file is only required when using --tier or --all (which need the full list).
    """
    if not PRIORITY_LANGUAGES_FILE.exists():
        return []
    with open(PRIORITY_LANGUAGES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["languages"]


def resolve_languages(args, all_languages: List[dict]) -> List[dict]:
    """Resolve CLI arguments into a filtered list of language entries."""
    selected = []

    if args.all:
        if not all_languages:
            log(f"--all requires {PRIORITY_LANGUAGES_FILE} to exist", "ERROR")
            sys.exit(1)
        selected = list(all_languages)
    elif args.tier is not None:
        if not all_languages:
            log(f"--tier requires {PRIORITY_LANGUAGES_FILE} to exist", "ERROR")
            sys.exit(1)
        selected = [l for l in all_languages if l["tier"] == args.tier]
    elif args.iso:
        iso_code = args.iso.lower()
        selected = [l for l in all_languages if l["iso"] == iso_code]
        if not selected:
            # Allow processing languages not in priority list
            selected = [_build_custom_language_entry(iso_code)]
    elif args.iso_list:
        iso_codes = [c.strip().lower() for c in args.iso_list.split(",")]
        for code in iso_codes:
            matches = [l for l in all_languages if l["iso"] == code]
            if matches:
                selected.extend(matches)
            else:
                selected.append(_build_custom_language_entry(code))
    else:
        log("No language selection specified. Use --iso, --iso-list, --tier, or --all", "ERROR")
        sys.exit(1)

    return selected


def _build_custom_language_entry(iso: str) -> dict:
    """Build a minimal language entry for a language not in the priority list."""
    return {
        "iso": iso,
        "language": iso.upper(),
        "tier": 0,
        "nt": {"syncable": [], "with_timecode": [], "audio_with_timecode": []},
        "ot": {"syncable": [], "with_timecode": [], "audio_with_timecode": []},
        "notes": "Custom language (not in priority list)",
    }


def get_whisper_language(iso: str) -> Optional[str]:
    """Map ISO 639-3 code to Whisper language code."""
    return ISO639_3_TO_WHISPER.get(iso)


# ─── Work Item Generation ──────────────────────────────────────────────────

def generate_work_items(
    languages: List[dict],
    testament: str,
    template_refs_by_canon: Optional[Dict[str, Dict[str, Set[int]]]] = None,
) -> List[dict]:
    """
    Generate work items from language entries.
    Each work item = one (iso, canon, distinct_id) combination to process.
    If template_refs_by_canon is provided, only canons with matching refs are included.

    Scans all categories (syncable, with-timecode, audio-with-timecode) to find
    filesets that have or could have audio+text pairs.
    Exclusion filtering is handled upstream — core omits excluded filesets
    before emitting the batch manifest.
    """

    work_items = []

    for lang in languages:
        iso = lang["iso"]
        lang_name = lang["language"]
        tier = lang["tier"]

        canons = []
        if testament in ("nt", "both"):
            canons.append("nt")
        if testament in ("ot", "both"):
            canons.append("ot")

        for canon in canons:
            # If template filtering is active, skip canons with no template refs
            if template_refs_by_canon and not template_refs_by_canon.get(canon):
                continue

            # Collect all known filesets from metadata
            canon_data = lang.get(canon, {})
            all_filesets = set()
            for cat_key in ("syncable", "with_timecode", "audio_with_timecode"):
                all_filesets.update(canon_data.get(cat_key, []))

            # Check what's actually on disk
            # (downloads/BB/, downloads/contrib/, downloads/helloao/aligned/)
            # Only consider filesets that have or can have audio — text-only
            # downloads (e.g. DEUL12, DEUD05) can't be aligned and shouldn't
            # generate work items.
            found_on_disk = set()
            for base in (DOWNLOADS_DIR, Path("downloads/contrib"), Path("downloads/helloao/aligned")):
                iso_dir = base / canon / iso
                if not iso_dir.is_dir():
                    continue
                for distinct_dir in iso_dir.iterdir():
                    if not distinct_dir.is_dir():
                        continue
                    has_local_audio = any(distinct_dir.rglob("*.mp3"))
                    has_external_audio = (distinct_dir / "audio.json").exists()
                    if has_local_audio or has_external_audio:
                        found_on_disk.add(distinct_dir.name)

            if found_on_disk:
                for distinct_id in sorted(found_on_disk):
                    work_items.append({
                        "iso": iso,
                        "language": lang_name,
                        "tier": tier,
                        "canon": canon,
                        "distinct_id": distinct_id,
                        "has_downloads": True,
                    })
            elif all_filesets:
                # Nothing on disk yet but metadata knows filesets — create work items
                # so auto-download can fetch them
                for fileset in sorted(all_filesets):
                    work_items.append({
                        "iso": iso,
                        "language": lang_name,
                        "tier": tier,
                        "canon": canon,
                        "distinct_id": fileset,
                        "has_downloads": False,
                    })
            else:
                # Custom language with no metadata and nothing on disk —
                # create a placeholder so auto-download can discover filesets
                work_items.append({
                    "iso": iso,
                    "language": lang_name,
                    "tier": tier,
                    "canon": canon,
                    "distinct_id": None,
                    "has_downloads": False,
                })

    return work_items


# ─── File Discovery ─────────────────────────────────────────────────────────

def _find_base_dir(canon: str, iso: str, distinct_id: str) -> Optional[Path]:
    """
    Find the download directory for a fileset.
    Searches in priority order:
      downloads/BB/                  (DBT)
      downloads/contrib/             (contributed: NORNBS, DEUSOL, ...)
      downloads/helloao/aligned/     (helloAO + reader: BSBHAY, BSBGIL, ...)
    """
    for base in (DOWNLOADS_DIR, Path("downloads/contrib"), Path("downloads/helloao/aligned")):
        candidate = base / canon / iso / distinct_id
        if candidate.exists():
            return candidate
    return None


def _find_helloao_id(iso: str, distinct_id: str) -> Optional[str]:
    """
    Find a helloAO translation ID that pairs with a DBT audio-only distinct_id.

    Convention: helloAO uses lowercase+underscore IDs (e.g., gaz_bib),
    DBT uses uppercase IDs (e.g., GAZBIB).
    """
    if "_" in distinct_id or not distinct_id.isupper():
        return None
    if not HELLOAO_DOWNLOADS.is_dir():
        return None

    for candidate in HELLOAO_DOWNLOADS.iterdir():
        if not candidate.is_dir() or "_" not in candidate.name:
            continue
        if candidate.name.replace("_", "").upper() == distinct_id:
            helloao_iso = candidate.name.split("_")[0]
            if helloao_iso == iso:
                return candidate.name
    return None


def _fetch_helloao_text(helloao_id: str, book: str, chapter_num: int, dest_dir: Path) -> Optional[Path]:
    """
    Fetch a chapter's text from helloAO API and write as ET-format .txt file.
    Returns the path to the written file, or None on failure.
    """
    import urllib.request

    chapter_str = f"{chapter_num:03d}"
    txt_filename = f"{book}_{chapter_str}_{helloao_id.replace('_', '').upper()}_ET.txt"
    txt_path = dest_dir / txt_filename

    if txt_path.exists():
        return txt_path

    url = f"{HELLOAO_API}/{helloao_id}/{book}/{chapter_num}.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        log(f"    Failed to fetch helloAO text {book} {chapter_num}: {e}", "WARN")
        return None

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
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(verses) + "\n", encoding="utf-8")
    return txt_path


def discover_chapter_files(
    iso: str,
    canon: str,
    distinct_id: str,
    output_dir: Path,
    force: bool = False,
    required_chapters: Optional[Dict[str, Set[int]]] = None,
) -> Tuple[List[dict], int]:
    """
    Discover audio+text file pairs for a given language/canon/fileset.
    Searches across all category directories (syncable, with-timecode, etc.).
    If required_chapters is given ({BOOK: {chapter_nums}}), only include those.
    Returns (chapter_list, skipped_count).
    """
    base_dir = _find_base_dir(canon, iso, distinct_id)
    if not base_dir:
        return [], 0

    chapters = []
    skipped = 0

    # If this version dir uses an external audio source (sermon-online via
    # audio.json, or a helloao/aligned/ subtree), the mp3 may be fetched on
    # demand. We synthesise the audio_fileset id from the directory structure
    # so chapter discovery still works without local mp3s.
    has_external_audio = False
    external_audio_fileset = None
    audio_json_path = base_dir / "audio.json"
    if audio_json_path.exists() or "helloao" in base_dir.parts:
        has_external_audio = True
        # Convention: contributed audio for canon=nt → {DISTINCT_ID}N2DA, OT → {DISTINCT_ID}O2DA
        # Matches what import_contrib.py / align_bsb.py write.
        suffix = "O2DA" if canon == "ot" else "N2DA"
        external_audio_fileset = f"{distinct_id}{suffix}"

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        book = book_dir.name

        # Skip books not in the required set
        if required_chapters is not None and book not in required_chapters:
            continue

        # Find audio files in this book directory
        mp3_files = sorted(book_dir.glob("*.mp3"))

        # Group mp3s by chapter, then pick standard vs drama
        chapter_mp3s = {}  # {(book, chapter_num): [(fileset_id, mp3_path, is_drama), ...]}
        for mp3_path in mp3_files:
            stem = mp3_path.stem  # e.g., REV_015_NLDHSVN1DA
            parts = stem.split("_", 2)
            if len(parts) < 3:
                continue
            file_book = parts[0]
            chapter_str = parts[1]
            audio_fileset = parts[2]

            try:
                chapter_num = int(chapter_str)
            except ValueError:
                continue

            if required_chapters is not None and chapter_num not in required_chapters.get(file_book, set()):
                continue

            is_drama = "2DA" in audio_fileset or "2SA" in audio_fileset
            key = (file_book, chapter_num)
            if key not in chapter_mp3s:
                chapter_mp3s[key] = []
            chapter_mp3s[key].append((audio_fileset, mp3_path, is_drama, chapter_str))

        # External audio: also discover chapters that only have text files.
        # The mp3 will be fetched on demand by ensure_chapter_audio().
        if has_external_audio and external_audio_fileset:
            for txt_path in sorted(book_dir.glob("*.txt")):
                stem = txt_path.stem  # e.g., JHN_001_DEUSOL_NT_ET
                parts = stem.split("_", 2)
                if len(parts) < 3:
                    continue
                file_book = parts[0]
                chapter_str = parts[1]
                try:
                    chapter_num = int(chapter_str)
                except ValueError:
                    continue
                if required_chapters is not None and chapter_num not in required_chapters.get(file_book, set()):
                    continue
                key = (file_book, chapter_num)
                if key in chapter_mp3s:
                    continue  # already covered by a real mp3
                # mp3_path is where the on-demand download will land
                synth_mp3 = book_dir / f"{file_book}_{chapter_str}_{external_audio_fileset}.mp3"
                is_drama = "2DA" in external_audio_fileset
                chapter_mp3s[key] = [
                    (external_audio_fileset, synth_mp3, is_drama, chapter_str)
                ]

        for (file_book, chapter_num), mp3_list in sorted(chapter_mp3s.items()):
            chapter_str = mp3_list[0][3]

            # Find matching text file (prefer _ET plain text over other formats)
            txt_candidates = list(book_dir.glob(f"{file_book}_{chapter_str}_*.txt"))
            if not txt_candidates:
                helloao_id = _find_helloao_id(iso, distinct_id)
                if helloao_id:
                    fetched = _fetch_helloao_text(helloao_id, file_book, chapter_num, book_dir)
                    if fetched:
                        txt_candidates = [fetched]
                if not txt_candidates:
                    log(f"  No text file for {file_book} {chapter_num} in {distinct_id}", "WARN")
                    continue
            txt_candidates.sort(key=lambda p: (0 if "_ET" in p.stem else 1, p.name))
            txt_path = txt_candidates[0]
            text_fileset = txt_path.stem.split("_", 2)[2] if len(txt_path.stem.split("_", 2)) >= 3 else ""

            # Create entries for ALL audio filesets (MMS runs on all)
            # Mark which one Whisper should prefer (standard over drama)
            standard = [m for m in mp3_list if not m[2]]
            drama = [m for m in mp3_list if m[2]]
            whisper_fileset = standard[0][0] if standard else mp3_list[0][0]

            for audio_fileset, mp3_path, is_drama, _ in mp3_list:
                out_book_dir = output_dir / canon / iso / distinct_id / file_book
                whisper_book_dir = WORD_TIMING_DIR / canon / iso / distinct_id / file_book
                whisper_words_filename = f"{file_book}_{chapter_str}_{audio_fileset}_whisper_words.json"
                whisper_words_path = whisper_book_dir / whisper_words_filename
                srt_filename = f"{file_book}_{chapter_str}_{audio_fileset}.srt"
                srt_path = out_book_dir / srt_filename

                # Skip only if the chapter is fully complete (final fused
                # output already exists) — NOT merely because Whisper's
                # word-timing exists. Whisper finishing doesn't mean MMS/
                # fusion did too (e.g. a crash mid-chapter); each step's own
                # skip-if-exists check in align_pipeline.py's main loop is
                # what actually short-circuits already-done steps, so this
                # must still surface the chapter for those checks to run.
                timing_filename = f"{file_book}_{chapter_str}_{audio_fileset}_timing.json"
                timing_path = out_book_dir / timing_filename
                if timing_path.exists() and not force:
                    skipped += 1
                    continue

                chapters.append({
                    "book": file_book,
                    "chapter": chapter_num,
                    "chapter_str": chapter_str,
                    "audio_path": mp3_path,
                    "text_path": txt_path,
                    "audio_fileset": audio_fileset,
                    "text_fileset": text_fileset,
                    "whisper_words_path": whisper_words_path,
                    "srt_path": srt_path,
                    "whisper_prefer": audio_fileset == whisper_fileset,
                    "is_drama": is_drama,
                })

    return chapters, skipped


def count_expected_chapters(canon: str) -> int:
    """Count total expected chapters for a testament."""
    books = NT_BOOKS if canon == "nt" else OT_BOOKS
    return sum(books.values())


# ─── Audio Availability ─────────────────────────────────────────────────────

def check_and_report_audio(
    work_items: List[dict],
    output_dir: Path,
    required_chapters: Optional[Dict[str, Dict[str, Set[int]]]] = None,
):
    """Report audio availability for all work items."""
    log("Audio availability report:")
    log("")
    for item in work_items:
        iso = item["iso"]
        canon = item["canon"]
        distinct_id = item["distinct_id"]
        canon_refs = required_chapters.get(canon) if required_chapters else None
        expected = sum(len(chs) for chs in canon_refs.values()) if canon_refs else count_expected_chapters(canon)

        if not item["has_downloads"]:
            log(f"  {iso}/{canon.upper()}/{distinct_id}: 0/{expected} chapters (no downloads)")
            continue

        chapters, skipped = discover_chapter_files(
            iso, canon, distinct_id, output_dir, force=True, required_chapters=canon_refs
        )
        total_available = len(chapters) + skipped
        books_found = sorted(set(c["book"] for c in chapters))

        if total_available == 0:
            log(f"  {iso}/{canon.upper()}/{distinct_id}: 0/{expected} chapters (no audio+text pairs)")
        elif total_available >= expected:
            done_info = f", {skipped} already complete" if skipped else ""
            log(f"  {iso}/{canon.upper()}/{distinct_id}: {total_available}/{expected} chapters (complete{done_info})")
        else:
            log(f"  {iso}/{canon.upper()}/{distinct_id}: {total_available}/{expected} chapters ({', '.join(books_found)})")


def download_audio_for_chapters(
    iso: str,
    canon: str,
    distinct_id: Optional[str],
    chapters_by_book: Dict[str, Set[int]],
    audio_fileset: Optional[str] = None,
    text_fileset: Optional[str] = None,
    text_source: Optional[str] = None,
) -> bool:
    """Fetch audio+text for specific chapters via download_language_content.download_job().

    audio_fileset/text_fileset/text_source come from an enriched batch job
    (see batch_manifest.py) when the caller has one — passed straight
    through so download_job() can skip its own catalog resolution entirely.
    """
    if distinct_id is None:
        log(
            f"  Skipping {iso}/{canon.upper()}: no distinct_id resolved "
            "(batch job must supply one — see batch_manifest.py)",
            "WARN",
        )
        return False
    job = {
        "iso": iso,
        "canon": canon.upper(),
        "distinct_id": distinct_id,
        "chapters": {book: sorted(chs) for book, chs in chapters_by_book.items()},
    }
    if audio_fileset:
        job["audio_fileset"] = audio_fileset
        job["text_fileset"] = text_fileset
        job["text_source"] = text_source
    log(f"Downloading: {iso}/{canon.upper()}/{distinct_id}")
    return download_job(job, content_types=["audio", "text"])


# ─── Whisper Transcription ──────────────────────────────────────────────────

# Module-level cache for the faster-whisper model so it is loaded only once.
_fw_model_cache: dict = {}


def load_whisper_model(model_name: str):
    """Load (or return cached) the Whisper model for the current backend.

    For mlx-whisper the model is loaded lazily per-call (MLX handles caching
    internally). For faster-whisper the model is cached in _fw_model_cache so
    it is only loaded from disk once per process.

    Returns the loaded model object, or None for the MLX backend (which does
    not expose a persistent model handle).
    """
    if _USE_MLX:
        return None  # mlx_whisper.transcribe() handles its own caching

    # Log current VRAM state before deciding (helps diagnose OOM issues)
    if _USE_CUDA:
        try:
            import torch
            free, total = torch.cuda.mem_get_info(0)
            log(f"VRAM before Whisper load: {free // (1024**2)} MB free / {total // (1024**2)} MB total")
        except Exception:
            pass

    # Prefer CUDA but fall back to CPU (int8) if VRAM is tight or CPU is forced.
    # _enough_vram_for_whisper() already checks _FORCE_WHISPER_CPU internally.
    if _enough_vram_for_whisper(model_name):
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    key = (model_name, device)
    if key not in _fw_model_cache:
        from faster_whisper import WhisperModel
        log(f"Loading faster-whisper model '{model_name}' on {device} ({compute_type}) ...")
        t0 = time.time()
        _fw_model_cache[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
        log(f"Whisper model loaded in {time.time() - t0:.1f}s")
    return _fw_model_cache[key]


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: Optional[str] = None,
    _model=None,
) -> dict:
    """Transcribe a single audio file with word timestamps.

    Backend selection (automatic):
      - Apple Silicon Mac  → mlx-whisper  (Metal GPU via MLX)
      - NVIDIA CUDA GPU    → faster-whisper (CUDA)
      - Other / CPU        → faster-whisper (CPU)

    Pass a pre-loaded faster-whisper model via `_model` to avoid reloading it
    on every call (use load_whisper_model() to obtain one).

    Returns a dict in the same shape as openai-whisper:
      {"segments": [{"start", "end", "text", "words": [{"word", "start", "end", "probability"}]}]}
    """
    if _USE_MLX:
        import mlx_whisper
        kwargs = {
            "path_or_hf_repo": model_name,
            "word_timestamps": True,
        }
        if language:
            kwargs["language"] = language
        return mlx_whisper.transcribe(str(audio_path), **kwargs)

    # ── faster-whisper path (CUDA or CPU) ──────────────────────────────────
    model = _model if _model is not None else load_whisper_model(model_name)

    # Free any cached but unused CUDA memory before inference to reduce
    # fragmentation-related OOM errors on GPUs with limited VRAM.
    if _USE_CUDA:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    fw_segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
    )

    # Convert faster-whisper output to openai-whisper-compatible dict
    segments = []
    for seg in fw_segments:
        words = []
        for w in (seg.words or []):
            words.append({
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            })
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": words,
        })

    return {"segments": segments}


# ─── Verse Alignment ────────────────────────────────────────────────────────

def build_word_timeline(segments: List[dict]) -> List[dict]:
    """
    Build a word-level timeline from Whisper segments.

    Each entry: {"text": word, "start": timestamp, "end": timestamp}

    Uses word-level timestamps from segments when available,
    otherwise interpolates word positions within each segment.
    """
    timeline = []

    for seg in segments:
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                entry = {
                    "text": w.get("word", w.get("text", "")),
                    "start": w["start"],
                    "end": w.get("end", w["start"]),
                }
                if "probability" in w:
                    entry["score"] = w["probability"]
                timeline.append(entry)
        else:
            text = seg.get("text", "").strip()
            if not text:
                continue
            words = text.split()
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_duration = seg_end - seg_start

            for i, word in enumerate(words):
                if len(words) == 1:
                    t_start = seg_start
                    t_end = seg_end
                else:
                    t_start = seg_start + seg_duration * (i / len(words))
                    t_end = seg_start + seg_duration * ((i + 1) / len(words))
                timeline.append({"text": word, "start": t_start, "end": t_end})

    return timeline


def _word_similarity(ref_words: List[str], whisper_window: List[str]) -> float:
    """Compute word-level similarity between reference words and a Whisper window.

    Returns fraction of reference words that have a close match (>=0.6 char similarity)
    in the whisper window, using greedy sequential matching.
    """
    if not ref_words or not whisper_window:
        return 0.0
    matches = 0
    used = set()
    for rw in ref_words:
        best = 0.0
        best_j = -1
        for j, ww in enumerate(whisper_window):
            if j in used:
                continue
            r = difflib.SequenceMatcher(None, rw, ww).ratio()
            if r > best:
                best = r
                best_j = j
        if best >= 0.6 and best_j >= 0:
            matches += 1
            used.add(best_j)
    return matches / len(ref_words)


def align_to_verses(
    segments: List[dict],
    verse_texts: List[str],
    book: str,
    chapter_str: str,
    config: Optional[LanguageConfig] = None,
) -> Tuple[List[dict], dict, int, List[dict]]:
    """
    Align Whisper segments to verse boundaries using the text file.

    Uses a two-pass approach:
      Pass 1: Find anchor verses with confident word-level similarity matches
      Pass 2: Interpolate timestamps for unmatched verses between anchors

    Returns (timing_entries, word_timing, matched_verse_count, word_timeline).

    word_timing is compact format:
      {"book": "REV", "chapter": "15", "verses": {"1": [3.27, 3.54, ...], "2": [...]}}
    where each array position corresponds to a word in the verse text.
    null means the word could not be confidently matched.

    word_timeline is the raw Whisper word-level output:
      [{"text": word, "start": timestamp}, ...]
    """
    if config is None:
        config = load_language_config("default")

    word_timeline = build_word_timeline(segments)

    word_timing = {"book": book, "chapter": chapter_str, "verses": {}}

    if not word_timeline:
        # No transcription output; generate interpolated timing + empty word timing
        timing = _interpolated_timing(verse_texts, segments, book, chapter_str)
        for verse_num, verse_text in enumerate(verse_texts, start=1):
            word_count = len(verse_text.strip().split()) if verse_text.strip() else 0
            word_timing["verses"][str(verse_num)] = [None] * word_count
        return timing, word_timing, 0, word_timeline

    total_duration = segments[-1]["end"] if segments else 0
    num_timeline = len(word_timeline)

    # Pre-normalize all timeline words once
    norm_timeline = [normalize_text(w["text"], config) for w in word_timeline]

    # ── Pass 1: Find anchor verses ──
    # Search for each verse around its expected proportional position
    anchors = {}  # verse_index -> timeline_index
    search_from = 0

    for vi, verse_text in enumerate(verse_texts):
        verse_text = verse_text.strip()
        if not verse_text:
            continue

        verse_words = verse_text.split()
        norm_vwords = [normalize_text(w, config) for w in verse_words]
        num_vwords = len(norm_vwords)

        # Expected position based on proportion through the text
        expected_pos = int(num_timeline * vi / len(verse_texts))
        # Search in a window around expected position, never before search_from
        window_start = max(search_from, expected_pos - num_vwords * 3)
        window_end = min(expected_pos + num_vwords * 5, num_timeline)

        best_sim = 0.0
        best_idx = -1

        for i in range(window_start, min(window_end, num_timeline - num_vwords + 1)):
            window = norm_timeline[i:i + num_vwords]
            sim = _word_similarity(norm_vwords, window)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
            if sim > 0.7:
                break

        if best_sim >= DEFAULT_MATCH_THRESHOLD and best_idx >= 0:
            anchors[vi] = best_idx
            search_from = best_idx + num_vwords

    matched = len(anchors)

    # ── Pass 2: Build timing entries + word timing ──
    # Verse 0 always at timestamp 0
    timing_entries = [{
        "book": book,
        "chapter": chapter_str,
        "verse_start": "0",
        "verse_start_alt": "0",
        "timestamp": 0,
    }]

    for vi, verse_text in enumerate(verse_texts):
        verse_num = vi + 1
        verse_text = verse_text.strip()

        if not verse_text:
            prev_time = timing_entries[-1]["timestamp"]
            timing_entries.append({
                "book": book,
                "chapter": chapter_str,
                "verse_start": str(verse_num),
                "verse_start_alt": str(verse_num),
                "timestamp": round(prev_time, 2),
            })
            word_timing["verses"][str(verse_num)] = []
            continue

        verse_words = verse_text.split()
        num_verse_words = len(verse_words)

        if vi in anchors:
            # Matched verse — use anchor timestamp
            best_idx = anchors[vi]
            timestamp = round(word_timeline[best_idx]["start"], 2)

            # Word-level alignment
            word_times = _align_verse_words(verse_words, word_timeline, best_idx, config)
        else:
            # Unmatched verse — interpolate between surrounding anchors
            timestamp = _interpolate_verse_time(
                vi, verse_texts, anchors, word_timeline, total_duration,
                timing_entries[-1]["timestamp"],
            )
            word_times = [None] * num_verse_words
            best_idx = -1

        timing_entries.append({
            "book": book,
            "chapter": chapter_str,
            "verse_start": str(verse_num),
            "verse_start_alt": str(verse_num),
            "timestamp": timestamp,
        })
        word_timing["verses"][str(verse_num)] = word_times

    return timing_entries, word_timing, matched, word_timeline


def _interpolate_verse_time(
    verse_idx: int,
    verse_texts: List[str],
    anchors: Dict[int, int],
    word_timeline: List[dict],
    total_duration: float,
    prev_timestamp: float,
) -> float:
    """Interpolate a timestamp for an unmatched verse between surrounding anchors."""
    # Find previous anchor
    prev_anchor_vi = None
    prev_anchor_time = 0.0
    for vi in range(verse_idx - 1, -1, -1):
        if vi in anchors:
            prev_anchor_vi = vi
            prev_anchor_time = word_timeline[anchors[vi]]["start"]
            break

    # Find next anchor
    next_anchor_vi = None
    next_anchor_time = total_duration
    for vi in range(verse_idx + 1, len(verse_texts)):
        if vi in anchors:
            next_anchor_vi = vi
            next_anchor_time = word_timeline[anchors[vi]]["start"]
            break

    # Count words between anchors for proportional interpolation
    start_vi = (prev_anchor_vi + 1) if prev_anchor_vi is not None else 0
    end_vi = next_anchor_vi if next_anchor_vi is not None else len(verse_texts)

    words_before = 0  # words from start_vi to verse_idx (exclusive)
    words_total = 0   # words from start_vi to end_vi (exclusive)
    for vi in range(start_vi, end_vi):
        vt = verse_texts[vi].strip()
        wc = len(vt.split()) if vt else 0
        if vi < verse_idx:
            words_before += wc
        words_total += wc

    if words_total > 0:
        proportion = words_before / words_total
    else:
        # Fallback to equal spacing
        span = end_vi - start_vi
        proportion = (verse_idx - start_vi) / span if span > 0 else 0

    time_range = next_anchor_time - prev_anchor_time
    timestamp = prev_anchor_time + proportion * time_range
    return round(max(timestamp, prev_timestamp), 2)


def _align_verse_words(
    verse_words: List[str],
    timeline: List[dict],
    timeline_start: int,
    config: Optional[LanguageConfig] = None,
) -> list:
    """
    Align individual verse words against the Whisper word timeline.

    Uses sequential fuzzy matching starting from timeline_start.
    Returns a list of start times (or null) for each verse word.

    Handles:
    - 1:1 match: verse word matches timeline word -> take timestamp
    - Whisper split: one verse word matches multiple timeline words -> take first timestamp
    - Whisper merge: multiple verse words match one timeline word -> same timestamp for both
    - No match: -> null
    """
    if config is None:
        config = load_language_config("default")

    num_verse_words = len(verse_words)
    result = [None] * num_verse_words
    ti = timeline_start  # current position in timeline
    max_ti = min(timeline_start + num_verse_words * 3, len(timeline))  # search window

    for vi, verse_word in enumerate(verse_words):
        if ti >= max_ti:
            break

        norm_verse = normalize_text(verse_word, config)
        if not norm_verse:
            continue

        # Try direct match at current position
        best_ratio = 0.0
        best_offset = -1
        # Look ahead up to 3 positions for a match
        look_ahead = min(3, max_ti - ti)
        for offset in range(look_ahead):
            candidate = normalize_text(timeline[ti + offset]["text"], config)
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(None, norm_verse, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_offset = offset

            # Also try merging consecutive timeline words (Whisper split a word)
            if offset > 0:
                merged = normalize_text(
                    " ".join(timeline[ti + j]["text"] for j in range(offset + 1)),
                    config,
                )
                merge_ratio = difflib.SequenceMatcher(None, norm_verse, merged).ratio()
                if merge_ratio > best_ratio:
                    best_ratio = merge_ratio
                    best_offset = 0  # take start of the merged group

        if best_ratio >= 0.4:
            result[vi] = round(timeline[ti + best_offset]["start"], 2)
            ti = ti + best_offset + 1
        else:
            # No match — check if timeline word matches next verse word (Whisper merged)
            # Leave this word as null and don't advance timeline
            pass

    return result


def _interpolated_timing(
    verse_texts: List[str], segments: List[dict], book: str, chapter_str: str
) -> List[dict]:
    """Generate purely interpolated timing when Whisper produces no output."""
    total_duration = segments[-1]["end"] if segments else 0
    total_words = sum(len(v.split()) for v in verse_texts)

    entries = [{
        "book": book,
        "chapter": chapter_str,
        "verse_start": "0",
        "verse_start_alt": "0",
        "timestamp": 0,
    }]

    cumulative_words = 0
    for verse_num, verse_text in enumerate(verse_texts, start=1):
        if total_words > 0:
            proportion = cumulative_words / total_words
            timestamp = round(proportion * total_duration, 2)
        else:
            timestamp = 0
        entries.append({
            "book": book,
            "chapter": chapter_str,
            "verse_start": str(verse_num),
            "verse_start_alt": str(verse_num),
            "timestamp": timestamp,
        })
        cumulative_words += len(verse_text.split())

    return entries


def write_word_timing_json(word_timing: dict, output_path: Path):
    """Write compact word-level timing data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(word_timing, f, separators=(",", ":"))


def write_whisper_words_json(
    word_timeline: List[dict], book: str, chapter: str, output_path: Path
):
    """Write raw Whisper word-level timeline as intermediate file for alignment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "book": book,
        "chapter": chapter,
        "words": [
            {
                "text": w["text"],
                "start": round(w["start"], 2),
                "end": round(w.get("end", w["start"]), 2),
                **({"score": round(w["score"], 3)} if "score" in w else {}),
            }
            for w in word_timeline
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── SRT Generation ────────────────────────────────────────────────────

def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: List[dict]) -> str:
    """Convert Whisper segments to SRT subtitle format."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def write_srt(segments: List[dict], output_path: Path):
    """Write Whisper segments as an SRT subtitle file."""
    srt_content = segments_to_srt(segments)
    if not srt_content:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)


# ─── Output ─────────────────────────────────────────────────────────────────

def write_timing_json(entries: List[dict], output_path: Path):
    """Write verse timing data in the standard format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


# ─── Chapter Processing ────────────────────────────────────────────────────

def process_chapter(
    chapter: dict,
    model_name: str,
    whisper_language: Optional[str],
    _whisper_model=None,
) -> dict:
    """
    Process a single chapter: transcribe audio and write intermediate files.

    Generates:
      - _whisper_words.json in word-timing-data/ (raw word timeline for alignment)
      - .srt in export/timing-data/ (subtitle file)

    Alignment to verses (_timing.json, _words.json) is done separately by
    language-specific alignment scripts (e.g. align_words.py).

    Pass a pre-loaded faster-whisper model via `_whisper_model` to avoid
    reloading it on every chapter (use load_whisper_model() to obtain one).

    Returns stats dict.
    """
    book = chapter["book"]
    chapter_num = chapter["chapter"]
    audio_path = chapter["audio_path"]
    whisper_words_path = chapter["whisper_words_path"]
    srt_path = chapter["srt_path"]

    # Transcribe (always with word timestamps)
    start_time = time.time()
    result = transcribe_audio(audio_path, model_name, whisper_language, _model=_whisper_model)
    transcribe_time = time.time() - start_time

    segments = result.get("segments", [])
    duration = segments[-1]["end"] if segments else 0

    # Build word timeline from Whisper output
    word_timeline = build_word_timeline(segments)

    # Write outputs
    write_whisper_words_json(word_timeline, book, str(chapter_num), whisper_words_path)
    write_srt(segments, srt_path)

    return {
        "duration": duration,
        "transcribe_time": transcribe_time,
        "words": len(word_timeline),
        "segments": len(segments),
    }


# ─── Main Pipeline ─────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser(
        description="Generate verse timing from Bible audio using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --iso nld
  %(prog)s --iso-list nld,pol,ell
  %(prog)s --tier 1
  %(prog)s --all
  %(prog)s --iso nld --template John
  %(prog)s --iso nld --testament nt
  %(prog)s --iso nld --dry-run
  %(prog)s --iso nld --no-download
  %(prog)s --tier 1 --check-audio
        """,
    )

    # Language selection
    lang_group = parser.add_argument_group("Language selection (at least one required)")
    lang_group.add_argument("--iso", type=str, help="Single language ISO 639-3 code (e.g., nld)")
    lang_group.add_argument("--iso-list", type=str, help="Comma-separated ISO codes (e.g., nld,pol,ell)")
    lang_group.add_argument("--tier", type=int, choices=[1, 2, 3], help="Process all languages of this tier")
    lang_group.add_argument("--all", action="store_true", help="Process all priority languages")

    # Scope
    scope_group = parser.add_argument_group("Scope")
    scope_group.add_argument(
        "--template", type=str,
        help="Only process chapters from this template (e.g., John, OBS). Default: all templates",
    )
    scope_group.add_argument(
        "--testament", type=str, choices=["nt", "ot", "both"], default="both",
        help="Which testament to process (default: both)",
    )
    scope_group.add_argument(
        "--book", type=str,
        help="Filter to a specific book (e.g., LUK, GEN)",
    )
    scope_group.add_argument(
        "--chapter", type=int,
        help="Filter to a specific chapter number (requires --book)",
    )

    # Whisper options
    whisper_group = parser.add_argument_group("Whisper options")
    whisper_group.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Whisper model (default: {DEFAULT_MODEL})",
    )
    whisper_group.add_argument(
        "--whisper-cpu", action="store_true",
        help="Force faster-whisper to run on CPU even when a CUDA GPU is available "
             "(useful when GPU RAM is needed by another model). "
             "No effect on Apple Silicon (mlx-whisper always uses Metal).",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing options")
    proc_group.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    proc_group.add_argument("--force", action="store_true", help="Re-process even if output exists")
    proc_group.add_argument("--check-audio", action="store_true", help="Only report audio availability")
    proc_group.add_argument(
        "--no-download", action="store_true",
        help="Skip auto-download (only process already-downloaded files)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # Validate selection
    if not any([args.iso, args.iso_list, args.tier is not None, args.all]):
        parser.error("Specify at least one of: --iso, --iso-list, --tier, --all")

    log("=" * 70)
    log("Whisper Bible Audio Transcription Pipeline")
    log("=" * 70)

    # ── Step 1: Scan templates to determine required chapters ──

    template_label = args.template if args.template else "all templates"
    log(f"Scanning {template_label} for Bible references...")
    all_refs = load_all_template_refs(args.template)

    if not all_refs:
        log(f"No Bible references found in {template_label}", "ERROR")
        sys.exit(1)

    refs_by_canon = classify_template_refs(all_refs)

    # Apply testament filter
    if args.testament == "nt":
        refs_by_canon["ot"] = {}
    elif args.testament == "ot":
        refs_by_canon["nt"] = {}

    # Apply --book / --chapter filter
    if args.book:
        book_upper = args.book.upper()
        for canon in ("nt", "ot"):
            canon_refs = refs_by_canon.get(canon, {})
            if book_upper in canon_refs:
                if args.chapter:
                    refs_by_canon[canon] = {book_upper: {args.chapter}}
                else:
                    refs_by_canon[canon] = {book_upper: canon_refs[book_upper]}
            else:
                refs_by_canon[canon] = {}

    total_chapters = sum(len(chs) for canon_refs in refs_by_canon.values() for chs in canon_refs.values())
    for canon in ("nt", "ot"):
        canon_refs = refs_by_canon.get(canon, {})
        if canon_refs:
            book_summary = ", ".join(
                f"{b}:{','.join(str(c) for c in sorted(chs))}"
                for b, chs in sorted(canon_refs.items())
            )
            log(f"  {canon.upper()}: {sum(len(chs) for chs in canon_refs.values())} chapters "
                f"({book_summary})")
    log(f"  Total: {total_chapters} chapters needed from templates")

    # ── Step 2: Resolve languages ──

    all_languages = load_priority_languages()
    languages = resolve_languages(args, all_languages)
    log(f"Languages selected: {len(languages)}")
    for lang in languages:
        log(f"  {lang['iso']} ({lang['language']}) - tier {lang['tier']}")

    # Generate work items (filtered by template canons)
    work_items = generate_work_items(languages, args.testament, refs_by_canon)
    if not work_items:
        log("No syncable filesets found for the selected languages/testament", "WARN")
        sys.exit(0)

    log(f"Work items: {len(work_items)} fileset(s)")

    # Check audio mode
    if args.check_audio:
        check_and_report_audio(work_items, args.output_dir, refs_by_canon)
        return

    # ── Step 3: Auto-download missing audio+text ──

    if not args.no_download and not args.dry_run:
        for item in work_items:
            iso = item["iso"]
            canon = item["canon"]
            canon_refs = refs_by_canon.get(canon, {})
            if not canon_refs:
                continue

            # Check what's already downloaded (flat structure)
            existing = 0
            iso_dir = DOWNLOADS_DIR / canon / iso
            if iso_dir.exists():
                for book, chapters in canon_refs.items():
                    for distinct_dir in iso_dir.iterdir():
                        if not distinct_dir.is_dir():
                            continue
                        book_dir = distinct_dir / book
                        if not book_dir.exists():
                            continue
                        for ch in chapters:
                            ch_str = f"{ch:03d}"
                            mp3s = list(book_dir.glob(f"{book}_{ch_str}_*.mp3"))
                            txts = list(book_dir.glob(f"{book}_{ch_str}_*.txt"))
                            if mp3s and txts:
                                existing += 1
                        break  # Only check first distinct_id per category

            needed = sum(len(chs) for chs in canon_refs.values())
            if existing >= needed:
                continue  # All files present, skip download

            log(f"Downloading audio+text for {iso}/{canon.upper()}...")
            download_audio_for_chapters(iso, canon, item["distinct_id"], canon_refs)

        # Regenerate work items after download (directories may now exist)
        work_items = generate_work_items(languages, args.testament, refs_by_canon)

    # ── Step 4: Transcribe ──

    # Apply --whisper-cpu override before any model loading
    if args.whisper_cpu:
        set_whisper_cpu(True)

    # Check that the selected backend is available (unless dry-run)
    whisper_model_instance = None
    if not args.dry_run:
        if _USE_MLX:
            try:
                import mlx_whisper  # noqa: F401
            except ImportError:
                log("mlx-whisper not installed. Run: pip install -r requirements-whisper.txt", "ERROR")
                sys.exit(1)
            log("Backend: mlx-whisper (Apple Silicon / Metal)")
        else:
            try:
                from faster_whisper import WhisperModel  # noqa: F401
            except ImportError:
                log("faster-whisper not installed. Run: pip install faster-whisper", "ERROR")
                sys.exit(1)
            # Pre-load the model once so it is not reloaded for every chapter.
            # load_whisper_model() automatically falls back to CPU if VRAM is tight.
            whisper_model_instance = load_whisper_model(args.model)
            # Report which device was actually selected after VRAM check
            if _FORCE_WHISPER_CPU:
                actual_device = "CPU (forced)"
            elif _enough_vram_for_whisper(args.model):
                actual_device = "CUDA"
            else:
                actual_device = "CPU (VRAM fallback)" if _USE_CUDA else "CPU"
            log(f"Backend: faster-whisper ({actual_device})")

    total_stats = {
        "languages_processed": 0,
        "chapters_processed": 0,
        "chapters_skipped": 0,
        "chapters_failed": 0,
        "total_audio_duration": 0,
        "total_processing_time": 0,
    }

    processed_languages = set()

    for item_idx, item in enumerate(work_items):
        iso = item["iso"]
        canon = item["canon"]
        distinct_id = item["distinct_id"]
        lang_name = item["language"]
        tier = item["tier"]
        canon_refs = refs_by_canon.get(canon, {})

        log("")
        log(f"[{item_idx + 1}/{len(work_items)}] {iso} ({lang_name}), tier {tier}, "
            f"{canon.upper()}/{distinct_id}")
        log("-" * 50)

        if not item["has_downloads"]:
            log("No audio downloaded", "WARN")
            continue

        # Discover chapters (filtered to template refs)
        chapters, skipped = discover_chapter_files(
            iso, canon, distinct_id, args.output_dir, args.force,
            required_chapters=canon_refs if canon_refs else None,
        )
        total_stats["chapters_skipped"] += skipped

        if skipped > 0:
            log(f"Skipping {skipped} already-processed chapter(s)")

        if not chapters:
            if skipped > 0:
                log("All chapters already processed")
            else:
                log("No audio+text pairs found", "WARN")
            continue

        expected = sum(len(chs) for chs in canon_refs.values()) if canon_refs else count_expected_chapters(canon)
        log(f"Chapters to process: {len(chapters)}/{expected}")

        if args.dry_run:
            for ch in chapters:
                log(f"  Would process: {ch['book']} {ch['chapter']}")
            continue

        # Get Whisper language hint
        whisper_lang = get_whisper_language(iso)
        if whisper_lang:
            log(f"Whisper language: {whisper_lang}")
        else:
            log("Whisper language: auto-detect")

        # Process each chapter
        for ch_idx, chapter in enumerate(chapters):
            book = chapter["book"]
            ch_num = chapter["chapter"]
            label = f"[{ch_idx + 1}/{len(chapters)}] {book} {ch_num}"

            log(f"{label} - Transcribing...")

            try:
                stats = process_chapter(
                    chapter, args.model, whisper_lang,
                    _whisper_model=whisper_model_instance,
                )
                duration_str = format_duration(stats["duration"])
                speed = stats["duration"] / stats["transcribe_time"] if stats["transcribe_time"] > 0 else 0
                log(f"{label} - {stats['segments']} segments, {stats['words']} words, "
                    f"{duration_str} audio, {speed:.1f}x realtime")
                log(f"{label} - Wrote: {chapter['whisper_words_path']}")

                total_stats["chapters_processed"] += 1
                total_stats["total_audio_duration"] += stats["duration"]
                total_stats["total_processing_time"] += stats["transcribe_time"]

            except KeyboardInterrupt:
                log("Interrupted by user", "WARN")
                _print_summary(total_stats)
                sys.exit(1)
            except Exception as e:
                log(f"{label} - Failed: {e}", "ERROR")
                total_stats["chapters_failed"] += 1

        processed_languages.add(iso)

    total_stats["languages_processed"] = len(processed_languages)
    _print_summary(total_stats)


def _print_summary(stats: dict):
    """Print processing summary."""
    log("")
    log("=" * 70)
    log("Transcription Summary")
    log("=" * 70)
    log(f"  Languages processed:  {stats['languages_processed']}")
    log(f"  Chapters processed:   {stats['chapters_processed']}")
    log(f"  Chapters skipped:     {stats['chapters_skipped']} (already done)")
    log(f"  Chapters failed:      {stats['chapters_failed']}")
    log(f"  Total audio duration: {format_duration(stats['total_audio_duration'])}")
    log(f"  Total processing time: {format_duration(stats['total_processing_time'])}")
    if stats["total_processing_time"] > 0:
        speed = stats["total_audio_duration"] / stats["total_processing_time"]
        log(f"  Avg speed:            {speed:.1f}x realtime")
    log("=" * 70)


if __name__ == "__main__":
    main()
