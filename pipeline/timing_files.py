"""Where pipeline output and DBT's downloaded timing live on disk, and how to
read them. Shared by align_pipeline.py's end-of-language checks and the
reporting tools in tools/ (quality_report, check_timing_quality,
three_way_arbiter, ...). Moved here from tools/quality_report.py on
2026-10-03 so pipeline code never imports from tools/.
"""
import json
from pathlib import Path

TIMING_DIR = Path("export/timing-data")
DOWNLOADS_DIR = Path("downloads/BB")
TIMECODE_CATEGORIES = ["with-timecode", "audio-with-timecode"]


def find_quality_files(iso: str, testament: str = None) -> list:
    """Find all *_words_quality.json files for a language."""
    files = []
    canons = _get_canons(testament)

    for canon in canons:
        canon_dir = TIMING_DIR / canon / iso
        if not canon_dir.exists():
            continue
        for qf in sorted(canon_dir.rglob("*_words_quality.json")):
            files.append((canon, qf))
    return files


def find_pipeline_timing_files(iso: str, testament: str = None) -> list:
    """Find all *_timing.json files in pipeline output for a language."""
    files = []
    canons = _get_canons(testament)

    for canon in canons:
        canon_dir = TIMING_DIR / canon / iso
        if not canon_dir.exists():
            continue
        for tf in sorted(canon_dir.rglob("*_timing.json")):
            files.append((canon, tf))
    return files


def find_downloaded_timecode(canon: str, iso: str, distinct_id: str, book: str, chapter_str: str) -> Path | None:
    """Find downloaded timecode file for a specific chapter."""
    for cat in TIMECODE_CATEGORIES:
        book_dir = DOWNLOADS_DIR / canon / cat / iso / distinct_id / book
        if not book_dir.exists():
            continue
        matches = list(book_dir.glob(f"{book}_{chapter_str}_*_timing.json"))
        if matches:
            return matches[0]
    return None


def find_all_downloaded_timecode(iso: str, testament: str = None) -> dict:
    """Find all downloaded timecode files for a language.

    Returns: {(canon, distinct_id, book, chapter_str): Path}
    """
    result = {}
    canons = _get_canons(testament)

    for canon in canons:
        # Search in category subdirs (with-timecode, audio-with-timecode)
        search_dirs = []
        for cat in TIMECODE_CATEGORIES:
            cat_dir = DOWNLOADS_DIR / canon / cat / iso
            if cat_dir.exists():
                search_dirs.append(cat_dir)
        # Also search direct language dir (downloads/BB/{canon}/{iso}/)
        direct_dir = DOWNLOADS_DIR / canon / iso
        if direct_dir.exists():
            search_dirs.append(direct_dir)

        for search_dir in search_dirs:
            for tf in search_dir.rglob("*_timing.json"):
                distinct_id = tf.parent.parent.name
                book = tf.parent.name
                # Parse chapter from filename: BOOK_CCC_FILESET_timing.json
                parts = tf.stem.replace("_timing", "").split("_", 2)
                if len(parts) >= 2:
                    chapter_str = parts[1]
                    key = (canon, distinct_id, book, chapter_str)
                    result[key] = tf
    return result


def load_timing_verses(path: Path) -> dict | None:
    """Load a timing.json file and return {verse_start: timestamp}.

    Handles both our own pipeline output (compact {"pos": [...]}) and
    DBT's original downloaded timecodes under downloads/BB/ (old verbose
    list-of-verse-dicts — an external source that stays in that format).

    Returns None (rather than raising) on a corrupt/empty file — a
    zero-byte or truncated timing.json is itself a real data-quality bug
    worth surfacing, not a reason to crash a whole corpus-wide comparison
    run over one bad file (confirmed 2026-09-24: export/timing-data/nt/
    xon/XONNVR/LUK/LUK_003_XO1NVRN1DA_timing.json was exactly this).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict) and "pos" in data:
        # pos[i] is verse (i+1)'s timestamp — no verse-0 slot, see
        # align_words.py's write_timing_json() docstring.
        return {str(i + 1): t for i, t in enumerate(data["pos"]) if t is not None}
    if isinstance(data, list):
        return {str(entry["verse_start"]): entry["timestamp"] for entry in data}
    # Neither shape — e.g. a {"status": "defer_to_dbt", ...} redirect record
    # (apply_arbiter_corrections.py's whole-chapter-redirect mechanism), or
    # some other non-timing dict. Not corrupt, just not verse timing at all.
    return None


def compare_verse_timings(downloaded: dict | None, pipeline: dict | None) -> dict | None:
    """Compare two verse timing dicts. Returns comparison stats.

    None input (a corrupt/unreadable timing.json from load_timing_verses)
    is treated the same as "no common verses" here — callers that need to
    tell "corrupt file" apart from "nothing in common" should check
    load_timing_verses()'s own return value before calling this.
    """
    if downloaded is None or pipeline is None:
        return None
    # Only compare non-zero verses that exist in both
    common_verses = []
    for v in sorted(downloaded.keys(), key=lambda x: int(x)):
        if v == "0":
            continue
        if v in pipeline:
            common_verses.append(v)

    if not common_verses:
        return None

    deltas = []
    verse_deltas = {}
    for v in common_verses:
        delta = abs(pipeline[v] - downloaded[v])
        deltas.append(delta)
        verse_deltas[v] = pipeline[v] - downloaded[v]

    mean_delta = sum(deltas) / len(deltas)
    max_delta = max(deltas)
    max_verse = common_verses[deltas.index(max_delta)]

    if mean_delta < 0.5:
        status = "GOOD"
    elif mean_delta < 2.0:
        status = "DRIFT"
    else:
        status = "BAD"

    return {
        "common_verses": len(common_verses),
        "mean_delta": mean_delta,
        "max_delta": max_delta,
        "max_verse": max_verse,
        "status": status,
        "verse_deltas": verse_deltas,
        "downloaded": downloaded,
        "pipeline": pipeline,
    }


def load_quality(path: Path) -> dict:
    """Load a quality JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_canons(testament: str = None) -> list:
    canons = []
    if testament in (None, "nt", "both"):
        canons.append("nt")
    if testament in (None, "ot", "both"):
        canons.append("ot")
    return canons


def _parse_timing_path(timing_path: Path) -> tuple:
    """Extract (distinct_id, book, chapter_str) from a pipeline timing path."""
    distinct_id = timing_path.parent.parent.name
    book = timing_path.parent.name
    parts = timing_path.stem.replace("_timing", "").split("_", 2)
    chapter_str = parts[1] if len(parts) >= 2 else None
    return distinct_id, book, chapter_str
