#!/usr/bin/env python3
"""
Hebrew word alignment fusion script (Step 2 of 2).

Reads pre-computed word timelines from word-timing-data/:
  - *_whisper_words.json  (from whisper_transcribe.py, Step 1a)
  - *_mms_words.json      (from mms_align_words.py, Step 1b)

Fuses both sources with reference text to produce verse-level and word-level
timing files in export/timing-data/ for consumption by export_templates.py.

When both sources are available, MMS-FA is used as primary (it aligns against
known text) and Whisper is used for validation/adjustment. When only one
source is available, that source is used alone.

This script is language-specific for Hebrew — it handles niqqud stripping,
maqaf splitting, parashah marker removal, and YHVH pronunciation mapping.

Usage:
    # Align all available Hebrew chapters
    python align_words_heb.py --iso heb

    # Only OT
    python align_words_heb.py --iso heb --testament ot

    # Re-align even if output exists
    python align_words_heb.py --iso heb --force

    # Preview what would be done
    python align_words_heb.py --iso heb --dry-run

    # Align a specific chapter
    python align_words_heb.py --iso heb --book GEN --chapter 17
"""

import argparse
import difflib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Constants ──────────────────────────────────────────────────────────────

DOWNLOADS_DIR = Path("downloads/BB")
OUTPUT_DIR = Path("export/timing-data")
WORD_TIMING_DIR = Path("word-timing-data")

WHISPER_MATCH_THRESHOLD = 0.5

# Categories that may contain reference text (searched in order of priority)
AUDIO_TEXT_CATEGORIES = ["syncable", "with-timecode", "audio-with-timecode"]


# ─── Logging ────────────────────────────────────────────────────────────────

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ─── File I/O ───────────────────────────────────────────────────────────────

def load_word_timeline(path: Path) -> List[dict]:
    """Load a *_whisper_words.json or *_mms_words.json file.

    Returns list of word dicts with keys: text, start, end (optional), score (optional).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = []
    for w in data.get("words", []):
        entry = {
            "text": w["text"],
            "start": w["start"],
            "end": w.get("end", w["start"]),
        }
        if "score" in w:
            entry["score"] = w["score"]
        words.append(entry)
    return words


def write_timing_json(entries: List[dict], output_path: Path):
    """Write verse timing data in the standard format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def write_word_timing_json(word_timing: dict, output_path: Path):
    """Write compact word-level timing data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(word_timing, f, separators=(",", ":"))


# ─── Text Normalization (Hebrew-specific) ───────────────────────────────────

def strip_non_spoken_markers(text: str) -> str:
    """Remove parashah/setumah markers and other non-spoken annotations."""
    text = re.sub(r"[{(][פס][})]", "", text)
    text = re.sub(r"(?:^|\s)[פס](?:\s|$)", " ", text)
    return text.strip()


_PRONUNCIATION_MAP = {
    "יהוה": "adonay",
}


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching.

    Hebrew-specific: strip niqqud, maqaf → space, apply pronunciation map,
    strip punctuation, collapse whitespace.
    """
    text = text.lower()
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("\u05be", " ")
    for heb, pron in _PRONUNCIATION_MAP.items():
        text = text.replace(heb, pron)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── MMS-FA Verse Mapping ──────────────────────────────────────────────────

def _map_mms_to_verses(
    mms_words: List[dict],
    verse_texts: List[str],
    book: str,
    chapter_str: str,
) -> Tuple[List[dict], dict, int]:
    """Map flat MMS word results to verse boundaries.

    The MMS words are aligned 1:1 with the cleaned reference text words
    (after strip_non_spoken_markers + normalize), so we count words per
    verse and slice accordingly.

    Returns (timing_entries, word_timing, verse_count).
    """
    timing_entries = [{
        "book": book,
        "chapter": chapter_str,
        "verse_start": "0",
        "verse_start_alt": "0",
        "timestamp": 0,
    }]
    word_timing = {"book": book, "chapter": chapter_str, "verses": {}}

    word_idx = 0
    for vi, verse_text in enumerate(verse_texts):
        verse_num = vi + 1
        cleaned = _clean_verse(verse_text)

        if not cleaned:
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

        verse_word_count = len(cleaned.split())
        verse_words = mms_words[word_idx:word_idx + verse_word_count]

        # First valid timestamp in this verse
        verse_time = None
        for w in verse_words:
            if w["start"] is not None and w["start"] > 0:
                verse_time = w["start"]
                break
        if verse_time is None:
            verse_time = timing_entries[-1]["timestamp"]

        timing_entries.append({
            "book": book,
            "chapter": chapter_str,
            "verse_start": str(verse_num),
            "verse_start_alt": str(verse_num),
            "timestamp": round(verse_time, 2),
        })

        word_times = []
        for w in verse_words:
            if w["start"] is not None and w["start"] > 0:
                word_times.append(round(w["start"], 2))
            else:
                word_times.append(None)
        word_timing["verses"][str(verse_num)] = word_times

        word_idx += verse_word_count

    return timing_entries, word_timing, len(verse_texts)


def _clean_verse(text: str) -> str:
    """Clean a verse for word counting (same transform as mms_align_words.py)."""
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("\u05be", " ")
    for heb, pron in _PRONUNCIATION_MAP.items():
        text = text.replace(heb, pron)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Whisper Verse Alignment ───────────────────────────────────────────────

def _word_similarity(ref_words: List[str], whisper_window: List[str]) -> float:
    """Compute word-level similarity between reference words and a Whisper window."""
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


def _align_whisper_to_verses(
    whisper_words: List[dict],
    verse_texts: List[str],
    book: str,
    chapter_str: str,
) -> Tuple[List[dict], dict, int]:
    """Align Whisper word timeline to verse boundaries using fuzzy matching.

    Returns (timing_entries, word_timing, matched_verse_count).
    """
    word_timing = {"book": book, "chapter": chapter_str, "verses": {}}

    total_duration = whisper_words[-1]["end"] if whisper_words else 0
    num_timeline = len(whisper_words)

    # Pre-normalize all timeline words
    norm_timeline = [normalize_text(w["text"]) for w in whisper_words]

    # Pass 1: Find anchor verses
    anchors = {}
    search_from = 0

    for vi, verse_text in enumerate(verse_texts):
        verse_text = verse_text.strip()
        if not verse_text:
            continue
        verse_words = verse_text.split()
        norm_vwords = [normalize_text(w) for w in verse_words]
        num_vwords = len(norm_vwords)

        expected_pos = int(num_timeline * vi / len(verse_texts))
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

        if best_sim >= WHISPER_MATCH_THRESHOLD and best_idx >= 0:
            anchors[vi] = best_idx
            search_from = best_idx + num_vwords

    matched = len(anchors)

    # Pass 2: Build timing entries
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
            best_idx = anchors[vi]
            timestamp = round(whisper_words[best_idx]["start"], 2)
            word_times = _align_verse_words_whisper(verse_words, whisper_words, best_idx)
        else:
            timestamp = _interpolate_verse_time(
                vi, verse_texts, anchors, whisper_words, total_duration,
                timing_entries[-1]["timestamp"],
            )
            word_times = [None] * num_verse_words

        timing_entries.append({
            "book": book,
            "chapter": chapter_str,
            "verse_start": str(verse_num),
            "verse_start_alt": str(verse_num),
            "timestamp": timestamp,
        })
        word_timing["verses"][str(verse_num)] = word_times

    return timing_entries, word_timing, matched


def _align_verse_words_whisper(
    verse_words: List[str],
    timeline: List[dict],
    timeline_start: int,
) -> list:
    """Align individual verse words against Whisper timeline using fuzzy matching."""
    num_verse_words = len(verse_words)
    result = [None] * num_verse_words
    ti = timeline_start
    max_ti = min(timeline_start + num_verse_words * 3, len(timeline))

    for vi, verse_word in enumerate(verse_words):
        if ti >= max_ti:
            break
        norm_verse = normalize_text(verse_word)
        if not norm_verse:
            continue

        best_ratio = 0.0
        best_offset = -1
        look_ahead = min(3, max_ti - ti)
        for offset in range(look_ahead):
            candidate = normalize_text(timeline[ti + offset]["text"])
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(None, norm_verse, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_offset = offset
            if offset > 0:
                merged = normalize_text(
                    " ".join(timeline[ti + j]["text"] for j in range(offset + 1))
                )
                merge_ratio = difflib.SequenceMatcher(None, norm_verse, merged).ratio()
                if merge_ratio > best_ratio:
                    best_ratio = merge_ratio
                    best_offset = 0

        if best_ratio >= 0.4:
            result[vi] = round(timeline[ti + best_offset]["start"], 2)
            ti = ti + best_offset + 1

    return result


def _interpolate_verse_time(
    verse_idx: int,
    verse_texts: List[str],
    anchors: Dict[int, int],
    word_timeline: List[dict],
    total_duration: float,
    prev_timestamp: float,
) -> float:
    """Interpolate a timestamp for an unmatched verse between surrounding anchors."""
    prev_anchor_vi = None
    prev_anchor_time = 0.0
    for vi in range(verse_idx - 1, -1, -1):
        if vi in anchors:
            prev_anchor_vi = vi
            prev_anchor_time = word_timeline[anchors[vi]]["start"]
            break

    next_anchor_vi = None
    next_anchor_time = total_duration
    for vi in range(verse_idx + 1, len(verse_texts)):
        if vi in anchors:
            next_anchor_vi = vi
            next_anchor_time = word_timeline[anchors[vi]]["start"]
            break

    start_vi = (prev_anchor_vi + 1) if prev_anchor_vi is not None else 0
    end_vi = next_anchor_vi if next_anchor_vi is not None else len(verse_texts)

    words_before = 0
    words_total = 0
    for vi in range(start_vi, end_vi):
        vt = verse_texts[vi].strip()
        wc = len(vt.split()) if vt else 0
        if vi < verse_idx:
            words_before += wc
        words_total += wc

    if words_total > 0:
        proportion = words_before / words_total
    else:
        span = end_vi - start_vi
        proportion = (verse_idx - start_vi) / span if span > 0 else 0

    time_range = next_anchor_time - prev_anchor_time
    timestamp = prev_anchor_time + proportion * time_range
    return round(max(timestamp, prev_timestamp), 2)


# ─── Per-Word Fusion Logic ─────────────────────────────────────────────────

def _find_whisper_match(
    mms_word: dict,
    whisper_words: List[dict],
    whisper_norm: List[str],
    search_start: int,
    search_end: int,
    max_time_diff: float = 2.0,
) -> Optional[Tuple[int, float]]:
    """Find the best matching Whisper word for an MMS word.

    Searches whisper_words[search_start:search_end] for the closest match
    by combined time proximity and text similarity. Rejects matches where
    the time difference exceeds max_time_diff seconds, to prevent mixing
    timestamps from completely different positions in the audio.

    Returns (whisper_index, match_quality) or None.
    """
    mms_start = mms_word["start"]
    mms_norm = normalize_text(mms_word["text"])
    if not mms_norm:
        return None

    best_idx = -1
    best_quality = 0.0

    for j in range(search_start, min(search_end, len(whisper_words))):
        w_norm = whisper_norm[j]
        if not w_norm:
            continue

        # Hard time proximity gate — reject if too far apart
        time_diff = abs(mms_start - whisper_words[j]["start"])
        if time_diff > max_time_diff:
            # If the Whisper word is already past the MMS word by more than
            # the threshold, no point looking further (they're time-sorted)
            if whisper_words[j]["start"] > mms_start + max_time_diff:
                break
            continue

        # Text similarity (0–1)
        text_sim = difflib.SequenceMatcher(None, mms_norm, w_norm).ratio()

        # Time proximity bonus (closer = better, max 1.0 at 0s, 0.0 at max_time_diff)
        time_score = 1.0 - time_diff / max_time_diff

        # Combined quality: text match matters most, time is a tiebreaker
        quality = text_sim * 0.7 + time_score * 0.3

        if quality > best_quality:
            best_quality = quality
            best_idx = j

    if best_idx >= 0 and best_quality >= 0.4:
        return (best_idx, best_quality)
    return None


def fuse_words_per_word(
    mms_words: List[dict],
    whisper_words: List[dict],
    verse_texts: List[str],
    book: str,
    chapter_str: str,
) -> Tuple[List[dict], dict, dict]:
    """Per-word fusion: MMS backbone + Whisper refinement.

    For each reference word (from MMS-FA, 1:1 with ref text):
    - Find matching Whisper word by time + text similarity
    - If both have timestamps: pick the one with higher confidence score
    - If only MMS: use MMS
    - Track statistics on how many words were improved by Whisper

    Returns (timing_entries, word_timing, fusion_stats).
    """
    # Pre-normalize Whisper words
    whisper_norm = [normalize_text(w["text"]) for w in whisper_words]

    # Walk MMS words (backbone, 1:1 with ref text), fuse per word
    fused_words = []
    whisper_search_start = 0
    words_from_mms = 0
    words_from_whisper = 0
    words_improved = 0

    for mms_w in mms_words:
        mms_score = mms_w.get("score", 0.0)
        fused = {
            "text": mms_w["text"],
            "start": mms_w["start"],
            "end": mms_w.get("end", mms_w["start"]),
            "score": mms_score,
            "source": "mms",
        }

        # Search a window around the MMS word's time position
        search_end = min(whisper_search_start + 30, len(whisper_words))
        match = _find_whisper_match(
            mms_w, whisper_words, whisper_norm,
            whisper_search_start, search_end,
        )

        if match is not None:
            w_idx, match_quality = match
            w_word = whisper_words[w_idx]
            w_score = w_word.get("score", 0.0)

            # Pick the source with higher confidence
            if w_score > mms_score:
                fused["start"] = w_word["start"]
                fused["end"] = w_word.get("end", w_word["start"])
                fused["score"] = w_score
                fused["source"] = "whisper"
                words_from_whisper += 1
                words_improved += 1
            else:
                words_from_mms += 1

            # Advance search window past matched Whisper word
            whisper_search_start = w_idx + 1
        else:
            words_from_mms += 1

        fused_words.append(fused)

    # Now map fused words to verses (same logic as _map_mms_to_verses)
    timing_entries = [{
        "book": book,
        "chapter": chapter_str,
        "verse_start": "0",
        "verse_start_alt": "0",
        "timestamp": 0,
    }]
    word_timing = {"book": book, "chapter": chapter_str, "verses": {}}

    word_idx = 0
    for vi, verse_text in enumerate(verse_texts):
        verse_num = vi + 1
        cleaned = _clean_verse(verse_text)

        if not cleaned:
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

        verse_word_count = len(cleaned.split())
        verse_words = fused_words[word_idx:word_idx + verse_word_count]

        # First valid timestamp in this verse
        verse_time = None
        for w in verse_words:
            if w["start"] is not None and w["start"] > 0:
                verse_time = w["start"]
                break
        if verse_time is None:
            verse_time = timing_entries[-1]["timestamp"]

        timing_entries.append({
            "book": book,
            "chapter": chapter_str,
            "verse_start": str(verse_num),
            "verse_start_alt": str(verse_num),
            "timestamp": round(verse_time, 2),
        })

        word_times = []
        for w in verse_words:
            if w["start"] is not None and w["start"] > 0:
                word_times.append(round(w["start"], 2))
            else:
                word_times.append(None)
        word_timing["verses"][str(verse_num)] = word_times

        word_idx += verse_word_count

    # Enforce monotonicity within each verse — if fusion picked timestamps
    # from different sources that are slightly out of order, fix them.
    mono_fixes = 0
    for vnum, times in word_timing["verses"].items():
        for i in range(1, len(times)):
            if times[i] is not None and times[i - 1] is not None:
                if times[i] < times[i - 1]:
                    times[i] = times[i - 1]
                    mono_fixes += 1

    fusion_stats = {
        "total_words": len(fused_words),
        "from_mms": words_from_mms,
        "from_whisper": words_from_whisper,
        "improved": words_improved,
        "mono_fixes": mono_fixes,
    }

    return timing_entries, word_timing, fusion_stats


# ─── Work Item Discovery ───────────────────────────────────────────────────

def discover_work_items(
    iso: str,
    testament: Optional[str] = None,
    force: bool = False,
    book_filter: Optional[str] = None,
    chapter_filter: Optional[int] = None,
) -> List[dict]:
    """Find chapters that have word timing data (MMS and/or Whisper).

    Scans word-timing-data/{canon}/{iso}/{distinct_id}/{book}/ for
    *_mms_words.json and *_whisper_words.json files.
    Groups by (book, chapter, audio_fileset).
    """
    items = []

    canons = []
    if testament in (None, "ot", "both"):
        canons.append("ot")
    if testament in (None, "nt", "both"):
        canons.append("nt")

    for canon in canons:
        canon_dir = WORD_TIMING_DIR / canon / iso
        if not canon_dir.exists():
            continue

        # Collect all word timing files
        file_groups = {}  # (book, chapter_str, fileset) -> {mms_path, whisper_path}

        for word_file in sorted(canon_dir.rglob("*_mms_words.json")):
            key = _parse_word_file(word_file, "_mms_words")
            if key:
                file_groups.setdefault(key, {})["mms_path"] = word_file

        for word_file in sorted(canon_dir.rglob("*_whisper_words.json")):
            key = _parse_word_file(word_file, "_whisper_words")
            if key:
                file_groups.setdefault(key, {})["whisper_path"] = word_file

        for (book, chapter_str, audio_fileset), paths in sorted(file_groups.items()):
            try:
                chapter_num = int(chapter_str)
            except ValueError:
                continue

            if book_filter and book != book_filter:
                continue
            if chapter_filter is not None and chapter_num != chapter_filter:
                continue

            # Extract distinct_id from path
            sample_path = paths.get("mms_path") or paths.get("whisper_path")
            distinct_id = sample_path.parent.parent.name

            # Find reference text
            ref_path = _find_reference_text(canon, iso, distinct_id, book, chapter_str)

            # Build output paths
            out_book_dir = OUTPUT_DIR / canon / iso / distinct_id / book
            timing_path = out_book_dir / f"{book}_{chapter_str}_{audio_fileset}_timing.json"
            words_path = out_book_dir / f"{book}_{chapter_str}_{audio_fileset}_words.json"

            if timing_path.exists() and not force:
                continue

            items.append({
                "mms_path": paths.get("mms_path"),
                "whisper_path": paths.get("whisper_path"),
                "ref_text_path": ref_path,
                "timing_path": timing_path,
                "words_path": words_path,
                "book": book,
                "chapter": chapter_num,
                "chapter_str": chapter_str,
                "canon": canon,
                "iso": iso,
                "distinct_id": distinct_id,
                "audio_fileset": audio_fileset,
            })

    return items


def _parse_word_file(path: Path, suffix: str) -> Optional[Tuple[str, str, str]]:
    """Parse a word timing filename into (book, chapter_str, audio_fileset)."""
    stem = path.stem.replace(suffix, "")
    parts = stem.split("_", 2)
    if len(parts) < 3:
        return None
    return (parts[0], parts[1], parts[2])


def _find_reference_text(
    canon: str, iso: str, distinct_id: str, book: str, chapter_str: str,
) -> Optional[Path]:
    """Find the reference .txt file for a chapter."""
    for category in AUDIO_TEXT_CATEGORIES:
        base_dir = DOWNLOADS_DIR / canon / category / iso / distinct_id / book
        if not base_dir.exists():
            continue
        candidates = list(base_dir.glob(f"{book}_{chapter_str}_*.txt"))
        if candidates:
            return candidates[0]
    return None


# ─── Chapter Processing ────────────────────────────────────────────────────

def process_chapter(item: dict) -> dict:
    """Fuse word timelines and produce final verse timing.

    Returns stats dict.
    """
    book = item["book"]
    chapter_str = item["chapter_str"]
    ref_text_path = item["ref_text_path"]
    timing_path = item["timing_path"]
    words_path = item["words_path"]

    if ref_text_path is None:
        return {"error": "No reference text found"}

    # Read verse texts, stripping non-spoken markers
    with open(ref_text_path, "r", encoding="utf-8") as f:
        verse_texts = [strip_non_spoken_markers(line.rstrip("\n")) for line in f.readlines()]
    while verse_texts and not verse_texts[-1].strip():
        verse_texts.pop()

    verse_count = len(verse_texts)

    # Load MMS words
    mms_words = None
    mms_avg_score = 0.0
    if item["mms_path"]:
        mms_words = load_word_timeline(item["mms_path"])
        if mms_words:
            scores = [w.get("score", 0) for w in mms_words if w.get("score", 0) > 0]
            mms_avg_score = sum(scores) / len(scores) if scores else 0

    # Load Whisper words
    whisper_words = None
    whisper_avg_score = 0.0
    if item["whisper_path"]:
        whisper_words = load_word_timeline(item["whisper_path"])
        if whisper_words:
            scores = [w.get("score", 0) for w in whisper_words if w.get("score", 0) > 0]
            whisper_avg_score = sum(scores) / len(scores) if scores else 0

    # Determine strategy and run fusion
    if mms_words and whisper_words:
        # Both available — per-word fusion
        final_timing, final_word_timing, fusion_stats = fuse_words_per_word(
            mms_words, whisper_words, verse_texts, book, chapter_str,
        )
        source = "fused"
    elif mms_words:
        # MMS only
        final_timing, final_word_timing, _ = _map_mms_to_verses(
            mms_words, verse_texts, book, chapter_str,
        )
        source = "mms"
        fusion_stats = None
    elif whisper_words:
        # Whisper only — use verse-level alignment
        final_timing, final_word_timing, whisper_matched = _align_whisper_to_verses(
            whisper_words, verse_texts, book, chapter_str,
        )
        source = "whisper"
        fusion_stats = None
    else:
        return {"error": "No usable word timing data"}

    # Write outputs
    write_timing_json(final_timing, timing_path)
    write_word_timing_json(final_word_timing, words_path)

    result = {
        "verses": verse_count,
        "source": source,
        "mms_score": round(mms_avg_score, 3) if mms_words else None,
        "whisper_score": round(whisper_avg_score, 3) if whisper_words else None,
    }
    if fusion_stats:
        result["fusion"] = fusion_stats

    return result


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fuse Whisper + MMS word timelines into verse timing for Hebrew",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--iso", type=str, default="heb", help="ISO 639-3 code (default: heb)")
    parser.add_argument("--testament", type=str, choices=["nt", "ot", "both"], default=None,
                        help="Which testament to process (default: all available)")
    parser.add_argument("--book", type=str, default=None, help="Filter to a specific book (e.g. GEN)")
    parser.add_argument("--chapter", type=int, default=None, help="Filter to a specific chapter number")
    parser.add_argument("--force", action="store_true", help="Re-align even if output exists")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")

    args = parser.parse_args()

    log("=" * 60)
    log(f"Hebrew Word Alignment Fusion — {args.iso}")
    log("=" * 60)

    items = discover_work_items(
        iso=args.iso,
        testament=args.testament,
        force=args.force,
        book_filter=args.book,
        chapter_filter=args.chapter,
    )

    if not items:
        log("No chapters to process (no word timing data or all done)")
        return

    log(f"Found {len(items)} chapter(s) to align")

    if args.dry_run:
        for item in items:
            sources = []
            if item["mms_path"]:
                sources.append("MMS")
            if item["whisper_path"]:
                sources.append("Whisper")
            ref_status = "OK" if item["ref_text_path"] else "MISSING ref"
            log(f"  {item['book']} {item['chapter']} ({item['distinct_id']}) — "
                f"sources: {'+'.join(sources)}, ref: {ref_status}")
        return

    processed = 0
    failed = 0

    for idx, item in enumerate(items):
        book = item["book"]
        ch = item["chapter"]
        label = f"[{idx + 1}/{len(items)}] {book} {ch}"

        try:
            stats = process_chapter(item)
            if "error" in stats:
                log(f"{label} — {stats['error']}", "ERROR")
                failed += 1
            else:
                parts = [f"{stats['verses']} verses", f"source={stats['source']}"]
                if stats["mms_score"] is not None:
                    parts.append(f"mms_score={stats['mms_score']}")
                if stats.get("whisper_score") is not None:
                    parts.append(f"whisper_score={stats['whisper_score']}")
                if stats.get("fusion"):
                    fs = stats["fusion"]
                    parts.append(
                        f"fusion: {fs['from_whisper']}/{fs['total_words']} from whisper "
                        f"({fs['improved']} improved)"
                    )
                    if fs.get("mono_fixes", 0) > 0:
                        parts.append(f"{fs['mono_fixes']} mono fixes")
                log(f"{label} — {', '.join(parts)}")
                processed += 1
        except Exception as e:
            log(f"{label} — Failed: {e}", "ERROR")
            failed += 1

    log("")
    log(f"Done: {processed} aligned, {failed} failed")


if __name__ == "__main__":
    main()
