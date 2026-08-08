#!/usr/bin/env python3
"""
Word alignment fusion script (Step 2 of 2).

Reads pre-computed word timelines from word-timing-data/:
  - *_whisper_words.json  (from whisper_transcribe.py, Step 1a)
  - *_mms_words.json      (from mms_align_words.py, Step 1b)

Fuses both sources with reference text to produce verse-level and word-level
timing files in export/timing-data/ for consumption by export_templates.py.

When both sources are available, MMS-FA is used as primary (it aligns against
known text) and Whisper is used for validation/adjustment. When only one
source is available, that source is used alone.

Language-specific text processing (diacritics, markers, pronunciation maps)
is loaded from config/languages/{iso}.toml via text_processing.py.

Usage:
    # Align all available chapters for a language
    python align_words.py --iso heb

    # Only OT
    python align_words.py --iso heb --testament ot

    # Re-align even if output exists
    python align_words.py --iso heb --force

    # Preview what would be done
    python align_words.py --iso heb --dry-run

    # Align a specific chapter
    python align_words.py --iso heb --book GEN --chapter 17
"""

import argparse
import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from text_processing import (
    LanguageConfig,
    clean_for_alignment,
    is_aramaic_chapter,
    load_language_config,
    normalize_text,
    strip_markers,
)

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


def write_quality_json(word_quality: dict, output_path: Path):
    """Write per-word quality data (scores and sources)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(word_quality, f, ensure_ascii=False, indent=2)


# ─── MMS-FA Verse Mapping ──────────────────────────────────────────────────

def _map_mms_to_verses(
    mms_words: List[dict],
    verse_texts: List[str],
    book: str,
    chapter_str: str,
    config: LanguageConfig,
) -> Tuple[List[dict], dict, int]:
    """Map flat MMS word results to verse boundaries.

    The MMS words are aligned 1:1 with the cleaned reference text words
    (after strip_markers + clean_for_alignment), so we count words per
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
    word_timing = {"book": book, "chapter": chapter_str, "verses": {}, "verse_ends": {}}

    word_idx = 0
    for vi, verse_text in enumerate(verse_texts):
        verse_num = vi + 1
        cleaned = clean_for_alignment(verse_text, config)

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
            word_timing["verse_ends"][str(verse_num)] = []
            continue

        verse_word_count = len(cleaned.split())
        verse_words = mms_words[word_idx:word_idx + verse_word_count]

        # First valid timestamp in this verse
        verse_time = None
        for w in verse_words:
            if w["start"] is not None:
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
        word_end_times = []
        for w in verse_words:
            if w["start"] is not None:
                word_times.append(round(w["start"], 2))
            else:
                word_times.append(None)
            end_val = w.get("end")
            if end_val is not None:
                word_end_times.append(round(end_val, 2))
            else:
                word_end_times.append(None)
        word_timing["verses"][str(verse_num)] = word_times
        word_timing["verse_ends"][str(verse_num)] = word_end_times

        word_idx += verse_word_count

    return timing_entries, word_timing, len(verse_texts)


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
    config: LanguageConfig,
) -> Tuple[List[dict], dict, int]:
    """Align Whisper word timeline to verse boundaries using fuzzy matching.

    Returns (timing_entries, word_timing, matched_verse_count).
    """
    word_timing = {"book": book, "chapter": chapter_str, "verses": {}, "verse_ends": {}}

    total_duration = whisper_words[-1]["end"] if whisper_words else 0
    num_timeline = len(whisper_words)

    # Pre-normalize all timeline words
    norm_timeline = [normalize_text(w["text"], config) for w in whisper_words]

    # Pass 1: Find anchor verses
    anchors = {}
    search_from = 0

    for vi, verse_text in enumerate(verse_texts):
        verse_text = verse_text.strip()
        if not verse_text:
            continue
        verse_words = verse_text.split()
        norm_vwords = [normalize_text(w, config) for w in verse_words]
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
            word_timing["verse_ends"][str(verse_num)] = []
            continue

        verse_words = verse_text.split()
        num_verse_words = len(verse_words)

        if vi in anchors:
            best_idx = anchors[vi]
            timestamp = round(whisper_words[best_idx]["start"], 2)
            word_times, word_end_times = _align_verse_words_whisper(verse_words, whisper_words, best_idx, config)
        else:
            timestamp = _interpolate_verse_time(
                vi, verse_texts, anchors, whisper_words, total_duration,
                timing_entries[-1]["timestamp"],
            )
            word_times = [None] * num_verse_words
            word_end_times = [None] * num_verse_words

        timing_entries.append({
            "book": book,
            "chapter": chapter_str,
            "verse_start": str(verse_num),
            "verse_start_alt": str(verse_num),
            "timestamp": timestamp,
        })
        word_timing["verses"][str(verse_num)] = word_times
        word_timing["verse_ends"][str(verse_num)] = word_end_times

    return timing_entries, word_timing, matched


def _align_verse_words_whisper(
    verse_words: List[str],
    timeline: List[dict],
    timeline_start: int,
    config: LanguageConfig,
) -> Tuple[list, list]:
    """Align individual verse words against Whisper timeline using fuzzy matching.

    Returns (start_times, end_times) — parallel lists, one entry per verse word.
    Unmatched words have None in both lists.
    """
    num_verse_words = len(verse_words)
    starts = [None] * num_verse_words
    ends = [None] * num_verse_words
    ti = timeline_start
    max_ti = min(timeline_start + num_verse_words * 3, len(timeline))

    for vi, verse_word in enumerate(verse_words):
        if ti >= max_ti:
            break
        norm_verse = normalize_text(verse_word, config)
        if not norm_verse:
            continue

        best_ratio = 0.0
        best_offset = -1
        look_ahead = min(3, max_ti - ti)
        for offset in range(look_ahead):
            candidate = normalize_text(timeline[ti + offset]["text"], config)
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(None, norm_verse, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_offset = offset
            if offset > 0:
                merged = normalize_text(
                    " ".join(timeline[ti + j]["text"] for j in range(offset + 1)),
                    config,
                )
                merge_ratio = difflib.SequenceMatcher(None, norm_verse, merged).ratio()
                if merge_ratio > best_ratio:
                    best_ratio = merge_ratio
                    best_offset = 0

        if best_ratio >= 0.4:
            matched_entry = timeline[ti + best_offset]
            starts[vi] = round(matched_entry["start"], 2)
            ends[vi] = round(matched_entry.get("end", matched_entry["start"]), 2)
            ti = ti + best_offset + 1

    return starts, ends


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


# ─── Header Detection ─────────────────────────────────────────────────────

def detect_audio_header(
    whisper_words: List[dict],
    verse_texts: List[str],
    config: LanguageConfig,
) -> Tuple[Optional[float], Optional[str]]:
    """Detect spoken audio header (e.g. book/chapter title, music) before verse text.

    Uses two signals:
    1. Gap detection: a silence/music gap > 2s in the first 80s of audio
    2. Text matching: sliding-window match of verse 1 text against Whisper words

    Returns (verse_start_time, header_text) if a header is detected, else (None, None).
    verse_start_time is where the actual verse content begins (use to skip header).
    header_text is the detected header words (for logging/diagnostics only).
    """
    if not whisper_words or not verse_texts:
        return None, None

    # Get first non-empty verse words for matching
    first_verse = None
    for vt in verse_texts:
        vt = vt.strip()
        if vt:
            first_verse = vt
            break
    if not first_verse:
        return None, None

    first_verse_words = [normalize_text(w, config) for w in first_verse.split()[:5]]
    first_verse_words = [w for w in first_verse_words if w]
    if not first_verse_words:
        return None, None

    # Signal 1: Find gaps > 2s in the first 80s — common between header and verse text
    gap_boundary = None
    for j in range(1, len(whisper_words)):
        if whisper_words[j]["start"] > 80.0:
            break
        gap = whisper_words[j]["start"] - whisper_words[j - 1].get("end", whisper_words[j - 1]["start"])
        if gap > 2.0:
            gap_boundary = j  # verse text likely starts at this word index
            break

    # Signal 2: Find where verse 1 text first matches in the Whisper timeline
    # Use sliding window of first_verse_words against whisper words
    text_match_idx = None
    search_limit = min(len(whisper_words), 50)  # search first 50 words
    for i in range(search_limit - len(first_verse_words) + 1):
        window = [normalize_text(whisper_words[i + k]["text"], config) for k in range(len(first_verse_words))]
        matches = 0
        for fvw in first_verse_words:
            best = max(
                (difflib.SequenceMatcher(None, fvw, ww).ratio() for ww in window),
                default=0.0,
            )
            if best >= 0.7:
                matches += 1
        if matches >= max(2, len(first_verse_words) * 0.6):
            text_match_idx = i
            break

    # Decide which signal to trust:
    # - Text match is the strongest signal (it identifies where verse 1 actually
    #   begins). When present, use it directly.
    # - Gap alone is unreliable: inter-verse pauses can exceed 2s in many
    #   recordings, especially after sentence breaks. So when text match is
    #   missing, only trust the gap if it occurs early enough to plausibly be
    #   a real header (typical title speech is < 8s).
    GAP_ONLY_MAX_SECONDS = 8.0
    boundary_idx = None
    if text_match_idx is not None and text_match_idx > 0:
        boundary_idx = text_match_idx
    elif (
        gap_boundary is not None
        and whisper_words[gap_boundary]["start"] <= GAP_ONLY_MAX_SECONDS
    ):
        boundary_idx = gap_boundary

    if boundary_idx is None or boundary_idx == 0:
        return None, None  # No header detected

    verse_start_time = whisper_words[boundary_idx]["start"]
    header_text_parts = [w["text"].strip() for w in whisper_words[:boundary_idx]]
    header_text = " ".join(header_text_parts)

    return verse_start_time, header_text


# ─── Per-Word Fusion Logic ─────────────────────────────────────────────────

def _find_whisper_match(
    mms_word: dict,
    whisper_words: List[dict],
    whisper_norm: List[str],
    search_start: int,
    search_end: int,
    config: LanguageConfig,
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
    mms_norm = normalize_text(mms_word["text"], config)
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


MMS_FALLBACK_THRESHOLD = 0.3  # Default; overridden by config.mms_fallback_threshold


def fuse_words_per_word(
    mms_words: List[dict],
    whisper_words: List[dict],
    verse_texts: List[str],
    book: str,
    chapter_str: str,
    config: LanguageConfig,
    audio_path: Optional[Path] = None,
    mms_components=None,
    gap_fill_dir: Optional[Path] = None,
) -> Tuple[List[dict], dict, dict]:
    """Per-word fusion: MMS primary, Whisper fallback only.

    MMS provides all word timestamps (aligned 1:1 with reference text).
    Whisper is only used as fallback when MMS score < fallback threshold
    (from language config), since MMS forced alignment is consistently
    more accurate for timing.

    If audio_path and mms_components are provided, gaps >3s between
    consecutive MMS words in the same verse trigger a segment re-alignment.
    Gap-fill results are saved to gap_fill_dir for future reference.

    Returns (timing_entries, word_timing, word_quality, fusion_stats).
    """
    # Use near-zero threshold for Aramaic passages (MMS timings are correct
    # despite low scores; Whisper is unreliable for Aramaic)
    fallback_threshold = config.mms_fallback_threshold
    try:
        ch_num = int(chapter_str)
    except ValueError:
        ch_num = 0
    if is_aramaic_chapter(book, ch_num, config):
        fallback_threshold = 0.01
        log(f"  Aramaic passage detected — using MMS-only (threshold={fallback_threshold})")

    # Detect spoken audio header (e.g. book title before verse text)
    header_end, header_text = detect_audio_header(whisper_words, verse_texts, config)
    if header_end is not None:
        log(f"  Header detected: {header_end:.2f}s "
            f"(Whisper non-matching words before verse text)")

    # Pre-normalize Whisper words
    whisper_norm = [normalize_text(w["text"], config) for w in whisper_words]

    # If MMS was run with header text prepended, the header words are
    # already stripped and the first verse word has correct timing.
    # If not (standalone MMS without header), the first word absorbs the
    # header duration (start=0.0, end=~3.88s). Detect this and fix it.
    if header_end is not None and mms_words:
        first = mms_words[0]
        # Only adjust if the first word starts well before the header end,
        # indicating it absorbed the header audio (tolerance: 0.5s)
        if first["start"] + 0.5 < header_end < first.get("end", first["start"]):
            mms_words = list(mms_words)  # avoid mutating original
            mms_words[0] = {**first, "start": header_end}

    # Walk MMS words (backbone, 1:1 with ref text)
    # MMS always wins unless its score is below the fallback threshold
    fused_words = []
    whisper_search_start = 0
    words_from_mms = 0
    words_from_whisper = 0

    for mms_w in mms_words:
        mms_score = mms_w.get("score", 0.0)
        fused = {
            "text": mms_w["text"],
            "start": mms_w["start"],
            "end": mms_w.get("end", mms_w["start"]),
            "score": mms_score,
            "source": "mms",
            "whisper_score": None,
        }

        # Search for a matching Whisper word (for fallback and diagnostics)
        search_end = min(whisper_search_start + 30, len(whisper_words))
        match = _find_whisper_match(
            mms_w, whisper_words, whisper_norm,
            whisper_search_start, search_end, config,
        )

        if match is not None:
            w_idx, match_quality = match
            w_word = whisper_words[w_idx]
            w_score = w_word.get("score", 0.0)
            fused["whisper_score"] = w_score

            # Only use Whisper when MMS score is below threshold
            if mms_score < fallback_threshold and w_score > mms_score:
                fused["start"] = w_word["start"]
                fused["end"] = w_word.get("end", w_word["start"])
                fused["score"] = w_score
                fused["source"] = "whisper"
                fused["mms_score"] = mms_score
                words_from_whisper += 1
            else:
                words_from_mms += 1

            # Advance search window past matched Whisper word
            whisper_search_start = w_idx + 1
        else:
            words_from_mms += 1

        fused_words.append(fused)

    # ── Gap detection and re-alignment ──
    # Detect gaps >3s between consecutive words. These indicate MMS anchored
    # a word to the header audio instead of the actual verse audio.
    GAP_THRESHOLD = 3.0
    MAX_GAP_FILLS = 3  # max iterations to avoid infinite loops

    gaps_fixed = 0
    attempted_gaps = set()  # track (word_index, original_start) to avoid retrying
    for gap_iter in range(MAX_GAP_FILLS):
        gap_found = False
        for i in range(len(fused_words) - 1):
            gap = fused_words[i + 1]["start"] - fused_words[i]["start"]
            gap_key = (i, round(fused_words[i]["start"], 2))
            if gap > GAP_THRESHOLD and fused_words[i]["source"] in ("mms", "mms_gap_fill") and gap_key not in attempted_gaps:
                attempted_gaps.add(gap_key)
                # Gap detected between word i and i+1
                gap_start = fused_words[i]["start"]
                gap_end = fused_words[i + 1]["start"]
                gap_text = fused_words[i]["text"]
                next_text = fused_words[i + 1]["text"]
                log(f"  Gap detected: {gap:.1f}s between "
                    f"'{gap_text}'@{gap_start:.2f}s and "
                    f"'{next_text}'@{gap_end:.2f}s")

                if audio_path and mms_components:
                    # Find the Whisper gap to determine where the header ends.
                    # The last Whisper word before a >5s silence marks the true
                    # header end — use its start time as the segment start.
                    whisper_header_end = None
                    if whisper_words:
                        for wi in range(1, min(len(whisper_words), 20)):
                            if whisper_words[wi]["start"] - whisper_words[wi-1]["start"] > 5.0:
                                whisper_header_end = whisper_words[wi-1]["start"] + 0.5
                                break

                    if whisper_header_end and whisper_header_end > gap_start:
                        segment_start = whisper_header_end
                    else:
                        # Fallback: start closer to gap_end
                        segment_start = max(gap_start + 1.0, gap_end - 2.0)
                    segment_end = gap_end + 1.0  # margin after next word
                    # Text for segment: just the gap word(s)
                    segment_text = fused_words[i]["text"]

                    bundle, model, tokenizer, aligner_obj, uroman = mms_components
                    from mms_align_words import align_segment
                    gap_results = align_segment(
                        audio_path, segment_text,
                        segment_start, segment_end,
                        bundle, model, tokenizer, aligner_obj, uroman,
                    )

                    if gap_results and len(gap_results) >= 1:
                        new_start = gap_results[0]["start"]
                        new_score = gap_results[0]["score"]
                        old_start = fused_words[i]["start"]

                        # Only use if the new position is between the old position
                        # and the next word (i.e., it closed the gap)
                        if old_start < new_start < gap_end and new_score > 0.3:
                            log(f"  Gap fixed: '{gap_text}' moved "
                                f"{old_start:.2f}s → {new_start:.2f}s "
                                f"(score={new_score:.2f})")
                            fused_words[i]["start"] = new_start
                            fused_words[i]["end"] = gap_results[0].get("end", new_start)
                            fused_words[i]["score"] = new_score
                            fused_words[i]["source"] = "mms_gap_fill"
                            gaps_fixed += 1
                            gap_found = True

                            # Save gap-fill result
                            if gap_fill_dir:
                                import json as _json
                                gap_fill_dir.mkdir(parents=True, exist_ok=True)
                                gap_file = gap_fill_dir / f"{book}_{chapter_str}_gap_{gap_start:.0f}s.json"
                                with open(gap_file, "w") as gf:
                                    _json.dump({
                                        "original_start": old_start,
                                        "new_start": new_start,
                                        "segment_start": segment_start,
                                        "segment_end": segment_end,
                                        "text": segment_text,
                                        "results": gap_results,
                                        "iteration": gap_iter,
                                    }, gf, indent=2)
                            break  # restart scan after fix
                        else:
                            log(f"  Gap fill rejected: new_start={new_start:.2f}s, "
                                f"score={new_score:.2f}")
                else:
                    log(f"  Gap cannot be fixed (no MMS components available)")
                break  # only handle one gap per iteration

        if not gap_found:
            break

    if gaps_fixed > 0:
        log(f"  Fixed {gaps_fixed} gap(s) via segment re-alignment")

    # ── Gap-triggered MMS drift correction ──
    # After gap detection, check if MMS drifted onto a narrator repeat.
    # Only scan near detected gaps — not the whole chapter.
    # When drift is found, re-run MMS from the last good word before the gap
    # through the end of the chapter (MMS can't be trusted after drift point).
    drift_fixed = 0

    if audio_path and mms_components and whisper_words and gaps_fixed >= 0:
        # Find gaps that remain (including ones that were "fixed" but may indicate drift)
        # Also check the original gap positions
        for i in range(len(fused_words) - 1):
            gap = fused_words[i + 1]["start"] - fused_words[i]["start"]
            if gap < GAP_THRESHOLD:
                continue

            # Found a remaining gap — scan words after it for drift
            # (Whisper significantly earlier than MMS)
            drift_start_idx = None
            last_good_idx = i  # last word before the gap is the anchor

            for j in range(i + 1, min(i + 10, len(fused_words))):
                fw = fused_words[j]
                mms_time = fw["start"]
                fw_text_norm = normalize_text(fw["text"], config)
                if not fw_text_norm:
                    continue

                # Search Whisper for this word at an earlier time
                best_wh_time = None
                for ww in whisper_words:
                    if ww["start"] > mms_time - 0.5:
                        break
                    ww_norm = normalize_text(ww["text"], config)
                    if not ww_norm:
                        continue
                    sim = difflib.SequenceMatcher(None, fw_text_norm, ww_norm).ratio()
                    if sim >= 0.6 and ww["score"] > 0.5:
                        time_diff = mms_time - ww["start"]
                        if time_diff > 2.0:
                            best_wh_time = ww["start"]

                if best_wh_time is not None:
                    drift_start_idx = j
                    log(f"  MMS drift near gap: word {j} '{fw['text']}' "
                        f"MMS@{mms_time:.2f}s, Whisper@{best_wh_time:.2f}s "
                        f"(drift={mms_time - best_wh_time:.1f}s)")
                    break

            if drift_start_idx is None:
                continue

            # Re-run MMS on a tight segment around the gap only.
            # Use last good word as anchor, end just after the gap
            # (before the narrator repeat confuses MMS again).
            last_good_time = fused_words[last_good_idx]["start"]
            gap_end_time = fused_words[i + 1]["start"]  # first word after gap
            restart_time = max(0, last_good_time - 0.5)
            # End the segment at the gap_end word + small margin
            # This forces MMS to find the words BEFORE the narrator repeat
            segment_end_time = gap_end_time + 0.5

            # Collect the gap words: from last_good_idx to the word after gap
            gap_word_count = (i + 1) - last_good_idx + 1
            gap_words = fused_words[last_good_idx:last_good_idx + gap_word_count]
            gap_text = " ".join(w["text"] for w in gap_words)

            bundle, model, tokenizer, aligner_obj, uroman_obj = mms_components
            from mms_align_words import realign_from_point, load_audio

            log(f"  Re-running MMS on segment {restart_time:.1f}-{segment_end_time:.1f}s "
                f"for {len(gap_words)} words around gap")

            waveform, sample_rate = load_audio(audio_path, bundle)
            try:
                new_results = realign_from_point(
                    waveform, sample_rate, restart_time, gap_text,
                    bundle, model, tokenizer, aligner_obj, uroman_obj,
                    end_time=segment_end_time,
                )
            except RuntimeError as e:
                # CTC forced-align raises (not returns a low score) when this
                # narrow, timestamp-derived window is too short for
                # gap_text's token count — the window's duration comes from
                # OTHER words' (possibly already-drifted) timestamps, which
                # has no guaranteed relationship to how much audio gap_text
                # actually needs. align_obs_words.py hit the identical CTC
                # constraint and solved it with a pre-expanded window; this
                # call site can't cheaply pre-validate the same way (the
                # window is anchored to detected timestamps, not a
                # proportional pace estimate), so instead: catch it and keep
                # the chapter's existing alignment for these words rather
                # than losing the ENTIRE chapter's output to one bad
                # gap-fill attempt (confirmed real: 35+ chapters lost this
                # way in a single run before this fix, 2026-08-08).
                log(f"  Gap-fill re-alignment failed (CTC: {e}), keeping existing alignment")
                new_results = []

            if new_results and len(new_results) == len(gap_words):
                improved = 0
                for k, (old_fw, new_r) in enumerate(zip(gap_words, new_results)):
                    idx = last_good_idx + k
                    if new_r["score"] > 0.3:
                        old_start = fused_words[idx]["start"]
                        if new_r["start"] < old_start - 0.5:
                            fused_words[idx]["start"] = new_r["start"]
                            fused_words[idx]["end"] = new_r.get("end", new_r["start"])
                            fused_words[idx]["score"] = new_r["score"]
                            fused_words[idx]["source"] = "mms_drift_fix"
                            improved += 1
                        elif abs(new_r["start"] - old_start) < 0.5:
                            if new_r["score"] > fused_words[idx]["score"]:
                                fused_words[idx]["score"] = new_r["score"]

                if improved > 0:
                    drift_fixed += improved
                    log(f"  Drift corrected: {improved}/{len(gap_words)} words moved earlier")

                    if gap_fill_dir:
                        import json as _json
                        gap_fill_dir.mkdir(parents=True, exist_ok=True)
                        drift_file = gap_fill_dir / f"{book}_{chapter_str}_drift_{restart_time:.0f}s.json"
                        with open(drift_file, "w") as df:
                            _json.dump({
                                "last_good_idx": last_good_idx,
                                "drift_start_idx": drift_start_idx,
                                "restart_time": restart_time,
                                "segment_end_time": segment_end_time,
                                "words_corrected": improved,
                                "total_gap_words": len(gap_words),
                            }, df, indent=2)
                else:
                    log(f"  Drift re-alignment: no improvement found")
            elif new_results:
                log(f"  Drift fix: word count mismatch ({len(new_results)} vs {len(gap_words)})")
            else:
                log(f"  Drift fix: no results from re-alignment")

            break  # only handle first drift per chapter

    if drift_fixed > 0:
        log(f"  Fixed {drift_fixed} word(s) via drift correction")

    # Now map fused words to verses (same logic as _map_mms_to_verses)
    timing_entries = [{
        "book": book,
        "chapter": chapter_str,
        "verse_start": "0",
        "verse_start_alt": "0",
        "timestamp": 0,
    }]
    word_timing = {"book": book, "chapter": chapter_str, "verses": {}, "verse_ends": {}}
    quality_verses = {}

    word_idx = 0
    for vi, verse_text in enumerate(verse_texts):
        verse_num = vi + 1
        cleaned = clean_for_alignment(verse_text, config)

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
            word_timing["verse_ends"][str(verse_num)] = []
            quality_verses[str(verse_num)] = []
            continue

        verse_word_count = len(cleaned.split())
        verse_words = fused_words[word_idx:word_idx + verse_word_count]

        # First valid timestamp in this verse
        verse_time = None
        for w in verse_words:
            if w["start"] is not None:
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
        word_end_times = []
        verse_quality = []
        for w in verse_words:
            if w["start"] is not None:
                word_times.append(round(w["start"], 2))
            else:
                word_times.append(None)
            end_val = w.get("end")
            if end_val is not None:
                word_end_times.append(round(end_val, 2))
            else:
                word_end_times.append(None)
            # Build quality entry — implicit MMS (no source field),
            # explicit whisper fallback (source + mms_score)
            q_entry = {"score": round(w["score"], 3)}
            if w["source"] == "whisper":
                q_entry["source"] = "whisper"
                q_entry["mms_score"] = round(w["mms_score"], 3)
            elif w["whisper_score"] is not None:
                q_entry["whisper_score"] = round(w["whisper_score"], 3)
            verse_quality.append(q_entry)
        word_timing["verses"][str(verse_num)] = word_times
        word_timing["verse_ends"][str(verse_num)] = word_end_times
        quality_verses[str(verse_num)] = verse_quality

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
        "mono_fixes": mono_fixes,
    }

    # Build quality summary
    all_scores = [w["score"] for w in fused_words]
    low_quality_threshold = fallback_threshold
    low_quality_count = sum(1 for s in all_scores if s < low_quality_threshold)
    null_count = sum(
        1 for times in word_timing["verses"].values()
        for t in times if t is None
    )
    low_quality_verses = []
    for vnum, qwords in quality_verses.items():
        if any(w["score"] < low_quality_threshold for w in qwords):
            low_quality_verses.append(vnum)

    word_quality = {
        "book": book,
        "chapter": chapter_str,
        "verses": quality_verses,
        "summary": {
            "total_words": len(fused_words),
            "avg_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
            "low_quality_count": low_quality_count,
            "null_count": null_count,
            "from_whisper": words_from_whisper,
            "from_mms": words_from_mms,
            "low_quality_verses": low_quality_verses,
        },
    }

    return timing_entries, word_timing, word_quality, fusion_stats


# ─── Work Item Discovery ───────────────────────────────────────────────────

def discover_work_items(
    iso: str,
    testament: Optional[str] = None,
    force: bool = False,
    redo_no_quality: bool = False,
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
                if redo_no_quality:
                    quality_path = Path(str(words_path).replace("_words.json", "_words_quality.json"))
                    if quality_path.exists():
                        continue  # already has quality — skip
                    # no quality file — include for re-fusion
                else:
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

def process_chapter(item: dict, config: LanguageConfig, mms_components=None) -> dict:
    """Fuse word timelines and produce final verse timing.

    Args:
        mms_components: Optional tuple (bundle, model, tokenizer, aligner, uroman)
            for gap-fill re-alignment. If None, gap detection still runs but
            cannot re-align — gaps are logged only.

    item.preserve_existing_timing (optional bool): when True, only write the
        word-level outputs (_words.json and the quality companion). The
        verse-level _timing.json is left untouched — used for contributions
        that ship authoritative verse boundaries (e.g. CSV-derived deu/DEUSOL).

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
        verse_texts = [strip_markers(line.rstrip("\n"), config) for line in f.readlines()]
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
    word_quality = None
    if mms_words and whisper_words:
        # Both available — per-word fusion
        # Resolve audio path for gap-fill re-alignment
        audio_path = item.get("audio_path")
        if audio_path and isinstance(audio_path, str):
            audio_path = Path(audio_path)

        # Gap-fill output directory (alongside other word-timing data)
        gap_fill_dir = None
        if item.get("mms_path"):
            gap_fill_dir = Path(item["mms_path"]).parent

        final_timing, final_word_timing, word_quality, fusion_stats = fuse_words_per_word(
            mms_words, whisper_words, verse_texts, book, chapter_str, config,
            audio_path=audio_path,
            mms_components=mms_components,
            gap_fill_dir=gap_fill_dir,
        )
        source = "fused"
    elif mms_words:
        # MMS only
        final_timing, final_word_timing, _ = _map_mms_to_verses(
            mms_words, verse_texts, book, chapter_str, config,
        )
        source = "mms"
        fusion_stats = None
    elif whisper_words:
        # Whisper only — use verse-level alignment
        final_timing, final_word_timing, whisper_matched = _align_whisper_to_verses(
            whisper_words, verse_texts, book, chapter_str, config,
        )
        source = "whisper"
        fusion_stats = None
    else:
        return {"error": "No usable word timing data"}

    # Write outputs (verse-level timing may be skipped when contributor data
    # already provides authoritative boundaries)
    if not item.get("preserve_existing_timing"):
        write_timing_json(final_timing, timing_path)
    write_word_timing_json(final_word_timing, words_path)

    # Write companion quality file (fusion path only)
    if word_quality:
        quality_path = Path(str(words_path).replace("_words.json", "_words_quality.json"))
        write_quality_json(word_quality, quality_path)

    result = {
        "verses": verse_count,
        "source": source,
        "mms_score": round(mms_avg_score, 3) if mms_words else None,
        "whisper_score": round(whisper_avg_score, 3) if whisper_words else None,
    }
    if fusion_stats:
        result["fusion"] = fusion_stats
    if word_quality:
        result["quality"] = word_quality["summary"]

    return result


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fuse Whisper + MMS word timelines into verse timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--iso", type=str, default="heb", help="ISO 639-3 code (default: heb)")
    parser.add_argument("--testament", type=str, choices=["nt", "ot", "both"], default=None,
                        help="Which testament to process (default: all available)")
    parser.add_argument("--book", type=str, default=None, help="Filter to a specific book (e.g. GEN)")
    parser.add_argument("--chapter", type=int, default=None, help="Filter to a specific chapter number")
    parser.add_argument("--force", action="store_true", help="Re-align even if output exists")
    parser.add_argument("--redo-no-quality", action="store_true",
                        help="Re-fuse only chapters that have timing but no quality file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")

    args = parser.parse_args()

    log("=" * 60)
    log(f"Word Alignment Fusion — {args.iso}")
    log("=" * 60)

    config = load_language_config(args.iso)

    items = discover_work_items(
        iso=args.iso,
        testament=args.testament,
        force=args.force,
        redo_no_quality=args.redo_no_quality,
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
            stats = process_chapter(item, config)
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
                        f"fusion: {fs['from_whisper']}/{fs['total_words']} from whisper (fallback)"
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
