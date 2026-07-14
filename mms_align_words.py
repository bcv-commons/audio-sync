#!/usr/bin/env python3
"""
MMS forced alignment for Bible audio (Step 1b of 2).

Uses torchaudio MMS_FA to align audio files directly against reference text,
producing word-level timing data in word-timing-data/ for consumption by
the alignment fusion script (align_words.py, Step 2).

This is a companion to whisper_transcribe.py (Step 1a) — both produce
word-level timelines in the same directory. The fusion step combines them.

Output format matches whisper_transcribe.py:
    {
        "book": "GEN",
        "chapter": "017",
        "words": [
            {"text": "word", "start": 1.2, "end": 1.5, "score": 0.85},
            ...
        ]
    }

Usage:
    # Align all available Hebrew chapters
    python mms_align_words.py --iso heb

    # Only OT
    python mms_align_words.py --iso heb --testament ot

    # Re-align even if output exists
    python mms_align_words.py --iso heb --force

    # Preview what would be done
    python mms_align_words.py --iso heb --dry-run

    # Align a specific chapter
    python mms_align_words.py --iso heb --book GEN --chapter 17

Prerequisites:
    pip install torch torchaudio uroman
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Allow MPS to fall back to CPU for ops not yet implemented on Metal.
# Must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio
from uroman import Uroman

from text_processing import load_language_config, strip_markers, clean_for_alignment
from align_words import detect_audio_header
from batch_manifest import load_batch, get_template_chapters_from_batch

# ─── Constants ──────────────────────────────────────────────────────────────

DOWNLOADS_DIR = Path("downloads/BB")
WORD_TIMING_DIR = Path("word-timing-data")

# Categories that may contain audio+text (searched in order of priority)
AUDIO_TEXT_CATEGORIES = ["syncable", "with-timecode", "audio-with-timecode"]


# ─── Logging ────────────────────────────────────────────────────────────────

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ─── MMS Forced Alignment ──────────────────────────────────────────────────

# Set to True (e.g. from align_pipeline.py --mms-cpu) to force CPU inference
# even when a CUDA GPU is available.
_MMS_FORCE_CPU: bool = False


def select_device(requested: Optional[str] = None) -> torch.device:
    """Pick the best available torch device.

    Priority: explicit request > CUDA > MPS (Apple Silicon) > CPU.
    Respects _MMS_FORCE_CPU override (forces CPU when set).
    """
    if requested:
        return torch.device(requested)
    if not _MMS_FORCE_CPU and torch.cuda.is_available():
        return torch.device("cuda")
    if not _MMS_FORCE_CPU and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mms_model(device: Optional[torch.device] = None):
    """Load MMS_FA model, tokenizer, aligner on the chosen device, init Uroman.

    The wav2vec2 model runs on the requested device (CUDA/MPS/CPU). The model's
    device can later be queried via `next(model.parameters()).device`.
    The aligner (torchaudio forced_align) is CPU-only — emissions are moved
    to CPU before alignment.
    """
    if device is None:
        device = select_device()
    bundle = torchaudio.pipelines.MMS_FA
    log(f"Loading MMS_FA model on {device.type.upper()} ...")
    t0 = time.time()
    model = bundle.get_model()   # Wav2Vec2FABundle.get_model() does not accept a device arg
    model = model.to(device)
    model.eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    uroman = Uroman()
    log(f"Model loaded in {time.time() - t0:.1f}s")
    return bundle, model, tokenizer, aligner, uroman


def _prepare_words(text: str, uroman: Uroman, tokenizer) -> Tuple[List[str], List[str]]:
    """Romanize text and filter to tokenizer dictionary.

    Returns (orig_words, clean_rom_words).
    """
    romanized = uroman.romanize_string(text)
    orig_words = text.split()
    rom_words = romanized.split()

    dict_keys = set(tokenizer.dictionary.keys())
    clean_rom_words = []
    for w in rom_words:
        cleaned = "".join(c for c in w if c in dict_keys)
        clean_rom_words.append(cleaned if cleaned else "*")

    return orig_words, clean_rom_words


# Maximum waveform samples to process in one model forward pass.
# wav2vec2 attention is O(N²) in sequence length, so memory scales fast.
# Per-device caps:
#   CPU: ~5 min @ 16 kHz (CPU can spill to system memory)
#   MPS: ~60 s @ 16 kHz (Metal buffers cap below ~14 GiB on most M-series)
#   CUDA: ~2 min @ 16 kHz (most GPUs handle larger chunks than MPS)
_CHUNK_SAMPLES_BY_DEVICE = {
    "cpu": 4_800_000,
    "mps": 960_000,
    "cuda": 1_920_000,
}

# Runtime override for --mms-chunk-minutes. When set (non-None), this value
# takes precedence over _CHUNK_SAMPLES_BY_DEVICE for all devices.
# Override at runtime via --mms-chunk-minutes (e.g. 2 for 6 GB desktop GPUs
# that also run a desktop environment consuming ~600 MB VRAM).
_MAX_CHUNK_SAMPLES: Optional[int] = None

# Overlap between chunks (in samples) to avoid boundary artifacts.
# 0.5 seconds at 16 kHz.
_CHUNK_OVERLAP = 8_000


def _max_chunk_samples(device: torch.device) -> int:
    """Return the max chunk size for the given device.

    If _MAX_CHUNK_SAMPLES is set (via --mms-chunk-minutes), it overrides
    the per-device defaults in _CHUNK_SAMPLES_BY_DEVICE.
    """
    if _MAX_CHUNK_SAMPLES is not None:
        return _MAX_CHUNK_SAMPLES
    return _CHUNK_SAMPLES_BY_DEVICE.get(device.type, 4_800_000)


def _compute_emission_chunked(waveform, model):
    """Run model forward pass, chunking long waveforms to limit memory.

    Chunk size is selected per device (MPS has tighter buffer limits than CPU).
    The waveform is moved to the model's device before each forward pass,
    and emissions are returned on CPU (the aligner is CPU-only).
    """
    model_device = next(model.parameters()).device
    max_chunk = _max_chunk_samples(model_device)
    total_samples = waveform.shape[1]

    if total_samples <= max_chunk:
        with torch.no_grad():
            emission, _ = model(waveform.to(model_device))
        return emission.cpu()

    # Process in chunks
    emissions = []
    offset = 0
    chunk_idx = 0

    while offset < total_samples:
        end = min(offset + max_chunk, total_samples)
        chunk = waveform[:, offset:end].to(model_device)

        with torch.no_grad():
            chunk_emission, _ = model(chunk)
        chunk_emission = chunk_emission.cpu()

        if chunk_idx == 0:
            # First chunk: keep all frames
            emissions.append(chunk_emission)
        else:
            # Subsequent chunks: skip overlap frames
            # Calculate how many emission frames correspond to the overlap
            overlap_samples = min(_CHUNK_OVERLAP, end - offset)
            overlap_ratio = overlap_samples / (end - offset)
            overlap_frames = int(chunk_emission.shape[1] * overlap_ratio)
            emissions.append(chunk_emission[:, overlap_frames:, :])

        if end >= total_samples:
            break

        offset = end - _CHUNK_OVERLAP
        chunk_idx += 1

    return torch.cat(emissions, dim=1)


def _align_waveform(
    waveform,
    text: str,
    bundle,
    model,
    tokenizer,
    aligner,
    uroman: Uroman,
) -> List[dict]:
    """Core MMS_FA alignment on a pre-loaded waveform.

    For long audio (>5 min), the model forward pass is chunked to avoid
    OOM errors while the aligner still operates on the full emission sequence.

    Returns list of dicts with keys: text, start, end, score.
    """
    # Romanize and prepare words
    orig_words, clean_rom_words = _prepare_words(text, uroman, tokenizer)

    # Tokenize and align full waveform
    tokens = tokenizer(clean_rom_words)

    emission = _compute_emission_chunked(waveform, model)

    token_spans = aligner(emission[0], tokens)
    ratio = waveform.shape[1] / emission.shape[1] / bundle.sample_rate

    results = []
    for word_i, word_spans in enumerate(token_spans):
        orig_word = orig_words[word_i] if word_i < len(orig_words) else "?"
        if not word_spans:
            results.append({
                "text": orig_word,
                "start": 0.0,
                "end": 0.0,
                "score": 0.0,
            })
            continue
        start_sec = word_spans[0].start * ratio
        end_sec = word_spans[-1].end * ratio
        score = sum(s.score for s in word_spans) / len(word_spans)

        results.append({
            "text": orig_word,
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "score": round(score, 3),
        })

    return results


def load_audio(audio_path: Path, bundle):
    """Load and resample audio, returning (waveform, sample_rate)."""
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    return waveform, bundle.sample_rate


def run_forced_alignment(
    audio_path: Path,
    text: str,
    bundle,
    model,
    tokenizer,
    aligner,
    uroman: Uroman,
) -> List[dict]:
    """Run MMS_FA forced alignment on full audio (no chunking).

    Returns list of dicts with keys: text, start, end, score.
    """
    waveform, _ = load_audio(audio_path, bundle)
    return _align_waveform(waveform, text, bundle, model, tokenizer, aligner, uroman)


def realign_from_point(
    waveform,
    sample_rate: int,
    restart_time: float,
    text: str,
    bundle,
    model,
    tokenizer,
    aligner,
    uroman,
    end_time: float = None,
) -> List[dict]:
    """Re-run MMS_FA on audio from restart_time onwards (or to end_time).

    Shared by collapse recovery, gap-fill, and drift correction.
    Slices the waveform, aligns text, adjusts timestamps back to original timeframe.

    Returns list of dicts with keys: text, start, end, score.
    """
    start_sample = int(restart_time * sample_rate)
    if end_time is not None:
        end_sample = int(end_time * sample_rate)
        segment = waveform[:, start_sample:end_sample]
    else:
        segment = waveform[:, start_sample:]

    if segment.shape[1] == 0:
        return []

    results = _align_waveform(segment, text, bundle, model, tokenizer, aligner, uroman)

    for r in results:
        r["start"] = round(r["start"] + restart_time, 2)
        r["end"] = round(r["end"] + restart_time, 2)

    return results


def align_segment(
    audio_path: Path,
    text: str,
    start_time: float,
    end_time: float,
    bundle,
    model,
    tokenizer,
    aligner,
    uroman,
) -> List[dict]:
    """Run MMS_FA on a segment of the audio between start_time and end_time.

    Convenience wrapper around realign_from_point that loads audio from a file.
    Returns list of dicts with keys: text, start, end, score.
    """
    waveform, sample_rate = load_audio(audio_path, bundle)
    return realign_from_point(
        waveform, sample_rate, start_time, text,
        bundle, model, tokenizer, aligner, uroman,
        end_time=end_time,
    )


# ─── Collapse Detection & Restart ─────────────────────────────────────────

COLLAPSE_THRESHOLD = 0.1   # Score at or below this = collapsed
COLLAPSE_MIN_RUN = 5       # Minimum consecutive collapsed words to trigger


def detect_collapse(word_results, threshold=COLLAPSE_THRESHOLD, min_run=COLLAPSE_MIN_RUN):
    """Find first collapse point: min_run consecutive words with score <= threshold.

    Returns the index of the first collapsed word, or None.
    """
    run_start = None
    run_len = 0

    for i, w in enumerate(word_results):
        if w["score"] <= threshold:
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len >= min_run:
                return run_start
        else:
            run_start = None
            run_len = 0

    return None


def _map_word_idx_to_verse(word_idx, cleaned_verses):
    """Map a flat word index to (verse_index, word_offset_in_verse)."""
    pos = 0
    for vi, verse in enumerate(cleaned_verses):
        words = verse.split() if verse else []
        if pos + len(words) > word_idx:
            return vi, word_idx - pos
        pos += len(words)
    return len(cleaned_verses) - 1, 0


def _find_whisper_restart_time(whisper_words, verse_texts, verse_idx, config):
    """Find Whisper's timestamp for the start of a verse.

    Estimates the Whisper word position by counting reference words in
    prior verses, then searches nearby for a text match.
    """
    from text_processing import normalize_text

    # Count reference words before this verse
    approx_word_pos = 0
    for vi in range(verse_idx):
        cleaned = clean_for_alignment(verse_texts[vi], config)
        if cleaned:
            approx_word_pos += len(cleaned.split())

    # Get the target verse's first word (normalized)
    target_verse = clean_for_alignment(verse_texts[verse_idx], config)
    if not target_verse:
        return None
    target_first_word = normalize_text(target_verse.split()[0], config)

    # Search in a window around the expected position
    search_start = max(0, approx_word_pos - 10)
    search_end = min(len(whisper_words), approx_word_pos + 30)

    for i in range(search_start, search_end):
        w = whisper_words[i]
        w_norm = normalize_text(w["text"], config)
        if w_norm == target_first_word and w.get("score", 0) > 0.3:
            return w["start"]

    # Fallback: use the Whisper word at the approximate position
    if approx_word_pos < len(whisper_words):
        w = whisper_words[approx_word_pos]
        if w.get("start", 0) > 0:
            return w["start"]

    return None


# ─── File I/O ───────────────────────────────────────────────────────────────

def write_mms_words_json(
    word_results: List[dict], book: str, chapter: str, output_path: Path,
):
    """Write MMS word-level timeline in the same format as whisper_words.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "book": book,
        "chapter": chapter,
        "words": [
            {
                "text": w["text"],
                "start": w["start"],
                "end": w["end"],
                "score": w["score"],
            }
            for w in word_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Template Helpers ─────────────────────────────────────────────────────

def get_template_chapters(template_ids: List[str]) -> set:
    """Return (BOOK, chapter_int) pairs from the current batch manifest."""
    batch = load_batch()
    return get_template_chapters_from_batch(batch, template_ids)


# ─── Collapse Detection for Existing Output ──────────────────────────────

def has_null_collapse(mms_path: Path) -> bool:
    """Check if an existing MMS output file has the null-collapse pattern.

    Returns True if >30% of words in the second half have score <= 0.1.
    """
    try:
        with open(mms_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        words = data.get("words", [])
        if len(words) < 4:
            return False
        half = len(words) // 2
        second_half = words[half:]
        bad = sum(1 for w in second_half if w.get("score", 0) <= 0.1)
        return bad / len(second_half) > 0.3
    except Exception:
        return False


# ─── Work Item Discovery ───────────────────────────────────────────────────

def discover_work_items(
    iso: Optional[str] = None,
    testament: Optional[str] = None,
    force: bool = False,
    redo_collapsed: bool = False,
    book_filter: Optional[str] = None,
    chapter_filter: Optional[int] = None,
    template_chapters: Optional[set] = None,
) -> List[dict]:
    """Find all audio+text pairs for a language (or all languages) and build work items.

    Scans downloads/BB/{canon}/{category}/{iso}/{distinct_id}/{book}/
    for .mp3 files with matching .txt reference text.
    If iso is None, scans all language directories.
    """
    items = []

    canons = []
    if testament in (None, "ot", "both"):
        canons.append("ot")
    if testament in (None, "nt", "both"):
        canons.append("nt")

    for canon in canons:
        # Search in category subdirs and also directly under canon
        search_bases = []
        for category in AUDIO_TEXT_CATEGORIES:
            cat_dir = DOWNLOADS_DIR / canon / category
            if cat_dir.exists():
                search_bases.append(cat_dir)
        # Also search direct language dirs (downloads/BB/{canon}/{iso}/)
        direct_dir = DOWNLOADS_DIR / canon
        if direct_dir.exists():
            if iso:
                direct_iso = direct_dir / iso
                if direct_iso.exists() and direct_iso.is_dir():
                    # Only add if it's a language dir (not a category dir)
                    if direct_iso.name not in AUDIO_TEXT_CATEGORIES:
                        search_bases.append(direct_dir)
            else:
                search_bases.append(direct_dir)

        for base_dir in search_bases:
            if iso:
                iso_dirs = [base_dir / iso] if (base_dir / iso).exists() else []
            else:
                iso_dirs = sorted(d for d in base_dir.iterdir()
                                  if d.is_dir() and d.name not in AUDIO_TEXT_CATEGORIES)

            for iso_dir in iso_dirs:
                lang_iso = iso_dir.name

                for distinct_dir in sorted(iso_dir.iterdir()):
                    if not distinct_dir.is_dir():
                        continue
                    distinct_id = distinct_dir.name

                    for book_dir in sorted(distinct_dir.iterdir()):
                        if not book_dir.is_dir():
                            continue
                        book = book_dir.name

                        if book_filter and book != book_filter:
                            continue

                        for audio_path in sorted(book_dir.glob("*.mp3")):
                            stem = audio_path.stem
                            parts = stem.split("_", 2)
                            if len(parts) < 3:
                                continue

                            book_code = parts[0]
                            chapter_str = parts[1]
                            audio_fileset = parts[2]

                            try:
                                chapter_num = int(chapter_str)
                            except ValueError:
                                continue

                            if chapter_filter is not None and chapter_num != chapter_filter:
                                continue

                            if template_chapters and (book_code, chapter_num) not in template_chapters:
                                continue

                            # Find matching text file
                            txt_candidates = list(book_dir.glob(f"{book_code}_{chapter_str}_*.txt"))
                            if not txt_candidates:
                                continue
                            text_path = txt_candidates[0]

                            # Build output path
                            out_book_dir = WORD_TIMING_DIR / canon / lang_iso / distinct_id / book_code
                            mms_path = out_book_dir / f"{book_code}_{chapter_str}_{audio_fileset}_mms_words.json"

                            # Skip if already aligned
                            if mms_path.exists() and not force:
                                if redo_collapsed and has_null_collapse(mms_path):
                                    pass  # include — needs redo
                                else:
                                    continue

                            items.append({
                                "audio_path": audio_path,
                                "text_path": text_path,
                                "mms_path": mms_path,
                                "book": book_code,
                                "chapter": chapter_num,
                                "chapter_str": chapter_str,
                                "canon": canon,
                                "iso": lang_iso,
                                "distinct_id": distinct_id,
                                "audio_fileset": audio_fileset,
                            })

    return items


# ─── Chapter Processing ────────────────────────────────────────────────────

def process_chapter(item: dict, bundle, model, tokenizer, aligner, uroman, config,
                    header_skip_time: Optional[float] = None,
                    whisper_path: Optional[Path] = None) -> dict:
    """Align a single chapter using MMS forced alignment.

    If header_skip_time is provided (detected from Whisper), the audio is sliced
    to skip the spoken header (book/chapter title, music) and alignment starts
    from that point. Timestamps are adjusted back to the original timeframe.

    If whisper_path is provided, collapse detection is enabled: when MMS loses
    track partway through, the waveform is sliced using Whisper timestamps and
    MMS is re-run on the remaining audio+text.

    Writes *_mms_words.json to word-timing-data/.
    Returns stats dict.
    """
    book = item["book"]
    chapter_str = item["chapter_str"]
    audio_path = item["audio_path"]
    text_path = item["text_path"]
    mms_path = item["mms_path"]

    # Read verse texts, stripping non-spoken markers
    with open(text_path, "r", encoding="utf-8") as f:
        verse_texts = [strip_markers(line.rstrip("\n"), config) for line in f.readlines()]

    # Remove trailing empty lines
    while verse_texts and not verse_texts[-1].strip():
        verse_texts.pop()

    # Clean verses for alignment and join into single text
    cleaned_verses = [clean_for_alignment(v, config) for v in verse_texts]
    non_empty_verses = [v for v in cleaned_verses if v]
    full_text = " ".join(non_empty_verses)
    total_words = len(full_text.split())

    if total_words == 0:
        return {"error": "No words in reference text after cleaning"}

    # Load audio once (reused for retry if needed)
    waveform, sample_rate = load_audio(audio_path, bundle)

    # Skip header: slice audio to start at verse_start_time
    t0 = time.time()
    if header_skip_time and header_skip_time > 0:
        word_results = realign_from_point(
            waveform, sample_rate, header_skip_time, full_text,
            bundle, model, tokenizer, aligner, uroman,
        )
    else:
        word_results = _align_waveform(
            waveform, full_text, bundle, model, tokenizer, aligner, uroman,
        )
    elapsed = time.time() - t0

    # ── Collapse detection & restart ──
    restarted = False
    collapse_idx = detect_collapse(word_results)

    if collapse_idx is not None and whisper_path is not None:
        # Load Whisper words for restart guidance
        whisper_words = _load_whisper_words(whisper_path)

        if whisper_words:
            # Map collapse word index to verse
            verse_idx, _ = _map_word_idx_to_verse(collapse_idx, non_empty_verses)

            # Find restart time from Whisper
            restart_time = _find_whisper_restart_time(
                whisper_words, verse_texts, verse_idx, config,
            )

            if restart_time and restart_time > 0:
                sample_offset = int(restart_time * sample_rate)
                if sample_offset < waveform.shape[1]:
                    remaining_text = " ".join(non_empty_verses[verse_idx:])

                    if remaining_text.strip():
                        t1 = time.time()
                        retry_results = realign_from_point(
                            waveform, sample_rate, restart_time, remaining_text,
                            bundle, model, tokenizer, aligner, uroman,
                        )
                        elapsed += time.time() - t1

                        # Stitch: keep words before collapse verse, use retry for the rest
                        pre_collapse_word_count = sum(
                            len(v.split()) for v in non_empty_verses[:verse_idx]
                        )
                        word_results = word_results[:pre_collapse_word_count] + retry_results
                        restarted = True

    # Write output
    write_mms_words_json(word_results, book, chapter_str, mms_path)

    # Stats
    scores = [w["score"] for w in word_results if w["score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0

    result = {
        "words": total_words,
        "aligned": len(scores),
        "avg_score": round(avg_score, 3),
        "elapsed": round(elapsed, 1),
    }
    if header_skip_time:
        result["header_skipped"] = round(header_skip_time, 1)
    if restarted:
        result["restarted"] = True
        result["restart_verse"] = verse_idx
    return result


def _load_whisper_words(whisper_path: Path) -> Optional[list]:
    """Load Whisper word-level data from a whisper_words.json file."""
    if not whisper_path or not whisper_path.exists():
        return None
    try:
        with open(whisper_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("words", [])
    except (json.JSONDecodeError, OSError):
        return None


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MMS forced alignment for Bible audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--iso", type=str, default=None, help="ISO 639-3 code (default: all languages)")
    parser.add_argument("--testament", type=str, choices=["nt", "ot", "both"], default=None,
                        help="Which testament to process (default: all available)")
    parser.add_argument("--book", type=str, default=None, help="Filter to a specific book (e.g. GEN)")
    parser.add_argument("--chapter", type=int, default=None, help="Filter to a specific chapter number")
    parser.add_argument("--force", action="store_true", help="Re-align even if output exists")
    parser.add_argument("--redo-collapsed", action="store_true",
                        help="Re-align only chapters whose existing output has collapsed null regions")
    parser.add_argument("--template", type=str, nargs="+", default=None,
                        help="Only process chapters used by these templates (e.g. John OBS)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "mps", "cuda"],
                        help="Device for MMS model forward pass (default: auto — mps on Apple Silicon, cuda if available, else cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument(
        "--mms-cpu", action="store_true",
        help="Force MMS to run on CPU even when a CUDA GPU is available. "
             "Useful on GPUs with limited VRAM shared with a desktop environment.",
    )
    parser.add_argument(
        "--mms-chunk-minutes", type=float, default=None,
        help="Maximum audio chunk size (in minutes) for MMS inference. "
             "Smaller values use less VRAM but may reduce alignment accuracy at chunk boundaries. "
             "Default: per-device (CPU=5 min, CUDA=2 min, MPS=1 min). Try 2 on a 6 GB GPU.",
    )

    args = parser.parse_args()

    if getattr(args, "mms_cpu", False):
        import mms_align_words as _self
        _self._MMS_FORCE_CPU = True

    if getattr(args, "mms_chunk_minutes", None) is not None:
        import mms_align_words as _self
        _self._MAX_CHUNK_SAMPLES = int(args.mms_chunk_minutes * 60 * 16000)
        log(f"MMS chunk size set to {args.mms_chunk_minutes:.1f} min "
            f"({_self._MAX_CHUNK_SAMPLES:,} samples)")

    log("=" * 60)
    log(f"MMS Forced Alignment — {args.iso or 'all languages'}")
    if args.template:
        log(f"Filtering to template(s): {', '.join(args.template)}")
    log("=" * 60)

    # Build template chapter filter if requested
    tmpl_chapters = get_template_chapters(args.template) if args.template else None

    # Discover work items
    items = discover_work_items(
        iso=args.iso,
        testament=args.testament,
        force=args.force,
        redo_collapsed=args.redo_collapsed,
        book_filter=args.book,
        chapter_filter=args.chapter,
        template_chapters=tmpl_chapters,
    )

    if not items:
        log("No chapters to process (all done or no audio+text pairs found)")
        return

    log(f"Found {len(items)} chapter(s) to align")

    if args.dry_run:
        for item in items:
            log(f"  {item['iso']} {item['book']} {item['chapter']} ({item['distinct_id']})")
        return

    # Load model (shared across languages)
    device = select_device(args.device)
    bundle, model, tokenizer, aligner, uroman = load_mms_model(device)

    # Load language config(s)
    config_cache = {}
    if args.iso:
        config_cache[args.iso] = load_language_config(args.iso)

    # Process
    processed = 0
    failed = 0

    for idx, item in enumerate(items):
        book = item["book"]
        ch = item["chapter"]
        lang_iso = item["iso"]
        label = f"[{idx + 1}/{len(items)}] {lang_iso} {book} {ch}"

        # Load config for this language if not cached
        if lang_iso not in config_cache:
            config_cache[lang_iso] = load_language_config(lang_iso)
        config = config_cache[lang_iso]

        # Look for existing Whisper data for header detection and collapse recovery
        whisper_path = Path(str(item["mms_path"]).replace("_mms_words.json", "_whisper_words.json"))
        if not whisper_path.exists():
            whisper_path = None

        # Detect header from Whisper output
        header_skip_time = None
        if whisper_path:
            whisper_words = _load_whisper_words(whisper_path)
            if whisper_words:
                with open(item["text_path"], "r", encoding="utf-8") as f:
                    verse_texts = [strip_markers(line.rstrip("\n"), config) for line in f.readlines()]
                verse_start, header_text = detect_audio_header(whisper_words, verse_texts, config)
                if verse_start:
                    header_skip_time = verse_start
                    log(f"{label} — header detected ({header_skip_time:.1f}s): \"{header_text}\"")

        try:
            stats = process_chapter(item, bundle, model, tokenizer, aligner, uroman, config,
                                    header_skip_time=header_skip_time,
                                    whisper_path=whisper_path)
            if "error" in stats:
                log(f"{label} — {stats['error']}", "ERROR")
                failed += 1
            else:
                log(f"{label} — {stats['aligned']}/{stats['words']} words, "
                    f"score={stats['avg_score']}, took={stats['elapsed']}s")
                processed += 1
        except KeyboardInterrupt:
            log("Interrupted by user", "WARN")
            break
        except Exception as e:
            log(f"{label} — Failed: {e}", "ERROR")
            failed += 1

    log("")
    log(f"Done: {processed} aligned, {failed} failed")


if __name__ == "__main__":
    main()
