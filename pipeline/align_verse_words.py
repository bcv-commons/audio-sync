#!/usr/bin/env python3
"""
Verse-anchored, MMS-only alignment — for languages where Whisper cannot be
trusted as a guide.

The default chapter pipeline (whisper_transcribe.py -> mms_align_words.py ->
align_words.py) leans on Whisper in two places: detecting a spoken header
before verse 1, and gap-fill/drift-correction during fusion. Both degrade
when Whisper's transcription is unreliable for a language — confirmed with
Hindi (HINBIB): real recognition errors (avg score 0.68 vs 0.93-0.94 for
French/Arabic), 3-9x realtime instead of 20-50x, and no code-level fix
available (a documented model-capability gap, not a bug). A blind guide is
worse than no guide.

This module applies the same fix already built for OBS narration
(align_obs_words.py's segment_anchored_align) to Bible verse boundaries
instead of OBS segment boundaries: align each verse independently within a
window anchored to an expected-pace position, falling back to that pace
estimate when local confidence is too low to trust. No Whisper involved at
all. See align_obs_words.py's module docstring for the original rationale
and the sweep that calibrated WINDOW_FRAC/MIN_LOCAL_SCORE; MIN_WINDOW_SECONDS
here is re-tuned for verse-length (not narration-segment-length) audio.

Toggled per-language via LanguageConfig.verse_only_mode (config/languages/
{iso}.toml) — this module itself has no awareness of that flag, so it stays
usable standalone for any language regardless of config state, same as
align_obs_words.py.

Output: reuses align_words.py's write_timing_json/write_word_timing_json/
write_quality_json as-is — same on-disk shape as fusion-mode output, so
downstream consumers (quality_report.py, check_timing_quality.py) need no
changes. No *_mms_words.json is written — there's no single continuous MMS
run to represent in that shape.

Usage:
    python align_verse_words.py --iso hin --book 1TH --chapter 1 --force
    python align_verse_words.py --iso hin --book 1TH
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from align_words import write_quality_json, write_timing_json, write_word_timing_json
from mms_align_words import load_audio, load_mms_model, realign_from_point, select_device
from text_processing import clean_for_alignment, load_language_config, strip_markers

DOWNLOADS_DIR = Path("downloads/BB")
OUTPUT_DIR = Path("export/timing-data")

# WINDOW_FRAC/MIN_LOCAL_SCORE reused as-is from align_obs_words.py's sweep —
# both are shape-invariant (proportional windowing; confidence threshold).
# MIN_WINDOW_SECONDS is NOT reused as-is: OBS segments are multi-sentence
# narration beats where exp_dur*0.8 usually already exceeds 20s, so the
# floor rarely binds. A short Bible verse's exp_dur*0.8 is often far below
# that, so a 20s floor would dominate for most verses — oversized windows
# relative to verse length, risking the exact adjacent-window-collision
# failure mode the causal floor exists to prevent. Needs a real sweep
# against Hindi verse-length audio (see pending work) before being trusted;
# this starting value is a placeholder, not a calibrated result.
WINDOW_FRAC = 0.8
MIN_WINDOW_SECONDS = 8.0
MIN_LOCAL_SCORE = 0.35


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def verse_anchored_align(
    audio_path: Path,
    non_empty_verses: List[str],
    config, bundle, model, tokenizer, aligner, uroman,
    window_frac: float = WINDOW_FRAC,
    min_window_seconds: float = MIN_WINDOW_SECONDS,
    min_local_score: float = MIN_LOCAL_SCORE,
) -> List[dict]:
    """Align each verse independently within a window anchored to an
    expected-pace position. Adapted from align_obs_words.py's
    segment_anchored_align() — same window/floor/overshoot-cap logic, see
    that module's docstring for the full rationale.

    Returns one dict per non-empty verse (same order as non_empty_verses):
    {verse_index, words, expected_start, local_score, start, end,
    word_results, source}. word_results is a list of {text, start, end,
    score} dicts — the raw per-word output for "local" verses, or a
    synthetic all-None list for "fallback" verses (no real per-word timing
    exists for a pace estimate).
    """
    waveform, sample_rate = load_audio(audio_path, bundle)
    total_duration = waveform.shape[1] / sample_rate

    word_counts = [len(v.split()) for v in non_empty_verses]
    total_words = sum(word_counts) or 1

    expected_starts = []
    cum_words = 0
    for wc in word_counts:
        expected_starts.append(total_duration * cum_words / total_words)
        cum_words += wc

    results = []
    floor = 0.0
    for i, (verse_text, wc) in enumerate(zip(non_empty_verses, word_counts)):
        exp_start = expected_starts[i]
        exp_dur = total_duration * wc / total_words
        window = max(exp_dur * window_frac, min_window_seconds)
        win_start = max(floor, exp_start - window)
        win_end = min(total_duration, exp_start + exp_dur + window)

        min_required = exp_dur * 1.3 + 2.0
        if win_end - win_start < min_required:
            win_end = min(total_duration, win_start + min_required)

        if win_end - win_start < 1.0:
            local_words = []
        else:
            try:
                local_words = realign_from_point(
                    waveform, sample_rate, win_start, verse_text,
                    bundle, model, tokenizer, aligner, uroman, end_time=win_end,
                )
            except RuntimeError as e:
                log(f"    verse {i + 1}: CTC align failed ({e}), using fallback", "WARNING")
                local_words = []

        if next(model.parameters()).device.type == "mps":
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass
        elif next(model.parameters()).device.type == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        scores = [w["score"] for w in local_words if w["score"] > 0]
        local_avg = sum(scores) / len(scores) if scores else 0.0

        if local_words and local_avg >= min_local_score:
            start = local_words[0]["start"]
            end = local_words[-1].get("end", start)
            if i < len(non_empty_verses) - 1:
                # Cap the floor's forward advance at the next verse's
                # pace-based expected start (prevents runaway overshoot —
                # see align_obs_words.py's docstring). But never cap it
                # *below this verse's own start*: when real pacing outruns
                # the uniform-pace estimate (a verse's actual position
                # already past where naive pacing expected the *next*
                # verse to begin), that cap could push the propagated
                # floor earlier than the timestamp just assigned to this
                # verse — the next verse's window would then anchor before
                # this one, producing a backwards timestamp. Confirmed
                # directly against real Hindi audio (HINBIB JHN 6:4→5 and
                # JHN 19:8→9, both had negative gaps before this fix).
                end = max(min(end, expected_starts[i + 1]), start)
            source = "local"
            word_results = local_words
        else:
            start = max(exp_start, floor)
            end = start
            local_avg = 0.0
            source = "fallback"
            word_results = [
                {"text": w, "start": None, "end": None, "score": 0.0}
                for w in verse_text.split()
            ]

        results.append({
            "verse_index": i,
            "words": wc,
            "expected_start": round(exp_start, 2),
            "local_score": round(local_avg, 3),
            "start": round(start, 2),
            "end": round(end, 2),
            "word_results": word_results,
            "source": source,
        })
        floor = end

    return results


def process_chapter_verse_only(
    item: dict, bundle, model, tokenizer, aligner, uroman, config,
) -> dict:
    """Verse-only equivalent of mms_align_words.process_chapter() /
    align_words.py's fusion process_chapter() combined — one step instead
    of two, since there's no Whisper source to fuse. Returns a stats dict:
    {"verses": N, "avg_score": F, "fallbacks": N, "elapsed": F} or
    {"error": msg}.
    """
    book = item["book"]
    chapter_str = item["chapter_str"]
    audio_path = item["audio_path"]
    text_path = item["text_path"]
    timing_path = item["timing_path"]
    words_path = item["words_path"]
    quality_path = item["quality_path"]

    with open(text_path, "r", encoding="utf-8") as f:
        verse_texts = [strip_markers(line.rstrip("\n"), config) for line in f.readlines()]
    while verse_texts and not verse_texts[-1].strip():
        verse_texts.pop()

    cleaned_verses = [clean_for_alignment(v, config) for v in verse_texts]
    non_empty_verses = [v for v in cleaned_verses if v]
    total_words = sum(len(v.split()) for v in non_empty_verses)

    if total_words == 0:
        return {"error": "No words in reference text after cleaning"}

    t0 = time.time()
    results = verse_anchored_align(
        audio_path, non_empty_verses, config, bundle, model, tokenizer, aligner, uroman,
    )
    elapsed = time.time() - t0

    # Reinsert empty verses (same walk-and-reuse-previous-timestamp pattern
    # as align_words.py's _map_mms_to_verses) and build timing/word/quality
    # payloads in the same on-disk shape fusion-mode already writes.
    fallback_threshold = config.mms_fallback_threshold
    timing_entries = [{
        "book": book, "chapter": chapter_str,
        "verse_start": "0", "verse_start_alt": "0", "timestamp": 0,
    }]
    word_timing = {"book": book, "chapter": chapter_str, "verses": {}, "verse_ends": {}}
    quality_verses = {}
    all_scores = []
    fallback_count = 0

    result_iter = iter(results)
    for vi, verse_text in enumerate(verse_texts):
        verse_num = vi + 1
        cleaned = clean_for_alignment(verse_text, config)

        if not cleaned:
            prev_time = timing_entries[-1]["timestamp"]
            timing_entries.append({
                "book": book, "chapter": chapter_str,
                "verse_start": str(verse_num), "verse_start_alt": str(verse_num),
                "timestamp": round(prev_time, 2),
            })
            word_timing["verses"][str(verse_num)] = []
            word_timing["verse_ends"][str(verse_num)] = []
            continue

        r = next(result_iter)
        timing_entries.append({
            "book": book, "chapter": chapter_str,
            "verse_start": str(verse_num), "verse_start_alt": str(verse_num),
            "timestamp": round(r["start"], 2),
        })

        word_times = []
        word_end_times = []
        verse_quality = []
        for w in r["word_results"]:
            word_times.append(round(w["start"], 2) if w["start"] is not None else None)
            end_val = w.get("end")
            word_end_times.append(round(end_val, 2) if end_val is not None else None)
            q_entry = {"score": r["local_score"], "source": r["source"]}
            verse_quality.append(q_entry)
            all_scores.append(r["local_score"])
        word_timing["verses"][str(verse_num)] = word_times
        word_timing["verse_ends"][str(verse_num)] = word_end_times
        quality_verses[str(verse_num)] = verse_quality

        if r["source"] == "fallback":
            fallback_count += 1

    null_count = sum(
        1 for times in word_timing["verses"].values() for t in times if t is None
    )
    low_quality_verses = [
        vnum for vnum, qwords in quality_verses.items()
        if any(w["score"] < fallback_threshold for w in qwords)
    ]
    low_quality_count = sum(1 for s in all_scores if s < fallback_threshold)

    word_quality = {
        "book": book,
        "chapter": chapter_str,
        "verses": quality_verses,
        "summary": {
            "total_words": len(all_scores),
            "avg_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
            "low_quality_count": low_quality_count,
            "null_count": null_count,
            "from_whisper": 0,
            "from_mms": len(all_scores),
            "low_quality_verses": low_quality_verses,
        },
    }

    write_timing_json(timing_entries, timing_path)
    write_word_timing_json(word_timing, words_path)
    write_quality_json(word_quality, quality_path)

    local_scores = [r["local_score"] for r in results if r["source"] == "local"]
    return {
        "verses": len(verse_texts),
        "avg_score": round(sum(local_scores) / len(local_scores), 3) if local_scores else 0.0,
        "fallbacks": fallback_count,
        "elapsed": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Verse-anchored MMS-only alignment (no Whisper)")
    parser.add_argument("--iso", required=True, help="Language ISO 639-3 code (e.g. hin)")
    parser.add_argument("--distinct-id", required=True, help="Fileset distinct_id (e.g. HINBIB)")
    parser.add_argument("--book", required=True, help="Book code (e.g. 1TH)")
    parser.add_argument("--canon", type=str, default="nt", choices=["nt", "ot"])
    parser.add_argument("--chapter", type=int, default=None, help="Single chapter number (default: all in book)")
    parser.add_argument("--force", action="store_true", help="Re-align even if output exists")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    from align_pipeline import needs_run
    from whisper_transcribe import discover_chapter_files

    config = load_language_config(args.iso)
    bundle, model, tokenizer, aligner, uroman = load_mms_model(select_device(args.device))

    required = {args.book: {args.chapter}} if args.chapter is not None else {args.book: set(range(1, 200))}
    chapters, skipped = discover_chapter_files(
        args.iso, args.canon, args.distinct_id, OUTPUT_DIR, force=args.force, required_chapters=required,
    )
    if not chapters:
        log(f"No chapters discovered for {args.iso}/{args.canon}/{args.distinct_id}/{args.book} "
            "— check downloads/BB/ for this edition", "ERROR")
        sys.exit(1)

    run_results = []
    for chapter in chapters:
        book = chapter["book"]
        ch_num = chapter["chapter"]
        chapter_str = chapter["chapter_str"]
        audio_fileset = chapter["audio_fileset"]
        canon = args.canon
        iso = args.iso
        distinct_id = args.distinct_id

        out_book_dir = OUTPUT_DIR / canon / iso / distinct_id / book
        timing_path = out_book_dir / f"{book}_{chapter_str}_{audio_fileset}_timing.json"
        words_path = out_book_dir / f"{book}_{chapter_str}_{audio_fileset}_words.json"
        quality_path = Path(str(words_path).replace("_words.json", "_words_quality.json"))

        if not needs_run(timing_path, force=args.force):
            log(f"{book} {ch_num}: skipped (exists)")
            continue

        item = {
            "book": book, "chapter": ch_num, "chapter_str": chapter_str,
            "audio_path": chapter["audio_path"], "text_path": chapter["text_path"],
            "timing_path": timing_path, "words_path": words_path, "quality_path": quality_path,
        }
        log(f"{book} {ch_num}: aligning (verse-only, MMS)...")
        stats = process_chapter_verse_only(item, bundle, model, tokenizer, aligner, uroman, config)
        if "error" in stats:
            log(f"{book} {ch_num}: ERROR: {stats['error']}", "ERROR")
            run_results.append({"book": book, "chapter": ch_num, "status": "failed", "error": stats["error"]})
            continue
        log(f"{book} {ch_num}: {stats['verses']} verses, avg_score={stats['avg_score']}, "
            f"fallbacks={stats['fallbacks']}, {stats['elapsed']}s")
        run_results.append({"book": book, "chapter": ch_num, "status": "ok", **stats})

    log(f"Done: {len(run_results)} chapter(s) processed")


if __name__ == "__main__":
    main()
