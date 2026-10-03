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
downstream consumers (tools/quality_report.py, tools/check_timing_quality.py) need no
changes. No *_mms_words.json is written — there's no single continuous MMS
run to represent in that shape.

Usage:
    python align_verse_words.py --iso hin --book 1TH --chapter 1 --force
    python align_verse_words.py --iso hin --book 1TH
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from align_words import write_quality_json, write_timing_json, write_word_timing_json
from gpu_health import CudaContextPoisonedError
from mms_align_words import (
    align_window,
    compute_file_emission,
    load_audio,
    load_mms_model,
    select_device,
)
from text_processing import (
    clean_for_alignment,
    format_verse_id,
    load_language_config,
    read_verse_texts,
)
from vowel_pacing import count_vowels

DOWNLOADS_DIR = Path("downloads/BB")
OUTPUT_DIR = Path("export/timing-data")

# Search window around each verse's pace-estimated position. Wide on
# purpose (2026-10-03): with wildcard-padded alignment the neighbours' speech
# inside the window is absorbed by <star> instead of smearing this verse, so
# a wide window costs nothing -- while a narrow one silently fails whenever
# the reading runs ahead of the pace estimate and the window starts after the
# verse does (acn ACT 9: eight verses 4-12 s late). Measured against DBT on
# 340 chapters / 12,349 verses / 154 languages: verses >1 s off went from
# 11.7% (0.8 / 8 s, the pre-padding values) to 6.6% (2.0 / 20 s); chapters
# failing the chapter gate 55 -> 26; better in 67 languages, worse in 7
# (all by a few verses).
WINDOW_FRAC = 2.0
MIN_WINDOW_SECONDS = 20.0

# Every verse is also aligned in a narrow window (the pre-padding size) and
# the better candidate kept: higher alignment score, minus WINDOW_GAP_PENALTY
# per second between the candidate's start and the previous verse's end. No
# external reference is involved. The wide window alone sometimes lets a verse
# match 10-30 s late (inside the next verse) and drags the following verses
# with it; the narrow candidate then scores higher and starts closer to where
# the previous verse ended. Measured against DBT (graded only, never used to
# choose) on 340 chapters / 12,349 verses / 154 languages: wide only 6.6% of
# verses >1 s off and worse than narrow in 7 languages; this rule 6.3% and
# worse in 1 (by one verse). Seven other DBT-free rules were tested (pace,
# continuity only, score only, model-transcript onset match, ...); none did
# better. Retrying low-score verses with a floor-free window added nothing.
ALT_WINDOW = (0.8, 8.0)
WINDOW_GAP_PENALTY = 0.01
MIN_LOCAL_SCORE = 0.35

# Written to every chapter's quality file (summary.method). Bump it whenever
# a change would make already-written output worth redoing; the redo tooling
# (tools/run_gpu_redo.py --redo-older-method) selects on it.
#   anchored-star-v1 (2026-10-03): wildcard-padded windows, emission once
#   per chapter, two-sided chunk context, chapter gate.
#   anchored-star-v2 (2026-10-03): wide search windows (see WINDOW_FRAC).
#   anchored-star-v3 (2026-10-03): wide + narrow candidate per verse
#   (see ALT_WINDOW).
ALIGNMENT_METHOD = "anchored-star-v3"

# Chapter gate (decided 2026-10-03 from a 121-language DBT comparison): a
# chapter where more than GATE_MAX_LOW_SHARE of its verses score below
# GATE_LOW_SCORE is not published as our own timing. In that sample, chapters
# with no such verses had 3.5% of verses >1 s off DBT; chapters with 10-25%
# had 28.5%, and above 25% had 55-72%. Raising the per-verse
# MIN_LOCAL_SCORE to 0.5 instead was measured too and made things slightly
# worse (interpolated verses are rarely right), so it stays at 0.35.
GATE_LOW_SCORE = 0.5
GATE_MAX_LOW_SHARE = 0.10


def _usable_dbt_timing(audio_path: Path) -> bool:
    """DBT's own verse timing for this exact audio fileset, if it is on disk
    and plausible (non-zero, never going backwards)."""
    dbt_path = audio_path.with_name(audio_path.stem + "_timing.json")
    try:
        entries = json.loads(dbt_path.read_text())
        ts = [float(e["timestamp"]) for e in entries
              if str(e.get("verse_start", "")).isdigit() and int(e["verse_start"]) >= 1]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return len(ts) >= 2 and max(ts) > 0 and all(b >= a for a, b in zip(ts, ts[1:]))


def _defer_record(timing_path: Path, reason: str) -> dict:
    p = timing_path
    return {
        "status": "defer_to_dbt", "reason": reason,
        "canon": p.parent.parent.parent.parent.name, "iso": p.parent.parent.parent.name,
        "distinct_id": p.parent.parent.name, "book": p.parent.name,
        "chapter": p.name.split("_", 2)[1],
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def _release_device_cache(model) -> None:
    """Hand cached GPU memory back after the chapter's single model pass."""
    device_type = next(model.parameters()).device.type
    try:
        import torch
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def verse_anchored_align(
    audio_path: Path,
    non_empty_verses: list[str],
    config, bundle, model, tokenizer, aligner, uroman,
    window_frac: float = WINDOW_FRAC,
    min_window_seconds: float = MIN_WINDOW_SECONDS,
    min_local_score: float = MIN_LOCAL_SCORE,
    alt_window: tuple[float, float] | None = ALT_WINDOW,
    choose_window=None,
    reanchor_below: float | None = None,
) -> list[dict]:
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
    # One model pass for the whole chapter; every verse window below is a
    # slice of this (see compute_file_emission).
    emission = compute_file_emission(waveform, model)
    del waveform
    _release_device_cache(model)

    # Pacing proxy for both the search-window anchor below and
    # _interpolate_fallback_runs()'s fallback interpolation: vowel count,
    # not raw word count. Validated 2026-09-26 against 9 real DBT-vs-
    # pipeline disputes (pipeline/vowel_pacing.py's own docstring) --
    # vowel-letter count (a long vowel counts twice) tracks a verse's real
    # spoken duration noticeably better than word count, which treats a
    # one-syllable and a five-syllable word identically.
    pace_weights = [count_vowels(uroman.romanize_string(v)) or len(v.split())
                    for v in non_empty_verses]
    total_pace_weight = sum(pace_weights) or 1

    expected_starts = []
    cum_weight = 0
    for w in pace_weights:
        expected_starts.append(total_duration * cum_weight / total_pace_weight)
        cum_weight += w

    results = []
    floor = 0.0
    for i, (verse_text, wc) in enumerate(zip(non_empty_verses, pace_weights)):
        exp_start = expected_starts[i]
        exp_dur = total_duration * wc / total_pace_weight
        def _try_window(frac, min_sec, lower=None):
            w = max(exp_dur * frac, min_sec)
            ws = max(floor if lower is None else lower, exp_start - w)
            we = min(total_duration, exp_start + exp_dur + w)
            if we - ws < min_required:
                we = min(total_duration, ws + min_required)
            if we - ws < 1.0:
                return w, ws, we, []
            try:
                return w, ws, we, align_window(
                    emission, total_duration, ws, we, verse_text,
                    bundle, tokenizer, aligner, uroman,
                )
            except CudaContextPoisonedError:
                raise
            except RuntimeError as e:
                log(f"    verse {i + 1}: CTC align failed ({e}), using fallback", "WARNING")
                return w, ws, we, []

        def _avg(words):
            sc = [x["score"] for x in words if x["score"] > 0]
            return sum(sc) / len(sc) if sc else 0.0

        min_required = exp_dur * 1.3 + 2.0
        window, win_start, win_end, local_words = _try_window(window_frac, min_window_seconds)
        if alt_window is not None:
            # Same verse, second window size; keep the better of the two
            # (default rule: see ALT_WINDOW / WINDOW_GAP_PENALTY).
            # choose_window can replace that rule (True = take the alternative).
            alt = _try_window(*alt_window)
            if choose_window is not None:
                take_alt = choose_window(
                    primary=local_words, alternative=alt[3], floor=floor,
                    exp_start=exp_start, exp_dur=exp_dur, verse_text=verse_text,
                    emission=emission, total_duration=total_duration,
                )
            else:
                def _merit(words):
                    if not words or words[0]["start"] is None:
                        return 0.0
                    return _avg(words) - WINDOW_GAP_PENALTY * abs(words[0]["start"] - floor)
                take_alt = bool(alt[3]) and (not local_words or _merit(alt[3]) > _merit(local_words))
            if take_alt:
                window, win_start, win_end, local_words = alt
        if reanchor_below is not None and _avg(local_words) < reanchor_below and results:
            # Low confidence usually means the floor (the previous verse's
            # end) is already past this verse -- an earlier overshoot that
            # would otherwise drag every following verse late. Retry with the
            # window allowed to start back at the previous verse's START.
            retry = _try_window(*(alt_window or (window_frac, min_window_seconds)),
                                lower=results[-1]["start"])
            if _avg(retry[3]) > _avg(local_words):
                window, win_start, win_end, local_words = retry

        scores = [w["score"] for w in local_words if w["score"] > 0]
        local_avg = sum(scores) / len(scores) if scores else 0.0

        # The chapter's last verse gets a floor-independent retry when the
        # primary (floor-anchored) attempt is about to fall back — floor
        # here is the PREVIOUS verse's own reported end, and for the last
        # verse that's uniquely unreliable in both directions: (a) that
        # previous verse's end may have been legitimately capped down to
        # avoid overshooting (see the cap below), pushing this verse's
        # floor earlier than reality, or (b) the previous verse's own CTC
        # match may itself have overshot PAST the true boundary, handing
        # this verse a floor later than reality. Confirmed both directly
        # 2026-09-03: ace/1CO16 v24 (floor 1.45s too early, from the cap)
        # and hak/1CH21 v30 (floor 8.5s too late, from verse 29's own
        # overshoot) both fell back with the primary attempt, but a
        # window built from exp_start alone — ignoring floor entirely —
        # found the real content in both cases (0.73 and 0.98 score).
        # Every other verse keeps relying on floor as before: this
        # unreliability is specific to the last verse, which is the only
        # one with no *following* verse whose own successful anchor could
        # otherwise correct for an errant floor on the next iteration.
        is_last_verse = i == len(non_empty_verses) - 1
        used_alt_window = False
        if is_last_verse and (not local_words or local_avg < min_local_score):
            alt_win_start = max(0.0, exp_start - window)
            alt_win_end = min(total_duration, exp_start + exp_dur + window)
            if alt_win_end - alt_win_start < min_required:
                alt_win_end = min(total_duration, alt_win_start + min_required)
            if alt_win_end - alt_win_start >= 1.0 and (alt_win_start, alt_win_end) != (win_start, win_end):
                try:
                    alt_words = align_window(
                        emission, total_duration, alt_win_start, alt_win_end, verse_text,
                        bundle, tokenizer, aligner, uroman,
                    )
                except CudaContextPoisonedError:
                    raise
                except RuntimeError:
                    alt_words = []
                alt_scores = [w["score"] for w in alt_words if w["score"] > 0]
                alt_avg = sum(alt_scores) / len(alt_scores) if alt_scores else 0.0
                if alt_avg > local_avg:
                    local_words, local_avg = alt_words, alt_avg
                    used_alt_window = True

        # Cross-check a "local" (accepted) match against the vowel-pacing
        # search center itself, not just its own confidence score. CTC only
        # ever searches inside [win_start, win_end], and win_end is widened
        # past the natural pacing-implied end (exp_start + exp_dur + window)
        # whenever min_required demands it -- a match that landed in that
        # WIDENED region was reachable only because of the widening, not
        # because pacing expected the verse there. That's exactly the
        # failure mode confirmed 2026-09-26 (mtr/MTRNLC MAT 13:53,
        # mal/MALNIB JHN 19:41): a genuinely repeated/formulaic phrase
        # elsewhere in the chapter, phonetically real, CTC-confident, with
        # totally normal-looking per-word scores -- so the score alone
        # never flags it, and neither does anything downstream: the
        # arbiter's own pacing check only ever runs against DBT's timing
        # after the fact, and only for editions DBT has timing for at all.
        # This is the only place in the whole pipeline that can catch this
        # class of error before it's ever written to disk. Excludes the
        # last verse's own alt-window rescue (used_alt_window) -- that path
        # already has its own tested, floor-independent rationale (see
        # above) and re-litigating it here isn't worth the added risk.
        natural_win_end = exp_start + exp_dur + window
        if local_words and local_avg >= min_local_score and not used_alt_window:
            cand_start = local_words[0]["start"]
            if cand_start > natural_win_end:
                narrow_start = max(floor, exp_start - window)
                narrow_end = min(total_duration, natural_win_end)
                narrow_words, narrow_avg = [], 0.0
                if narrow_end - narrow_start >= 1.0:
                    try:
                        narrow_words = align_window(
                            emission, total_duration, narrow_start, narrow_end, verse_text,
                            bundle, tokenizer, aligner, uroman,
                        )
                    except CudaContextPoisonedError:
                        raise
                    except RuntimeError:
                        narrow_words = []
                    narrow_scores = [w["score"] for w in narrow_words if w["score"] > 0]
                    narrow_avg = sum(narrow_scores) / len(narrow_scores) if narrow_scores else 0.0
                if narrow_words and narrow_avg >= min_local_score:
                    # A comparably good match exists inside the natural
                    # pacing window -- prefer it; it doesn't depend on the
                    # widened search having found something else instead.
                    local_words, local_avg = narrow_words, narrow_avg
                else:
                    # Nothing decent near where pacing expects this verse --
                    # don't trust the far/wide match either. Falls through
                    # to the fallback branch below, so
                    # _interpolate_fallback_runs() places it by vowel-count
                    # interpolation between real neighbors instead.
                    log(f"    verse {i + 1}: local match at {cand_start:.2f}s is "
                        f"{cand_start - natural_win_end:.2f}s beyond the natural pacing window "
                        f"(narrow re-search found nothing as good) -- treating as fallback", "WARNING")
                    local_words, local_avg = [], 0.0

        if local_words and local_avg >= min_local_score:
            start = local_words[0]["start"]
            end = local_words[-1].get("end", start)
            if used_alt_window and start < floor:
                # The alt window deliberately ignores floor to escape a
                # possibly-errant one (see above) — but floor is also the
                # previous verse's own reported end, and reporting this
                # verse starting before that would be a backwards
                # timestamp (confirmed 2026-09-03: awa/LUK24 and
                # khk/PSA57 both regressed this way before this clamp).
                # Only the verse-level boundary is clamped; word_results
                # below keeps the real per-word timestamps the alt window
                # found, so the richer detail isn't lost, just the single
                # reported verse start.
                start = floor
                end = max(end, start)
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
            "words": len(verse_text.split()),  # genuine word count for diagnostics -- pacing math uses vowel-count pace_weights instead
            "expected_start": round(exp_start, 2),
            "local_score": round(local_avg, 3),
            "start": round(start, 2),
            "end": round(end, 2),
            "word_results": word_results,
            "source": source,
        })
        floor = end

    _interpolate_fallback_runs(results, pace_weights, total_duration)
    return results


def _interpolate_fallback_runs(results: list[dict], pace_weights: list[int], total_duration: float) -> None:
    """Second pass: replace each run of "fallback" verses' start/end with a
    position interpolated between the nearest REAL (source="local") verse
    before and after the run, proportional to vowel count (pace_weights)
    within the run — instead of the whole-chapter uniform-pace estimate the
    first pass used (max(exp_start, floor), see the loop above). Mutates
    results in place; relabels handled verses "interpolated".

    Why: the whole-chapter pace assumption is wrong whenever speech pacing
    varies within the chapter, and a run of consecutive fallbacks
    compounds it — each one's floor is itself an estimate, not real audio
    evidence (confirmed 2026-09-24 against xtn/ACT3 and aaa/AAAMLT REV22:
    both showed 5-9s of drift across a run of several consecutive verses,
    recovering only once a real anchor reappeared). Interpolating between
    real neighbors uses only evidence already produced for THIS chapter,
    so it works identically regardless of whether DBT ships comparison
    timing for the language at all — unlike anchoring on DBT's own data,
    which only covers the subset of editions DBT has timing for.

    A run touching either end of the chapter (no real verse before/after)
    falls back to the chapter boundary (0.0 or total_duration) as that
    side's anchor — still better than nothing, though less certain than a
    run with real anchors on both sides.
    """
    n = len(results)
    i = 0
    while i < n:
        if results[i]["source"] != "fallback":
            i += 1
            continue

        run_start = i
        while i < n and results[i]["source"] == "fallback":
            i += 1
        run_end = i  # exclusive

        prev_anchor = results[run_start - 1]["end"] if run_start > 0 else 0.0
        next_anchor = results[run_end]["start"] if run_end < n else total_duration
        if next_anchor < prev_anchor:
            next_anchor = prev_anchor

        run_weights = pace_weights[run_start:run_end]
        total_run_weight = sum(run_weights) or 1

        cum = 0
        for j, wc in zip(range(run_start, run_end), run_weights):
            frac_start = cum / total_run_weight
            cum += wc
            frac_end = cum / total_run_weight
            v_start = prev_anchor + (next_anchor - prev_anchor) * frac_start
            v_end = prev_anchor + (next_anchor - prev_anchor) * frac_end
            results[j]["start"] = round(v_start, 2)
            results[j]["end"] = round(v_end, 2)
            results[j]["source"] = "interpolated"


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

    verse_texts = read_verse_texts(text_path, config)

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
    #
    # No Whisper transcript exists in verse_only_mode (that's the whole
    # point of this module — see module docstring), so audio-intro
    # detection isn't possible here; intro_end is never set.
    fallback_threshold = config.mms_fallback_threshold
    pos = []  # pos[i] is verse (i+1)'s timestamp — see write_timing_json()
    timing = {"id": format_verse_id(book, chapter_str), "pos": pos}
    word_timing = {"id": format_verse_id(book, chapter_str), "beg": {}, "end": {}}
    quality_verses = {}
    all_scores = []
    fallback_count = 0
    prev_time = 0.0

    result_iter = iter(results)
    for vi, verse_text in enumerate(verse_texts):
        verse_num = vi + 1
        cleaned = clean_for_alignment(verse_text, config)

        if not cleaned:
            pos.append(round(prev_time, 2))
            word_timing["beg"][str(verse_num)] = []
            word_timing["end"][str(verse_num)] = []
            continue

        r = next(result_iter)
        pos.append(round(r["start"], 2))
        prev_time = r["start"]

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
        word_timing["beg"][str(verse_num)] = word_times
        word_timing["end"][str(verse_num)] = word_end_times
        quality_verses[str(verse_num)] = verse_quality

        if r["source"] in ("fallback", "interpolated"):
            fallback_count += 1

    null_count = sum(
        1 for times in word_timing["beg"].values() for t in times if t is None
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
            "method": ALIGNMENT_METHOD,
        },
    }

    low_share = sum(1 for r in results if r["local_score"] < GATE_LOW_SCORE) / len(results)
    gate = "pass"
    if low_share > GATE_MAX_LOW_SHARE:
        gate = "defer_to_dbt" if _usable_dbt_timing(Path(audio_path)) else "held_back"
    word_quality["summary"]["low_score_share"] = round(low_share, 3)
    word_quality["summary"]["gate"] = gate

    if gate == "defer_to_dbt":
        record = _defer_record(Path(timing_path), f"chapter gate: {low_share:.0%} of verses scored below {GATE_LOW_SCORE}")
        Path(timing_path).parent.mkdir(parents=True, exist_ok=True)
        Path(timing_path).write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
        Path(words_path).write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    else:
        write_timing_json(timing, timing_path)
        write_word_timing_json(word_timing, words_path)
    write_quality_json(word_quality, quality_path)

    local_scores = [r["local_score"] for r in results if r["source"] == "local"]
    return {
        "verses": len(verse_texts),
        "avg_score": round(sum(local_scores) / len(local_scores), 3) if local_scores else 0.0,
        "fallbacks": fallback_count,
        "elapsed": round(elapsed, 1),
        "gate": gate,
    }
