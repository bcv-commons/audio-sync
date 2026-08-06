#!/usr/bin/env python3
"""
OBS narration alignment — segment-anchored MMS forced alignment.

Aligns each [[segment:N]] boundary (an OBS story's per-image narration
beat) to its start position in that story's single audio clip. This is
NOT the same strategy as the DBT chapter pipeline (whisper_transcribe.py /
mms_align_words.py / align_words.py): those force-align a whole chapter as
one continuous word sequence and derive verse starts from word offsets
afterward, which works because DBT audio reliably tracks its reference
text word-for-word. OBS narration audio doesn't have that guarantee
(YouTube-extracted, "retelling" source text, unverified fidelity per
language) — a single bad stretch in a whole-file alignment cascades
through every downstream segment.

Spot-checked directly against 11 real published OBS-TLF languages before
building this (2026-08-04/05): whole-file alignment produced physically
impossible results for at least one language (mai: 7/15 gaps averaged
>6 words/sec, a monotonicity violation, i.e. narration would need to be
inhumanly fast). Segment-anchored alignment — align each segment
independently within a window around an expected-pace position, falling
back to that pace estimate when local confidence is too low to trust —
fixed every language in that sweep. Parameters below (WINDOW_FRAC=0.8,
MIN_WINDOW_SECONDS=20.0, MIN_LOCAL_SCORE=0.35) came from a 7-config sweep
on the two lowest-scoring languages (mai, sat), not a guess.

That sweep only checked strict monotonicity (start[i+1] > start[i]),
which isn't the same as *plausible* — wiring this into the real module
(not the throwaway sweep prototype) surfaced two more failure modes
monotonicity alone doesn't catch, both fixed here and confirmed against
real data (ahr, mai, sat, hoc, all re-verified after each fix):
  1. Adjacent windows can fully overlap when min_window_seconds is large
     relative to a short segment's expected duration, letting MMS anchor
     one segment into its neighbor's actual audio — "monotonic" but
     wrong (two 40-word segments landing 0.08s apart). Fixed by flooring
     each window at the *actual* end of the previous segment (causal,
     not a static uniform-pace partition — that was tried first and
     measurably hurt scores).
  2. CTC forced-align raises (not a low score) when a window is too
     short for a segment's token count — word-count-based duration
     estimates don't reliably predict token count across scripts. Caught
     and treated as a fallback rather than crashing the story. Left
     unguarded, an early overshoot can also pin every remaining
     segment's floor near end-of-file (seen directly: 5 segments
     collapsing to one identical timestamp) — fixed by capping how far
     the floor can advance at the next segment's own expected start.

Output (Contract B raw-timing shape, story-scoped instead of book/chapter):
    export/timing-data/obs/<iso>/<story_id>_timing.json
    [
      {"story": "01", "segment": 1, "timestamp": 1.92, "score": 0.82, "source": "local"},
      {"story": "01", "segment": 2, "timestamp": 24.53, "score": 0.16, "source": "fallback"},
      ...
    ]
    "source" is "local" (real windowed MMS alignment, score is that
    alignment's confidence) or "fallback" (word-count-proportional pace
    estimate used because local confidence was below MIN_LOCAL_SCORE;
    score is 0.0 — there is no alignment confidence for a fallback).

Usage:
    python align_obs_words.py --iso ahr              # align every story
    python align_obs_words.py --iso ahr --story 01    # one story
    python align_obs_words.py --iso ahr --force       # re-align existing output
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from mms_align_words import load_audio, load_mms_model, realign_from_point, select_device
from obs_batch_manifest import fetch_story_text, get_stories, load_obs_batch
from text_processing import clean_for_alignment, load_language_config, strip_markers

OBS_DOWNLOADS_DIR = Path("downloads/obs")
OBS_OUTPUT_DIR = Path("export/timing-data/obs")
OBS_RUNS_DIR = Path("_obs_runs")

# Calibrated 2026-08-05 against mai/sat (lowest-scoring of 11 spot-checked
# languages) — see module docstring.
WINDOW_FRAC = 0.8
MIN_WINDOW_SECONDS = 20.0
MIN_LOCAL_SCORE = 0.35

IMAGE_RE = re.compile(r"!\[OBS Image\]\((?P<url>[^)]+)\)")
CITATION_RE = re.compile(r"^_.+_$")


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def parse_story_md(raw: str) -> List[str]:
    """Split an OBS story .md into its per-segment narration text.

    Segment boundaries are OBS-TLF's image markers (the only structural
    boundary in the source — see MONO's internal-docs/obs-narration-
    import-spec.md). Drops the title line and the trailing italicized
    Bible-reference citation line if present.

    Verified against real content from OBS-TLF/ahr_obs, BCS/mai_obs,
    BCS/sat_obs, BCS/bho_obs, and 7 more languages spot-checked
    2026-08-04 — the citation-line pattern held on all of them.
    """
    lines = raw.split("\n")
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    body = "\n".join(lines[idx + 1:])  # drop the "# N. Title" line
    body_lines = body.rstrip().split("\n")
    if body_lines and CITATION_RE.match(body_lines[-1].strip()):
        body_lines = body_lines[:-1]
    body = "\n".join(body_lines).strip()
    parts = IMAGE_RE.split(body)
    segments = []
    for i in range(1, len(parts), 2):
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if text:
            segments.append(text)
    return segments


def ensure_story_audio(iso: str, story_id: str, audio_url: str, force: bool = False) -> Path:
    """Download (and cache) one story's audio clip. Not mirrored by MONO —
    the manifest just carries the door43 URL through, so this repo is
    what fetches and caches it locally, same principle as contrib/'s
    audio_url external-audio pattern.
    """
    from download_language_content import _get_with_retry

    dest = OBS_DOWNLOADS_DIR / iso / f"{story_id}.m4a"
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = _get_with_retry(audio_url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def segment_anchored_align(
    audio_path: Path,
    segments: List[str],
    config,
    bundle, model, tokenizer, aligner, uroman,
    window_frac: float = WINDOW_FRAC,
    min_window_seconds: float = MIN_WINDOW_SECONDS,
    min_local_score: float = MIN_LOCAL_SCORE,
) -> List[dict]:
    """Align each segment independently within a window anchored to an
    expected-pace position, instead of one continuous whole-file
    alignment. See module docstring for why.

    Returns one dict per segment: {segment, words, expected_start,
    local_score, start, source}. source is "local" or "fallback".
    """
    waveform, sample_rate = load_audio(audio_path, bundle)
    total_duration = waveform.shape[1] / sample_rate

    word_counts = [len(s.split()) for s in segments]
    total_words = sum(word_counts) or 1

    # Precompute every segment's expected start up front so windows can be
    # clamped against neighbors (see below) — needs lookahead/lookbehind,
    # not available from a running cumulative sum alone.
    expected_starts = []
    cum_words = 0
    for wc in word_counts:
        expected_starts.append(total_duration * cum_words / total_words)
        cum_words += wc

    results = []
    # Causal floor: the *actual* end of the previous segment (or its
    # fallback position), not just its theoretical pace estimate. Using
    # real propagated information instead of a static uniform-pace
    # partition — see below for why a static midpoint-clamp isn't the
    # right fix.
    floor = 0.0
    for i, (seg_text, wc) in enumerate(zip(segments, word_counts)):
        exp_start = expected_starts[i]
        exp_dur = total_duration * wc / total_words
        window = max(exp_dur * window_frac, min_window_seconds)
        win_start = max(floor, exp_start - window)
        win_end = min(total_duration, exp_start + exp_dur + window)

        # win_start is floored at the previous segment's REAL end, which
        # prevents the collision this loop used to be vulnerable to
        # (confirmed directly on real data: ahr story 01's segments 1 and
        # 2 — 42 and 39 words — landed 0.08s apart because both windows
        # independently anchored near 0; monotonicity alone didn't catch
        # it, since 5.00 <= 5.08 "passes"). An earlier fix tried clamping
        # win_end to the midpoint with the NEXT segment too, using only
        # the uniform-pace prior — that's the wrong data to clamp against
        # (the prior is often meaningfully wrong; real narration doesn't
        # pace evenly) and measurably hurt scores. So win_end here is
        # bounded only by the window itself, not by a guess about a
        # segment not yet aligned.
        #
        # But flooring win_start can leave too little room for THIS
        # segment's own text — CTC forced-align needs at least roughly
        # one frame per token (plus extra for consecutive repeats), and a
        # squeezed window crashes with "targets length is too long for
        # CTC" rather than failing gracefully (confirmed directly: a
        # ~2.1s window for a segment needing far more). Guarantee a
        # generous minimum duration by extending win_end; if even that
        # doesn't fit before total_duration (floor pinned near the end of
        # the file), skip straight to the fallback instead of risking the
        # crash.
        min_required = exp_dur * 1.3 + 2.0
        if win_end - win_start < min_required:
            win_end = min(total_duration, win_start + min_required)

        if win_end - win_start < 1.0:
            local_words = []
        else:
            try:
                local_words = realign_from_point(
                    waveform, sample_rate, win_start, seg_text,
                    bundle, model, tokenizer, aligner, uroman, end_time=win_end,
                )
            except RuntimeError as e:
                # CTC forced-align raises (not returns a low score) when the
                # window is too short for this segment's token count —
                # confirmed directly: word-count-based duration estimates
                # don't reliably predict token count across languages/
                # scripts, so the min_required expansion above is a
                # best-effort reduction of how often this happens, not a
                # guarantee. Treat it exactly like a low-confidence result
                # (fall back to the pace estimate) rather than letting one
                # segment's edge case crash the whole story.
                log(f"    segment {i + 1}: CTC align failed ({e}), using fallback", "WARNING")
                local_words = []
        # Release cached MPS memory after every windowed call — this loop
        # runs up to ~50x per story (once per segment). process_chapter()
        # in mms_align_words.py does this once per whole chapter; that's
        # not enough here since we're not going through process_chapter()
        # at all. Skipping this exhausted unified memory on a real sweep
        # run (2026-08-05) — confirmed the fix, not just a guess.
        if next(model.parameters()).device.type == "mps":
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass

        scores = [w["score"] for w in local_words if w["score"] > 0]
        local_avg = sum(scores) / len(scores) if scores else 0.0

        if local_words and local_avg >= min_local_score:
            start = local_words[0]["start"]
            end = local_words[-1].get("end", start)
            # Cap how far the floor can advance at the NEXT segment's own
            # theoretical starting point. Without this, a forced-align
            # artifact (CTC stretching the last word to consume the rest
            # of a generous window) propagates an unrealistically-late
            # floor forward — confirmed directly: one such overshoot on
            # ahr story 01 pinned every remaining segment (12-16) to the
            # exact same timestamp near the end of the file, since each
            # fallback's floor was already maxed out and never recovered.
            # Still allows genuine forward progress (end can exceed this
            # segment's own exp_start, tightening the next window a bit,
            # which is correct) — just not runaway overshoot.
            if i < len(segments) - 1:
                end = min(end, expected_starts[i + 1])
            source = "local"
        else:
            start = max(exp_start, floor)
            end = start
            local_avg = 0.0
            source = "fallback"

        results.append({
            "segment": i + 1,
            "words": wc,
            "expected_start": round(exp_start, 2),
            "local_score": round(local_avg, 3),
            "start": round(start, 2),
            "source": source,
        })
        floor = end

    return results


def write_obs_timing_json(iso: str, story_id: str, results: List[dict], output_path: Path):
    entries = [
        {
            "story": story_id,
            "segment": r["segment"],
            "timestamp": r["start"],
            "score": r["local_score"],
            "source": r["source"],
        }
        for r in results
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def process_story(
    iso: str, story_id: str, story_job: dict, batch: dict, config,
    bundle, model, tokenizer, aligner, uroman, force: bool = False,
) -> dict:
    """Align one story. Returns a stats dict (fusion_words.py-style) —
    {"segments": N, "avg_score": F, "fallbacks": N} or {"error": msg}.
    """
    output_path = OBS_OUTPUT_DIR / iso / f"{story_id}_timing.json"
    if output_path.exists() and not force:
        return {"skipped": True}

    raw_md = fetch_story_text(batch, story_id)
    segments = parse_story_md(raw_md)
    if not segments:
        return {"error": "No segments parsed from story .md"}

    config_lang = load_language_config(iso) if config is None else config
    cleaned = [clean_for_alignment(strip_markers(s, config_lang), config_lang) for s in segments]

    audio_path = ensure_story_audio(iso, story_id, story_job["audio_url"], force=force)

    t0 = time.time()
    results = segment_anchored_align(
        audio_path, cleaned, config_lang, bundle, model, tokenizer, aligner, uroman,
    )
    elapsed = time.time() - t0

    write_obs_timing_json(iso, story_id, results, output_path)

    local_scores = [r["local_score"] for r in results if r["source"] == "local"]
    fallbacks = sum(1 for r in results if r["source"] == "fallback")
    return {
        "segments": len(results),
        "avg_score": round(sum(local_scores) / len(local_scores), 3) if local_scores else 0.0,
        "fallbacks": fallbacks,
        "elapsed": round(elapsed, 1),
    }


def write_run_manifest(iso: str, results: List[dict]) -> Path:
    manifest = {
        "iso": iso,
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }
    OBS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = OBS_RUNS_DIR / f"{iso}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Align OBS narration segments to audio")
    parser.add_argument("--iso", required=True, help="Language ISO code (e.g. ahr)")
    parser.add_argument("--story", type=str, default=None, help="Align only this story id (e.g. 01)")
    parser.add_argument("--force", action="store_true", help="Re-align even if output exists")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    batch = load_obs_batch(args.iso)
    stories = get_stories(batch)
    if args.story:
        if args.story not in stories:
            log(f"Story {args.story} not in batch for {args.iso}", "ERROR")
            sys.exit(1)
        stories = {args.story: stories[args.story]}

    log(f"{args.iso}: {len(stories)} story/stories to process "
        f"(source={batch.get('source_repo')}, license={batch.get('license')})")

    bundle, model, tokenizer, aligner, uroman = load_mms_model(select_device(args.device))
    config = load_language_config(args.iso)

    run_results = []
    for story_id in sorted(stories):
        story_job = stories[story_id]
        log(f"[{story_id}] aligning ({story_job.get('segment_count', '?')} segments)...")
        stats = process_story(
            args.iso, story_id, story_job, batch, config,
            bundle, model, tokenizer, aligner, uroman, force=args.force,
        )
        if stats.get("skipped"):
            log(f"[{story_id}] skipped (exists)")
            continue
        if "error" in stats:
            log(f"[{story_id}] ERROR: {stats['error']}", "ERROR")
            run_results.append({"story_id": story_id, "status": "failed", "error": stats["error"]})
            continue
        log(f"[{story_id}] {stats['segments']} segments, avg_score={stats['avg_score']}, "
            f"fallbacks={stats['fallbacks']}, {stats['elapsed']}s")
        run_results.append({
            "story_id": story_id, "status": "ok",
            "segments": stats["segments"], "avg_score": stats["avg_score"],
            "fallbacks": stats["fallbacks"],
        })

    if run_results:
        path = write_run_manifest(args.iso, run_results)
        log(f"Run manifest written: {path} ({len(run_results)} result(s))")


if __name__ == "__main__":
    main()
