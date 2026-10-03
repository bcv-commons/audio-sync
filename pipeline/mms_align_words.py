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

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Allow MPS to fall back to CPU for ops not yet implemented on Metal.
# Must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio
from gpu_health import CudaContextPoisonedError, note_alignment_success, wrap_if_poisoned
from text_processing import clean_for_alignment, read_verse_texts
from uroman import Uroman

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


def select_device(requested: str | None = None) -> torch.device:
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


def load_mms_model(device: torch.device | None = None):
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


def _prepare_words(text: str, uroman: Uroman, tokenizer) -> tuple[list[str], list[str]]:
    """Romanize text and filter to tokenizer dictionary.

    Excludes the CTC blank token's own character from the "valid" set even
    though it's technically a dictionary key — it's a structural separator
    in the vocabulary, not a real phoneme, so it must never appear in an
    alignment target. uroman's CJK romanization can legitimately emit it as
    a syllable separator (e.g. Chinese "共负一轭" -> "gong-eyi...") —
    confirmed 2026-08-12 against real cmn/yue 2CO 6 text: torchaudio's
    forced_align() raises "targets Tensor shouldn't contain blank index"
    when a target sequence contains it, since a leftover '-' from the old
    unfiltered dict_keys would otherwise pass straight through (it IS a
    real dictionary key — just the one reserved for blank, index 0).

    Returns (orig_words, clean_rom_words).
    """
    romanized = uroman.romanize_string(text)
    orig_words = text.split()
    rom_words = romanized.split()

    blank_char = next(k for k, v in tokenizer.dictionary.items() if v == 0)
    dict_keys = set(tokenizer.dictionary.keys()) - {blank_char}
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
_MAX_CHUNK_SAMPLES: int | None = None

# wav2vec2's convolutional front end: 400-sample receptive field, 320-sample
# hop (20 ms @ 16 kHz) -- so a pass over n samples yields
# (n - 400) // 320 + 1 frames, and frame k starts at sample k * 320.
_FRAME_HOP = 320

# Context given to each chunk on BOTH sides and then discarded (3 s @ 16 kHz).
# This is the design every established long-audio CTC pipeline uses -- fairseq's
# own MMS aligner (30 s windows + 3 s each side), ctc-forced-aligner (30 + 2),
# Hugging Face's ASR pipeline (chunk/6 each side) -- because wav2vec2's
# self-attention sees the whole forward pass, so frames near a cut are computed
# with truncated context. The earlier scheme here (0.5 s of left-only lookback)
# matched none of them. The context counts toward the forward-pass size, so
# peak memory is unchanged: each pass is still at most _max_chunk_samples().
_CHUNK_CONTEXT = 48_000


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

    Long input is cut into consecutive "core" spans; each core is run with
    _CHUNK_CONTEXT of real audio on both sides (so the pass is never larger
    than the device's max chunk) and only the core's own frames are kept.
    Cores and context are whole multiples of the frame hop, so the kept
    frames land on exactly the same frame grid as a single un-chunked pass
    and the total frame count is identical.
    """
    model_device = next(model.parameters()).device
    max_chunk = _max_chunk_samples(model_device)
    total_samples = waveform.shape[1]

    if total_samples <= max_chunk:
        with torch.no_grad():
            emission, _ = model(waveform.to(model_device))
        return emission.cpu()

    context = min(_CHUNK_CONTEXT, max_chunk // 4) // _FRAME_HOP * _FRAME_HOP
    core = max(_FRAME_HOP, (max_chunk - 2 * context) // _FRAME_HOP * _FRAME_HOP)

    emissions = []
    core_start = 0
    while core_start < total_samples:
        core_end = min(core_start + core, total_samples)
        in_start = max(0, core_start - context)
        in_end = min(total_samples, core_end + context)
        is_last = core_end >= total_samples

        with torch.no_grad():
            chunk_emission, _ = model(waveform[:, in_start:in_end].to(model_device))
        chunk_emission = chunk_emission.cpu()

        first_frame = (core_start - in_start) // _FRAME_HOP
        last_frame = chunk_emission.shape[1] if is_last else (core_end - in_start) // _FRAME_HOP
        emissions.append(chunk_emission[:, first_frame:last_frame, :])
        core_start = core_end

    return torch.cat(emissions, dim=1)


# MMS_FA's wav2vec2 architecture has a fixed 20ms/320-sample stride @16kHz —
# confirmed empirically 2026-08-10 (measured emission.shape[1] against real
# audio duration: 49.9989 fps, not a heuristic).
FRAMES_PER_SECOND = 50.0

# DP-table-cell estimate (frames x total_tokens) above which torchaudio's
# forced_align_impl is at risk of SEGFAULTING — not raising a catchable
# exception — confirmed directly 2026-08-10 (PYTHONFAULTHANDLER=1, 3/3
# reproductions) against Indonesian edition INDASV's Psalm 119 (the
# longest chapter in the Bible): 106,035 frames x 12,294 tokens ~= 1.30
# billion cells crashed; the same chapter succeeded elsewhere at 62,613 x
# 8,896 ~= 557 million cells. 300M leaves ~1.9x margin below the known-good
# point (not just below the crash point) since we only have one confirmed-
# safe data point and don't know how close to the true boundary it sits.
# Typical chapters are several orders of magnitude below this — this is a
# rare fallback path (a handful of unusually long chapters across the
# whole corpus), not a change to normal-case behavior. See
# _align_or_chunk()/_align_chapter_chunked() below.
#
# This is a crash-safety ceiling, not a runtime target — greedy chunk
# packing prefers the largest chunk that stays under it, so on slow/CPU
# hardware a near-maximal chunk (e.g. 65 of 66 verses in one chunk) can
# still take a long time despite never risking the crash. Override via
# --ctc-chunk-cells or conf/hw.local.json's ctc_chunk_threshold_cells for
# more evenly-sized (but more boundary-transition) chunks on slow boxes;
# 300M is calibrated safe to leave as the default for GPU hardware.
_CTC_CHUNK_THRESHOLD_CELLS = 300_000_000


def _ctc_min_frames_required(tokens: list[list[int]]) -> int:
    """Minimum emission frames CTC needs to align this token sequence.

    CTC requires one frame per target token, plus one extra frame between
    any two consecutive identical tokens (a blank is mandatory there, or
    they'd collapse into a single emission during decoding). This mirrors
    exactly what torchaudio's forced_align_impl checks internally — its
    own error message reports these same two terms ("targets length" and
    "number of repeats"), which is how this formula was reverse-derived
    (confirmed 2026-08-08 against real failures: e.g. targets length 245,
    repeats 2, required 247, available only 149).
    """
    flat = [t for word_tokens in tokens for t in word_tokens]
    if not flat:
        return 0
    repeats = sum(1 for i in range(1, len(flat)) if flat[i] == flat[i - 1])
    return len(flat) + repeats


# The wildcard token MMS_FA's emission carries as its last column (always
# log-prob 0, i.e. "matches any frame"). torchaudio's tokenizer maps "*" to it.
STAR_WORD = "*"


def _align_emission(
    emission,
    n_samples: int,
    text: str,
    bundle,
    tokenizer,
    aligner,
    uroman: Uroman,
    star_edges: bool = False,
) -> list[dict]:
    """CTC-align text against an already-computed emission (no model pass).

    star_edges pads the target with a wildcard before and after the text.
    Required whenever the audio window is wider than the text it is being
    aligned to (every windowed/anchored caller): without it CTC must account
    for every frame using only this text's tokens, so it smears the first
    and last words across the neighbouring units' speech. Confirmed
    2026-10-02 against DBT's own verse timings (ory REV 15: 5 of 8 verse
    starts 3-5.6 s early without it, all 8 within a constant offset with
    it) and on OBS narration (12 of 16 segment starts 1-3.5 s early). This
    is what the token exists for -- fairseq's MMS aligner and torchaudio's
    forced-alignment tutorial both use it for audio not covered by the text.

    Returns list of dicts with keys: text, start, end, score (times in
    seconds relative to the start of the emission).
    """
    orig_words, clean_rom_words = _prepare_words(text, uroman, tokenizer)
    target_words = [STAR_WORD] + clean_rom_words + [STAR_WORD] if star_edges else clean_rom_words
    tokens = tokenizer(target_words)

    # Proactive CTC feasibility check — compute whether this call CAN
    # succeed before attempting it, instead of attempting it and catching
    # the RuntimeError forced_align_impl would raise. Same outcome (no
    # crash), but this way we never call into a configuration we already
    # know is impossible, and the audio genuinely doesn't have enough
    # frames for this much text (e.g. an audio/text length mismatch, or
    # unusually fast speech) — not a bug to work around per call site.
    available_frames = emission.shape[1]
    required_frames = _ctc_min_frames_required(tokens)
    if required_frames > available_frames:
        log(
            f"CTC alignment infeasible: {available_frames} frames available, "
            f"{required_frames} required (text too long for this audio's "
            f"duration) — returning unaligned words instead of attempting it.",
            "WARNING",
        )
        return [
            {"text": w, "start": None, "end": None, "score": 0.0}
            for w in orig_words
        ]

    try:
        token_spans = aligner(emission[0], tokens)
    except RuntimeError as e:
        raise wrap_if_poisoned(e) from e
    note_alignment_success()
    if star_edges:
        token_spans = token_spans[1:-1]
    ratio = n_samples / emission.shape[1] / bundle.sample_rate

    results = []
    for word_i, word_spans in enumerate(token_spans):
        orig_word = orig_words[word_i] if word_i < len(orig_words) else "?"
        if not word_spans:
            results.append({
                "text": orig_word,
                "start": None,
                "end": None,
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


def _align_waveform(
    waveform,
    text: str,
    bundle,
    model,
    tokenizer,
    aligner,
    uroman: Uroman,
    star_edges: bool = False,
) -> list[dict]:
    """Core MMS_FA alignment on a pre-loaded waveform.

    The model forward pass is chunked for long audio (see
    _compute_emission_chunked) while the aligner still operates on the full
    emission sequence. See _align_emission for star_edges.

    Returns list of dicts with keys: text, start, end, score.
    """
    try:
        emission = _compute_emission_chunked(waveform, model)
    except RuntimeError as e:
        raise wrap_if_poisoned(e) from e
    return _align_emission(
        emission, waveform.shape[1], text, bundle, tokenizer, aligner, uroman,
        star_edges=star_edges,
    )


def _load_audio_via_ffmpeg_repair(audio_path: Path):
    """Re-encode through ffmpeg into a temp WAV, then load that.

    ffmpeg's own decoder tolerates malformed packets (logs a warning,
    drops the frame, keeps going) where torchaudio's stricter one raises
    outright — confirmed 2026-08-12 against a real corrupted source file
    (ron/RONBSR MRK 4: a couple of malformed packets right at the start of
    the file). Re-encoding recovered 377.3s of the original 380s — only
    the actually-corrupt lead-in packets were lost, not the whole file.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(audio_path), str(tmp_path)],
            capture_output=True, text=True, timeout=300, check=False,
        )
        if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg repair transcode failed: {result.stderr[-500:]}")
        return torchaudio.load(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def load_audio(audio_path: Path, bundle):
    """Load and resample audio, returning (waveform, sample_rate).

    Falls back to an ffmpeg repair transcode (see
    _load_audio_via_ffmpeg_repair) when torchaudio's own decoder raises —
    no overhead for the normal case, only triggers on an actual failure.
    """
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
    except Exception as e:
        log(f"  torchaudio.load() failed ({e}) — retrying via ffmpeg repair transcode", "WARNING")
        waveform, sample_rate = _load_audio_via_ffmpeg_repair(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    return waveform, bundle.sample_rate


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
    end_time: float | None = None,
    star_edges: bool = False,
) -> list[dict]:
    """Re-run MMS_FA on audio from restart_time onwards (or to end_time).

    Shared by collapse recovery, gap-fill, and drift correction.
    Slices the waveform, aligns text, adjusts timestamps back to original timeframe.

    star_edges: pass True when the window is deliberately wider than the
    text (a search window around one verse/segment) -- see _align_emission.

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

    # Defense-in-depth CTC-cells guard (see _CTC_CHUNK_THRESHOLD_CELLS):
    # this function is the one chokepoint every per-window caller funnels
    # through (verse-only's verse_anchored_align, fusion's gap-fill/drift
    # realign, OBS's segment_anchored_align), so guarding here protects
    # all of them without each needing its own chunking logic. The real
    # 2026-10-02 PSA 119 crash turned out to be a text-parsing bug (a
    # whole chapter's JSON misread as one ~29,000-char "verse" -- see
    # text_processing.py's _is_sofria_json) rather than a genuinely long
    # single verse/segment, so this guard has no known real trigger
    # today -- it exists for whatever's next, not as the fix for that bug.
    duration = segment.shape[1] / sample_rate
    _, clean_rom_words = _prepare_words(text, uroman, tokenizer)
    tokens = tokenizer(clean_rom_words)
    required_frames = _ctc_min_frames_required(tokens)
    available_frames_est = duration * FRAMES_PER_SECOND
    if available_frames_est * required_frames > _CTC_CHUNK_THRESHOLD_CELLS and len(clean_rom_words) > 1:
        words = text.split()
        if len(words) > 1:
            mid_word = len(words) // 2
            text_a = " ".join(words[:mid_word])
            text_b = " ".join(words[mid_word:])
            # Proportional-pace split of the window, same technique
            # _align_chapter_chunked() uses for verse-group boundaries --
            # weight by word count, not a blind time midpoint.
            frac = mid_word / len(words)
            mid_time = restart_time + duration * frac
            results_a = realign_from_point(
                waveform, sample_rate, restart_time, text_a,
                bundle, model, tokenizer, aligner, uroman, end_time=mid_time,
                star_edges=star_edges,
            )
            results_b = realign_from_point(
                waveform, sample_rate, mid_time, text_b,
                bundle, model, tokenizer, aligner, uroman, end_time=end_time,
                star_edges=star_edges,
            )
            return results_a + results_b

    results = _align_waveform(
        segment, text, bundle, model, tokenizer, aligner, uroman, star_edges=star_edges,
    )

    # None-safe: _align_waveform's fallback path (word/clip unalignable)
    # returns start/end=None, not a real timestamp — offsetting by
    # restart_time must preserve that "no timestamp" meaning rather than
    # crash (None + float) or silently turn it into a fake real value.
    for r in results:
        r["start"] = round(r["start"] + restart_time, 2) if r["start"] is not None else None
        r["end"] = round(r["end"] + restart_time, 2) if r["end"] is not None else None

    return results


def compute_file_emission(waveform, model):
    """One acoustic pass over a whole file (chunked to the device's limit).

    For callers that align many overlapping search windows from the same
    file (verse_anchored_align, segment_anchored_align): running the model
    per window pushes every second of audio through it several times over,
    because neighbouring windows overlap heavily. Compute the emission once
    and hand slices of it to align_window() instead. A side effect worth
    having: every unit is matched against the same acoustic evidence, so
    where a window happens to start can no longer perturb the result.
    """
    try:
        return _compute_emission_chunked(waveform, model)
    except RuntimeError as e:
        raise wrap_if_poisoned(e) from e


def align_window(
    emission,
    total_seconds: float,
    win_start: float,
    win_end: float,
    text: str,
    bundle,
    tokenizer,
    aligner,
    uroman,
) -> list[dict]:
    """Align text inside the [win_start, win_end) slice of a whole-file
    emission (see compute_file_emission). Always wildcard-padded -- a search
    window is by definition wider than its text (see _align_emission).

    Returns list of dicts with keys: text, start, end, score, with times in
    the file's own timeframe.
    """
    total_frames = emission.shape[1]
    fps = total_frames / total_seconds
    f0 = max(0, int(win_start * fps))
    f1 = min(total_frames, int(win_end * fps))
    if f1 <= f0:
        return []

    # Same crash-safety ceiling realign_from_point() applies (see
    # _CTC_CHUNK_THRESHOLD_CELLS): split the text and the window in
    # proportion rather than hand torchaudio an oversized DP table.
    _, clean_rom_words = _prepare_words(text, uroman, tokenizer)
    required_frames = _ctc_min_frames_required(tokenizer(clean_rom_words))
    words = text.split()
    if (f1 - f0) * required_frames > _CTC_CHUNK_THRESHOLD_CELLS and len(words) > 1:
        mid_word = len(words) // 2
        mid_time = win_start + (win_end - win_start) * mid_word / len(words)
        return (
            align_window(emission, total_seconds, win_start, mid_time, " ".join(words[:mid_word]),
                         bundle, tokenizer, aligner, uroman)
            + align_window(emission, total_seconds, mid_time, win_end, " ".join(words[mid_word:]),
                           bundle, tokenizer, aligner, uroman)
        )

    n_samples = round((f1 - f0) / fps * bundle.sample_rate)
    results = _align_emission(
        emission[:, f0:f1, :], n_samples, text, bundle, tokenizer, aligner, uroman,
        star_edges=True,
    )
    offset = f0 / fps
    for r in results:
        r["start"] = round(r["start"] + offset, 2) if r["start"] is not None else None
        r["end"] = round(r["end"] + offset, 2) if r["end"] is not None else None
    return results


# Verses of forward context given to each non-last chunk in
# _align_chapter_chunked() — aligned along with the chunk's own core
# verses for CTC trailing context, then discarded (that content gets its
# own full/proper alignment as the PRIMARY content of the next chunk).
LOOKAHEAD_VERSES = 2


def _align_chapter_chunked(
    waveform,
    sample_rate: int,
    non_empty_verses: list[str],
    bundle,
    model,
    tokenizer,
    aligner,
    uroman: Uroman,
    start_time: float = 0.0,
) -> tuple[list[dict], int]:
    """Align an oversized chapter by adaptively grouping verses into the
    largest chunks that stay safely under _CTC_CHUNK_THRESHOLD_CELLS, with
    forward lookahead overlap at each chunk boundary for CTC trailing
    context.

    Only reached via _align_or_chunk() when a whole-chapter alignment call
    would risk torchaudio's forced_align_impl segfault — a rare path for
    unusually long chapters, confirmed 2026-08-10 against Indonesian
    edition INDASV's Psalm 119 (see _CTC_CHUNK_THRESHOLD_CELLS's comment).

    start_time anchors the whole chapter's start (e.g. header_skip_time, or
    a collapse-recovery restart_time) — all internal chunk-boundary math is
    relative to the audio from start_time onward, and returned timestamps
    are shifted back to the original (start_time=0) timeframe, same
    convention as realign_from_point().

    Returns (word_results, chunk_count). word_results is a flat list of
    per-word dicts (text, start, end, score), one entry per word across
    ALL non_empty_verses in order — same shape _align_waveform()/
    realign_from_point() return, so callers (collapse detection,
    _map_word_idx_to_verse(), the JSON writer) need no changes.
    """
    total_duration = waveform.shape[1] / sample_rate - start_time
    word_counts = [len(v.split()) for v in non_empty_verses]
    total_words = sum(word_counts) or 1

    # Proportional-pace estimate of each verse's expected position — same
    # technique align_obs_words.py's segment_anchored_align() uses for its
    # own windowing, reused here for chunk-membership decisions instead.
    expected_starts = []
    cum_words = 0
    for wc in word_counts:
        expected_starts.append(total_duration * cum_words / total_words)
        cum_words += wc
    expected_ends = expected_starts[1:] + [total_duration]

    # Per-verse token counts (cheap — tokenizer lookups only, no model
    # forward pass) — needed for both the chunk-grouping loop below and
    # the min-window-duration guard per chunk.
    verse_token_counts = []
    for v in non_empty_verses:
        _, clean_rom = _prepare_words(v, uroman, tokenizer)
        verse_token_counts.append(_ctc_min_frames_required(tokenizer(clean_rom)))

    # ── Adaptive grouping: greedily pack consecutive verses into the
    # largest chunk that stays under the threshold. Larger chunks are
    # preferred over small fixed-size ones — more CTC context, fewer
    # boundary-transition points — since crossing the threshold at all is
    # rare enough that maximizing chunk size while staying safe is worth
    # the extra bookkeeping. ──
    chunks = []  # list of (start_idx, end_idx_exclusive)
    cur_start = 0
    cur_tokens = 0
    cur_frames_est = 0.0
    for i, wc in enumerate(word_counts):
        v_tokens = verse_token_counts[i]
        v_frames_est = (expected_ends[i] - expected_starts[i]) * FRAMES_PER_SECOND
        cand_tokens = cur_tokens + v_tokens
        cand_frames = cur_frames_est + v_frames_est
        if cur_tokens > 0 and cand_frames * cand_tokens > _CTC_CHUNK_THRESHOLD_CELLS:
            chunks.append((cur_start, i))
            cur_start = i
            cur_tokens = v_tokens
            cur_frames_est = v_frames_est
        else:
            cur_tokens = cand_tokens
            cur_frames_est = cand_frames
    chunks.append((cur_start, len(non_empty_verses)))
    # A single verse whose own token x frame estimate alone exceeds the
    # threshold would still risk the crash — the cur_tokens > 0 guard above
    # means the grouping loop can't split BELOW one verse. No real Bible
    # verse is anywhere near this scale (worst case a few dozen words), so
    # this is a theoretical residual, not an observed one.

    log(f"    Chapter exceeds CTC size threshold — splitting into "
        f"{len(chunks)} chunk(s) of {[e - s for s, e in chunks]} verse(s)")

    all_words: list[dict] = []
    for chunk_i, (v_start, v_end) in enumerate(chunks):
        is_last = chunk_i == len(chunks) - 1
        lookahead_end = v_end if is_last else min(v_end + LOOKAHEAD_VERSES, len(non_empty_verses))
        extended_verses = non_empty_verses[v_start:lookahead_end]
        extended_text = " ".join(extended_verses)
        core_word_count = sum(word_counts[v_start:v_end])

        chunk_start_time = start_time + expected_starts[v_start]
        chunk_end_time = start_time + expected_ends[lookahead_end - 1]

        # Same min-window-duration guard segment_anchored_align() uses
        # (align_obs_words.py), to keep the window from being too tight
        # for CTC on this chunk specifically.
        exp_dur = chunk_end_time - chunk_start_time
        min_required_dur = exp_dur * 1.3 + 2.0
        if exp_dur < min_required_dur:
            chunk_end_time = min(start_time + total_duration, chunk_start_time + min_required_dur)

        try:
            chunk_words = realign_from_point(
                waveform, sample_rate, chunk_start_time, extended_text,
                bundle, model, tokenizer, aligner, uroman, end_time=chunk_end_time,
            )
        except CudaContextPoisonedError:
            raise
        except RuntimeError as e:
            log(f"    Chunk {chunk_i + 1}/{len(chunks)} alignment failed ({e}), "
                f"using unaligned fallback for this chunk", "WARNING")
            chunk_words = [
                {"text": w, "start": None, "end": None, "score": 0.0}
                for v in extended_verses for w in v.split()
            ]

        # Keep only the core chunk's own words — the lookahead-only words
        # get their own full/proper alignment as the PRIMARY content of
        # the next chunk's own call.
        all_words.extend(chunk_words[:core_word_count])

        # Same per-window cache-release rationale as segment_anchored_align()
        # — this loop runs once per chunk per oversized chapter (rare, but
        # can still be dozens of chunks for something Psalm-119-scale).
        device_type = next(model.parameters()).device.type
        if device_type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        elif device_type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    return all_words, len(chunks)


def _align_or_chunk(
    waveform,
    sample_rate: int,
    non_empty_verses: list[str],
    bundle,
    model,
    tokenizer,
    aligner,
    uroman: Uroman,
    start_time: float = 0.0,
) -> tuple[list[dict], int | None]:
    """Dispatch to chunked or direct whole-text alignment based on a cheap
    size estimate — the shared gate used by process_chapter()'s initial
    alignment call AND its collapse-restart retry call, so both are
    protected against the CTC segfault (see _CTC_CHUNK_THRESHOLD_CELLS).
    Without this shared entry point, a collapse-triggered retry over a
    large remaining suffix of an oversized chapter would hit the exact
    same crash the chunking exists to prevent.

    Avoids paying for a full-chapter acoustic forward pass just to
    discover chunking is needed: available_frames is estimated from audio
    duration x FRAMES_PER_SECOND (a fixed architecture constant, not a
    heuristic — see its own comment), and required tokens via the
    already-cheap tokenizer() + _ctc_min_frames_required() (no model
    forward pass either).

    Returns (word_results, chunk_count) — chunk_count is None when the
    normal (non-chunked) path was used, purely for caller log/stats
    visibility into how often the rare chunked path actually fires.
    """
    text = " ".join(non_empty_verses)
    remaining_duration = waveform.shape[1] / sample_rate - start_time
    available_frames_est = remaining_duration * FRAMES_PER_SECOND

    _, clean_rom_words = _prepare_words(text, uroman, tokenizer)
    tokens = tokenizer(clean_rom_words)
    required_frames = _ctc_min_frames_required(tokens)

    if available_frames_est * required_frames > _CTC_CHUNK_THRESHOLD_CELLS:
        return _align_chapter_chunked(
            waveform, sample_rate, non_empty_verses,
            bundle, model, tokenizer, aligner, uroman, start_time=start_time,
        )

    word_results = realign_from_point(
        waveform, sample_rate, start_time, text,
        bundle, model, tokenizer, aligner, uroman,
    )
    return word_results, None


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
    star_edges: bool = False,
) -> list[dict]:
    """Run MMS_FA on a segment of the audio between start_time and end_time.

    Convenience wrapper around realign_from_point that loads audio from a file.
    Returns list of dicts with keys: text, start, end, score.
    """
    waveform, sample_rate = load_audio(audio_path, bundle)
    return realign_from_point(
        waveform, sample_rate, start_time, text,
        bundle, model, tokenizer, aligner, uroman,
        end_time=end_time, star_edges=star_edges,
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
    word_results: list[dict], book: str, chapter: str, output_path: Path,
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

# ─── Collapse Detection for Existing Output ──────────────────────────────

# ─── Work Item Discovery ───────────────────────────────────────────────────

# ─── Chapter Processing ────────────────────────────────────────────────────

def process_chapter(item: dict, bundle, model, tokenizer, aligner, uroman, config,
                    header_skip_time: float | None = None,
                    whisper_path: Path | None = None) -> dict:
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
    verse_texts = read_verse_texts(text_path, config)

    # Clean verses for alignment and join into single text
    cleaned_verses = [clean_for_alignment(v, config) for v in verse_texts]
    non_empty_verses = [v for v in cleaned_verses if v]
    full_text = " ".join(non_empty_verses)
    total_words = len(full_text.split())

    if total_words == 0:
        return {"error": "No words in reference text after cleaning"}

    # Load audio once (reused for retry if needed)
    waveform, sample_rate = load_audio(audio_path, bundle)

    # Skip header: slice audio to start at verse_start_time.
    # _align_or_chunk() handles both the normal whole-chapter case and the
    # rare oversized-chapter case (chunked alignment, see
    # _CTC_CHUNK_THRESHOLD_CELLS) transparently — same call either way.
    t0 = time.time()
    word_results, chunk_count = _align_or_chunk(
        waveform, sample_rate, non_empty_verses,
        bundle, model, tokenizer, aligner, uroman,
        start_time=header_skip_time or 0.0,
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
                        # _align_or_chunk(), not a direct realign_from_point()
                        # call: the remaining suffix of an already-oversized
                        # chapter can itself be large enough to risk the same
                        # CTC segfault this whole mechanism exists to avoid —
                        # without this, only the FIRST alignment attempt
                        # would be protected, not a collapse-triggered retry.
                        retry_results, retry_chunk_count = _align_or_chunk(
                            waveform, sample_rate, non_empty_verses[verse_idx:],
                            bundle, model, tokenizer, aligner, uroman,
                            start_time=restart_time,
                        )
                        elapsed += time.time() - t1
                        if retry_chunk_count is not None:
                            chunk_count = retry_chunk_count

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
    if chunk_count is not None:
        result["chunked"] = True
        result["chunk_count"] = chunk_count

    # Release cached-but-unused device memory after each chapter. Chapters
    # vary widely in length, so each one's chunked forward pass allocates
    # differently-sized tensors — without freeing back to the allocator
    # between chapters, PyTorch's CUDA caching allocator accumulates
    # fragmented free blocks that don't coalesce for a later chapter's
    # larger allocation, even when nvidia-smi shows free VRAM (this is
    # exactly what PyTorch's own "reserved but unallocated" OOM hint
    # points at). Same rationale on MPS: unified memory shared with
    # everything else the process holds (including Whisper's model), so
    # an unbounded cache here is a real OOM contributor on a long
    # multi-chapter run either way.
    device_type = next(model.parameters()).device.type
    if device_type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif device_type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    return result


def _load_whisper_words(whisper_path: Path) -> list | None:
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
