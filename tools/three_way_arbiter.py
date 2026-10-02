#!/usr/bin/env python3
"""Three-way verse-boundary arbitration: our pipeline vs. DBT's own timing
vs. Whisper's independent transcript, for chapters where all three exist.

Motivation (2026-09-24 session): tools/compare_timing.py treats DBT's
timing as ground truth, but DBT's own alignment can itself be wrong —
confirmed directly against real chapters (xtn/ACT3: verse 12's DBT
timestamp lands mid-sentence in Whisper's independent transcript while ours
lands right at a sentence start; verse 11 is the reverse). Neither source
is reliably authoritative alone. Whisper transcribes straight from the
audio with zero knowledge of either our verse boundaries or DBT's, so it's
a genuinely independent third signal.

PRIMARY method — content matching (added 2026-09-24, per explicit request
to strengthen this beyond a structural heuristic): for each disputed verse,
take its own reference text (the same text the alignment pipeline itself
used), romanize it with uroman (the same phonetic-normalization tool
mms_align_words.py already uses for CTC alignment — text and Whisper's
transcript are very unlikely to share a script/spelling convention
directly, especially for the low-resource languages this pipeline mostly
handles, so literal string comparison would fail even for a perfect
match), and search a window of Whisper's own (also uroman-romanized) word
stream for the best-matching contiguous run. The search starts centered
between DBT's and our claimed timestamps (deliberately not biased toward
either) and widens geometrically (see WIDEN_FACTOR/MAX_RADIUS) if no
confident match is found, then refines the best window's boundary by
nudging it a few words earlier/later to pin down the tightest fit. This
gives a content-VERIFIED timestamp, not just a plausible-looking boundary.

FALLBACK method — the original structural heuristic (a capitalized word
preceded by a real silence gap, i.e. plausibly sentence-initial): used
only when content matching can't find a confident match anywhere in the
widened search (e.g. Whisper's transcription is too garbled in that
region for text comparison to mean anything, which is common for the
lower-resource languages this pipeline targets).

Only covers the subset of the corpus with all three data sources: DBT
timing + our pipeline timing + Whisper transcription (fusion-mode chapters
only — verse_only_mode deliberately skips Whisper, see
align_verse_words.py's module docstring). See correlate_arbiter_labels.py
for turning this subset's verdicts into signals that might generalize to
chapters without DBT/Whisper data at all.

Usage:
    python tools/three_way_arbiter.py --testament nt
    python tools/three_way_arbiter.py --iso xtn
    python tools/three_way_arbiter.py --iso xtn --book ACT --chapter 3 --detail
"""
import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

from quality_report import (
    TIMING_DIR, DOWNLOADS_DIR, find_all_downloaded_timecode, find_pipeline_timing_files,
    load_timing_verses, compare_verse_timings, _parse_timing_path, _get_canons,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from text_processing import read_verse_texts, clean_for_alignment, load_language_config  # noqa: E402
from vowel_pacing import verse_vowel_counts, compare_pacing, detect_repeated_phrases  # noqa: E402
from whisper_quality_guard import is_low_whisper_quality_language, HIGH_BAR_MATCH_RATIO  # noqa: E402

WORD_TIMING_DIR = Path("word-timing-data")

DISPUTE_THRESHOLD = 2.0       # seconds — same bar compare_timing.py uses for BAD

# --- structural fallback (capitalized word + silence gap) ---
MIN_GAP = 0.20
MAX_CANDIDATE_DISTANCE = 3.0
TIE_MARGIN = 0.5

# --- content-matching primary method ---
INITIAL_RADIUS = 20.0    # seconds either side of the search center, first try
WIDEN_FACTOR = 2.0        # multiply radius by this each time nothing confident is found
MAX_RADIUS = 160.0        # give up widening past this (seconds either side)
MIN_MATCH_RATIO = 0.35    # difflib ratio below this isn't trusted as a real match —
                           # lowered from 0.45 2026-09-24: a 400-language sample found
                           # 39.4% of STRUCTURAL-fallback verses (the ones this
                           # threshold rejects) scored 0.35-0.45, and spot-checking
                           # confirmed genuine (if noisy) content matches at that
                           # level for low-resource languages where Whisper's own
                           # transcription is inherently rough — e.g. aaa/AAAMLT
                           # ACT 1:5 at ratio 0.43: reference "...ijonu re ame..."
                           # vs Whisper's "...gionnu ...re mu..." is a real, if
                           # imperfect, match, not noise. TEXT_MATCH verses are
                           # ambiguous only ~1% of the time regardless of ratio
                           # within its current range, vs. ~80% for the STRUCTURAL
                           # fallback these near-misses would otherwise drop into —
                           # rescuing them should sharply cut overall ambiguity.
WEAK_MATCH_RATIO = 0.5    # a TEXT_MATCH verdict below this ratio is confident enough
                           # to keep (it beat MIN_MATCH_RATIO) but weak enough that a
                           # confident, disagreeing vowel-pacing verdict downgrades it
                           # to AMBIGUOUS rather than being ignored -- see
                           # arbitrate_chapter()'s cross-check, added 2026-09-26 after
                           # mxt/MXTTBL ACT 2:3 confidently resolved wrong at ratio=0.40
REFINE_WORDS = 3           # after finding the best window, nudge its start by up to
                            # this many words either way to tighten the boundary
PER_VERSE_TIME_CAP = 3.0   # seconds — hard wall-clock cap on one verse's search;
                            # a long chapter at MAX_RADIUS can have hundreds of
                            # candidate windows, each an O(window) difflib call —
                            # without this cap one outlier chapter can stall an
                            # entire corpus run with zero visibility (confirmed
                            # 2026-09-24: a ~50min-estimated run was still going
                            # after 80+ minutes)

DRIFT_GAP_MARGIN = 30.0    # seconds — how far past a chapter's own recorded
                            # drift/gap-fix window (see align_words.py's
                            # gap-fill / drift-correction step) its effect is
                            # still treated as untrustworthy. The fix mechanism
                            # only re-anchors the few words immediately around
                            # the detected gap; DBT's own reference timing is
                            # never re-synced against that same narrator-repeat
                            # confusion, so DBT can stay drifted well past the
                            # fix's own narrow segment. Confirmed directly
                            # 2026-09-24: hla/HLAPNG ACT 4:17-19 — three already-
                            # correct verses (verified word-for-word against the
                            # real MMS transcript) got "corrected" toward a DBT
                            # timestamp that was itself ~155s off, landing
                            # entirely inside the wrong verse's content, because
                            # the search center below was seeded from that same
                            # bad DBT number. 30s is a deliberately generous
                            # margin — the confirmed bad cases extended well
                            # beyond the fix file's own segment_end_time.


def _whisper_path_for(pipeline_timing_path: Path) -> Path | None:
    """Derive word-timing-data/.../*_whisper_words.json from an
    export/timing-data/.../*_timing.json path — same relative structure,
    different root and filename suffix (see align_words.py's WORD_TIMING_DIR
    / align_pipeline.py's word_book_dir construction).
    """
    try:
        rel = pipeline_timing_path.relative_to(TIMING_DIR)
    except ValueError:
        return None
    stem = pipeline_timing_path.name.replace("_timing.json", "")
    whisper_path = WORD_TIMING_DIR / rel.parent / f"{stem}_whisper_words.json"
    return whisper_path


def _reference_text_path_for(pipeline_timing_path: Path) -> Path | None:
    """Find the .txt reference text file this chapter's alignment used —
    same flat layout as the audio/DBT-timing files (downloads/BB/{canon}/
    {iso}/{distinct_id}/{book}/{book}_{chapter}_{text_fileset}.txt), text
    fileset commonly but not always == distinct_id, so glob rather than
    assume the name.
    """
    try:
        rel = pipeline_timing_path.relative_to(TIMING_DIR)
    except ValueError:
        return None
    canon, iso, distinct_id, book = rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
    stem = pipeline_timing_path.name.replace("_timing.json", "")
    parts = stem.split("_", 2)  # BOOK, CCC, fileset
    if len(parts) < 2:
        return None
    chapter_str = parts[1]
    book_dir = DOWNLOADS_DIR / canon / iso / distinct_id / book
    if not book_dir.exists():
        return None
    matches = list(book_dir.glob(f"{book}_{chapter_str}_*.txt"))
    return matches[0] if matches else None


def _drift_gap_windows_for(pipeline_timing_path: Path) -> list[tuple[str, float, float]]:
    """Load this chapter's own recorded drift/gap-fix events — written by
    align_words.py's gap-fill / MMS-drift-correction step (same directory
    as the whisper words file: word-timing-data/.../{BOOK}_{chapter}_
    {drift,gap}_{N}s.json) — as (kind, start, end) windows. See
    DRIFT_GAP_MARGIN for why these can't be trusted as a DBT-comparison
    anchor even well past their own recorded segment_end.
    """
    try:
        rel = pipeline_timing_path.relative_to(TIMING_DIR)
    except ValueError:
        return []
    stem = pipeline_timing_path.name.replace("_timing.json", "")
    parts = stem.split("_", 2)
    if len(parts) < 2:
        return []
    book, chapter_str = parts[0], parts[1]
    word_dir = WORD_TIMING_DIR / rel.parent
    if not word_dir.exists():
        return []
    windows = []
    for f in word_dir.glob(f"{book}_{chapter_str}_drift_*s.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        start, end = d.get("restart_time"), d.get("segment_end_time")
        if start is not None and end is not None:
            windows.append(("drift", start, end))
    for f in word_dir.glob(f"{book}_{chapter_str}_gap_*s.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        start, end = d.get("segment_start"), d.get("segment_end")
        if start is not None and end is not None:
            windows.append(("gap", start, end))
    return windows


def _in_drift_gap_window(t: float, windows: list[tuple[str, float, float]]) -> bool:
    return any(s - DRIFT_GAP_MARGIN <= t <= e + DRIFT_GAP_MARGIN for _kind, s, e in windows)


def _load_whisper_words(whisper_path: Path) -> list[dict] | None:
    try:
        data = json.loads(whisper_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    words = data.get("words", [])
    return words or None


def _load_whisper_candidates(words: list[dict]) -> list[float]:
    """Structural fallback candidates: capitalized words preceded by a real
    silence gap — see module docstring."""
    candidates = []
    prev_end = None
    for w in words:
        text = w.get("text", "")
        start = w.get("start")
        end = w.get("end")
        if start is None:
            continue
        is_cap = bool(re.match(r"^\s*[A-Z]", text))
        has_gap = prev_end is not None and (start - prev_end) >= MIN_GAP
        if is_cap and has_gap:
            candidates.append(start)
        prev_end = end if end is not None else prev_end
    return candidates


def _nearest_distance(t: float, candidates: list[float]) -> float | None:
    if not candidates:
        return None
    return min(abs(c - t) for c in candidates)


def _romanize_whisper_words(words: list[dict], uroman) -> list[dict]:
    """Pre-romanize a chapter's whole Whisper word list once — reused
    across every disputed verse in that chapter, since romanizing per-verse
    per-search-window would redo the same work many times over."""
    out = []
    for w in words:
        start = w.get("start")
        if start is None:
            continue
        rom = uroman.romanize_string(w.get("text", "")).strip().lower()
        if rom:
            out.append({"start": start, "end": w.get("end"), "rom": rom})
    return out


def _romanize_verse_text(verse_text: str, config, uroman) -> str:
    cleaned = clean_for_alignment(verse_text, config)
    return uroman.romanize_string(cleaned).strip().lower()


def _window_ratio(verse_rom: str, whisper_rom_words: list[dict], start_idx: int, n: int) -> float:
    window_text = " ".join(w["rom"] for w in whisper_rom_words[start_idx:start_idx + n])
    if not window_text:
        return 0.0
    return difflib.SequenceMatcher(None, verse_rom, window_text).ratio()


def _text_match_search(verse_rom: str, verse_word_count: int, whisper_rom_words: list[dict],
                        center: float) -> dict | None:
    """Slide a window (sized to the verse's own word count) across Whisper's
    romanized word stream, searching a region centered on `center` that
    widens geometrically until a confident match is found or MAX_RADIUS is
    hit. Refines the best window's start by nudging a few words either way.

    Returns {"start": float, "ratio": float, "word_idx": int} for the best
    match found, or None if nothing reached MIN_MATCH_RATIO even at max
    radius.
    """
    if not whisper_rom_words or verse_word_count <= 0:
        return None

    n = max(1, verse_word_count)
    radius = INITIAL_RADIUS
    best = None
    deadline = time.monotonic() + PER_VERSE_TIME_CAP

    while radius <= MAX_RADIUS:
        if time.monotonic() > deadline:
            # A long chapter + a wide radius means many candidate windows,
            # each an O(window) difflib comparison — pathologically slow on
            # an outlier chapter otherwise (confirmed 2026-09-24: a
            # full-corpus run that should've taken ~50min was still going
            # after 80+ with no visibility into which chapter). Bail to
            # whatever's best so far (may be None) rather than let one
            # chapter stall the whole run; caller falls back to the
            # structural heuristic when this returns nothing confident.
            break
        # candidate start indices: any word whose own start falls in range
        lo_t, hi_t = center - radius, center + radius
        idxs = [i for i, w in enumerate(whisper_rom_words) if lo_t <= w["start"] <= hi_t]
        if idxs:
            for i in idxs:
                if time.monotonic() > deadline:
                    break
                ratio = _window_ratio(verse_rom, whisper_rom_words, i, n)
                if best is None or ratio > best["ratio"]:
                    best = {"start": whisper_rom_words[i]["start"], "ratio": ratio, "word_idx": i}
            if best and best["ratio"] >= MIN_MATCH_RATIO:
                break
        radius *= WIDEN_FACTOR

    if best is None:
        return None

    # Refine: nudge the window start a few words either way for the tightest fit.
    # Done even for a sub-threshold best — the diagnostic ratio should
    # reflect the tightest fit search found, not an unrefined guess, so
    # near-miss analysis (see correlate_arbiter_labels.py-style digging)
    # isn't comparing refined "hits" against unrefined "misses".
    i0 = best["word_idx"]
    for delta in range(-REFINE_WORDS, REFINE_WORDS + 1):
        i = i0 + delta
        if i < 0 or i >= len(whisper_rom_words):
            continue
        ratio = _window_ratio(verse_rom, whisper_rom_words, i, n)
        if ratio > best["ratio"]:
            best = {"start": whisper_rom_words[i]["start"], "ratio": ratio, "word_idx": i}

    # Caller decides whether best["ratio"] clears MIN_MATCH_RATIO — always
    # returning the best-found result (not None below threshold) lets
    # callers log a diagnostic ratio even for a rejected match.
    return best


def resolve_verse(v: str, dl_t: float, pl_t: float, ref_verses, whisper_rom_words,
                   struct_candidates: list[float], config, uroman,
                   center: float | None = None,
                   drift_gap_windows: list | None = None) -> dict:
    """Resolve one verse's DBT-vs-pipeline dispute — the per-verse body of
    arbitrate_chapter()'s loop, extracted so a caller can also re-invoke it
    for a SPECIFIC verse with a custom search center (see
    apply_arbiter_corrections.py's cascade resolution: when a confident
    correction is blocked by a neighbor that was never itself confidently
    resolved — AMBIGUOUS, not a hard "already agrees with DBT" boundary —
    re-running that neighbor's own search anchored nearer the now-corrected
    verse can resolve it too, unlocking the whole run instead of stopping
    at the first blocked verse).

    Returns the same per-verse detail dict arbitrate_chapter() puts in its
    "detail" list (verse, dbt_t, ours_t, method, match_t, match_ratio,
    best_ratio_seen, best_t_seen, dbt_dist, ours_dist, verdict).
    """
    method = None
    match_t = None
    match_ratio = None
    best_ratio_seen = None  # diagnostic: best ratio found even if rejected
    best_t_seen = None      # diagnostic: its timestamp — lets a future threshold
                             # change be re-evaluated from saved --out data instead
                             # of requiring a re-run (a gap that cost a re-run
                             # 2026-09-24, fixed here)

    vi = int(v) - 1
    if whisper_rom_words is not None and ref_verses is not None and 0 <= vi < len(ref_verses):
        verse_text = ref_verses[vi]
        cleaned = clean_for_alignment(verse_text, config) if verse_text else ""
        word_count = len(cleaned.split()) if cleaned else 0
        if word_count > 0:
            verse_rom = _romanize_verse_text(verse_text, config, uroman)
            if center is not None:
                search_center = center
            elif drift_gap_windows and (_in_drift_gap_window(dl_t, drift_gap_windows)
                                         or _in_drift_gap_window(pl_t, drift_gap_windows)):
                # DBT's own timestamp can't be trusted as half of the search
                # anchor here (see DRIFT_GAP_MARGIN) — center on our own
                # pipeline position alone rather than the (DBT+ours)/2
                # midpoint, so a DBT-side drift doesn't drag the search into
                # the wrong neighborhood.
                search_center = pl_t
            else:
                search_center = (dl_t + pl_t) / 2.0
            match = _text_match_search(verse_rom, word_count, whisper_rom_words, search_center)
            if match is not None:
                best_ratio_seen = match["ratio"]
                best_t_seen = match["start"]
                if match["ratio"] >= MIN_MATCH_RATIO:
                    method = "TEXT_MATCH"
                    match_t = match["start"]
                    match_ratio = match["ratio"]

    if match_t is not None:
        dl_dist = abs(dl_t - match_t)
        pl_dist = abs(pl_t - match_t)
        if abs(dl_dist - pl_dist) < TIE_MARGIN:
            verdict = "AMBIGUOUS"
        else:
            verdict = "DBT" if dl_dist < pl_dist else "OURS"
    else:
        # Fallback: structural (capitalized word + gap) heuristic.
        method = "STRUCTURAL"
        dl_dist = _nearest_distance(dl_t, struct_candidates)
        pl_dist = _nearest_distance(pl_t, struct_candidates)
        dl_ok = dl_dist is not None and dl_dist <= MAX_CANDIDATE_DISTANCE
        pl_ok = pl_dist is not None and pl_dist <= MAX_CANDIDATE_DISTANCE
        if not dl_ok and not pl_ok:
            verdict = "AMBIGUOUS"
        elif dl_ok and not pl_ok:
            verdict = "DBT"
        elif pl_ok and not dl_ok:
            verdict = "OURS"
        elif abs(dl_dist - pl_dist) < TIE_MARGIN:
            verdict = "AMBIGUOUS"
        else:
            verdict = "DBT" if dl_dist < pl_dist else "OURS"

    return {
        "verse": v, "dbt_t": dl_t, "ours_t": pl_t,
        "method": method, "match_t": match_t, "match_ratio": match_ratio,
        "best_ratio_seen": best_ratio_seen, "best_t_seen": best_t_seen,
        "dbt_dist": dl_dist, "ours_dist": pl_dist, "verdict": verdict,
    }


def prepare_chapter_matching_context(whisper_path: Path, text_path: Path | None, config, uroman):
    """Load and romanize everything resolve_verse() needs for one chapter —
    shared setup so arbitrate_chapter() and a cascade-resolution caller
    don't each redo the whisper-word romanization pass. Returns
    (whisper_rom_words, struct_candidates, ref_verses) — the first two may
    be None/empty if whisper data is missing (verse_only_mode chapters, by
    design, never have it -- see align_verse_words.py's module docstring);
    ref_verses is None only if no reference text was found or it failed to
    parse, INDEPENDENT of whisper availability -- a verse_only_mode chapter
    still has real reference text, and arbitrate_chapter() now falls back to
    pipeline/vowel_pacing.py's Whisper-independent signal using exactly that
    text when whisper data is absent, instead of giving up on the whole
    chapter (confirmed 2026-09-26: this was previously ~95% of all
    AMBIGUOUS/unarbitrated chapters -- see the arbiter's own investigation
    notes -- because verse_only_mode chapters were skipped before ever
    reaching a verdict).
    """
    ref_verses = None
    if text_path is not None and text_path.exists():
        try:
            ref_verses = read_verse_texts(text_path, config)
        except Exception:
            ref_verses = None

    whisper_words = _load_whisper_words(whisper_path)
    if whisper_words is None:
        return None, None, ref_verses
    struct_candidates = _load_whisper_candidates(whisper_words)

    whisper_rom_words = None
    if ref_verses is not None:
        whisper_rom_words = _romanize_whisper_words(whisper_words, uroman)

    return whisper_rom_words, struct_candidates, ref_verses


def resolve_chapter_by_pacing(dl_verses: dict, pl_verses: dict, ref_verses: list[str],
                               config, uroman, common: list[str], strict: bool = False) -> dict | None:
    """Whisper-independent fallback for verse_only_mode (and any other
    no-Whisper) chapters: one chapter-wide vowel-pacing comparison (see
    pipeline/vowel_pacing.py) applied to every disputed verse, since the
    fingerprint is inherently a whole-chapter judgment, not a per-verse one.

    strict=True raises the bar for committing to a verdict (see
    pipeline/vowel_pacing.py's STRICT_MIN_RATIO/STRICT_MIN_ABS_MARGIN) --
    used when Whisper data DOES exist for this chapter (this is only a
    fallback for one verse's own weak/failed TEXT_MATCH, not for a
    language with no Whisper coverage at all). Confirmed 2026-09-30
    (spa/SPAWTC JHN 5): letting a weak pacing margin overrule a language
    where Whisper is otherwise trustworthy produced a confirmed-wrong
    verdict. strict=False (the default) keeps the original, more lenient
    bar for languages with no Whisper signal to fall back on at all --
    some evidence beats none there.

    Returns None if the pacing signal itself can't be computed (e.g. no
    vowels at all after cleaning) -- callers should treat that exactly like
    "no evidence", same as today's no-Whisper AMBIGUOUS.
    """
    vowel_counts = verse_vowel_counts(ref_verses, config, uroman)
    if not any(vowel_counts.values()):
        return None
    from vowel_pacing import STRICT_MIN_RATIO, STRICT_MIN_ABS_MARGIN
    kwargs = {"min_ratio": STRICT_MIN_RATIO, "min_abs_margin": STRICT_MIN_ABS_MARGIN} if strict else {}
    result = compare_pacing(dl_verses, pl_verses, vowel_counts, label_a="DBT", label_b="OURS", **kwargs)
    repeats = detect_repeated_phrases(ref_verses, config, uroman)

    verdict_map = {"DBT": "DBT", "OURS": "OURS", None: "AMBIGUOUS"}
    verdict = verdict_map[result["pick"]]
    fp_a, fp_b = result["fp_a"], result["fp_b"]
    return {
        "verdict": verdict,
        "method": "VOWEL_PACING",
        "reason": result["reason"],
        "dbt_rms": fp_a.rms_dev if fp_a else None,
        "ours_rms": fp_b.rms_dev if fp_b else None,
        "repeated_phrases_nearby": repeats,
    }


def arbitrate_chapter(dl_path: Path, pl_path: Path, whisper_path: Path,
                       text_path: Path | None, config, uroman, iso: str | None = None) -> dict | None:
    """Returns {verses_disputed, votes: {OURS, DBT, AMBIGUOUS}, detail: [...]}
    or None if there's nothing to arbitrate (no dispute, or no evidence of
    any kind -- no Whisper AND no usable reference text).

    `iso`, when given, gates every verdict through
    whisper_quality_guard.is_low_whisper_quality_language() -- see that
    function's docstring (confirmed 2026-10-01 on acd/ACDWBT MAT 22: a
    language the backfill already flagged as broadly unreliable can still
    produce a TEXT_MATCH verdict that LOOKS confident but is wrong by ear).
    """
    dl_verses = load_timing_verses(dl_path)
    pl_verses = load_timing_verses(pl_path)
    if dl_verses is None or pl_verses is None:
        return None

    whisper_rom_words, struct_candidates, ref_verses = prepare_chapter_matching_context(
        whisper_path, text_path, config, uroman)
    if struct_candidates is None and ref_verses is None:
        return None  # truly nothing to arbitrate with
    drift_gap_windows = _drift_gap_windows_for(pl_path)

    common = sorted((set(dl_verses) & set(pl_verses)) - {"0"}, key=lambda x: int(x))
    disputed = [v for v in common if abs(dl_verses[v] - pl_verses[v]) >= DISPUTE_THRESHOLD]
    if not disputed:
        return None

    # Chapter-wide vowel-pacing verdict, computed once up front whenever
    # there's reference text to compute it from -- used as the LAST-tier
    # fallback per verse below (after TEXT_MATCH and STRUCTURAL both had
    # their chance), not only when Whisper is totally absent. Confirmed
    # 2026-09-26: several verse_only_mode chapters (e.g. gat/GATNTM ACT 11,
    # mir/MIRTBL ACT 4) still carry a STALE whisper_words.json from before
    # the language was flagged verse_only_mode, so struct_candidates is NOT
    # None for them -- but that stale Whisper data is itself unreliable
    # enough that STRUCTURAL resolves 0% of their disputed verses, leaving
    # 100% AMBIGUOUS. Gating vowel-pacing on "no Whisper file exists" alone
    # would miss exactly this population; gating it on "the existing
    # methods came back AMBIGUOUS" catches it too.
    pacing = None
    if ref_verses is not None:
        # strict=True whenever the LANGUAGE is fusion-mode (not
        # verse_only_mode) -- NOT whenever struct_candidates happens to be
        # non-None for this one chapter. Confirmed 2026-09-30 (ind/INDASV
        # MAT 5:33 and ACT 8:15): gating on struct_candidates let a chapter
        # whose whisper_words.json file was simply missing from disk (never
        # generated, or purged -- not a verse_only_mode design choice) fall
        # through to the lenient bar, where a barely-1.5x-ratio pacing call
        # confidently picked the wrong side per direct ear verification.
        # verse_only_mode is a deliberate per-language flag that Whisper is
        # never trustworthy there at all; a missing per-chapter file on an
        # otherwise-fusion-mode language doesn't change that Whisper SHOULD
        # be trustworthy for this language, so the strict bar still applies.
        pacing = resolve_chapter_by_pacing(dl_verses, pl_verses, ref_verses, config, uroman, disputed,
                                            strict=not getattr(config, "verse_only_mode", False))

    votes = {"OURS": 0, "DBT": 0, "AMBIGUOUS": 0}
    detail = []

    for v in disputed:
        dl_t, pl_t = dl_verses[v], pl_verses[v]
        if struct_candidates is not None:
            entry = resolve_verse(v, dl_t, pl_t, ref_verses, whisper_rom_words,
                                   struct_candidates, config, uroman,
                                   drift_gap_windows=drift_gap_windows)
        else:
            entry = {
                "verse": v, "dbt_t": dl_t, "ours_t": pl_t,
                "method": None, "match_t": None, "match_ratio": None,
                "best_ratio_seen": None, "best_t_seen": None,
                "dbt_dist": None, "ours_dist": None, "verdict": "AMBIGUOUS",
            }

        if entry["verdict"] == "AMBIGUOUS" and pacing is not None:
            entry = {
                **entry,
                "method": pacing["method"], "dbt_dist": pacing["dbt_rms"], "ours_dist": pacing["ours_rms"],
                "verdict": pacing["verdict"], "pacing_reason": pacing["reason"],
            }
        elif (entry["verdict"] != "AMBIGUOUS" and pacing is not None and pacing["verdict"] != "AMBIGUOUS"
              and pacing["verdict"] != entry["verdict"]
              and (entry["method"] == "STRUCTURAL"
                   or (entry["method"] == "TEXT_MATCH" and (entry.get("match_ratio") or 1.0) < WEAK_MATCH_RATIO))):
            # A confident verdict from the weaker methods (STRUCTURAL, or a
            # TEXT_MATCH barely above MIN_MATCH_RATIO) that an independent,
            # content-agnostic signal actively contradicts. Confirmed
            # 2026-09-26: mxt/MXTTBL ACT 2:3 -- TEXT_MATCH confidently
            # resolved OURS at ratio=0.40 (just above the 0.35 floor), which
            # is wrong per direct ear verification; vowel-pacing correctly
            # favored DBT here. Don't flip the verdict on a secondary
            # heuristic's say-so -- just refuse to certify a weak primary
            # one that a second, independent signal disagrees with.
            entry = {**entry, "verdict": "AMBIGUOUS",
                     "pacing_reason": f"downgraded: {entry['method']} said {entry['verdict']}, "
                                       f"pacing said {pacing['verdict']} ({pacing['reason']})"}

        if (iso is not None and is_low_whisper_quality_language(iso)
                and entry["verdict"] != "DBT"
                and not (entry["method"] == "TEXT_MATCH" and (entry.get("match_ratio") or 0.0) >= HIGH_BAR_MATCH_RATIO)):
            # This language's own Whisper output was already broadly
            # unreliable per the backfill's quality guard -- don't let a
            # merely-moderate TEXT_MATCH/STRUCTURAL/pacing verdict (or no
            # verdict at all) override DBT's own timing. Confirmed
            # 2026-10-01: acd's 3 "OURS" verdicts at ratio 0.53-0.66 were
            # all wrong by direct ear verification.
            entry = {**entry, "verdict": "DBT",
                     "pacing_reason": f"low-whisper-quality language override: "
                                       f"{entry['method']} said {entry['verdict']} "
                                       f"(ratio={entry.get('match_ratio')}), defaulting to DBT"}

        votes[entry["verdict"]] += 1
        detail.append(entry)

    if not detail:
        return None
    return {"verses_disputed": len(detail), "votes": votes, "detail": detail}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", type=str)
    parser.add_argument("--iso-list", type=str)
    parser.add_argument("--testament", type=str, choices=["nt", "ot", "both"], default=None)
    parser.add_argument("--book", type=str, default=None)
    parser.add_argument("--chapter", type=str, default=None)
    parser.add_argument("--detail", action="store_true", help="Print per-verse verdicts")
    parser.add_argument("--out", type=str, default=None, help="Write full per-chapter results as JSON")
    parser.add_argument("--progress-every", type=int, default=100,
                         help="Print progress to stderr every N chapters checked (0 to disable)")
    args = parser.parse_args()

    from uroman import Uroman
    uroman = Uroman()

    if args.iso:
        isos = [args.iso]
    elif args.iso_list:
        isos = [c.strip() for c in args.iso_list.split(",")]
    else:
        canons = _get_canons(args.testament)
        isos = set()
        for canon in canons:
            d = TIMING_DIR / canon
            if d.exists():
                isos.update(p.name for p in d.iterdir() if p.is_dir())
        isos = sorted(isos)

    total_votes = {"OURS": 0, "DBT": 0, "AMBIGUOUS": 0}
    method_counts = {"TEXT_MATCH": 0, "STRUCTURAL": 0, "VOWEL_PACING": 0}
    chapters_arbitrated = 0
    chapters_checked = 0
    all_results = []
    run_start = time.monotonic()

    for iso in isos:
        try:
            config = load_language_config(iso)
        except Exception:
            config = load_language_config("default")
        for canon in _get_canons(args.testament):
            downloaded_tc = find_all_downloaded_timecode(iso, canon)
            if not downloaded_tc:
                continue
            pipeline_files = find_pipeline_timing_files(iso, canon)
            for c, tf in pipeline_files:
                distinct_id, book, chapter_str = _parse_timing_path(tf)
                if not chapter_str:
                    continue
                if args.book and book != args.book:
                    continue
                if args.chapter and chapter_str.lstrip("0") != args.chapter.lstrip("0"):
                    continue
                key = (c, distinct_id, book, chapter_str)
                if key not in downloaded_tc:
                    continue
                whisper_path = _whisper_path_for(tf)
                if whisper_path is None:
                    continue
                text_path = _reference_text_path_for(tf)
                chapters_checked += 1
                if args.progress_every and chapters_checked % args.progress_every == 0:
                    elapsed = time.monotonic() - run_start
                    print(f"  ... checked {chapters_checked} chapters ({chapters_arbitrated} arbitrated), "
                          f"currently at {iso}/{distinct_id} {book} {chapter_str}, "
                          f"{elapsed:.0f}s elapsed", file=sys.stderr)
                result = arbitrate_chapter(downloaded_tc[key], tf, whisper_path, text_path, config, uroman, iso=iso)
                if result is None:
                    continue
                chapters_arbitrated += 1
                for k in total_votes:
                    total_votes[k] += result["votes"][k]
                for d in result["detail"]:
                    if d["method"] in method_counts:
                        method_counts[d["method"]] += 1
                all_results.append({
                    "iso": iso, "canon": c, "distinct_id": distinct_id,
                    "book": book, "chapter": chapter_str, **result,
                })
                if args.detail:
                    print(f"\n{iso}/{distinct_id} {book} {chapter_str} ({c}): "
                          f"{result['votes']}")
                    for d in result["detail"]:
                        ratio_str = f" ratio={d['match_ratio']:.2f}" if d["match_ratio"] is not None else ""
                        print(f"  v{d['verse']}: DBT={d['dbt_t']:.2f} OURS={d['ours_t']:.2f} "
                              f"[{d['method']}{ratio_str}] -> {d['verdict']}")

    print(f"\nChapters arbitrated (had DBT + Whisper + a real dispute): {chapters_arbitrated}")
    total = sum(total_votes.values())
    if total:
        for k, v in total_votes.items():
            print(f"  {k}: {v} ({100*v/total:.1f}%)")
        print(f"  method used: TEXT_MATCH={method_counts['TEXT_MATCH']} STRUCTURAL={method_counts['STRUCTURAL']} "
              f"VOWEL_PACING={method_counts['VOWEL_PACING']}")
    else:
        print("  No disputed verses found with arbitrable data.")

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
