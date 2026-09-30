"""Vowel-density pacing signal — a Whisper-independent plausibility check on
verse timing, usable both as a merge-time input (alongside/instead of
Whisper's text-match candidates) and as a post-hoc diagnostic (alongside the
DBT-vs-pipeline arbiter).

Motivation: for `verse_only_mode` languages (and any chapter where Whisper's
transcript is unusable), the arbiter has no independent evidence to check a
disputed verse against — it falls back to a structural heuristic or gives up
(AMBIGUOUS). But the chapter's own reference text already encodes an
approximate speech-rate signal: assuming roughly constant speech rate within
a chapter, a verse's share of the chapter's total vowel count should roughly
match its share of the chapter's total audio duration. That doesn't require
a single word to be recognized — it only needs the reference text and *some*
verse-to-timestamp mapping to test for internal consistency.

Confirmed 2026-09-26 against 9 ear-verified real disputes (spanning 8 real
low-resource languages, `verse_only_mode` in 7 of 9): anchored on a source's
*own* first/last verse timestamps, the deviation-from-constant-rate curve
cleanly separated a genuinely well-paced source (small, flat, noisy-but-
bounded deviation) from a badly-drifted one (large, structured, bow-shaped
deviation, e.g. -76s at the trough over one third of a 58-verse chapter) —
even in a case (`mtr` MAT 13) where the wrong source's own *word-level* MMS
confidence score showed no drop at all at the disputed verse. The two
signals are complementary: confidence catches "these words don't sound
right," this catches "these words are confidently in the wrong place."

Known failure mode, not fixed by this module: a genuinely repeated/
formulaic phrase nearby (e.g. Matthew's recurring "when Jesus had finished
these parables..." discourse-closing formula) has genuinely similar vowel
density at more than one place in the chapter, so pacing alone can't
disambiguate between them either — `detect_repeated_phrases()` flags this
so a caller can lower its trust in ANY content-based signal for that
chapter, not just this one.

Known bias, guarded against: verse 1's own timestamp is frequently corrupted
by chapter-header/intro misdetection (see `align_words.py`'s
`detect_audio_header()` docstring) — anchoring the whole fingerprint on a
bad verse-1 timestamp would bias every single verse's expected time. Anchor
selection therefore skips verse 1 by default when a later verse is available
to anchor on instead.

Second known bias, found and fixed 2026-09-26 (`gat` ACT 11 false near-tie):
a source that has completely failed past some point (MMS confidence
collapsed to zero) doesn't necessarily leave its verse-position field empty
-- it can leave every verse from the failure point onward at the exact same
frozen fallback timestamp, never advancing. Anchoring on that source's own
LAST verse -- which is itself one of the frozen values -- trivially forces
the deviation curve to converge back to ~0 right at the anchor regardless of
how meaningless the frozen stretch is, making a total-failure source look
almost as "smooth" as a genuinely well-paced one (confirmed: 5.83s vs
5.84s RMS, indistinguishable, on data where one source was frozen for 28
consecutive verses). `fingerprint()` now detects a run of `DEGENERATE_RUN_LEN`
or more consecutive verses sharing a byte-identical timestamp, excludes that
whole run from both anchor selection and the deviation curve, and reports it
back via `degenerate_verses` so a caller (the arbiter, the merge step) can
treat "this verse's timestamp is inside a frozen run" as its own hard signal
-- independent of, and stronger than, any deviation number -- rather than
silently computing a deviation against a fabricated value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

VOWELS = set("aeiouAEIOU")
DEGENERATE_RUN_LEN = 3   # 3+ consecutive verses at the identical timestamp -> frozen, not real
DEGENERATE_EPS = 0.005   # seconds; timestamps this close count as "identical" (float rounding)


def find_degenerate_verses(verse_times: dict[str, float], ordered_verses: list[str]) -> set[str]:
    """Return the set of verses that belong to a run of DEGENERATE_RUN_LEN or
    more CONSECUTIVE (by verse number, per ordered_verses) verses sharing a
    byte-identical timestamp -- the fingerprint that a fallback/failure path
    got stuck and just repeated its last real (or placeholder) value instead
    of producing new positions. See module docstring for why this matters:
    without this check, a frozen run can make total failure look like smooth,
    well-paced output once the fingerprint's own anchor lands inside it."""
    degenerate = set()
    run = [ordered_verses[0]] if ordered_verses else []
    for prev, cur in zip(ordered_verses, ordered_verses[1:]):
        if abs(verse_times[cur] - verse_times[prev]) <= DEGENERATE_EPS:
            run.append(cur)
        else:
            if len(run) >= DEGENERATE_RUN_LEN:
                degenerate.update(run)
            run = [cur]
    if len(run) >= DEGENERATE_RUN_LEN:
        degenerate.update(run)
    return degenerate


def count_vowels(romanized_text: str) -> int:
    """Vowel-letter count in an already-romanized (uroman) string. Counting
    letters rather than collapsing long vowels (e.g. "aa") to one unit is
    deliberate: a long vowel is acoustically longer to say than a short one,
    so letter-count is a reasonable proxy for duration, not just an artifact
    to normalize away."""
    return sum(1 for c in romanized_text if c in VOWELS)


def verse_vowel_counts(ref_verses: list[str], config, uroman) -> dict[str, int]:
    """Romanize + count vowels for every verse in a chapter's reference text.

    ref_verses is 0-indexed (ref_verses[0] is verse 1), matching
    text_processing.read_verse_texts()'s own convention. Returns a
    {verse_str: vowel_count} dict, 1-indexed, matching the verse-string keys
    used throughout this codebase's timing/quality JSON (load_timing_verses,
    _words_quality.json, etc). A missing/empty verse counts as 0, not an
    error -- callers already treat 0-vowel verses as contributing nothing to
    the cumulative fraction, which is correct (e.g. a verse-number-only
    placeholder with no real text).
    """
    from text_processing import clean_for_alignment  # local import: pipeline/ sibling

    counts = {}
    for i, text in enumerate(ref_verses):
        v = str(i + 1)
        if not text:
            counts[v] = 0
            continue
        cleaned = clean_for_alignment(text, config)
        rom = uroman.romanize_string(cleaned) if cleaned else ""
        counts[v] = count_vowels(rom)
    return counts


@dataclass
class PacingFingerprint:
    anchor_lo: str
    anchor_hi: str
    per_verse: dict[str, dict] = field(default_factory=dict)  # verse -> {t, expected, dev}
    rms_dev: float = 0.0
    max_abs_dev: float = 0.0
    max_abs_dev_verse: str | None = None
    degenerate_verses: set = field(default_factory=set)  # verses excluded as a frozen run


def fingerprint(verse_times: dict[str, float], vowel_counts: dict[str, int],
                skip_verse_1: bool = True) -> PacingFingerprint | None:
    """Compute the constant-speech-rate deviation curve for ONE source's own
    per-verse timestamps (DBT's, our pipeline's, or any candidate set --
    this function is source-agnostic by design, so the same call works for
    either side of an arbiter dispute, or for a raw MMS candidate before
    fusion commits to it).

    Anchors on the first and last usable verse present in verse_times
    (intersected with vowel_counts), skipping verse "1" as a start anchor
    when a later verse is available -- verse 1's timestamp is
    disproportionately likely to carry chapter-header contamination (see
    module docstring) -- and skipping any verse inside a frozen/degenerate
    run (see find_degenerate_verses()) entirely, from both anchor selection
    and the deviation curve; those verses are reported separately via
    `degenerate_verses` rather than silently scored. Returns None if fewer
    than 3 usable, non-degenerate verses remain.
    """
    ordered = sorted(
        (v for v in verse_times if v in vowel_counts),
        key=lambda x: int(x),
    )
    degenerate = find_degenerate_verses(verse_times, ordered)
    common = [v for v in ordered if v not in degenerate]
    if skip_verse_1 and len(common) > 3 and common[0] == "1":
        common = common[1:]
    if len(common) < 3:
        return None

    lo, hi = common[0], common[-1]
    lo_t, hi_t = verse_times[lo], verse_times[hi]
    span = hi_t - lo_t
    total_vowels = sum(vowel_counts[v] for v in common if int(lo) < int(v) <= int(hi))
    if total_vowels <= 0 or span <= 0:
        return None

    fp = PacingFingerprint(anchor_lo=lo, anchor_hi=hi, degenerate_verses=degenerate)
    cum = 0
    devs = []
    for v in common:
        if int(v) <= int(lo):
            continue
        # frac uses vowels accumulated BEFORE this verse -- it predicts
        # where v STARTS. Adding vowel_counts[v] first would predict where
        # v ENDS (i.e. where v+1 starts) and silently mislabel it as v's
        # own expected position -- exactly the off-by-one bug found
        # 2026-09-26 in tools/refit_verse_only_fallback.py (a single-verse
        # fallback run was placed at literally the SAME timestamp as the
        # next real verse, zero duration, because of this mistake).
        frac = cum / total_vowels
        expected = lo_t + frac * span
        actual = verse_times[v]
        dev = actual - expected
        fp.per_verse[v] = {"t": actual, "expected": expected, "dev": dev}
        devs.append(dev)
        cum += vowel_counts[v]

    if not devs:
        return None
    fp.rms_dev = math.sqrt(sum(d * d for d in devs) / len(devs))
    worst = max(devs, key=abs)
    fp.max_abs_dev = abs(worst)
    fp.max_abs_dev_verse = next(v for v, d in zip(
        (v for v in common if int(v) > int(lo)), devs) if d == worst)
    return fp


DEFAULT_MIN_RATIO = 1.5     # worse source's RMS must be at least this many times
                             # the better source's, not just numerically higher
DEFAULT_MIN_ABS_MARGIN = 2.0  # ...and the absolute gap must clear this floor too,
                               # so two near-zero RMS values don't pass on ratio alone
STRICT_MIN_RATIO = 2.5      # used when the OTHER source (e.g. Whisper) is generally
STRICT_MIN_ABS_MARGIN = 5.0  # available for this chapter -- see compare_pacing()'s
                             # docstring for why a weak pacing margin shouldn't be
                             # allowed to overrule a signal that's usually trustworthy


def compare_pacing(times_a: dict[str, float], times_b: dict[str, float],
                    vowel_counts: dict[str, int], label_a: str = "a", label_b: str = "b",
                    min_ratio: float = DEFAULT_MIN_RATIO,
                    min_abs_margin: float = DEFAULT_MIN_ABS_MARGIN) -> dict:
    """Compare two candidate verse-timing sources (DBT vs pipeline, pipeline
    vs a raw pre-fusion MMS candidate, etc.) via their pacing fingerprints.
    This is the one entry point the arbiter and the merge step should both
    call -- it encodes the degenerate-source handling `fingerprint()` now
    exposes, so a caller never has to re-derive "no fingerprint means no
    trust" logic itself.

    Confirmed 2026-09-30 (spa/SPAWTC JHN 5): a fixed absolute tie-margin
    (originally 0.5s) let a genuinely weak signal commit to a verdict --
    DBT's RMS (9.79s) vs ours' (11.85s) is only a ~2s gap, and BOTH numbers
    were already elevated (dialogue-heavy chapter breaking the constant-
    rate assumption for either source), not one clean source vs one bad
    one. Ear-verified: the pick was wrong. Every validated TRUE positive
    this signal has ever produced (mtr, mxt, ndv, ...) had a much larger
    RATIO between the two RMS values (2.6x-27x), not just a numerically
    bigger gap -- so requiring both a minimum ratio AND a minimum absolute
    margin keeps every known-good case while correctly rejecting a
    marginal one like JHN 5.

    Returns {"pick": label_a|label_b|None, "reason": str,
             "fp_a": PacingFingerprint|None, "fp_b": PacingFingerprint|None}.
    pick is None when neither source has enough signal to compare (both
    fingerprints unavailable), or when the margin doesn't clear the bar --
    callers should treat that exactly like today's AMBIGUOUS, not as a
    vote either way.
    """
    fp_a = fingerprint(times_a, vowel_counts)
    fp_b = fingerprint(times_b, vowel_counts)

    if fp_a is None and fp_b is None:
        return {"pick": None, "reason": "neither source has a usable fingerprint", "fp_a": None, "fp_b": None}
    if fp_a is None:
        return {"pick": label_b, "reason": f"{label_a} has no usable fingerprint (degenerate/insufficient) -- {label_b} wins by default", "fp_a": None, "fp_b": fp_b}
    if fp_b is None:
        return {"pick": label_a, "reason": f"{label_b} has no usable fingerprint (degenerate/insufficient) -- {label_a} wins by default", "fp_a": fp_a, "fp_b": None}

    better, worse = (fp_a, fp_b) if fp_a.rms_dev <= fp_b.rms_dev else (fp_b, fp_a)
    ratio = (worse.rms_dev / better.rms_dev) if better.rms_dev > 0 else float("inf")
    abs_margin = worse.rms_dev - better.rms_dev

    if ratio < min_ratio or abs_margin < min_abs_margin:
        return {"pick": None,
                "reason": f"RMS deviation too close to call ({fp_a.rms_dev:.2f}s vs {fp_b.rms_dev:.2f}s, "
                          f"ratio={ratio:.2f}x, need >={min_ratio}x and >={min_abs_margin}s gap)",
                "fp_a": fp_a, "fp_b": fp_b}

    pick = label_a if fp_a.rms_dev < fp_b.rms_dev else label_b
    return {"pick": pick, "reason": f"lower RMS deviation ({fp_a.rms_dev:.2f}s vs {fp_b.rms_dev:.2f}s, ratio={ratio:.2f}x)",
            "fp_a": fp_a, "fp_b": fp_b}


def detect_repeated_phrases(ref_verses: list[str], config, uroman,
                             min_words: int = 4, jaccard_thresh: float = 0.6) -> list[tuple[str, str, float]]:
    """Flag pairs of NON-ADJACENT verses in a chapter whose romanized word
    sets overlap heavily -- a real, common case in Biblical narrative
    (discourse-closing formulas, refrains, genealogies' "X begat Y"
    template) where content-based signals (this module, Whisper text-match,
    even MMS's own confidence) genuinely cannot distinguish one occurrence
    from another, because the audio really does sound similar in both
    places. Returns [(verse_a, verse_b, jaccard_similarity), ...] for any
    pair scoring at or above jaccard_thresh, so a caller can lower its trust
    in a content-based signal for the verses involved rather than silently
    trusting a coin-flip.
    """
    from text_processing import clean_for_alignment

    word_sets: dict[str, set[str]] = {}
    for i, text in enumerate(ref_verses):
        if not text:
            continue
        v = str(i + 1)
        cleaned = clean_for_alignment(text, config)
        rom = uroman.romanize_string(cleaned) if cleaned else ""
        words = [w for w in rom.split() if len(w) > 1]
        if len(words) >= min_words:
            word_sets[v] = set(words)

    flagged = []
    verses = sorted(word_sets, key=lambda x: int(x))
    for i, va in enumerate(verses):
        for vb in verses[i + 1:]:
            if int(vb) - int(va) <= 1:
                continue  # adjacent verses sharing connective words is normal, not a repetition risk
            a, b = word_sets[va], word_sets[vb]
            inter = len(a & b)
            union = len(a | b)
            if union == 0:
                continue
            jac = inter / union
            if jac >= jaccard_thresh:
                flagged.append((va, vb, round(jac, 3)))
    return flagged
