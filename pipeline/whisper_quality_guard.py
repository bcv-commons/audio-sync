"""Post-hoc quality gate for freshly-transcribed Whisper output, used by
tools/whisper_backfill.py to catch chapters (and whole languages) where
Whisper produced garbage instead of a real transcript.

Why this exists: nothing else in the codebase does this at the point a
chapter is transcribed. whisper_transcribe.py's
`hallucination_silence_threshold` decode option is a MITIGATION applied
during decoding (tells the decoder to re-skip silence around a likely
hallucination) -- it doesn't tell you afterward whether it worked.
tools/verify_whisper_hint.py's score_transcript() computes a match_ratio
against reference text, but confirmed 2026-09-30 against real samples that
this doesn't discriminate at whole-chapter granularity -- an excellent
Czech transcript scored 0.022, a transcript that was 10% verbatim-looping
garbage scored 0.011, not a usable absolute threshold. align_pipeline.py's
AUTO_VERSE_ONLY_* mechanism is the right SHAPE (per-language rolling
tracker, auto-abort after enough evidence) but only fires as a side effect
of the later fusion/alignment step, not from a standalone transcription
pass like this backfill.

Calibrated 2026-09-30 against 11 real sample chapters across 11 languages
newly transcribed for the missing-whisper backfill (2 genuinely bad:
`mca` -- a "San Mateo 3" x7 hallucination loop, 21 words for a 217s
chapter; `ban` -- a real 33-word "Matius Paus Kali" loop at the start
plus an unrelated hallucinated-English drift ["calculate", "vaccinations",
"improving", "woman", "earth", ...] near the end, diluted by otherwise-real
Balinese content elsewhere in the same chapter; 9 genuinely good). Two
independent metrics, not one:

  - words_per_second: word_count / audio_duration. The single cleanest
    signal found -- the 9 good chapters clustered at 1.52-2.31 words/sec,
    `ban` (partially bad) sat at 0.83, `mca` (mostly bad) at 0.097. A
    chapter-wide compression-ratio check (Whisper's own internal
    hallucination heuristic, extended here to whole-chapter granularity)
    was tried first and rejected -- normal chapters' ordinary short-word
    repetition (function words) put several genuinely-good samples above
    Whisper's own default 2.4 threshold, so it doesn't separate cleanly at
    this granularity. A 3-gram repetition-ratio scan was also tried and
    rejected as the PRIMARY signal -- it caught `ban`'s opening loop
    (0.031 vs 0.004-0.014 for good chapters) but completely missed the
    hallucinated-English drift later in the same chapter, since that
    portion doesn't repeat at all, it fabricates. Kept as a SECONDARY
    signal since it's cheap and catches a different failure shape (a loop
    that doesn't depress word count, e.g. looping on short function
    words at normal cadence).
  - repetition_ratio: fraction of 3-grams equal to the most common 3-gram
    in the chapter.

Neither metric is trusted alone on a single chapter -- thresholds are
deliberately conservative (only mca's extreme case fails words_per_second
outright) and the real backstop is the per-language ROLLING tracker in
tools/whisper_backfill.py, which mirrors AUTO_VERSE_ONLY_MIN_CHAPTERS's
"don't decide on one data point" design: only after several chapters of a
language look bad does the backfill stop spending GPU time on it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Below this, a chapter is flagged as suspect (see module docstring for the
# calibration data this came from -- good chapters: 1.52-2.31 w/s;
# ban (partially bad): 0.83 w/s; mca (mostly bad): 0.097 w/s). Set below
# the lowest observed good sample with real margin, not right at the edge.
MIN_WORDS_PER_SECOND = 1.0

# 3-gram repetition ratio above this is flagged as a secondary signal.
# ban's opening loop measured 0.031 against a 0.004-0.014 band for good
# chapters -- this threshold sits well above that band, below ban's loop.
MAX_REPETITION_RATIO = 0.10

# A chapter needs essentially no words at all to be an instant, no-margin
# reject regardless of duration (mca: 21 words / 217s chapter).
MIN_ABSOLUTE_WORDS = 15

# Per-language rolling tracker (mirrors align_pipeline.py's
# AUTO_VERSE_ONLY_MIN_CHAPTERS/AUTO_VERSE_ONLY_HEADER_FAIL_RATE pattern):
# don't judge a language on fewer than this many transcribed chapters...
LANG_MIN_CHAPTERS_BEFORE_JUDGING = 8
# ...and only abort the language for the rest of this run if at least this
# fraction of those chapters were flagged.
LANG_ABORT_FLAGGED_FRACTION = 0.5


def repetition_ratio(words: list[dict], n: int = 3) -> float:
    """Fraction of n-grams equal to the single most common n-gram in the
    transcript -- catches a tight 'X Y Z X Y Z ...' hallucination loop."""
    toks = [w["text"].strip().lower() for w in words if w.get("text", "").strip()]
    if len(toks) < n * 2:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    most_common = counts.most_common(1)[0][1] if counts else 0
    return most_common / len(grams)


@dataclass
class ChapterQuality:
    word_count: int
    duration: float
    words_per_second: float
    repetition_ratio_3gram: float
    flagged: bool
    reasons: list[str] = field(default_factory=list)


def assess_chapter(words: list[dict], duration: float) -> ChapterQuality:
    n_words = len(words)
    wps = n_words / duration if duration > 0 else 0.0
    rep = repetition_ratio(words, n=3)

    reasons = []
    if n_words < MIN_ABSOLUTE_WORDS and duration > 20:
        reasons.append(f"only {n_words} words for a {duration:.0f}s chapter")
    if wps < MIN_WORDS_PER_SECOND:
        reasons.append(f"words/sec {wps:.2f} < {MIN_WORDS_PER_SECOND}")
    if rep > MAX_REPETITION_RATIO:
        reasons.append(f"3-gram repetition ratio {rep:.2f} > {MAX_REPETITION_RATIO}")

    return ChapterQuality(
        word_count=n_words, duration=duration, words_per_second=round(wps, 3),
        repetition_ratio_3gram=round(rep, 3), flagged=bool(reasons), reasons=reasons,
    )


@dataclass
class LanguageTracker:
    chapters_judged: int = 0
    chapters_flagged: int = 0
    aborted: bool = False
    abort_reason: str | None = None


def update_language_tracker(tracker: LanguageTracker, quality: ChapterQuality) -> None:
    tracker.chapters_judged += 1
    if quality.flagged:
        tracker.chapters_flagged += 1


def should_abort_language(tracker: LanguageTracker) -> bool:
    """Call after update_language_tracker(). Returns True the first time
    this language crosses the abort bar -- caller should stop processing
    its remaining chapters for this run (does NOT touch any config file;
    that's a separate, bigger decision for a human, not this backfill)."""
    if tracker.aborted:
        return True
    if tracker.chapters_judged < LANG_MIN_CHAPTERS_BEFORE_JUDGING:
        return False
    flagged_fraction = tracker.chapters_flagged / tracker.chapters_judged
    if flagged_fraction >= LANG_ABORT_FLAGGED_FRACTION:
        tracker.aborted = True
        tracker.abort_reason = (
            f"{tracker.chapters_flagged}/{tracker.chapters_judged} chapters "
            f"flagged ({flagged_fraction:.0%}) >= {LANG_ABORT_FLAGGED_FRACTION:.0%} threshold"
        )
        return True
    return False
