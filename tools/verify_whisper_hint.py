#!/usr/bin/env python3
"""Verify a candidate Whisper language-hint fix before adding it to
ISO639_3_TO_WHISPER (pipeline/whisper_transcribe.py).

Motivation: an iso missing from that table gets no language hint at all,
which for many low-resource languages makes Whisper auto-detect the WRONG
language and produce garbage (wrong script, hallucinated phrases, near-
total silence-miss) — confirmed directly this session on real chapters
(word count/score for `urb/ACT 7`, `lcp/ACT 7`, etc.). But an "obvious"
fix isn't automatically safe: `azb`'s mapping turned out to be a no-op
(Whisper's own model can't produce the right script for it regardless of
hint), and `bak.toml` documents a case where an explicit hint actively
REGRESSED output (forced wrong-script transliteration versus a better
auto-detect result). Both are real, already-encountered failure modes,
not hypothetical.

Method: for each candidate (iso, whisper_hint), re-transcribe a few real
chapters TWICE — once with no hint (today's behavior) and once with the
candidate hint — and score each transcript against how well it matches
the chapter's own known reference text (a whole-chapter uroman-romanized
difflib ratio — coarse but exactly the signal that distinguishes "real
matching content" from "wrong-language garbage", the same test used by
hand throughout this session). Classify CONFIRMED_IMPROVED / NO_OP /
REGRESSED / INCONCLUSIVE. Only ever reports — never touches
ISO639_3_TO_WHISPER or any config file.

Usage:
    python tools/verify_whisper_hint.py --iso slk:sk,ukr:uk,hrv:hr \
        --chapters-per-iso 3 --out _runs/whisper_hint_verify.json
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from quality_report import TIMING_DIR, DOWNLOADS_DIR, find_pipeline_timing_files, _parse_timing_path, _get_canons  # noqa: E402
from whisper_transcribe import (  # noqa: E402
    DEFAULT_MODEL, load_whisper_model, transcribe_audio, build_word_timeline,
)
from text_processing import read_verse_texts, clean_for_alignment, load_language_config  # noqa: E402
from download_language_content import download_audio, DownloadStats, ErrorLogger  # noqa: E402

IMPROVED_MARGIN = 0.15   # hinted ratio must beat baseline by at least this to count as improved
REGRESSED_MARGIN = 0.05  # hinted ratio below baseline by at least this counts as regressed

_download_stats = DownloadStats()
_error_logger = ErrorLogger()


def _audio_path_for(pipeline_timing_path: Path) -> Path | None:
    """Mirror three_way_arbiter._reference_text_path_for but for the .mp3."""
    try:
        rel = pipeline_timing_path.relative_to(TIMING_DIR)
    except ValueError:
        return None
    canon, iso, distinct_id, book = rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
    stem = pipeline_timing_path.name.replace("_timing.json", "")
    parts = stem.split("_", 2)
    if len(parts) < 2:
        return None
    chapter_str = parts[1]
    book_dir = DOWNLOADS_DIR / canon / iso / distinct_id / book
    if not book_dir.exists():
        return None
    matches = list(book_dir.glob(f"{book}_{chapter_str}_*.mp3"))
    return matches[0] if matches else None


def _reference_text_path_for(pipeline_timing_path: Path) -> Path | None:
    try:
        rel = pipeline_timing_path.relative_to(TIMING_DIR)
    except ValueError:
        return None
    canon, iso, distinct_id, book = rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
    stem = pipeline_timing_path.name.replace("_timing.json", "")
    parts = stem.split("_", 2)
    if len(parts) < 2:
        return None
    chapter_str = parts[1]
    book_dir = DOWNLOADS_DIR / canon / iso / distinct_id / book
    if not book_dir.exists():
        return None
    matches = list(book_dir.glob(f"{book}_{chapter_str}_*.txt"))
    return matches[0] if matches else None


def _try_download_audio(pipeline_timing_path: Path) -> Path | None:
    """The chapter has a real pipeline timing.json (proof it was
    successfully aligned before) but its source .mp3 has since been
    cleaned up locally. Extract the audio fileset id from the timing
    filename itself (same id the pipeline already fetched successfully
    once, e.g. `ACT_001_TAMDIPN1DA_timing.json` -> fileset `TAMDIPN1DA`)
    and re-fetch just that one file via the existing download_audio()."""
    try:
        rel = pipeline_timing_path.relative_to(TIMING_DIR)
    except ValueError:
        return None
    canon, iso, distinct_id, book = rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
    stem = pipeline_timing_path.name.replace("_timing.json", "")
    parts = stem.split("_", 2)
    if len(parts) < 3:
        return None
    _book2, chapter_str, fileset_id = parts
    try:
        chapter_num = int(chapter_str)
    except ValueError:
        return None
    out_dir = DOWNLOADS_DIR / canon / iso / distinct_id / book
    out_path = out_dir / f"{book}_{chapter_str}_{fileset_id}.mp3"
    if out_path.exists():
        return out_path
    ok = download_audio(fileset_id, book, chapter_num, out_path, iso, distinct_id,
                         _download_stats, _error_logger)
    return out_path if ok and out_path.exists() else None


def pick_sample_chapters(iso: str, n: int, allow_download: bool = True) -> list[Path]:
    """Pick up to n chapters with real audio + reference text on disk,
    preferring longer (more substantial) chapters for a more reliable
    signal. Tries NT then OT. Attempts an on-demand re-download of the
    source audio (see _try_download_audio) when it's missing locally —
    most fusion-mode isos had their raw audio cleaned up after the
    original alignment run, only the timing.json output remains."""
    on_disk = []
    needs_download = []
    for canon in _get_canons("both"):
        for c, tf in find_pipeline_timing_files(iso, canon):
            text = _reference_text_path_for(tf)
            if text is None:
                continue
            audio = _audio_path_for(tf)
            if audio is not None:
                try:
                    size = audio.stat().st_size
                except OSError:
                    continue
                on_disk.append((size, tf, audio, text))
            else:
                needs_download.append((tf, text))

    on_disk.sort(key=lambda x: -x[0])
    chosen = on_disk[:n]

    if len(chosen) < n and allow_download:
        for tf, text in needs_download:
            if len(chosen) >= n:
                break
            audio = _try_download_audio(tf)
            if audio is None:
                continue
            try:
                size = audio.stat().st_size
            except OSError:
                continue
            chosen.append((size, tf, audio, text))

    return chosen


def score_transcript(words: list[dict], ref_text: str, config, uroman) -> dict:
    transcript_text = " ".join(w["text"] for w in words)
    transcript_rom = uroman.romanize_string(transcript_text).strip().lower()
    ref_cleaned = clean_for_alignment(ref_text, config)
    ref_rom = uroman.romanize_string(ref_cleaned).strip().lower()
    ratio = difflib.SequenceMatcher(None, ref_rom, transcript_rom).ratio() if ref_rom and transcript_rom else 0.0
    probs = [w.get("score", 0.0) for w in words if w.get("score") is not None]
    avg_score = sum(probs) / len(probs) if probs else 0.0
    return {"word_count": len(words), "avg_score": round(avg_score, 3), "match_ratio": round(ratio, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", type=str, required=True,
                         help="Comma-separated iso:hint pairs, e.g. slk:sk,ukr:uk")
    parser.add_argument("--chapters-per-iso", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    pairs = []
    for item in args.iso.split(","):
        iso, hint = item.split(":")
        pairs.append((iso.strip(), hint.strip()))

    from uroman import Uroman
    uroman = Uroman()
    model = load_whisper_model(DEFAULT_MODEL)

    results = []
    for iso, hint in pairs:
        try:
            config = load_language_config(iso)
        except Exception:
            config = load_language_config("default")

        chapters = pick_sample_chapters(iso, args.chapters_per_iso)
        if not chapters:
            print(f"{iso}: no chapters with audio+text found, skipping")
            continue

        chapter_results = []
        for size, tf, audio, text in chapters:
            did, book, chapter_str = _parse_timing_path(tf)
            try:
                ref_verses = read_verse_texts(text, config)
                ref_text = " ".join(v for v in ref_verses if v)
            except Exception as e:
                print(f"  {iso} {book} {chapter_str}: ref text load failed ({e}), skipping")
                continue

            print(f"  {iso} {book} {chapter_str}: transcribing (baseline, no hint)...")
            baseline_raw = transcribe_audio(audio, DEFAULT_MODEL, language=None, _model=model)
            baseline_words = build_word_timeline(baseline_raw["segments"])
            baseline_score = score_transcript(baseline_words, ref_text, config, uroman)

            print(f"  {iso} {book} {chapter_str}: transcribing (hint={hint})...")
            hinted_raw = transcribe_audio(audio, DEFAULT_MODEL, language=hint, _model=model)
            hinted_words = build_word_timeline(hinted_raw["segments"])
            hinted_score = score_transcript(hinted_words, ref_text, config, uroman)

            delta = hinted_score["match_ratio"] - baseline_score["match_ratio"]
            chapter_results.append({
                "book": book, "chapter": chapter_str, "distinct_id": did,
                "baseline": baseline_score, "hinted": hinted_score, "delta": round(delta, 4),
            })
            print(f"    baseline match_ratio={baseline_score['match_ratio']} "
                  f"hinted match_ratio={hinted_score['match_ratio']} delta={delta:+.4f}")

        if not chapter_results:
            continue

        deltas = [c["delta"] for c in chapter_results]
        avg_delta = sum(deltas) / len(deltas)
        if avg_delta >= IMPROVED_MARGIN:
            verdict = "CONFIRMED_IMPROVED"
        elif avg_delta <= -REGRESSED_MARGIN:
            verdict = "REGRESSED"
        elif abs(avg_delta) < REGRESSED_MARGIN:
            verdict = "NO_OP"
        else:
            verdict = "INCONCLUSIVE"

        print(f"{iso} -> {hint}: {verdict} (avg delta={avg_delta:+.4f}, n={len(chapter_results)})\n")
        results.append({"iso": iso, "hint": hint, "verdict": verdict, "avg_delta": round(avg_delta, 4),
                         "chapters": chapter_results})

    print("=== SUMMARY ===")
    for r in results:
        print(f"  {r['iso']:5s} -> {r['hint']:4s}  {r['verdict']:20s}  avg_delta={r['avg_delta']:+.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
