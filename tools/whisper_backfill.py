#!/usr/bin/env python3
"""Backfill missing Whisper transcripts for fusion-mode languages.

Confirmed 2026-09-30: 32,304 of 56,860 NT chapters (56.8%) across
fusion-mode isos (verse_only_mode=false) have MMS timing.json output but
NO whisper_words.json at all. Traced the cause -- NOT the quality-based
auto-demotion in align_pipeline.py (that always persists a dated
config/languages/<iso>.toml when it fires; none exists for this
population, proving it never ran) but a scheduling gap: the September
`priority-fill-*` batches (_batches/priority-fill-both.json,
priority-fill-phase2.json) were scoped for full fusion ("both" in the
filename) and cover languages like `mak`, `min`, `ces` (tier 1!), but only
`make align-mms` was ever actually run for them -- `make align-whisper`
never was. This script finishes that dropped step.

Loads the Whisper model ONCE (same rationale as tools/run_gpu_redo.py --
per-chapter CLI invocation wastes model-load time at this scale) and
processes chapters grouped by iso, so the per-language quality tracker
(pipeline/whisper_quality_guard.py) has a coherent population to judge and
purge_iso_audio() can run once per language instead of globally.

Quality gating (see pipeline/whisper_quality_guard.py's module docstring
for the full calibration writeup against real samples): a flagged
chapter's transcript is NOT written -- writing known-garbage output would
let it be picked up later as if it were real TEXT_MATCH evidence for the
arbiter, which is worse than the chapter staying in its current
"no Whisper data, falls back to vowel-pacing" state. A language whose
early chapters are mostly flagged gets its REMAINING chapters skipped for
this run (not retried until a human looks at it) rather than burning GPU
time transcribing 250+ more chapters of a language Whisper evidently
can't handle -- mirrors align_pipeline.py's AUTO_VERSE_ONLY_* rolling-
tracker pattern, but deliberately does NOT touch any config/languages/
*.toml file itself; auto-flipping verse_only_mode is a bigger decision
than this backfill's job (add Whisper evidence where it's missing) and
stays a human call.

Non-DBT editions (config/helloao.toml, e.g. eng/ENGBSBHAY = helloAO BSB
read by Hays): audio and text come from helloAO instead of DBT, into
downloads/helloao/aligned/; their audio is deleted right after transcription
(purge_iso_audio() only sweeps downloads/BB), the text is kept for fusion.
Added 2026-10-03 after all 260 ENGBSBHAY chapters failed with "no audio
available".

Disk safety: mirrors tools/run_gpu_redo.py -- audio for this population
was already purged once (that's WHY it needs re-download at all; MMS
alignment completed for all of it already), so purge each language's
mp3s again via purge_aligned_audio.purge_iso_audio() once that language's
chapters are done (its own precondition -- both _timing.json and
_words.json already exist -- was already true before this script even
starts, since MMS alignment already completed for this population).

Resumable for free: needs_run() by way of "does whisper_words.json
already exist" skips completed chapters; a rejected (quality-flagged)
chapter gets a marker file (`<...>_whisper_rejected.json` next to where
the words file would go) so it isn't re-attempted every run without a
human clearing it.

Usage:
    python tools/whisper_backfill.py --dry-run            # scope only, no GPU work
    python tools/whisper_backfill.py --limit 20            # smoke test
    python tools/whisper_backfill.py                       # full run
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent))

from timing_files import TIMING_DIR, DOWNLOADS_DIR, find_all_downloaded_timecode  # noqa: E402
from three_way_arbiter import _whisper_path_for  # noqa: E402
from whisper_transcribe import (  # noqa: E402
    DEFAULT_MODEL_FASTER, load_whisper_model, transcribe_audio, build_word_timeline,
    write_whisper_words_json, get_whisper_language, set_whisper_cpu,
)
from whisper_quality_guard import (  # noqa: E402
    assess_chapter, LanguageTracker, update_language_tracker, should_abort_language,
)
from download_language_content import ensure_chapter_ready  # noqa: E402
from text_processing import load_language_config  # noqa: E402
from purge_aligned_audio import purge_iso_audio  # noqa: E402
from remote_audio import ensure_chapter_audio, manual_import_audio_info  # noqa: E402

# Editions whose audio is not a DBT fileset (config/helloao.toml, e.g.
# eng/ENGBSBHAY = helloAO "BSB" read by Hays) live here, same layout as
# whisper_transcribe._find_base_dir() expects. DBT's fetch can never find
# them, so they take the helloAO route below instead.
NON_DBT_BASE = Path("downloads/helloao/aligned")

WORD_TIMING_DIR = Path("word-timing-data")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def rejected_marker_path(whisper_words_path: Path) -> Path:
    return Path(str(whisper_words_path).replace("_whisper_words.json", "_whisper_rejected.json"))


def find_missing_whisper_chapters(testament: str = "nt", isos: list[str] | None = None,
                                   dispute_free_only: bool = False,
                                   skip_dramatized: bool = False) -> dict[str, list[Path]]:
    """Returns {iso: [pipeline_timing_path, ...]} for fusion-mode isos with
    a timing.json but no whisper_words.json AND no prior rejection marker.

    `isos`, when given, restricts the scan to just those language codes
    (still applies the same verse_only_mode/missing-file filtering) --
    used by --iso/--iso-list to scope a single-language pilot run instead
    of scanning the whole corpus.

    `dispute_free_only`, when True, additionally drops any chapter where
    DBT already has its own downloaded timecode -- confirmed 2026-10-01
    (acd): roughly 59% of this backlog is chapters where our new Whisper
    evidence would have to compete with DBT's own timing (and often loses,
    per the low-whisper-quality-language gate in
    pipeline/whisper_quality_guard.py), vs. the other ~41% where DBT has
    no timing at all and Whisper is simply, unambiguously additive -- no
    dispute possible, so no risk of the extra effort being wasted or
    actively harmful. Re-prioritization call: do the dispute-free
    population first, defer the dispute-relevant one.
    """
    result = defaultdict(list)
    base = TIMING_DIR / testament
    if not base.exists():
        return result
    iso_dirs = sorted(base / iso for iso in isos) if isos else sorted(base.iterdir())
    for iso_dir in iso_dirs:
        if not iso_dir.is_dir():
            continue
        iso = iso_dir.name
        try:
            config = load_language_config(iso)
        except Exception:
            continue
        if getattr(config, "verse_only_mode", False):
            continue

        dbt_keys = find_all_downloaded_timecode(iso, testament) if dispute_free_only else None

        for tf in iso_dir.rglob("*_timing.json"):
            wp = _whisper_path_for(tf)
            if wp is None:
                continue
            if wp.exists():
                continue
            if rejected_marker_path(wp).exists():
                continue
            if skip_dramatized and ("2DA" in tf.name or "2SA" in tf.name):
                # Dramatized filesets are aligned without Whisper (padded
                # verse-only path, tools/run_gpu_redo.py); Whisper on music +
                # multiple voices is mostly rejected by the quality guard.
                continue
            if dbt_keys is not None:
                rel = tf.relative_to(TIMING_DIR)
                canon, distinct_id, book = rel.parts[0], rel.parts[2], rel.parts[3]
                chapter_str = tf.name.replace("_timing.json", "").split("_", 2)[1]
                if (canon, distinct_id, book, chapter_str) in dbt_keys:
                    continue  # DBT has its own timing here -- dispute-relevant, deferred
            result[iso].append(tf)
    return result


def audio_and_text_paths(tf: Path, testament: str) -> tuple[Path | None, Path | None, str, str, str, str]:
    """(audio_path_or_None, text_path_or_None, canon, distinct_id, book, chapter_str)."""
    rel = tf.relative_to(TIMING_DIR)
    canon, iso, distinct_id, book = rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
    stem = tf.name.replace("_timing.json", "")
    parts = stem.split("_", 2)
    chapter_str = parts[1]
    book_dir = DOWNLOADS_DIR / canon / iso / distinct_id / book
    if manual_import_audio_info(distinct_id):
        book_dir = NON_DBT_BASE / canon / iso / distinct_id / book
    audio_matches = list(book_dir.glob(f"{book}_{chapter_str}_*.mp3")) if book_dir.exists() else []
    text_matches = list(book_dir.glob(f"{book}_{chapter_str}_*.txt")) if book_dir.exists() else []
    return (
        audio_matches[0] if audio_matches else None,
        text_matches[0] if text_matches else None,
        canon, distinct_id, book, chapter_str,
    )


def _fetch_non_dbt_chapter(info: dict, canon: str, iso: str, distinct_id: str,
                           book: str, chapter_str: str) -> bool:
    """Audio + text for an edition served by helloAO instead of DBT.
    Text is written as {BOOK}_{CCC}_{TRANSLATION}_ET.txt next to the audio,
    where fusion_only_redo.py / discover_chapter_files() look for it."""
    from download_language_content import _fetch_helloao_chapter
    book_dir = NON_DBT_BASE / canon / iso / distinct_id / book
    book_dir.mkdir(parents=True, exist_ok=True)
    ch = int(chapter_str)
    audio_ok = ensure_chapter_audio(book_dir / f"{book}_{chapter_str}_{distinct_id}.mp3", book, ch)
    text_ok = _fetch_helloao_chapter(info["translation"], book, ch,
                                     book_dir / f"{book}_{chapter_str}_{info['translation']}_ET.txt")
    if not (audio_ok and text_ok):
        log(f"  {iso}/{distinct_id} {book} {chapter_str}: helloAO fetch failed "
            f"(audio={audio_ok}, text={text_ok})")
    return audio_ok and text_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testament", type=str, default="nt", choices=["nt", "ot"])
    parser.add_argument("--iso", type=str, default=None, help="Scope to a single iso (pilot run)")
    parser.add_argument("--iso-list", type=str, default=None, help="Comma-separated isos")
    parser.add_argument("--dry-run", action="store_true", help="Report scope only, no GPU work")
    parser.add_argument("--dispute-free-only", action="store_true",
                         help="Skip any chapter where DBT already has its own timing "
                              "(defers the dispute-relevant population to a later pass)")
    parser.add_argument("--skip-dramatized", action="store_true",
                         help="Leave out dramatized (2DA/2SA) filesets -- they are re-aligned by "
                              "the verse-only redo instead (no Whisper).")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N chapters (smoke test)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_FASTER)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--report", type=str, default="_runs/whisper_backfill_report.json")
    args = parser.parse_args()

    isos = None
    if args.iso:
        isos = [args.iso]
    elif args.iso_list:
        isos = [c.strip() for c in args.iso_list.split(",")]
    missing = find_missing_whisper_chapters(args.testament, isos=isos, dispute_free_only=args.dispute_free_only,
                                            skip_dramatized=args.skip_dramatized)
    total_chapters = sum(len(v) for v in missing.values())
    log(f"Scope: {total_chapters} chapters missing whisper_words.json across {len(missing)} fusion-mode isos")

    if args.dry_run:
        by_size = sorted(missing.items(), key=lambda kv: -len(kv[1]))
        for iso, files in by_size[:30]:
            log(f"  {iso}: {len(files)} chapters")
        if len(by_size) > 30:
            log(f"  ... and {len(by_size) - 30} more isos")
        return

    if args.device == "cpu":
        set_whisper_cpu(True)
    model = load_whisper_model(args.model)

    report = {"isos": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    total_ok = 0
    total_flagged = 0
    total_failed = 0
    processed = 0

    iso_items = sorted(missing.items())
    for iso, files in iso_items:
        try:
            config = load_language_config(iso)
        except Exception:
            config = load_language_config("default")
        whisper_lang = get_whisper_language(iso)
        tracker = LanguageTracker()
        iso_ok = iso_flagged = iso_failed = 0

        for tf in files:
            if args.limit and processed >= args.limit:
                break
            if should_abort_language(tracker):
                break

            audio_path, text_path, canon, distinct_id, book, chapter_str = audio_and_text_paths(tf, args.testament)
            non_dbt = manual_import_audio_info(distinct_id)
            if audio_path is None and non_dbt:
                fetched = _fetch_non_dbt_chapter(non_dbt, canon, iso, distinct_id, book, chapter_str)
                audio_path, text_path, *_ = audio_and_text_paths(tf, args.testament)
                if not fetched:
                    audio_path = None
            elif audio_path is None:
                try:
                    ensure_chapter_ready(iso, canon, distinct_id, book, int(chapter_str))
                except Exception as e:
                    log(f"  {iso}/{distinct_id} {book} {chapter_str}: fetch failed ({e})")
                    total_failed += 1
                    iso_failed += 1
                    processed += 1
                    continue
                audio_path, text_path, *_ = audio_and_text_paths(tf, args.testament)
            if audio_path is None:
                log(f"  {iso}/{distinct_id} {book} {chapter_str}: no audio available after fetch, skipping")
                total_failed += 1
                iso_failed += 1
                processed += 1
                continue

            wp = _whisper_path_for(tf)
            try:
                result = transcribe_audio(audio_path, args.model, whisper_lang, _model=model)
            except Exception as e:
                log(f"  {iso}/{distinct_id} {book} {chapter_str}: transcribe EXCEPTION ({e})")
                total_failed += 1
                iso_failed += 1
                processed += 1
                continue

            segments = result.get("segments", [])
            duration = segments[-1]["end"] if segments else 0.0
            words = build_word_timeline(segments)
            quality = assess_chapter(words, duration)
            update_language_tracker(tracker, quality)

            if quality.flagged:
                marker = rejected_marker_path(wp)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({
                    "iso": iso, "distinct_id": distinct_id, "book": book, "chapter": chapter_str,
                    "reasons": quality.reasons, "words_per_second": quality.words_per_second,
                    "repetition_ratio_3gram": quality.repetition_ratio_3gram,
                    "word_count": quality.word_count, "duration": quality.duration,
                }, indent=2))
                log(f"  {iso}/{distinct_id} {book} {chapter_str}: FLAGGED ({', '.join(quality.reasons)}) -- not written")
                total_flagged += 1
                iso_flagged += 1
            else:
                write_whisper_words_json(words, book, chapter_str, wp)
                total_ok += 1
                iso_ok += 1
            if non_dbt:
                # purge_iso_audio() only sweeps downloads/BB; fusion needs the
                # text (kept), not the audio.
                audio_path.unlink(missing_ok=True)

            processed += 1

        if tracker.aborted:
            log(f"{iso}: ABORTED for this run -- {tracker.abort_reason}")

        report["isos"][iso] = {
            "ok": iso_ok, "flagged": iso_flagged, "failed": iso_failed,
            "total_in_scope": len(files), "aborted": tracker.aborted,
            "abort_reason": tracker.abort_reason,
        }

        try:
            deleted, freed = purge_iso_audio(iso)
            if deleted:
                log(f"  purged {deleted} mp3(s) for {iso}, {freed / 1e9:.2f} GB freed")
        except Exception as e:
            log(f"  purge failed for {iso} (non-fatal): {e}")

        log(f"... {iso} done: ok={iso_ok} flagged={iso_flagged} failed={iso_failed} "
            f"(running totals: ok={total_ok} flagged={total_flagged} failed={total_failed}, "
            f"{processed}/{total_chapters if not args.limit else min(args.limit, total_chapters)} processed)")

        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))

        if args.limit and processed >= args.limit:
            break

    log(f"DONE. ok={total_ok} flagged={total_flagged} failed={total_failed}")


if __name__ == "__main__":
    main()
