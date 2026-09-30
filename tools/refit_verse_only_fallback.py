#!/usr/bin/env python3
"""Refit stale verse_only_mode fallback/interpolated verse positions using
vowel-count-proportional interpolation between the nearest REAL ("local")
neighboring verses -- entirely from data already on disk (the current
_timing.json pos[], _words_quality.json's per-verse source tags, and the
original reference text). No audio file and no GPU needed.

This retroactively applies the same idea pipeline/align_verse_words.py's
_interpolate_fallback_runs() already applies to NEW alignment runs (added
2026-09-24, switched from word-count to vowel-count pacing 2026-09-26) to
the large backlog of already-aligned chapters whose output predates that
fix (confirmed 2026-09-26: 108,608 of 125,752 verse_only_mode chapter
outputs across 720 of 747 such languages; 18,468 of those actually contain
a fallback/interpolated verse, 18,338 with a real anchor on at least one
side) -- without spending any GPU time or re-downloading any audio.

Known approximation vs. a live GPU re-run: the live version anchors a run's
start-side boundary on the preceding real verse's own recorded END time;
_timing.json never persists per-verse end times, so this retroactive
version anchors on that verse's START time instead. That folds one real
verse's worth of speech duration into the run's proportional split -- a
small source of imprecision, strictly still better than the frozen/
uniform-pace value it replaces.

Only fixes runs bounded by a real ("local") verse on BOTH sides. A run
touching either edge of the chapter (verse 1, or the chapter's last verse,
is itself part of the fallback run) is left untouched -- there's no real
start-side anchor, and getting the far boundary right needs the audio
file's total duration, which is out of scope for a no-download pass. These
join the ~130 chapters with no real anchor at all as needing a genuine
re-run.

Usage:
    python tools/refit_verse_only_fallback.py --dry-run
    python tools/refit_verse_only_fallback.py --apply
    python tools/refit_verse_only_fallback.py --iso mtr --apply
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from quality_report import DOWNLOADS_DIR  # noqa: E402
from text_processing import read_verse_texts, load_language_config  # noqa: E402
from vowel_pacing import verse_vowel_counts  # noqa: E402

FIX_DATE = datetime(2026, 9, 24, tzinfo=timezone.utc)
TIMING = Path("export/timing-data")


def _vo_isos() -> set:
    isos = set()
    for cfg in Path("pipeline/config/languages").glob("*.toml"):
        if "verse_only_mode = true" in cfg.read_text():
            isos.add(cfg.stem)
    return isos


def _reference_text_path(canon: str, iso: str, distinct_id: str, book: str, chapter_str: str):
    book_dir = DOWNLOADS_DIR / canon / iso / distinct_id / book
    matches = list(book_dir.glob(f"{book}_{chapter_str}_*.txt"))
    return matches[0] if matches else None


def _quick_has_fallback(quality_path: Path) -> dict | None:
    """Cheap pre-check, no uroman/vowel-counting: does this chapter's
    quality data contain any fallback/interpolated verse at all? Returns
    the parsed quality_data dict if so (reused by refit_chapter(), avoiding
    a second read), None otherwise. This matters: most stale-but-pre-fix
    chapters have ZERO fallback verses (the fix only ever touches chapters
    that needed a fallback at all), so skipping the expensive uroman work
    for the ones that don't cuts real corpus-wide runtime by ~5-6x
    (confirmed 2026-09-26: an earlier version called verse_vowel_counts()
    -- full-chapter romanization -- on every one of 108,608 stale chapters
    instead of just the 18,468 that actually have a fallback verse, and
    was still running after 80+ minutes of CPU time)."""
    try:
        quality_data = json.loads(quality_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    verses_q = quality_data.get("verses", {})
    for words in verses_q.values():
        for w in words:
            if w.get("source") in ("fallback", "interpolated"):
                return quality_data
    return None


def refit_chapter(timing_path: Path, quality_data: dict, ref_verses: list[str],
                   config, uroman) -> dict | None:
    """Returns a report dict (and mutates nothing unless apply=True is
    handled by the caller writing timing_data back out) or None if there's
    nothing to do for this chapter. quality_data is the already-parsed
    dict from _quick_has_fallback() -- callers only reach here once that
    cheap check has confirmed a fallback verse exists."""
    try:
        timing_data = json.loads(timing_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    pos = timing_data.get("pos")
    if not isinstance(pos, list):
        return None
    verses_q = quality_data.get("verses", {})

    vowel_counts = verse_vowel_counts(ref_verses, config, uroman)

    # Build the ordered list of verse numbers that actually have quality
    # data (i.e. non-empty verses) with their source tag.
    verse_source = {}
    for vstr, words in verses_q.items():
        if not words:
            continue
        sources = {w.get("source") for w in words}
        # A verse's words all share one source by construction (see
        # align_verse_words.py's process_chapter_verse_only) -- if that
        # somehow isn't true, treat it conservatively as not a clean
        # fallback run member.
        verse_source[int(vstr)] = sources.pop() if len(sources) == 1 else "mixed"

    ordered = sorted(verse_source)
    if not ordered:
        return None

    changes = []
    i = 0
    n = len(ordered)
    while i < n:
        v = ordered[i]
        if verse_source[v] not in ("fallback", "interpolated"):
            i += 1
            continue
        run_start_idx = i
        while i < n and verse_source[ordered[i]] in ("fallback", "interpolated"):
            i += 1
        run_end_idx = i  # exclusive, index into `ordered`
        run_verses = ordered[run_start_idx:run_end_idx]

        prev_verse = ordered[run_start_idx - 1] if run_start_idx > 0 else None
        next_verse = ordered[run_end_idx] if run_end_idx < n else None
        # Must be immediately adjacent (verse number, not just list
        # position) to the run, and a real ("local") verse, to anchor on.
        if (prev_verse is None or verse_source[prev_verse] != "local"
                or prev_verse != run_verses[0] - 1):
            continue
        if (next_verse is None or verse_source[next_verse] != "local"
                or next_verse != run_verses[-1] + 1):
            continue

        prev_t = pos[prev_verse - 1]
        next_t = pos[next_verse - 1]
        span = next_t - prev_t
        if span <= 0:
            continue

        run_weights = [vowel_counts.get(str(v), 0) or 1 for v in run_verses]
        total_weight = sum(run_weights)
        cum = 0
        for v, w in zip(run_verses, run_weights):
            # frac uses weight accumulated BEFORE this verse (predicts
            # where v STARTS), not including it (which would predict where
            # v ENDS and place it flush against the NEXT anchor instead --
            # confirmed 2026-09-26: ayo/AYONTM GEN 4:20, a single-verse run,
            # landed at literally the same timestamp as verse 21, zero
            # duration, before this fix).
            frac = cum / total_weight
            new_t = round(prev_t + frac * span, 2)
            old_t = pos[v - 1]
            if abs(new_t - old_t) > 0.01:
                changes.append({"verse": v, "old_t": old_t, "new_t": new_t})
            pos[v - 1] = new_t
            cum += w

    if not changes:
        return None
    return {"timing_data": timing_data, "changes": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", type=str, default=None)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument("--report", type=str, default="_runs/refit_verse_only_fallback_report.json")
    args = parser.parse_args()

    vo_isos = {args.iso} if args.iso else _vo_isos()

    from uroman import Uroman
    uroman = Uroman()
    config_cache = {}

    total_chapters_changed = 0
    total_verses_changed = 0
    total_chapters_scanned = 0
    all_reports = []

    for canon in ("nt", "ot"):
        canon_dir = TIMING / canon
        if not canon_dir.exists():
            continue
        for iso_dir in sorted(canon_dir.iterdir()):
            if iso_dir.name not in vo_isos:
                continue
            iso = iso_dir.name
            if iso not in config_cache:
                try:
                    config_cache[iso] = load_language_config(iso)
                except Exception:
                    config_cache[iso] = load_language_config("default")
            config = config_cache[iso]

            for wf in iso_dir.rglob("*_words_quality.json"):
                mtime = datetime.fromtimestamp(wf.stat().st_mtime, tz=timezone.utc)
                if mtime >= FIX_DATE:
                    continue
                timing_path = wf.with_name(wf.name.replace("_words_quality.json", "_timing.json"))
                if not timing_path.exists():
                    continue

                total_chapters_scanned += 1
                quality_data = _quick_has_fallback(wf)
                if quality_data is None:
                    continue  # no fallback/interpolated verse here -- skip the expensive part entirely

                stem = wf.name.replace("_words_quality.json", "")
                parts = stem.split("_", 2)
                if len(parts) < 2:
                    continue
                book = wf.parent.name
                distinct_id = wf.parent.parent.name
                chapter_str = parts[1]
                text_path = _reference_text_path(canon, iso, distinct_id, book, chapter_str)
                if text_path is None:
                    continue
                try:
                    ref_verses = read_verse_texts(text_path, config)
                except Exception:
                    continue

                result = refit_chapter(timing_path, quality_data, ref_verses, config, uroman)
                if result is None:
                    continue

                total_chapters_changed += 1
                total_verses_changed += len(result["changes"])
                all_reports.append({
                    "iso": iso, "canon": canon, "distinct_id": distinct_id,
                    "book": book, "chapter": chapter_str,
                    "changes": result["changes"],
                })

                if args.apply:
                    timing_path.write_text(json.dumps(result["timing_data"], separators=(",", ":")))

    verb = "Would fix" if not args.apply else "Fixed"
    print(f"Scanned {total_chapters_scanned} pre-fix chapters with a real anchor on both sides checked")
    print(f"{verb} {total_verses_changed} verse(s) across {total_chapters_changed} chapter(s)")
    if not args.apply:
        print("\nDRY RUN -- no files written. Re-run with --apply to write.")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(all_reports, indent=2))
    print(f"Full report written to {args.report}")


if __name__ == "__main__":
    main()
