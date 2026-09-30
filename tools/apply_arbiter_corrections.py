#!/usr/bin/env python3
"""Apply three_way_arbiter.py's verdicts as real corrections to our own
timing output, instead of just reporting them.

Turns the arbiter from a diagnostic into a fix: wherever it finds confident
evidence we were wrong (verdict "DBT"), patch our stored verse-level
timestamp to the better-supported value — the content-verified TEXT_MATCH
position when available (an independent, audio-grounded timestamp, not
just "trust DBT"), or DBT's own timestamp when only the structural
fallback found evidence. Verses verdicted "OURS" or "AMBIGUOUS" are left
untouched — this only ever moves a verse toward stronger evidence, never
away from it, and never touches a verse the arbiter can't confidently
resolve.

CPU-only (uroman romanization + word-list search, no GPU realignment) — see
three_way_arbiter.py's own docstring for the underlying method. Only
touches verse-level start times (pos[] in *_timing.json) and, to keep the
file internally consistent, shifts that verse's own word-level timestamps
in *_words.json by the same delta (real per-word re-alignment isn't run —
a uniform shift is the safe minimum that keeps word offsets from
disagreeing with the corrected verse boundary; it's better than leaving
them anchored to a timestamp we just determined was wrong, though it's
still an approximation, not a re-alignment).

Safety:
  - Dry-run by default (--apply to actually write).
  - A correction is skipped (not just logged) if applying it would make
    this chapter's pos[] non-monotonic against its NEW neighbors (checked
    sequentially, verse by verse, against already-applied corrections in
    the same chapter) — never trade one timing bug for a worse one.
  - Every correction (applied or skipped-for-safety) is logged to the
    report file with before/after values for audit.

Usage:
    python tools/apply_arbiter_corrections.py --iso-list aaa,xtn --testament nt
    python tools/apply_arbiter_corrections.py --testament nt --apply --report corrections.json
"""
import argparse
import json
import sys
from pathlib import Path

from quality_report import (
    TIMING_DIR, find_all_downloaded_timecode, find_pipeline_timing_files,
    load_timing_verses, _parse_timing_path, _get_canons,
)
from three_way_arbiter import (
    _whisper_path_for, _reference_text_path_for,
    prepare_chapter_matching_context, resolve_verse, DISPUTE_THRESHOLD,
    _drift_gap_windows_for, _in_drift_gap_window,
    resolve_chapter_by_pacing, WEAK_MATCH_RATIO,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from text_processing import load_language_config  # noqa: E402

def _words_path_for(timing_path: Path) -> Path:
    return timing_path.with_name(timing_path.name.replace("_timing.json", "_words.json"))


def _build_batches(vnums: list, targets: dict) -> list:
    """Group a sorted list of DBT-verdict verse numbers into batches that
    are safe to apply atomically: consecutive verse numbers AND strictly
    increasing proposed target times. A break in either starts a new
    batch — a break in verse number is an unrelated dispute elsewhere in
    the chapter; a break in target ordering means this particular pair's
    own proposed positions disagree with each other (most often one bad
    text-match in an otherwise-solid run, e.g. a repeated/formulaic phrase
    confusing the search — see three_way_arbiter.py's docstring), a real
    conflict that batching the rest of the run together should not paper
    over by guessing which side is at fault.
    """
    batches = []
    cur = [vnums[0]]
    for vnum in vnums[1:]:
        if vnum == cur[-1] + 1 and targets[vnum][0] > targets[cur[-1]][0]:
            cur.append(vnum)
        else:
            batches.append(cur)
            cur = [vnum]
    batches.append(cur)
    return batches


def correct_chapter(dl_path: Path, pl_path: Path, whisper_path: Path,
                     text_path: Path | None, config, uroman, apply: bool) -> list[dict]:
    """Resolve every disputed verse in one chapter, apply any DBT-verdict
    correction to its on-disk *_timing.json / *_words.json (if apply=True),
    and return a log of every correction attempted (applied or skipped).

    Corrections are batched by consecutive run of DBT-verdict verses, not
    applied one verse at a time: the dominant real pattern is a whole
    stretch of correlated verses that need to move together (e.g. a
    multi-verse quote or a run of short verses DBT itself mis-timed), and
    checking each verse sequentially against a not-yet-processed neighbor's
    stale position produces false "would break monotonicity" conflicts on
    exactly those cases. Each run's *whole* proposed sequence is checked
    for internal consistency and compatibility with the nearest hard
    boundary on either side (a confirmed-good verse, or the edge of the
    chapter) before any of it is applied — it never crosses a verse already
    confirmed to agree with DBT.

    (An earlier version of this tool also tried to rescue an isolated
    AMBIGUOUS blocking neighbor by re-searching it anchored on the run's
    edge. Removed 2026-09-24: across the full corpus it only ever fired 4
    times, and direct word-level verification found 3 of those 4 wrong —
    including one case where it moved an already-correct verse to a worse
    position. Negligible volume, poor precision, no clean ratio threshold
    separated the one good case from the bad ones — not worth the risk.)"""
    dl_verses = load_timing_verses(dl_path)
    pl_verses = load_timing_verses(pl_path)
    if dl_verses is None or pl_verses is None:
        return []

    whisper_rom_words, struct_candidates, ref_verses = prepare_chapter_matching_context(
        whisper_path, text_path, config, uroman)
    if struct_candidates is None and ref_verses is None:
        return []  # truly nothing to arbitrate with
    drift_gap_windows = _drift_gap_windows_for(pl_path)

    common = sorted((set(dl_verses) & set(pl_verses)) - {"0"}, key=lambda x: int(x))
    disputed = [v for v in common if abs(dl_verses[v] - pl_verses[v]) >= DISPUTE_THRESHOLD]

    # Same tiered fallback as three_way_arbiter.arbitrate_chapter(): a
    # chapter-wide vowel-pacing verdict, used only where TEXT_MATCH/
    # STRUCTURAL either have nothing (no Whisper at all -- verse_only_mode's
    # normal state) or a confident-but-weak result vowel-pacing actively
    # contradicts (a stale/unreliable Whisper file that still resolves
    # something, just not correctly -- see three_way_arbiter's own
    # docstring update for why gating on "no Whisper file" alone isn't
    # enough). Mirrors arbitrate_chapter() exactly so this tool's real
    # corrections and that tool's diagnostics never silently disagree.
    pacing = None
    if ref_verses is not None:
        # strict=True whenever the LANGUAGE is fusion-mode (not
        # verse_only_mode), not whenever struct_candidates happens to be
        # non-None for this one chapter -- see three_way_arbiter.py's
        # arbitrate_chapter() for the full rationale (confirmed 2026-09-30,
        # ind/INDASV MAT 5:33 + ACT 8:15: a missing per-chapter whisper file
        # on an otherwise fusion-mode language must not fall back to the
        # lenient bar). Mirrors arbitrate_chapter() exactly so this tool's
        # real corrections and that tool's diagnostics never silently
        # disagree.
        pacing = resolve_chapter_by_pacing(dl_verses, pl_verses, ref_verses, config, uroman, disputed,
                                            strict=not getattr(config, "verse_only_mode", False))

    resolved: dict = {}
    for v in disputed:
        dl_t, pl_t = dl_verses[v], pl_verses[v]
        vnum = int(v)
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
            entry = {**entry, "method": pacing["method"], "verdict": pacing["verdict"]}
        elif (entry["verdict"] != "AMBIGUOUS" and pacing is not None and pacing["verdict"] != "AMBIGUOUS"
              and pacing["verdict"] != entry["verdict"]
              and (entry["method"] == "STRUCTURAL"
                   or (entry["method"] == "TEXT_MATCH" and (entry.get("match_ratio") or 1.0) < WEAK_MATCH_RATIO))):
            entry = {**entry, "verdict": "AMBIGUOUS"}

        resolved[vnum] = entry

    dbt_verses = sorted(vnum for vnum, e in resolved.items() if e["verdict"] == "DBT")
    if not dbt_verses:
        return []

    # Load the raw timing.json directly (not via load_timing_verses, which
    # normalizes away the file shape) so we can patch pos[] in place.
    try:
        timing_data = json.loads(pl_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    pos = timing_data.get("pos")
    if not isinstance(pos, list):
        return []

    words_path = _words_path_for(pl_path)
    words_data = None
    if words_path.exists():
        try:
            words_data = json.loads(words_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            words_data = None

    def shift_words(vn: int, delta: float):
        if words_data is None:
            return
        key = str(vn)
        for field in ("beg", "end"):
            values = words_data.get(field, {}).get(key)
            if values:
                words_data[field][key] = [
                    round(v + delta, 2) if v is not None else None for v in values
                ]

    # Prefer the content-verified TEXT_MATCH position (an independent,
    # audio-grounded timestamp) over DBT's own number when we have it; fall
    # back to DBT's timestamp when only the structural heuristic found
    # evidence (no independent verified position available). Computed once
    # for every DBT-verdict verse up front so batch-splitting can use it.
    targets = {}
    for vnum in dbt_verses:
        d = resolved[vnum]
        if d["method"] == "TEXT_MATCH" and d.get("match_t") is not None:
            targets[vnum] = (d["match_t"], "text_match")
        else:
            targets[vnum] = (d["dbt_t"], "dbt_timestamp")

    log = []
    n = len(pos)
    # The "next real neighbor" boundary check below must stop at the last
    # verse DBT actually has data for, not at the end of our own pos[]
    # array -- confirmed 2026-09-26 (mtr/MTRNLC MAT 13): our pipeline's
    # pos[] had 72 entries for a 58-verse chapter (a stale/orphaned tail,
    # unrelated to this dispute), and comparing the run's last proposed
    # value against pos[58] -- an array slot with no real verse behind it
    # -- rejected every single correction in the chapter as "would be
    # nonmonotonic" against data that was never a real neighbor at all.
    max_common_verse = max((int(v) for v in common), default=0)

    # A verse whose old position, proposed new position, or DBT's own claim
    # falls inside this chapter's own recorded drift/gap-fix window can't be
    # trusted either way — DBT's reference timing isn't re-synced past the
    # fix's own narrow segment (see three_way_arbiter.DRIFT_GAP_MARGIN), so
    # "closer to DBT" stops meaning "more correct" right here. Confirmed
    # directly 2026-09-24: hla/HLAPNG ACT 4:17-19, three already-correct
    # verses (verified word-for-word against the real MMS transcript)
    # "corrected" toward a DBT number that was itself ~155s off. Filtered
    # out before batching, not mid-batch, so one flagged verse can't also
    # drag down neighbors that would otherwise batch cleanly with it.
    # Verse 1 of a chapter is a second, distinct untrustworthy zone: DBT's
    # own verse-1 timestamp appears to systematically measure a point near
    # the chapter's spoken intro/header rather than the true first word,
    # and both the search center and (sometimes) the TEXT_MATCH result
    # inherit that bias. Confirmed 2026-09-25 on a live apply run: 4/4
    # verse-1 corrections checked against real MMS words were wrong — 3 of
    # them moved an already-correct original alignment to a wrong one —
    # while a random sample of 3 non-verse-1 corrections from the same run
    # were 3/3 correct. No root cause identified yet (only one of the 4 had
    # a recorded intro_end boundary), but the empirical pattern is strong
    # enough to act on without one.
    safe_dbt_verses = []
    for vnum in dbt_verses:
        d = resolved[vnum]
        old_t = pos[vnum - 1]
        new_t = targets[vnum][0]
        if vnum == 1:
            log.append({
                "verse": vnum, "old_t": old_t, "new_t": new_t, "dbt_t": d["dbt_t"],
                "evidence": targets[vnum][1], "method": d["method"], "match_ratio": d.get("match_ratio"),
                "status": "SKIPPED_verse_one",
            })
            continue
        if drift_gap_windows and any(_in_drift_gap_window(t, drift_gap_windows)
                                      for t in (old_t, new_t, d["dbt_t"])):
            log.append({
                "verse": vnum, "old_t": old_t, "new_t": new_t, "dbt_t": d["dbt_t"],
                "evidence": targets[vnum][1], "method": d["method"], "match_ratio": d.get("match_ratio"),
                "status": "SKIPPED_drift_gap_window",
            })
            continue
        safe_dbt_verses.append(vnum)

    if not safe_dbt_verses:
        return log

    runs = _build_batches(safe_dbt_verses, targets)

    def _skip_verses(vnums, reason):
        for vnum in vnums:
            d = resolved[vnum]
            new_t, evidence = targets[vnum]
            log.append({
                "verse": vnum, "old_t": pos[vnum - 1], "new_t": new_t, "dbt_t": d["dbt_t"],
                "evidence": evidence, "method": d["method"], "match_ratio": d.get("match_ratio"),
                "status": reason,
            })

    def _apply_batch(apply_list):
        # In-memory pos[] is updated even in dry-run so a later batch in
        # this same chapter sees the would-be-corrected state, not a stale
        # uncorrected one; only the on-disk WRITE is gated on --apply.
        for vnum in apply_list:
            idx = vnum - 1
            old_t = pos[idx]
            d = resolved[vnum]
            new_t, evidence = targets[vnum]
            dbt_t, ratio, method = d["dbt_t"], d.get("match_ratio"), d["method"]
            status = "APPLIED" if apply else "WOULD_APPLY"
            delta = new_t - old_t
            pos[idx] = new_t
            if apply:
                shift_words(vnum, delta)
            log.append({
                "verse": vnum, "old_t": old_t, "new_t": new_t, "dbt_t": dbt_t,
                "delta": round(delta, 2), "evidence": evidence, "method": method,
                "match_ratio": ratio, "status": status,
            })

    def _process_run(run):
        """Try to place `run` (a batch of consecutive, internally
        monotonic DBT-verdict targets) atomically. If a boundary conflict
        with a hard neighbor can't be avoided and the run has more than one
        verse, the conflicting EDGE verse is peeled off and retried on its
        own — the rest of the run, which was never in conflict with each
        other, still gets applied as a batch. This is what keeps one bad
        outlier (e.g. a mismatched text-search target) from dragging down
        an otherwise-solid run, while still getting the batching benefit
        for the genuinely correlated majority.
        """
        if not run:
            return
        if run[0] - 1 < 0 or run[-1] - 1 >= n:
            return
        seq = [targets[vnum][0] for vnum in run]
        first_idx, last_idx = run[0] - 1, run[-1] - 1

        if first_idx > 0 and seq[0] <= pos[first_idx - 1]:
            if len(run) > 1:
                _process_run(run[1:])
                _process_run([run[0]])
            else:
                _skip_verses(run, "SKIPPED_would_be_nonmonotonic")
            return

        if last_idx + 1 < n and (last_idx + 2) <= max_common_verse and seq[-1] >= pos[last_idx + 1]:
            if len(run) > 1:
                _process_run(run[:-1])
                _process_run([run[-1]])
            else:
                _skip_verses(run, "SKIPPED_would_be_nonmonotonic")
            return

        _apply_batch(list(run))

    for run in runs:
        _process_run(run)

    if apply and any(entry["status"] == "APPLIED" for entry in log):
        pl_path.write_text(json.dumps(timing_data, separators=(",", ":")), encoding="utf-8")
        if words_data is not None:
            words_path.write_text(json.dumps(words_data, separators=(",", ":")), encoding="utf-8")

    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", type=str)
    parser.add_argument("--iso-list", type=str)
    parser.add_argument("--testament", type=str, choices=["nt", "ot", "both"], default=None)
    parser.add_argument("--book", type=str, default=None)
    parser.add_argument("--chapter", type=str, default=None)
    parser.add_argument("--apply", action="store_true", help="Actually write corrections (default: dry-run)")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--report", type=str, default=None, help="Write full per-chapter correction log as JSON")
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

    mode = "APPLYING (writing real changes)" if args.apply else "DRY RUN (no files will be modified)"
    print(f"=== {mode} ===\n")

    chapters_checked = 0
    chapters_with_corrections = 0
    verses_applied = 0
    verses_skipped_nonmono = 0
    verses_skipped_drift_gap = 0
    verses_skipped_verse_one = 0
    verses_would_apply = 0
    all_results = []

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
                    print(f"  ... checked {chapters_checked} chapters, currently at "
                          f"{iso}/{distinct_id} {book} {chapter_str}", file=sys.stderr)

                log = correct_chapter(downloaded_tc[key], tf, whisper_path, text_path, config, uroman, args.apply)
                if not log:
                    continue
                chapters_with_corrections += 1
                for entry in log:
                    if entry["status"] == "APPLIED":
                        verses_applied += 1
                    elif entry["status"] == "WOULD_APPLY":
                        verses_would_apply += 1
                    elif entry["status"] == "SKIPPED_would_be_nonmonotonic":
                        verses_skipped_nonmono += 1
                    elif entry["status"] == "SKIPPED_drift_gap_window":
                        verses_skipped_drift_gap += 1
                    elif entry["status"] == "SKIPPED_verse_one":
                        verses_skipped_verse_one += 1
                all_results.append({
                    "iso": iso, "canon": c, "distinct_id": distinct_id,
                    "book": book, "chapter": chapter_str, "corrections": log,
                })

    print(f"\nChapters checked: {chapters_checked}")
    print(f"Chapters with at least one correction: {chapters_with_corrections}")
    if args.apply:
        print(f"Verses corrected: {verses_applied}")
    else:
        print(f"Verses that WOULD be corrected (dry-run): {verses_would_apply}")
    print(f"Verses skipped for safety (would break monotonicity): {verses_skipped_nonmono}")
    print(f"Verses skipped (inside a chapter's own drift/gap-fix window): {verses_skipped_drift_gap}")
    print(f"Verses skipped (verse 1 — DBT's own timestamp untrustworthy there): {verses_skipped_verse_one}")

    if args.report:
        Path(args.report).write_text(json.dumps(all_results, indent=2))
        print(f"\nFull correction log written to {args.report}")


if __name__ == "__main__":
    main()
