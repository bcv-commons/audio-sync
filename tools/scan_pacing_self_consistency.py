#!/usr/bin/env python3
"""Scan verse_only_mode chapters for a vowel-pacing self-consistency
anomaly, using ONLY our own pipeline's own verse positions -- no DBT
timing required at all.

Motivation: category 2's arbiter-corrections dry-run (2026-09-26) found
that 75% of the verse_only_mode chapters it corrected (11,984 of 15,952)
had shown ZERO visible error before -- no fallback tag, normal-looking
per-word confidence scores. That measurement only covers chapters with
DBT timing to compare against (Phase 2/3). Phase-1-only chapters (no DBT
timing at all) can't be checked that way -- but pipeline/vowel_pacing.py's
fingerprint() never actually needed a second source: it measures whether
ONE source's own verse positions track a plausible constant-speech-rate
model implied by the chapter's own reference text. Confirmed all session:
every known-bad case's OWN self-consistency RMS (mtr 31.94s, mxt 46.01s,
ndv 129.22s) was already dramatically higher than every known-good case's
(DBT's own 4.78-12.6s across examples; our own FIXED re-alignments'
7.11-12.87s) -- with zero DBT involved in that number. This scanner
applies that same self-only check across every verse_only_mode chapter,
Phase 1 included, to get real coverage of the population category 2
structurally cannot measure.

This is a DIAGNOSTIC/PRIORITIZATION tool only -- for a Phase-1-only
chapter there is no second source to correct toward, so a flagged chapter
here is a candidate for GPU re-alignment, not something this tool can fix
itself (unlike apply_arbiter_corrections.py, which has DBT to correct
toward for Phase 2/3 disputes).

Usage:
    python tools/scan_pacing_self_consistency.py --iso-list mtr,mxt
    python tools/scan_pacing_self_consistency.py  # all verse_only_mode isos
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from quality_report import load_timing_verses, TIMING_DIR, DOWNLOADS_DIR  # noqa: E402
from text_processing import read_verse_texts, load_language_config  # noqa: E402
from vowel_pacing import verse_vowel_counts, fingerprint  # noqa: E402

RMS_THRESHOLD = 20.0  # seconds -- see module docstring: known-good examples topped
                       # out at 12.87s, known-bad ones started at 31.94s. 20s sits
                       # cleanly in the gap between those two groups.


def _vo_isos() -> set:
    isos = set()
    for cfg in Path("pipeline/config/languages").glob("*.toml"):
        if "verse_only_mode = true" in cfg.read_text():
            isos.add(cfg.stem)
    return isos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", type=str, default=None)
    parser.add_argument("--iso-list", type=str, default=None)
    parser.add_argument("--report", type=str, default="_runs/pacing_self_consistency_report.json")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    if args.iso:
        isos = {args.iso}
    elif args.iso_list:
        isos = {c.strip() for c in args.iso_list.split(",")}
    else:
        isos = _vo_isos()

    from uroman import Uroman
    uroman = Uroman()
    config_cache = {}

    flagged = []
    checked = 0
    no_signal = 0
    ok = 0

    for canon in ("nt", "ot"):
        canon_dir = TIMING_DIR / canon
        if not canon_dir.exists():
            continue
        for iso_dir in sorted(canon_dir.iterdir()):
            if iso_dir.name not in isos:
                continue
            iso = iso_dir.name
            if iso not in config_cache:
                try:
                    config_cache[iso] = load_language_config(iso)
                except Exception:
                    config_cache[iso] = load_language_config("default")
            config = config_cache[iso]

            for timing_path in iso_dir.rglob("*_timing.json"):
                checked += 1
                if args.progress_every and checked % args.progress_every == 0:
                    print(f"  ... checked {checked} chapters, {len(flagged)} flagged so far, "
                          f"currently at {iso}/{timing_path.parent.parent.name} {timing_path.parent.name}",
                          file=sys.stderr)
                    # Checkpoint the actual findings, not just the progress
                    # count -- confirmed 2026-09-26: a run of this length
                    # (2h30m+ for the full corpus) only ever wrote its
                    # report at the very end, so a crash/kill/session
                    # teardown at any point before completion lost the
                    # entire in-memory flagged list, not just wall-clock
                    # time -- the stderr progress log survived (aggregate
                    # counts only), but the per-chapter detail this tool
                    # exists to produce did not. Every progress tick now
                    # also persists the flagged list so far, so a resumed
                    # or aborted run has real, chapter-level partial
                    # results, not just a number.
                    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.report).write_text(json.dumps(flagged, indent=2))

                pl_verses = load_timing_verses(timing_path)
                if pl_verses is None:
                    continue

                book = timing_path.parent.name
                distinct_id = timing_path.parent.parent.name
                stem = timing_path.name.replace("_timing.json", "")
                parts = stem.split("_", 2)
                if len(parts) < 2:
                    continue
                chapter_str = parts[1]

                text_matches = list((DOWNLOADS_DIR / canon / iso / distinct_id / book)
                                     .glob(f"{book}_{chapter_str}_*.txt"))
                if not text_matches:
                    continue
                try:
                    ref_verses = read_verse_texts(text_matches[0], config)
                except Exception:
                    continue

                vowel_counts = verse_vowel_counts(ref_verses, config, uroman)
                if not any(vowel_counts.values()):
                    continue

                fp = fingerprint(pl_verses, vowel_counts)
                if fp is None:
                    no_signal += 1
                    flagged.append({
                        "iso": iso, "canon": canon, "distinct_id": distinct_id,
                        "book": book, "chapter": chapter_str,
                        "rms_dev": None, "reason": "no_fingerprint (degenerate/insufficient verses)",
                    })
                    continue

                if fp.rms_dev > RMS_THRESHOLD:
                    flagged.append({
                        "iso": iso, "canon": canon, "distinct_id": distinct_id,
                        "book": book, "chapter": chapter_str,
                        "rms_dev": round(fp.rms_dev, 2), "max_abs_dev": round(fp.max_abs_dev, 2),
                        "reason": "high_self_rms",
                    })
                else:
                    ok += 1

    print(f"\nChecked {checked} verse_only_mode chapters across {len(isos)} isos")
    print(f"  self-consistency OK (RMS <= {RMS_THRESHOLD}s): {ok}")
    print(f"  flagged -- high self-RMS deviation: {len([f for f in flagged if f['reason']=='high_self_rms'])}")
    print(f"  flagged -- no fingerprint at all (degenerate/insufficient, likely total failure): {no_signal}")
    print(f"  TOTAL flagged: {len(flagged)}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(flagged, indent=2))
    print(f"\nFull flagged list written to {args.report}")


if __name__ == "__main__":
    main()
