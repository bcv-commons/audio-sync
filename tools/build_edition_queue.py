#!/usr/bin/env python3
"""
Build a phase-ordered edition queue for the "audio-sync becomes the producer
of DBT timing" effort — the shared work list both batch_download_audio.py
and the alignment pipeline (via internal-scripts/watchdog-align-parallel.sh)
walk in the same order.

Three phases, per (iso, canon, distinct_id) edition:

  Phase 1 — zero dbtTiming anywhere for this edition. DBT has no timing
  claim at all, verified or not, so this repo is the only possible source.

  Phase 2 — dbtTiming == "mms_align" on at least one audio variant (may
  also carry other codes). Real spot-check (2026-08-13, tools/compare_timing.py
  against our own already-produced output) confirmed this is DBT's own
  automated CTC-style alignment, not a curated source: 26% BAD, worst
  average drift (7.92s) of any category, and its timestamp precision
  (89% 3-decimal-place, e.g. 16.611s) matches raw algorithmic output.
  Treated with the same urgency as Phase 1, just ordered after it, rather
  than deferred as "trusted."

  Phase 3 — dbtTiming present but never "mms_align" (i.e. only "4"/"0"/"5").
  The same spot-check found a materially different timestamp fingerprint
  (~80-89% clean 2-decimal-place values, e.g. 25.8s) consistent with a
  more curated production process — "0" scored 63% GOOD vs. mms_align's
  16%, and "4" was mostly DRIFT (a small, consistent offset) rather than
  BAD. Deferred to last as the most-likely-already-fine category, but
  FLAGGED, not fully trusted: "0" was bimodal (63% GOOD but still 25%
  BAD), so a real quarter of it needs the same follow-up before being
  written off — see flagged_editions.

Within Phase 1, ordered by audio-variant count (the closest available
proxy — no real population/priority data exists locally; the `tier` field
elsewhere in this codebase is a hardcoded placeholder, not real data).

Source: cdn.bibel.wiki/catalog/audio.json, fetched (and cached) via
download_language_content.py's _load_dbt_catalog() — same catalog this
repo already uses for fileset resolution, no new data source.

Usage:
    python tools/build_edition_queue.py
    python tools/build_edition_queue.py --out _runs/edition_queue.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from download_language_content import _load_dbt_catalog  # noqa: E402

DEFAULT_OUT = Path("_runs/edition_queue.json")


def _phase_for(dbttiming_values: set[str]) -> int:
    if not dbttiming_values:
        return 1
    if "mms_align" in dbttiming_values:
        return 2
    return 3


def build_queue() -> dict:
    audio = _load_dbt_catalog("catalog-audio")
    entries = audio.get("entries", {})

    # One row per (iso, canon, distinct_id) edition — the granularity both
    # the downloader and the aligner actually operate at.
    editions = []
    variant_counts = defaultdict(int)
    dbttiming_values = defaultdict(set)

    for key, versions in entries.items():
        iso, canon = key.split(":")
        for distinct_id, variants in versions.items():
            edition_key = (iso, canon, distinct_id)
            variant_counts[edition_key] += len(variants)
            for v in variants:
                if "dbtTiming" in v:
                    dbttiming_values[edition_key].add(v["dbtTiming"])

    for edition_key, count in variant_counts.items():
        iso, canon, distinct_id = edition_key
        vals = dbttiming_values[edition_key]
        editions.append({
            "iso": iso,
            "canon": canon,
            "distinct_id": distinct_id,
            "phase": _phase_for(vals),
            "dbtTiming_values": sorted(vals),
            "audio_variants": count,
            # Phase 3's "0" code was bimodal in the 2026-08-13 spot-check
            # (63% GOOD, 25% BAD) — every phase-3 edition still needs a
            # real check before being treated as done, not just deferred.
            "flagged_for_verification": _phase_for(vals) == 3,
        })

    # Phase order, then higher audio-variant count first within a phase
    # (the priority proxy described in the module docstring).
    editions.sort(key=lambda e: (e["phase"], -e["audio_variants"]))

    counts = {p: sum(1 for e in editions if e["phase"] == p) for p in (1, 2, 3)}

    return {
        "source": "https://cdn.bibel.wiki/catalog/audio.json",
        "spot_check_reference": "2026-08-13 tools/compare_timing.py run — see this script's module docstring for the numbers behind the phase split",
        "phase_1_definition": "zero dbtTiming anywhere for this edition (sole potential source)",
        "phase_2_definition": "dbtTiming includes \"mms_align\" (DBT's own unverified automated alignment — spot-checked unreliable, 26% BAD)",
        "phase_3_definition": "dbtTiming present, never \"mms_align\" (likely more curated — spot-checked better but not clean, still flagged for verification)",
        "phase_1_count": counts[1],
        "phase_2_count": counts[2],
        "phase_3_count": counts[3],
        "editions": editions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                         help=f"Where to write the queue JSON (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    queue = build_queue()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(queue, f, indent=2)

    print(f"Wrote {args.out}")
    print(f"  Phase 1 (zero dbtTiming, sole source):        {queue['phase_1_count']} editions")
    print(f"  Phase 2 (mms_align, spot-checked unreliable):  {queue['phase_2_count']} editions")
    print(f"  Phase 3 (4/0/5, deferred but flagged):         {queue['phase_3_count']} editions")


if __name__ == "__main__":
    main()
