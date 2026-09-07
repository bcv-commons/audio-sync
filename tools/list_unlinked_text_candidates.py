#!/usr/bin/env python3
"""
Report, per (iso, canon, audio distinct_id), what pipeline/download_language_
content.py's text-source resolution finds beyond a direct verified link —
the human-facing counterpart to _find_unlinked_text_candidate().

Three outcomes matter here, and only the first is now handled automatically
by the pipeline (see resolve_preferred_text_source()'s "dbt-other" source
and _find_unlinked_text_candidate()'s docstring for the Tier 0/1 reasoning):

  AUTO-USABLE   Tier 0 (cross-verified cluster) or Tier 1 (dialect_variant/
                orthography_convention) candidate found — has_usable_text_
                source() now returns True for this edition without any
                further action needed here.
  FLAGGED       Only a Tier 2 ("distinct_translation") candidate exists —
                real text, but the bibles repo's own comparator says it's a
                genuinely different translation, which carries real forced-
                alignment risk if used unverified. Not auto-used.
  NOTHING       No catalog-overlap.json opinion at all for this iso:canon
                (or only failed/"r": false probes) — there's nothing here
                to widen into; the only way forward is a fresh comparison,
                either a probe the bibles repo runs against this specific
                distinct_id and publishes back to catalog-overlap.json
                (see docs on the two global probe chapter sets, "probes" in
                that file: NT REV15, OT PSA117+PSA51), or a local audio-vs-
                text verification step in this repo.

Also flags a fourth, unrelated case: OWN-TEXT-UNUSABLE-FORMAT — the audio
edition has its own catalog-text.json entry, but every format listed is
USX ("u"), which get_best_fileset_from_catalog()/_resolve_text_fileset()
deliberately drop (DBT's USX endpoint often 404s even when listed). That's
a format problem, not a linkage problem — no catalog-overlap.json candidate
search fixes it.

Usage:
    python tools/list_unlinked_text_candidates.py --iso-list ita,ukr,ces,hrv,dga
    python tools/list_unlinked_text_candidates.py --iso-list ita --json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from download_language_content import (  # noqa: E402
    _catalog_entries,
    _find_unlinked_text_candidate,
    _resolve_text_fileset,
    audio_distinct_ids_from_catalog,
    get_best_fileset_from_catalog,
    has_usable_text_source,
)


def classify(iso: str, canon: str, distinct_id: str) -> dict:
    usable = has_usable_text_source(iso, canon, distinct_id)
    fileset_info = get_best_fileset_from_catalog(iso, canon, distinct_id)
    own_text_entries = _catalog_entries(iso, canon, distinct_id, "catalog-text")
    own_fileset, _ = _resolve_text_fileset(own_text_entries, distinct_id)

    if own_text_entries and not own_fileset:
        return {"status": "OWN-TEXT-UNUSABLE-FORMAT",
                "detail": f"catalog-text.json has {own_text_entries} for {distinct_id} "
                          f"itself, but every format is USX (dropped)"}

    if usable is True or (hasattr(usable, "__len__") and len(usable) > 0):
        if fileset_info and fileset_info.get("text_fileset"):
            return {"status": "OWN-TEXT-OK", "detail": f"own DBT text fileset {fileset_info['text_fileset']}"}
        return {"status": "AUTO-USABLE", "detail": "resolved via a direct/verified/Tier-0/Tier-1 text source"}

    candidate = _find_unlinked_text_candidate(iso, canon, distinct_id)
    if candidate:
        # Shouldn't happen (has_usable_text_source would have returned True
        # for a Tier 0/1 hit) — surfaced anyway in case of future drift.
        return {"status": "AUTO-USABLE", "detail": str(candidate)}

    # Distinguish FLAGGED (a real Tier 2 candidate exists) from NOTHING.
    from download_language_content import _load_dbt_catalog
    overlap = _load_dbt_catalog("catalog-overlap").get("entries", {})
    canon_l = canon.lower()
    tier2 = []
    for key in (f"{iso}:{canon_l}", f"{iso}:{canon_l}p"):
        for group in overlap.get(key, []):
            if group.get("r") is False:
                continue
            if group.get("likely") == "distinct_translation":
                tier2.append(group)
    if tier2:
        best = max(tier2, key=lambda g: g.get("score", 0.0))
        ids = ",".join(best["ids"])
        return {"status": "FLAGGED",
                "detail": f"only distinct_translation candidate(s), best: {ids} "
                          f"(score={best.get('score')})"}

    return {"status": "NOTHING", "detail": "no catalog-overlap.json opinion at all"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso-list", required=True, help="Comma-separated iso codes")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = ap.parse_args()

    isos = [s.strip() for s in args.iso_list.split(",") if s.strip()]
    results = {}
    for iso in isos:
        results[iso] = {}
        for canon in ("NT", "OT"):
            dids = audio_distinct_ids_from_catalog(iso, canon)
            for did in dids:
                results[iso][f"{canon}:{did}"] = classify(iso, canon, did)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for iso, by_key in results.items():
        for key, info in by_key.items():
            print(f"{iso:5} {key:24} {info['status']:24} {info['detail']}")


if __name__ == "__main__":
    main()
