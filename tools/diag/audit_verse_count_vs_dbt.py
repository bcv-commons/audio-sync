#!/usr/bin/env python3
"""Self-audit: for every chapter where we have BOTH our own current
export/timing-data output AND DBT's own raw reference timing
(downloads/BB/.../*_timing.json), compare verse counts.

DBT's raw timing file is a flat list of {"verse_start": "N", ...} entries,
one per verse plus a "0" intro marker — so len(entries)-1 is DBT's own
authoritative verse count for that chapter (independent of alignment
quality; this is metadata about the reference text, not something our
alignment could get right/wrong on its own — a mismatch here means our
own text-ingestion split the chapter into the wrong number of verses,
exactly the ENGNKJ/JHN16 bug found 2026-09-15 investigating bibles'
BB-vs-legacy report).

Our own current timing file is the compact {"pos": [...]} format —
len(pos) is our own current verse count for that chapter.

Any mismatch is a confirmed, verifiable correctness bug in what we are
CURRENTLY publishing, independent of whether "legacy" (bibles' pre-2026-07-15
import of our old export/timing-data) has the same bug or not.
"""
import json
from pathlib import Path

BB_ROOT = Path("downloads/BB")
OUR_ROOT = Path("export/timing-data")

mismatches = []
checked = 0
our_missing = 0
bb_parse_fail = 0
our_parse_fail = 0

for bb_path in BB_ROOT.rglob("*_timing.json"):
    rel = bb_path.relative_to(BB_ROOT)
    our_path = OUR_ROOT / rel
    if not our_path.exists():
        our_missing += 1
        continue

    try:
        bb_data = json.loads(bb_path.read_text())
    except Exception:
        bb_parse_fail += 1
        continue
    if not isinstance(bb_data, list) or not bb_data:
        continue
    bb_verse_count = len(bb_data) - 1  # minus the "0" intro marker

    try:
        our_data = json.loads(our_path.read_text())
    except Exception:
        our_parse_fail += 1
        continue
    if "pos" not in our_data:
        continue
    our_verse_count = len(our_data["pos"])

    checked += 1
    if our_verse_count != bb_verse_count:
        mismatches.append({
            "path": str(rel),
            "bb_verses": bb_verse_count,
            "our_verses": our_verse_count,
            "diff": our_verse_count - bb_verse_count,
        })

print(f"Checked: {checked}")
print(f"Our file missing (not yet aligned locally): {our_missing}")
print(f"BB parse failures: {bb_parse_fail}")
print(f"Our parse failures: {our_parse_fail}")
print(f"Verse-count MISMATCHES: {len(mismatches)}")

out_path = Path("_runs/verse_count_audit.json")
out_path.write_text(json.dumps(mismatches, indent=2))
print(f"Full mismatch list written to {out_path}")

# Quick breakdown by distinct_id
from collections import Counter
by_edition = Counter()
for m in mismatches:
    parts = m["path"].split("/")
    # nt/eng/ENGNKJ/JHN/JHN_016_..._timing.json
    if len(parts) >= 3:
        by_edition[f"{parts[1]}/{parts[2]}"] += 1

print("\nTop 20 editions by mismatch count:")
for k, v in by_edition.most_common(20):
    print(f"  {k}: {v}")
