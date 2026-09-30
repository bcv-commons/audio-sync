#!/usr/bin/env python3
"""Build the final delete+redo list from _runs/gpu_rerun_needed_master.json,
and (with --apply) actually delete the targeted chapters' export/timing-data
output so they fall back to "never aligned" and get picked up fresh by the
next align_verse_words.py/align_pipeline.py run using today's improved code
(vowel-count window anchoring + same-run pacing cross-check).

Two-tier policy, in this order:

1. SEVERE EDITIONS FIRST: an edition ((iso, distinct_id)) where more than
   SEVERE_FRACTION of its own chapters are already in the master list gets
   ALL of its chapters marked for redo, not just the flagged ones.
   Confirmed 2026-09-26: 91% of severely-affected editions (172 of 189,
   >50% flagged, >=10 chapters) share the same "72-chapter NT-highlights"
   book set and standard (non-dramatized) narration style -- when an
   edition is already majority-wrong, the remaining "clean" chapters are
   far more likely to be false negatives (this signal's own blind spots --
   e.g. the repeated-phrase case) than genuinely fine, and a full redo is
   simpler and safer than trying to surgically preserve a minority that
   may not deserve it.

2. EVERYTHING ELSE: chapters from the master list that belong to a
   non-severe edition are redone individually, as already scoped.

Only ever deletes export/timing-data/*.json output (_timing.json,
_words.json, _words_quality.json) -- never downloads/, never reference
text, never audio. Deleting the OUTPUT is what makes a chapter look
"never aligned" to align_pipeline.py's own needs_run()/
_chapter_output_exists() checks, so the next real run picks it up fresh
with no other code change required.

Usage:
    python tools/prepare_gpu_redo.py                      # dry run (default)
    python tools/prepare_gpu_redo.py --apply              # actually delete
    python tools/prepare_gpu_redo.py --severe-fraction 0.3  # tune the threshold
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

MASTER_LIST = Path("_runs/gpu_rerun_needed_master.json")
TIMING_DIR = Path("export/timing-data")
SEVERE_FRACTION = 0.5
MIN_CHAPTERS_FOR_SEVERE = 10


def edition_key(c):
    return (c["iso"], c["canon"], c["distinct_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run)")
    parser.add_argument("--severe-fraction", type=float, default=SEVERE_FRACTION)
    parser.add_argument("--report", type=str, default="_runs/gpu_redo_final_report.json")
    args = parser.parse_args()

    master = json.loads(MASTER_LIST.read_text())
    flagged_by_edition = defaultdict(set)  # (iso, canon, distinct_id) -> {(book, chapter), ...}
    for c in master:
        flagged_by_edition[edition_key(c)].add((c["book"], c["chapter"]))

    # Total chapter count per edition, from what's actually on disk.
    total_by_edition = {}
    all_files_by_edition = {}
    for canon in ("nt", "ot"):
        d = TIMING_DIR / canon
        if not d.exists():
            continue
        for iso_dir in d.iterdir():
            for did_dir in iso_dir.iterdir():
                files = list(did_dir.rglob("*_timing.json"))
                if not files:
                    continue
                key = (iso_dir.name, canon, did_dir.name)
                total_by_edition[key] = len(files)
                all_files_by_edition[key] = files

    severe_editions = set()
    for key, flagged_chapters in flagged_by_edition.items():
        total = total_by_edition.get(key, 0)
        if total >= MIN_CHAPTERS_FOR_SEVERE and len(flagged_chapters) / total > args.severe_fraction:
            severe_editions.add(key)

    to_delete = []  # list of timing_path
    reasons = {}

    for key in severe_editions:
        for f in all_files_by_edition[key]:
            to_delete.append(f)
            reasons[str(f)] = "severe_edition_full_redo"

    for key, flagged_chapters in flagged_by_edition.items():
        if key in severe_editions:
            continue  # already covered above, whole edition
        iso, canon, distinct_id = key
        for book, chapter in flagged_chapters:
            matches = list((TIMING_DIR / canon / iso / distinct_id / book).glob(f"{book}_{chapter}_*_timing.json"))
            for f in matches:
                to_delete.append(f)
                reasons[str(f)] = "specific_chapter_flagged"

    verb = "Would delete" if not args.apply else "Deleting"
    print(f"Severe editions (>{args.severe_fraction*100:.0f}% flagged, >={MIN_CHAPTERS_FOR_SEVERE} chapters): "
          f"{len(severe_editions)}")
    severe_chapter_count = sum(1 for r in reasons.values() if r == "severe_edition_full_redo")
    specific_chapter_count = sum(1 for r in reasons.values() if r == "specific_chapter_flagged")
    print(f"  -> {severe_chapter_count} chapters (full-edition redo)")
    print(f"Non-severe editions: {specific_chapter_count} chapters (specific flagged chapters only)")
    print(f"{verb} {len(to_delete)} chapters' export/timing-data output total")

    deleted = 0
    for timing_path in to_delete:
        stem = timing_path.name.replace("_timing.json", "")
        siblings = [timing_path,
                    timing_path.with_name(f"{stem}_words.json"),
                    timing_path.with_name(f"{stem}_words_quality.json")]
        if args.apply:
            for p in siblings:
                if p.exists():
                    p.unlink()
            deleted += 1

    if args.apply:
        print(f"\nActually deleted output for {deleted} chapters.")
    else:
        print("\nDRY RUN -- no files deleted. Re-run with --apply to delete for real.")

    report = {
        "severe_editions": [{"iso": k[0], "canon": k[1], "distinct_id": k[2]} for k in severe_editions],
        "chapters": [{"path": str(f), "reason": reasons[str(f)]} for f in to_delete],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"Full report written to {args.report}")


if __name__ == "__main__":
    main()
