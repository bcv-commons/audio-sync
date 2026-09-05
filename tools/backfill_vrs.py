#!/usr/bin/env python3
"""
Stamp "vrs" onto timing.json files that were written before we declared it.

align_words.write_timing_json() now records the versification scheme every
chapter's numbering is expressed in, but the corpus already holds ~99k
timing files written without it. Re-aligning them just to add a field is
absurd (it's days of GPU time), and leaving them unstamped keeps the exact
ambiguity the field exists to remove: our pos[] is keyed to the source
text's numbering, and 28 published editions use a non-eng scheme, so
kaz/KAZKAZ PSA 34 is rso Psalm 34 = org Psalm 35 with nothing saying so.

This rewrites those files in place, adding only the "vrs" key. It touches
nothing else — no re-derivation, no reformatting of timings — so it is safe
to re-run and safe on files already stamped.

Editions whose scheme isn't positively known are skipped rather than
guessed: a wrong declaration is worse than none (see versification.py).

    python tools/backfill_vrs.py                 # dry run, whole corpus
    python tools/backfill_vrs.py --iso bul
    python tools/backfill_vrs.py --apply
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from versification import lookup_vrs  # noqa: E402

TIMING_DIR = Path("export/timing-data")


def iter_timing_files(iso_filter: str | None):
    for canon_dir in sorted(TIMING_DIR.glob("*")):
        if not canon_dir.is_dir():
            continue
        for iso_dir in sorted(canon_dir.glob("*")):
            if iso_filter and iso_dir.name != iso_filter:
                continue
            for did_dir in sorted(iso_dir.glob("*")):
                for book_dir in sorted(did_dir.glob("*")):
                    if not book_dir.is_dir():
                        continue
                    for f in sorted(book_dir.glob("*_timing.json")):
                        yield iso_dir.name, did_dir.name, f


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", help="Only this language")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write (default: dry run)")
    args = parser.parse_args()

    # One lookup per edition, not per chapter.
    scheme_cache: dict[tuple[str, str], str | None] = {}
    stamped = Counter()
    skipped_unknown = Counter()
    already = 0
    unreadable = 0
    legacy_format = 0
    total = 0

    for iso, did, path in iter_timing_files(args.iso):
        total += 1
        key = (iso, did)
        if key not in scheme_cache:
            scheme_cache[key] = lookup_vrs(iso, did)
        scheme = scheme_cache[key]
        if not scheme:
            skipped_unknown[f"{iso}/{did}"] += 1
            continue

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            unreadable += 1
            continue
        if not isinstance(data, dict):
            # Legacy verbose format ([{verse_start, timestamp}, ...]) from
            # before the compact redesign — valid, just has nowhere to put a
            # top-level key. Left alone deliberately: quietly restructuring
            # published artifacts is well outside "add one field", and these
            # need a real format migration instead.
            legacy_format += 1
            continue
        if data.get("vrs") == scheme:
            already += 1
            continue

        stamped[scheme] += 1
        if args.apply:
            data["vrs"] = scheme
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, separators=(",", ":"))
            except OSError:
                unreadable += 1

    print(f"timing.json files scanned:      {total}")
    print(f"  already stamped:              {already}")
    print(f"  edition scheme unknown:       {sum(skipped_unknown.values())} "
          f"across {len(skipped_unknown)} edition(s)")
    print(f"  legacy list format (skipped): {legacy_format}")
    print(f"  unreadable:                   {unreadable}")
    print(f"  {'STAMPED' if args.apply else 'would stamp'}:                {sum(stamped.values())}")
    if stamped:
        print(f"      by scheme: {dict(stamped)}")
    if not args.apply and sum(stamped.values()):
        print("\nDRY RUN — re-run with --apply to write.")


if __name__ == "__main__":
    main()
