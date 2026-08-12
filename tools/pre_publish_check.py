#!/usr/bin/env python3
"""
Pre-publish plausibility gate for scripts/publish-align.sh.

Scans every *_timing.json under export/timing-data/ (the exact tree about
to be published) for backwards timestamp jumps — a verse whose timestamp
is earlier than the previous verse's. Unlike dupes/gaps/tiny-steps (which
check_timing_quality.py also tracks but which can be legitimate — two
verses spoken back-to-back, a genuinely long intro), a backwards jump is
never correct: it means the alignment placed a verse at the wrong point
in the audio entirely (see the ENGBSB/ZLMAVB ISA 51 case that motivated
this check — verse 1 landed at 118.7s, verse 2 at 45.24s).

Rather than blocking the whole publish over one bad chapter (this repo's
existing philosophy — see ruff.toml's BLE001 comment: one failure must
never take down the rest of a run), flagged chapters are quarantined:
written to an rclone --exclude-from file so publish-align.sh can skip
just those files and publish everything else.

Usage:
    python tools/pre_publish_check.py [--out PATH]

Exit code 0 always (advisory — publish-align.sh decides what to do with
the exclude file); prints a one-line summary either way.
"""

import argparse
import json
import sys
from pathlib import Path

TIMING_DIR = Path("export/timing-data")
DEFAULT_OUT = Path("_runs/pre_publish_quarantine.txt")


def has_backwards_jump(timing_path: Path) -> bool:
    """True if any verse's timestamp precedes the previous verse's.

    Handles the new compact format ({"pos": [...]}), the old verbose
    per-verse-dict list this whole corpus is still in until reprocessed,
    and safely no-ops on shapes that are neither (e.g. OBS's story/segment
    timing files, which also live under export/timing-data/ but aren't
    Bible-chapter timing at all).
    """
    try:
        with open(timing_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if isinstance(data, dict) and "pos" in data:
        verses = [t for t in data["pos"] if t is not None]
    elif isinstance(data, list) and data and isinstance(data[0], dict) and "verse_start" in data[0]:
        verses = [e["timestamp"] for e in data if str(e.get("verse_start")) != "0"]
    else:
        return False

    return any(verses[i] < verses[i - 1] for i in range(1, len(verses)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                         help=f"Where to write the rclone exclude list (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    if not TIMING_DIR.exists():
        print(f"[pre-publish-check] {TIMING_DIR} not found, nothing to check")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("")
        return

    flagged_stems = []
    for tf in sorted(TIMING_DIR.rglob("*_timing.json")):
        if has_backwards_jump(tf):
            flagged_stems.append(tf.stem.replace("_timing", ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # One glob pattern per chapter, matches all its output files
    # (_timing.json, _words.json, _words_quality.json) at any depth.
    args.out.write_text("".join(f"**/{stem}_*\n" for stem in flagged_stems))

    if flagged_stems:
        print(f"[pre-publish-check] {len(flagged_stems)} chapter(s) with a backwards "
              f"timestamp jump — quarantined, not publishing:")
        for stem in flagged_stems[:20]:
            print(f"  {stem}")
        if len(flagged_stems) > 20:
            print(f"  ... and {len(flagged_stems) - 20} more")
        print(f"[pre-publish-check] full list: {args.out}")
    else:
        print("[pre-publish-check] no backwards-jump chapters found — nothing quarantined")


if __name__ == "__main__":
    sys.exit(main())
