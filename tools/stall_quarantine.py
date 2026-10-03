#!/usr/bin/env python3
"""Per-iso stall quarantine — stops a genuinely-poisonous language (one
whose alignment hangs indefinitely, triggering watchdog-align-parallel.sh's
300s stall-kill) from blocking every language behind it in the list,
forever, on every restart.

Root cause this exists for: confirmed 2026-09-16 via a real incident —
`cmo` was the active language at TWO CONSECUTIVE stall-kills (07:33 and
11:02 the same day), each restart re-scanning the full ~1,170-iso list
from #1 (shard_align.py's own design — see its module docstring) and
re-hitting the same hang. A single genuinely-stuck chapter can therefore
stall the ENTIRE remaining list indefinitely, not just waste time on its
own language. This module breaks that: after N consecutive stall-kills
land on the same iso, it's quarantined — shard_align.py skips it (logged,
not silently) on every subsequent run until a human clears it, rather
than retrying forever.

Deliberately does NOT track GPU-wedge kills the same way — a wedge is a
hardware/driver event that can land on whichever language happens to be
running at the time, not evidence that THIS language's content is the
problem (unlike a stall, which is specifically "this language's own
worker made no progress for 300s"). Conflating the two would quarantine
innocent languages that just had the bad luck to be active during an
unrelated GPU wedge.

State lives in two small JSON files under _runs/ (same convention as the
existing tools/pre_publish_check.py's _runs/pre_publish_quarantine.txt —
quarantine-not-block, human-reviewable, not a blocking gate):

  _runs/stall_tracking.json   — {iso: {"count": N, "last_stall_at": ...}}
    Running count of CONSECUTIVE stall-kills where this iso was the last
    one to start ("===== Language: X =====") before the kill. Reset to
    absent the moment that iso next completes successfully — a single
    old stall should never count against a language that's since proven
    it can finish.

  _runs/stall_quarantine.json — {iso: {"quarantined_at": ..., "consecutive_stalls": N,
                                        "last_stall_at": ...}}
    Once QUARANTINE_THRESHOLD is reached, the iso moves here and
    shard_align.py refuses to process it (skip + log) until a human
    removes its entry — see this module's __main__ block for the
    review/clear CLI.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

TRACKING_PATH = Path("_runs/stall_tracking.json")
QUARANTINE_PATH = Path("_runs/stall_quarantine.json")
QUARANTINE_THRESHOLD = 2  # consecutive stall-kills on the same iso


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def last_active_iso(logfile: Path) -> str | None:
    """The iso from the last "===== Language: X =====" line in a
    shard_align.py top-level log — i.e. whichever language was active
    (started, not necessarily finished) right before a kill."""
    try:
        text = logfile.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = re.findall(r"===== Language: (\S+) =====", text)
    return matches[-1] if matches else None


def record_stall(logfile: Path) -> str | None:
    """Call from watchdog-align-parallel.sh right after a STALL-triggered
    kill_tree(). Bumps the consecutive-stall count for whichever iso was
    active, escalating to quarantine at QUARANTINE_THRESHOLD. Returns a
    human-readable status line for the caller to log, or None if no
    active language could be identified (nothing to record)."""
    iso = last_active_iso(logfile)
    if not iso:
        return None

    tracking = _load(TRACKING_PATH)
    entry = tracking.get(iso, {"count": 0})
    entry["count"] += 1
    entry["last_stall_at"] = _now()
    tracking[iso] = entry
    _save(TRACKING_PATH, tracking)

    if entry["count"] >= QUARANTINE_THRESHOLD:
        quarantine = _load(QUARANTINE_PATH)
        quarantine[iso] = {
            "quarantined_at": _now(),
            "consecutive_stalls": entry["count"],
            "last_stall_at": entry["last_stall_at"],
        }
        _save(QUARANTINE_PATH, quarantine)
        del tracking[iso]
        _save(TRACKING_PATH, tracking)
        return (f"{iso}: {entry['count']} consecutive stall-kills — QUARANTINED "
                f"(see {QUARANTINE_PATH}; won't be attempted again until cleared)")
    return f"{iso}: stall-kill {entry['count']}/{QUARANTINE_THRESHOLD} (not yet quarantined)"


def is_quarantined(iso: str) -> bool:
    return iso in _load(QUARANTINE_PATH)


def clear_tracking_on_success(iso: str) -> None:
    """Call right after an iso finishes OK (align_pipeline.py's own
    per-language lifecycle block, since 2026-10-02 -- previously only
    shard_align.py called this) -- a language that's since proven it can
    complete shouldn't have an old, resolved stall count held against it
    indefinitely."""
    tracking = _load(TRACKING_PATH)
    if iso in tracking:
        del tracking[iso]
        _save(TRACKING_PATH, tracking)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_record = sub.add_parser("record-stall", help="Record a stall-kill (called by the bash watchdog)")
    p_record.add_argument("--logfile", required=True, type=Path)

    sub.add_parser("list", help="List current quarantine + tracking state")

    p_clear = sub.add_parser("clear", help="Remove an iso from quarantine (after investigating/fixing the cause)")
    p_clear.add_argument("iso")

    args = parser.parse_args()

    if args.cmd == "record-stall":
        msg = record_stall(args.logfile)
        if msg:
            print(msg)
        sys.exit(0)

    elif args.cmd == "list":
        q = _load(QUARANTINE_PATH)
        t = _load(TRACKING_PATH)
        print(f"Quarantined ({len(q)}):")
        for iso, info in q.items():
            print(f"  {iso}: {info}")
        print(f"Tracking, not yet quarantined ({len(t)}):")
        for iso, info in t.items():
            print(f"  {iso}: {info}")

    elif args.cmd == "clear":
        q = _load(QUARANTINE_PATH)
        if args.iso in q:
            del q[args.iso]
            _save(QUARANTINE_PATH, q)
            print(f"{args.iso}: cleared from quarantine")
        else:
            print(f"{args.iso}: not currently quarantined")
