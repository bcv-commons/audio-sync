#!/usr/bin/env python3
"""Spot-check the verse-only redo while it runs.

For every chapter already tagged with the current ALIGNMENT_METHOD:
  - per language: chapters redone, chapter-gate outcome (pass / held_back /
    defer_to_dbt), share of verses scoring below the gate score;
  - where DBT has its own timing for the same fileset on disk: share of
    verses more than 1 s from DBT (after removing the chapter's constant
    offset), for the new output and for the pre-redo backup of the same
    chapter, so a regression shows up as new > old.

Read-only. Usage:
    python tools/diag/redo_spot_check.py [--backup-dir DIR] [--json OUT]
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from align_verse_words import ALIGNMENT_METHOD, GATE_LOW_SCORE  # noqa: E402

TIMING_DIR = Path("export/timing-data")
DOWNLOADS_DIR = Path("downloads/BB")


def _dbt_starts(path: Path) -> dict[int, float] | None:
    try:
        entries = json.loads(path.read_text())
        dbt = {int(e["verse_start"]): float(e["timestamp"]) for e in entries
               if str(e.get("verse_start", "")).isdigit()}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    ts = [dbt[k] for k in sorted(dbt) if k >= 1]
    if len(ts) < 5 or max(ts) == 0 or any(b < a for a, b in zip(ts, ts[1:])):
        return None
    return dbt


def _off_by_more_than_1s(pos: list, dbt: dict[int, float]) -> tuple[int, int] | None:
    pairs = [(v, pos[v - 1]) for v in range(2, len(pos) + 1) if v in dbt and pos[v - 1] is not None]
    if len(pairs) < 5:
        return None
    off = statistics.median(t - dbt[v] for v, t in pairs)
    return sum(abs(t - dbt[v] - off) > 1.0 for v, t in pairs), len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backup-dir", default="export/timing-data-backup-pre-redo-2026-10-03")
    ap.add_argument("--json", default=None, help="Also write the per-language table here")
    args = ap.parse_args()
    backup = Path(args.backup_dir)

    langs = defaultdict(lambda: {"chapters": 0, "pass": 0, "held_back": 0, "defer_to_dbt": 0,
                                 "low_verses": 0, "verses": 0,
                                 "dbt_ch": 0, "dbt_v": 0, "new_off": 0, "old_off": 0})
    for dp, _dn, fn in os.walk(TIMING_DIR):
        for f in fn:
            if not f.endswith("_words_quality.json"):
                continue
            q = Path(dp) / f
            try:
                data = json.loads(q.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            summary = data.get("summary") or {}
            if summary.get("method") != ALIGNMENT_METHOD:
                continue
            rel = q.relative_to(TIMING_DIR)
            L = langs[rel.parts[1]]
            L["chapters"] += 1
            L[summary.get("gate", "pass")] += 1
            for words in (data.get("verses") or {}).values():
                if words:
                    L["verses"] += 1
                    L["low_verses"] += words[0].get("score", 0) < GATE_LOW_SCORE

            timing = q.with_name(f.replace("_words_quality.json", "_timing.json"))
            dbt = _dbt_starts(DOWNLOADS_DIR / timing.relative_to(TIMING_DIR))
            if dbt is None:
                continue
            try:
                new = json.loads(timing.read_text())
                old = json.loads((backup / timing.relative_to(TIMING_DIR)).read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if "pos" not in new or "pos" not in old:
                continue
            n, o = _off_by_more_than_1s(new["pos"], dbt), _off_by_more_than_1s(old["pos"], dbt)
            if n and o:
                L["dbt_ch"] += 1
                L["dbt_v"] += n[1]
                L["new_off"] += n[0]
                L["old_off"] += o[0]

    tot = defaultdict(int)
    for L in langs.values():
        for k, v in L.items():
            tot[k] += v
    print(f"Redone chapters (method {ALIGNMENT_METHOD}): {tot['chapters']} in {len(langs)} languages")
    print(f"  gate: pass {tot['pass']}, held_back {tot['held_back']}, defer_to_dbt {tot['defer_to_dbt']}")
    if tot["verses"]:
        print(f"  verses scoring below {GATE_LOW_SCORE}: {tot['low_verses']}/{tot['verses']} "
              f"({100 * tot['low_verses'] / tot['verses']:.1f}%)")
    if tot["dbt_v"]:
        print(f"  vs DBT ({tot['dbt_ch']} chapters, {tot['dbt_v']} verses): off by >1 s "
              f"new {100 * tot['new_off'] / tot['dbt_v']:.1f}% | before redo {100 * tot['old_off'] / tot['dbt_v']:.1f}%")

    def flag(L):
        reasons = []
        if L["chapters"] >= 5 and L["held_back"] + L["defer_to_dbt"] > 0.3 * L["chapters"]:
            reasons.append("over 30% of chapters gated")
        if L["dbt_v"] >= 50 and L["new_off"] > L["old_off"]:
            reasons.append("worse than before vs DBT")
        return reasons

    print("\nper language (chapters, gated, low-score verses, vs DBT new/old):")
    for iso, L in sorted(langs.items(), key=lambda kv: -kv[1]["chapters"]):
        gated = L["held_back"] + L["defer_to_dbt"]
        low = 100 * L["low_verses"] / L["verses"] if L["verses"] else 0
        dbt = (f"{100 * L['new_off'] / L['dbt_v']:.0f}% / {100 * L['old_off'] / L['dbt_v']:.0f}%"
               if L["dbt_v"] else "-")
        reasons = flag(L)
        print(f"  {iso:6s} {L['chapters']:5d} ch  gated {gated:4d}  low {low:4.1f}%  dbt {dbt:>11s}"
              f"{'  <-- ' + '; '.join(reasons) if reasons else ''}")
    if args.json:
        Path(args.json).write_text(json.dumps(langs, indent=1))


if __name__ == "__main__":
    main()
