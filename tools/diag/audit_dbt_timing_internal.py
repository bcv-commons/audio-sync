#!/usr/bin/env python3
"""Sanity-check DBT's own raw timing files (downloads/BB/.../*_timing.json)
for INTERNAL consistency — duplicate verse_start values, non-monotonic
ordering, or gaps in the verse_start sequence. This has been treated as
ground truth for verse count throughout this session's audit; this checks
whether that assumption itself holds.

A well-formed file: "0" (intro marker) then 1..N with no gaps, no dupes,
strictly increasing (see ENGKJV/JHN 3 as a clean baseline: 0,1,2,...,36).
"""
import json
from pathlib import Path

BB_ROOT = Path("downloads/BB")

total = 0
clean = 0
has_dupes = 0
has_gaps = 0
non_monotonic = 0
parse_fail = 0

dupe_samples = []
gap_samples = []
nonmono_samples = []

for f in BB_ROOT.rglob("*_timing.json"):
    try:
        d = json.loads(f.read_text())
    except Exception:
        parse_fail += 1
        continue
    if not isinstance(d, list) or not d:
        continue

    total += 1
    # verse_start values, keep "0" (intro) separate
    raw_starts = [e.get("verse_start") for e in d]
    try:
        nums = [int(s) for s in raw_starts]
    except (TypeError, ValueError):
        continue

    verse_nums = [n for n in nums if n != 0]  # drop intro marker
    if not verse_nums:
        continue

    is_dupe = len(verse_nums) != len(set(verse_nums))
    is_monotonic = all(verse_nums[i] < verse_nums[i+1] for i in range(len(verse_nums)-1))
    lo, hi = min(verse_nums), max(verse_nums)
    expected_count = hi - lo + 1
    has_gap = len(set(verse_nums)) != expected_count

    if is_dupe:
        has_dupes += 1
        if len(dupe_samples) < 10:
            dupe_samples.append((str(f.relative_to(BB_ROOT)), raw_starts))
    if has_gap and not is_dupe:
        has_gaps += 1
        if len(gap_samples) < 10:
            missing = sorted(set(range(lo, hi+1)) - set(verse_nums))
            gap_samples.append((str(f.relative_to(BB_ROOT)), missing))
    if not is_monotonic:
        non_monotonic += 1
        if len(nonmono_samples) < 10:
            nonmono_samples.append((str(f.relative_to(BB_ROOT)), raw_starts))
    if not is_dupe and not has_gap and is_monotonic:
        clean += 1

print(f"Total DBT timing files checked: {total}")
print(f"Parse failures: {parse_fail}")
print(f"Clean (no dupes, no gaps, monotonic): {clean} ({100*clean/total:.2f}%)")
print(f"Has duplicate verse_start values: {has_dupes}")
print(f"Has gaps in verse_start sequence (excl. dupes): {has_gaps}")
print(f"Non-monotonic order: {non_monotonic}")
print()
print("=== duplicate samples ===")
for path, starts in dupe_samples:
    print(" ", path, starts)
print()
print("=== gap samples ===")
for path, missing in gap_samples:
    print(" ", path, "missing:", missing)
print()
print("=== non-monotonic samples ===")
for path, starts in nonmono_samples:
    print(" ", path, starts)
