#!/usr/bin/env python3
"""Re-verify applied corrections against the measuring rod: a corrected
verse must end up at least as close to DBT's own timestamp as it was
before. Anything that comes out WORSE (new distance to DBT > old distance)
needs detailed manual review — it should never happen by construction
(every correction target is derived from evidence at least as strong as
DBT's own claim), but the point of re-verifying is not to assume that.
"""
import json
import sys
from pathlib import Path

APPLIED_LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-1000/scratch/corrections_applied_sample400.json")

data = json.loads(APPLIED_LOG.read_text())

improved = []
same = []
worse = []
skipped_for_safety = 0

for r in data:
    for c in r["corrections"]:
        if c["status"] == "SKIPPED_would_be_nonmonotonic":
            skipped_for_safety += 1
            continue
        if c["status"] != "APPLIED":
            continue
        dbt_t = c["dbt_t"]
        old_dist = abs(c["old_t"] - dbt_t)
        new_dist = abs(c["new_t"] - dbt_t)
        entry = {**c, "iso": r["iso"], "distinct_id": r["distinct_id"],
                 "book": r["book"], "chapter": r["chapter"],
                 "old_dist_to_dbt": round(old_dist, 3), "new_dist_to_dbt": round(new_dist, 3)}
        if new_dist < old_dist - 0.005:
            improved.append(entry)
        elif new_dist > old_dist + 0.005:
            worse.append(entry)
        else:
            same.append(entry)

total = len(improved) + len(same) + len(worse)
print(f"Applied corrections verified: {total}")
print(f"  IMPROVED (closer to DBT than before): {len(improved)} ({100*len(improved)/max(total,1):.1f}%)")
print(f"  SAME (within 5ms, effectively no change): {len(same)} ({100*len(same)/max(total,1):.1f}%)")
print(f"  WORSE (further from DBT than before — FLAG FOR REVIEW): {len(worse)} ({100*len(worse)/max(total,1):.1f}%)")
print(f"Skipped for monotonicity safety (not applied at all): {skipped_for_safety}")

if worse:
    print("\n=== WORSE cases — needs detailed review ===")
    for e in worse:
        print(f"  {e['iso']}/{e['distinct_id']} {e['book']} {e['chapter']} v{e['verse']}: "
              f"old_dist={e['old_dist_to_dbt']} new_dist={e['new_dist_to_dbt']} "
              f"(old_t={e['old_t']} new_t={e['new_t']} dbt_t={e['dbt_t']}, "
              f"method={e['method']}, ratio={e.get('match_ratio')})")

# Summary stats on how much closer, for the improved set
if improved:
    import statistics
    deltas = [e["old_dist_to_dbt"] - e["new_dist_to_dbt"] for e in improved]
    print(f"\nFor IMPROVED verses: mean reduction in distance-to-DBT = {statistics.mean(deltas):.2f}s, "
          f"median = {statistics.median(deltas):.2f}s")
