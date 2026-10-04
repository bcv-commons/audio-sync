#!/usr/bin/env bash
# One-page health + progress summary of the verse-only redo, written to
# _runs/daily_redo_report_<date>.txt (and printed). Read-only.
#
# Run by cron every morning (see CLAUDE.local.md), or by hand:
#     tools/diag/daily_redo_report.sh
set -u
cd "$(dirname "$0")/../.."
OUT="_runs/daily_redo_report_$(date -u +%Y%m%d).txt"
PY=.venv/bin/python

{
echo "== redo report $(date -u '+%Y-%m-%d %H:%M UTC') =="
echo
echo "-- processes"
for pat in "run_gpu_redo.py --report _runs/redo_worker_a" "run_gpu_redo.py --report _runs/redo_worker_b" "redo_hang_watch.py"; do
    if pgrep -f "$pat" > /dev/null; then echo "  up    $pat"; else echo "  DOWN  $pat"; fi
done
echo
echo "-- progress (chapters written, last 24 h, by hour)"
find export/timing-data -name '*_words_quality.json' -newermt "$(date -u -d '24 hours ago' '+%Y-%m-%d %H:%M')" \
    -printf '%TH\n' 2>/dev/null | sort | uniq -c | awk '{printf "  %sh: %d\n", $2, $1}' | tail -24
echo
echo "-- worker logs: hangs, crashes, errors (last 24 h)"
grep -h "HANG\|SIGSEGV\|Exited with\|Traceback\|OutOfMemory\|EXCEPTION" /tmp/gpu_redo_worker_a.log /tmp/gpu_redo_worker_b.log 2>/dev/null | tail -8 | cut -c1-200
echo "  quarantined (crash/hang): $(cat _runs/verse_only_quarantine_a.json _runs/verse_only_quarantine_b.json 2>/dev/null | grep -c '"reason"')"
echo
echo "-- hang watcher (last lines)"
tail -3 /tmp/redo_hang_watch.log 2>/dev/null | cut -c1-200
echo
echo "-- disk"
df -h / | tail -1 | awk '{print "  free " $4 " (" $5 " used)"}'
echo
echo "-- gpu"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'
echo
echo "-- per language: gate outcomes, low-score share, vs DBT (flags: <--)"
CUDA_VISIBLE_DEVICES="" $PY tools/diag/redo_spot_check.py 2>&1 | grep -v "^\[hw"
} > "$OUT" 2>&1
cat "$OUT"
