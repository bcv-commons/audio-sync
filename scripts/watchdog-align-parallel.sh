#!/usr/bin/env bash
#
# Unattended watchdog wrapper around shard_align.py (N parallel workers per
# language), same safety model as watchdog-align.sh (see that file for the
# full rationale — signal.alarm can't interrupt a hang inside a blocking
# CUDA call, so an external OS-level kill is the only reliable stall guard;
# a real GPU wedge needs `sudo reboot`, which this script cannot do itself).
#
# The one real difference from watchdog-align.sh: stall detection can't
# just watch one log file. shard_align.py's own top-level log barely grows
# between per-worker completions — all the interesting per-chapter output
# goes to each worker's own _runs/shard-<iso>-<ts>/worker-N.log. So this
# watches the *combined size* of every worker log across every shard
# directory instead — if that total stops growing, something's actually
# stuck (a genuinely-running set of workers is always writing *something*
# across N logs every few seconds).
#
# Also has to kill the whole process tree on stall/wedge, not just the
# shard_align.py parent — it spawns N align_pipeline.py children via
# subprocess.Popen, which don't die on their own if the parent is killed.
#
# Usage:
#   scripts/watchdog-align-parallel.sh "--iso-list fra,arb,hin --workers 2 --books ALL"
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

ARGS="${1:?Usage: $0 \"<shard_align.py args>\"}"
STALL_TIMEOUT=300      # seconds with no growth across ALL worker logs = considered hung
CHECK_INTERVAL=15
LOG_DIR="_runs/logs"
STATUS_FILE="$LOG_DIR/watchdog-parallel-status.txt"
SNAPSHOT_FILE="$LOG_DIR/watchdog-parallel-debug-snapshots.txt"
GPU_MON_FILE="$LOG_DIR/watchdog-parallel-gpu-monitor.csv"
NEEDS_REBOOT_FLAG="$LOG_DIR/watchdog-parallel-needs-reboot.flag"

mkdir -p "$LOG_DIR"
rm -f "$NEEDS_REBOOT_FLAG"

log_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$STATUS_FILE"
}

total_worker_log_bytes() {
    # Sum of every worker-*.log across every shard-*/ dir, 0 if none yet.
    find _runs/shard-*/worker-*.log -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'
}

kill_tree() {
    local pid="$1"
    # Direct children first (shard_align.py's align_pipeline.py workers),
    # then the parent itself.
    pkill -9 -P "$pid" 2>/dev/null
    kill -9 "$pid" 2>/dev/null
}

snapshot_debug_state() {
    local reason="$1"
    local logfile="$2"
    {
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') — $reason ====="
        echo "--- last 30 lines of top-level log ($logfile) ---"
        tail -30 "$logfile" 2>/dev/null
        echo "--- last 15 lines of each worker log touched in the last 2 minutes ---"
        find _runs/shard-*/worker-*.log -type f -mmin -2 2>/dev/null | while read -r f; do
            echo "-- $f --"
            tail -15 "$f" 2>/dev/null
        done
        echo "--- nvidia-smi ---"
        nvidia-smi 2>&1
        echo "===== end snapshot ====="
        echo
    } >> "$SNAPSHOT_FILE"
}

nohup nvidia-smi --query-gpu=timestamp,memory.used,memory.free,memory.reserved,utilization.gpu,utilization.memory,temperature.gpu,power.draw,pstate,clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown \
    --format=csv -l 2 > "$GPU_MON_FILE" 2>&1 < /dev/null &
GPU_MON_PID=$!
disown

cleanup() {
    kill "$GPU_MON_PID" 2>/dev/null
}
trap cleanup EXIT

log_status "Parallel watchdog started. ARGS: $ARGS"

attempt=0
while true; do
    attempt=$((attempt + 1))
    LOGFILE="$LOG_DIR/watchdog-parallel-run-$(date +%Y%m%dT%H%M%S).log"
    log_status "Attempt $attempt: launching shard_align.py $ARGS -> $LOGFILE"

    # shellcheck disable=SC2086
    PYTHONUNBUFFERED=1 .venv/bin/python pipeline/shard_align.py $ARGS > "$LOGFILE" 2>&1 < /dev/null &
    PID=$!

    last_total=0
    stall_time=0
    wedged=0

    while kill -0 "$PID" 2>/dev/null; do
        sleep "$CHECK_INTERVAL"

        if ! nvidia-smi > /dev/null 2>&1; then
            log_status "GPU WEDGED while shard_align.py still running (PID $PID)."
            snapshot_debug_state "GPU wedged (mid-run)" "$LOGFILE"
            kill_tree "$PID"
            wedged=1
            break
        fi

        cur_total=$(total_worker_log_bytes)
        if [ "$cur_total" = "$last_total" ]; then
            stall_time=$((stall_time + CHECK_INTERVAL))
        else
            stall_time=0
        fi
        last_total=$cur_total

        if [ "$stall_time" -ge "$STALL_TIMEOUT" ]; then
            log_status "STALL detected — no growth across any worker log for ${STALL_TIMEOUT}s (PID $PID). Killing."
            snapshot_debug_state "Stall (${STALL_TIMEOUT}s no output across all workers)" "$LOGFILE"
            kill_tree "$PID"
            sleep 2
            break
        fi
    done

    wait "$PID" 2>/dev/null
    exit_code=$?

    if [ "$wedged" -eq 0 ] && ! nvidia-smi > /dev/null 2>&1; then
        wedged=1
        log_status "GPU found wedged after process exit (exit_code=$exit_code)."
        snapshot_debug_state "GPU wedged (post-exit)" "$LOGFILE"
    fi

    if [ "$wedged" -eq 1 ]; then
        log_status "GPU is wedged — cannot self-heal (no passwordless sudo). Run 'sudo reboot', then re-launch this script to resume. Stopping watchdog."
        touch "$NEEDS_REBOOT_FLAG"
        exit 1
    fi

    if [ "$exit_code" -eq 0 ] && [ "$stall_time" -lt "$STALL_TIMEOUT" ]; then
        log_status "shard_align.py completed successfully (exit_code=0). Watchdog done."
        exit 0
    fi

    log_status "Resumable failure (exit_code=$exit_code, stalled=$([ "$stall_time" -ge "$STALL_TIMEOUT" ] && echo yes || echo no)) — restarting in 10s..."
    sleep 10
done
