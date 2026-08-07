#!/usr/bin/env bash
#
# Unattended watchdog wrapper around align_pipeline.py.
#
# Runs the pipeline, monitors its log output for stalls (Python's own
# STEP_TIMEOUT_SECONDS/signal.alarm guard can't interrupt a hang inside a
# blocking CUDA/C-extension call, so this exists as an OS-level kill from
# outside the process instead), kills+restarts on stall or non-wedging
# crash, and stops cleanly with a flag file if the GPU actually wedges
# (that needs `sudo reboot` — this script has no way to run that itself).
#
# Since align_pipeline.py always skips already-completed chapters, killing
# and restarting mid-run is safe — it just resumes where it left off.
#
# Usage:
#   scripts/watchdog-align.sh "--iso-list fra,arb,hin,cmn,ben,kor,tur,uzn --books ALL"
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

ARGS="${1:?Usage: $0 \"<align_pipeline.py args>\"}"
STALL_TIMEOUT=300      # seconds with no new log output = considered hung
CHECK_INTERVAL=15
LOG_DIR="_runs/logs"
STATUS_FILE="$LOG_DIR/watchdog-status.txt"
SNAPSHOT_FILE="$LOG_DIR/watchdog-debug-snapshots.txt"
GPU_MON_FILE="$LOG_DIR/watchdog-gpu-monitor.csv"
NEEDS_REBOOT_FLAG="$LOG_DIR/watchdog-needs-reboot.flag"

mkdir -p "$LOG_DIR"
rm -f "$NEEDS_REBOOT_FLAG"

log_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$STATUS_FILE"
}

snapshot_debug_state() {
    local reason="$1"
    local logfile="$2"
    {
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') — $reason ====="
        echo "--- last 40 log lines ($logfile) ---"
        tail -40 "$logfile" 2>/dev/null
        echo "--- nvidia-smi ---"
        nvidia-smi 2>&1
        echo "===== end snapshot ====="
        echo
    } >> "$SNAPSHOT_FILE"
}

# Continuous GPU telemetry for the whole watchdog session (survives across
# restarts) — same fields used for the manual crash diagnosis earlier today.
nohup nvidia-smi --query-gpu=timestamp,memory.used,memory.free,memory.reserved,utilization.gpu,utilization.memory,temperature.gpu,power.draw,pstate,clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown \
    --format=csv -l 2 > "$GPU_MON_FILE" 2>&1 < /dev/null &
GPU_MON_PID=$!
disown

cleanup() {
    kill "$GPU_MON_PID" 2>/dev/null
}
trap cleanup EXIT

log_status "Watchdog started. ARGS: $ARGS"

attempt=0
while true; do
    attempt=$((attempt + 1))
    LOGFILE="$LOG_DIR/watchdog-run-$(date +%Y%m%dT%H%M%S).log"
    log_status "Attempt $attempt: launching align_pipeline.py $ARGS -> $LOGFILE"

    # shellcheck disable=SC2086
    PYTHONUNBUFFERED=1 .venv/bin/python pipeline/align_pipeline.py $ARGS > "$LOGFILE" 2>&1 < /dev/null &
    PID=$!

    last_size=0
    stall_time=0
    wedged=0

    while kill -0 "$PID" 2>/dev/null; do
        sleep "$CHECK_INTERVAL"

        # Check for GPU wedge *while still running*, not just after exit —
        # catches it as early as possible.
        if ! nvidia-smi > /dev/null 2>&1; then
            log_status "GPU WEDGED while pipeline still running (PID $PID)."
            snapshot_debug_state "GPU wedged (mid-run)" "$LOGFILE"
            kill -9 "$PID" 2>/dev/null
            wedged=1
            break
        fi

        cur_size=$(stat -c %s "$LOGFILE" 2>/dev/null || echo 0)
        if [ "$cur_size" = "$last_size" ]; then
            stall_time=$((stall_time + CHECK_INTERVAL))
        else
            stall_time=0
        fi
        last_size=$cur_size

        if [ "$stall_time" -ge "$STALL_TIMEOUT" ]; then
            log_status "STALL detected — no log output for ${STALL_TIMEOUT}s (PID $PID). Killing."
            snapshot_debug_state "Stall (${STALL_TIMEOUT}s no output)" "$LOGFILE"
            kill -9 "$PID" 2>/dev/null
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
        log_status "Pipeline completed successfully (exit_code=0). Watchdog done."
        exit 0
    fi

    log_status "Resumable failure (exit_code=$exit_code, stalled=$([ "$stall_time" -ge "$STALL_TIMEOUT" ] && echo yes || echo no)) — restarting in 10s..."
    sleep 10
done
