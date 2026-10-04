#!/usr/bin/env python3
"""Hang detection for supervised run_gpu_redo.py workers.

run_gpu_redo.py rewrites its --inflight-marker file right before every
chapter it aligns and touches <marker>.hb at every chapter and group it
reaches (skipped ones too), so the newest of the two is how long the worker
has gone without progress. A normal chapter takes seconds; even Psalm 119 takes a few minutes.
If a marker stays unchanged for longer than --stale-minutes, the worker is
treated as hung (a GPU wedge, a stuck download, a deadlock): this script
sends that worker SIGUSR1 (no core dump of a multi-GB process). The
supervisor (run_gpu_redo_supervisor.py) handles that exit like a native crash -- it quarantines the in-flight
chapter and restarts the worker, which resumes where it stopped.

If the same worker is restarted for a hang --max-restarts times within an
hour, the cause is probably not one bad chapter (e.g. the GPU itself is
wedged): the watcher stops intervening and only logs, so a human can look.

Usage (one watcher for all workers):
    python tools/redo_hang_watch.py \\
        --marker _runs/verse_only_quarantine_a.inflight.json \\
        --marker _runs/verse_only_quarantine_b.inflight.json
"""
import argparse
import json
import os
import signal
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def worker_pid(marker: Path) -> int | None:
    """PID of the run_gpu_redo.py process writing this marker."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in argv]
        if any(a.endswith("run_gpu_redo.py") for a in args) and "--inflight-marker" in args:
            i = args.index("--inflight-marker")
            if i + 1 < len(args):
                try:
                    cwd = Path(os.readlink(entry / "cwd"))
                except OSError:
                    continue
                if (cwd / args[i + 1]).resolve() == marker.resolve():
                    return int(entry.name)
    return None


def process_start_epoch(pid: int) -> float:
    """Wall-clock start of a process (from /proc/<pid>/stat, not the mtime of
    /proc/<pid>, which is not the start time)."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    start_ticks = int(fields[19])                       # field 22 of stat
    boot = next(float(l.split()[1]) for l in Path("/proc/stat").read_text().splitlines() if l.startswith("btime"))
    return boot + start_ticks / os.sysconf("SC_CLK_TCK")


def progress_age_minutes(marker: Path, process_started: float, now: float) -> float:
    """Minutes since the worker last showed progress: the newest of its
    in-flight marker (rewritten per aligned chapter), its heartbeat file
    <marker>.hb (touched for every chapter and group it reaches, skipped ones
    too -- without it a long run of skips looked like a hang), and the process
    start (a fresh worker has not written either yet). The marker may not
    exist: the worker removes it between chapters."""
    beats = [process_started]
    for f in (marker, Path(str(marker) + ".hb")):
        if f.exists():
            beats.append(f.stat().st_mtime)
    return (now - max(beats)) / 60


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marker", action="append", required=True, help="A worker's --inflight-marker path")
    ap.add_argument("--stale-minutes", type=float, default=30.0)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--max-restarts", type=int, default=3, help="Per worker, within one hour")
    args = ap.parse_args()

    markers = [Path(m) for m in args.marker]
    restarts: dict[Path, list[float]] = {m: [] for m in markers}
    log(f"Watching {len(markers)} worker(s), stale after {args.stale_minutes:g} min")

    while True:
        now = time.time()
        for m in markers:
            pid = worker_pid(m)
            if pid is None:
                continue
            # A freshly started worker has not rewritten the marker yet; age
            # it from the later of the marker and the process start.
            age_min = progress_age_minutes(m, process_start_epoch(pid), now)
            if age_min < args.stale_minutes:
                continue
            restarts[m] = [t for t in restarts[m] if now - t < 3600]
            try:
                chapter = json.loads(m.read_text()).get("timing_path", "none (not aligning a chapter)")
            except (OSError, json.JSONDecodeError):
                chapter = "none (not aligning a chapter)"
            if len(restarts[m]) >= args.max_restarts:
                log(f"HUNG again ({age_min:.0f} min) pid {pid} on {chapter} -- already restarted "
                    f"{len(restarts[m])}x this hour, NOT intervening; check the GPU (nvidia-smi)")
                continue
            log(f"HUNG: pid {pid}, no new chapter for {age_min:.1f} min (on {chapter}) -- "
                f"sending SIGUSR1 so the supervisor quarantines it and restarts the worker")
            try:
                os.kill(pid, signal.SIGUSR1)
                restarts[m].append(now)
            except ProcessLookupError:
                pass
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
