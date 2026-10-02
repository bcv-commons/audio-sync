#!/usr/bin/env python3
"""Supervise tools/run_gpu_redo.py across the kind of crash Python can't
catch or recover from on its own: a SIGSEGV inside a native library
(confirmed 2026-10-02, root-caused via PYTHONFAULTHANDLER=1 to a crash
inside torchaudio's forced_align() CUDA/CTC kernel, reached via
mms_align_words.py's gap-fill retry path -- a signal handler bypasses
Python's try/except entirely, so the parent process has no way to skip
just the one bad chapter and keep going on its own).

Confirmed NOT caused by VRAM exhaustion (no CUDA allocator warnings on the
crashing runs) or by plain GPU/CPU resource contention from the
lexeme-aligner sibling project (an isolated single-chapter run of the
exact same code succeeded; a 48-group reproduction containing that same
chapter crashed every time) -- most consistent with a flaky native
extension under concurrent multi-process GPU use. Given a segfault can't
be caught, the practical fix is process-level: run run_gpu_redo.py as a
subprocess, and if it dies to SIGSEGV, read the --inflight-marker file it
was overwriting before each chapter to learn exactly which chapter it was
running, quarantine THAT ONE chapter (append to an exclude-file,
--exclude-chapters-file on the next attempt), and relaunch -- the rest of
the population resumes for free via needs_run(), so no progress is lost
beyond the one bad chapter.

Usage:
    python tools/run_gpu_redo_supervisor.py \
        --report _runs/never_aligned_verse_only_subset.json \
        --quarantine _runs/verse_only_quarantine.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [supervisor] {msg}", flush=True)


def load_quarantine(path: Path) -> list[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())["chapters"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    return []


def save_quarantine(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"chapters": entries}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=str, required=True)
    parser.add_argument("--quarantine", type=str, required=True,
                         help="Persistent exclude-list of chapters confirmed to crash the worker. "
                              "Grows across restarts; never cleared automatically.")
    parser.add_argument("--inflight-marker", type=str, default=None,
                         help="Default: <quarantine>.inflight.json next to --quarantine")
    parser.add_argument("--max-consecutive-crashes", type=int, default=20,
                         help="Stop (not loop forever) if this many crashes happen with zero new "
                              "chapters successfully written in between -- signals something more "
                              "systemic than a handful of individually-bad chapters.")
    parser.add_argument("--extra-args", type=str, default="",
                         help="Extra args forwarded verbatim to run_gpu_redo.py, space-separated "
                              "(e.g. '--no-download --keep-fusion-mode-audio')")
    args = parser.parse_args()

    quarantine_path = Path(args.quarantine)
    inflight_path = Path(args.inflight_marker) if args.inflight_marker else quarantine_path.with_suffix(".inflight.json")

    quarantined = load_quarantine(quarantine_path)
    consecutive_crashes = 0
    attempt = 0

    while True:
        attempt += 1
        if inflight_path.exists():
            inflight_path.unlink()

        cmd = [
            ".venv/bin/python", "tools/run_gpu_redo.py",
            "--report", args.report,
            "--inflight-marker", str(inflight_path),
        ]
        if quarantined:
            save_quarantine(quarantine_path, quarantined)
            cmd += ["--exclude-chapters-file", str(quarantine_path)]
        if args.extra_args:
            cmd += args.extra_args.split()

        log(f"Attempt {attempt}: launching ({len(quarantined)} chapter(s) quarantined so far) ...")
        result = subprocess.run(cmd, cwd=Path.cwd())

        if result.returncode == 0:
            log(f"Completed successfully after {attempt} attempt(s), "
                f"{len(quarantined)} chapter(s) permanently quarantined.")
            break

        if result.returncode == -11 or result.returncode == 139:
            if inflight_path.exists():
                try:
                    bad = json.loads(inflight_path.read_text())
                    quarantined.append({"path": bad["timing_path"], "reason": "SIGSEGV"})
                    save_quarantine(quarantine_path, quarantined)
                    log(f"SIGSEGV -- quarantining {bad['timing_path']} "
                        f"({bad['iso']}/{bad['distinct_id']} {bad['book']} {bad['chapter']})")
                    consecutive_crashes = 0  # made real progress: identified a new bad chapter
                except (OSError, json.JSONDecodeError, KeyError) as e:
                    log(f"SIGSEGV but couldn't read inflight marker ({e}) -- "
                        f"can't identify which chapter crashed, nothing to quarantine this round")
                    consecutive_crashes += 1
            else:
                log("SIGSEGV but no inflight marker found -- crashed before reaching any real chapter?")
                consecutive_crashes += 1
        else:
            log(f"Exited with code {result.returncode} (not a segfault) -- treating as a real failure, "
                f"not retrying blindly")
            consecutive_crashes += 1

        if consecutive_crashes >= args.max_consecutive_crashes:
            log(f"{consecutive_crashes} consecutive crashes with no new chapter identified -- "
                f"stopping rather than looping forever. Check manually.")
            sys.exit(1)

        time.sleep(5)


if __name__ == "__main__":
    main()
