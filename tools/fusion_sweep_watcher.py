#!/usr/bin/env python3
"""Chain a fusion-only pass onto tools/whisper_backfill.py's per-language
completions, so a dispute-free language's newly-backfilled Whisper
transcript actually gets combined with its existing MMS output into
published-ready final timing -- not just sitting as unused raw material.

Why this exists: whisper_backfill.py only ever calls
write_whisper_words_json() -- it writes the RAW Whisper transcript and
nothing else. The existing _timing.json/_words.json still reflect the
OLD MMS-only alignment until a separate fusion step (align_words.py's
process_chapter(), invoked here via align_pipeline.py's
--skip-whisper --skip-mms) combines the two. Confirmed 2026-10-01: a
language finishing in the backfill is NOT automatically publish-complete
without this.

Safe to run continuously, unlike the old arbiter-sweep-watcher this is
modeled on: fusion only ever combines MMS + Whisper for the SAME chapters
the backfill just touched -- no DBT involved, so there's no way for this
to reach into the deferred dispute-relevant population (fusion itself is
also CPU-only, no GPU model inference, so no contention risk with the
other running GPU jobs either).

Usage:
    python tools/fusion_sweep_watcher.py \
        --backfill-report _runs/whisper_backfill_disputefree_report.json \
        --backfill-pid 1387470 \
        --also-now abq,aca,ach,acu,ade,agx,alt,aly,aoz
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPORT_DIR = Path("_runs/fusion_sweep")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {"processed": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def parse_pipeline_summary(stdout: str) -> dict:
    out = {}
    patterns = {
        "fused": r"fused=(\d+)",
        "skipped_fresh": r"skipped_fresh=(\d+)",
        "skipped_no_whisper": r"skipped_no_whisper=(\d+)",
        "skipped_no_text": r"skipped_no_text=(\d+)",
        "failed": r"failed=(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, stdout)
        if m:
            out[key] = int(m.group(1))
    return out


def fuse_iso(iso: str) -> dict:
    result = {"iso": iso, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # Not align_pipeline.py --skip-whisper --skip-mms: its chapter discovery
    # globs for *.mp3, which this population's audio was already purged
    # long before whisper_backfill.py ever ran, so it finds zero chapters.
    # tools/fusion_only_redo.py discovers from the existing _timing.json/
    # whisper_words.json files directly instead -- confirmed 2026-10-01.
    log(f"{iso}: running fusion-only pass (tools/fusion_only_redo.py) ...")
    try:
        p = subprocess.run(
            [".venv/bin/python", "tools/fusion_only_redo.py", "--iso", iso],
            capture_output=True, text=True, timeout=3600,
        )
        result["returncode"] = p.returncode
        summary = parse_pipeline_summary(p.stdout)
        result["summary"] = summary
        (REPORT_DIR / f"{iso}.log").write_text(p.stdout + "\n" + p.stderr)
        if p.returncode != 0:
            log(f"{iso}: FAILED (exit {p.returncode}) -- see {REPORT_DIR / f'{iso}.log'}")
        else:
            log(f"{iso}: {p.stdout.strip().splitlines()[-1] if p.stdout.strip() else 'done'}")
    except Exception as e:
        log(f"{iso}: EXCEPTION ({e})")
        result["error"] = str(e)
    result["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return result


def backfill_process_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        import os
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backfill-report", type=str, default="_runs/whisper_backfill_disputefree_report.json")
    parser.add_argument("--backfill-pid", type=int, default=None)
    parser.add_argument("--state", type=str, default="_runs/fusion_sweep_watcher_state.json")
    parser.add_argument("--also-now", type=str, default=None,
                         help="Comma-separated isos to fuse immediately (catch-up for languages "
                              "already finished in the backfill before this watcher started)")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    state_path = Path(args.state)
    state = load_state(state_path)

    if args.also_now:
        for iso in [c.strip() for c in args.also_now.split(",")]:
            if iso in state["processed"]:
                log(f"{iso}: already fused, skipping (--also-now)")
                continue
            result = fuse_iso(iso)
            state["processed"][iso] = result
            save_state(state_path, state)

    report_path = Path(args.backfill_report)
    log(f"Watching {report_path} (backfill pid={args.backfill_pid}), polling every {args.poll_seconds}s "
        f"-- fusing each newly-completed language")

    while True:
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError):
                report = None
            if report:
                for iso, info in report.get("isos", {}).items():
                    if iso in state["processed"]:
                        continue
                    if info.get("ok", 0) == 0:
                        state["processed"][iso] = {"skipped": "no ok chapters", "backfill_info": info}
                        save_state(state_path, state)
                        continue
                    log(f"New completed iso from backfill: {iso} "
                        f"(ok={info.get('ok')} flagged={info.get('flagged')} aborted={info.get('aborted')})")
                    result = fuse_iso(iso)
                    result["backfill_info"] = info
                    state["processed"][iso] = result
                    save_state(state_path, state)

        backfill_done = not backfill_process_alive(args.backfill_pid)
        if backfill_done:
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text())
                    pending = [iso for iso in report.get("isos", {}) if iso not in state["processed"]]
                except (OSError, json.JSONDecodeError):
                    pending = []
            else:
                pending = []
            if not pending:
                log("Backfill process has exited and every reported iso has been fused -- stopping.")
                break
            log(f"Backfill process exited but {len(pending)} iso(s) still pending fusion: {pending}")

        time.sleep(args.poll_seconds)

    log(f"DONE. {len(state['processed'])} isos fused total.")


if __name__ == "__main__":
    main()
