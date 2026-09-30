#!/usr/bin/env python3
"""Actually re-align every chapter listed in _runs/gpu_redo_final_report.json
(the output of tools/prepare_gpu_redo.py --apply) with today's improved
code -- vowel-count window anchoring + same-run pacing cross-check in
pipeline/align_verse_words.py's verse_anchored_align().

Loads the MMS_FA model exactly ONCE (it's torchaudio's universal
multilingual forced-alignment checkpoint -- no per-language reload needed)
and reuses it across every chapter, every language, for the whole run --
per-chapter CLI invocation would waste ~7-9s per chapter on model loading
alone, which at ~34,000 chapters is 65+ hours of pure overhead.

Disk safety: given only ~12GB free and this population's audio was mostly
already purged (that's WHY these chapters need re-download at all),
purges each edition's newly-downloaded mp3s immediately after that
edition's chapters are done (reusing tools/purge_aligned_audio.py's own
purge_iso_audio(), which only ever deletes an mp3 whose _timing.json +
_words.json both already exist) -- keeps disk usage bounded across the
whole run instead of accumulating everything before any cleanup.

Crash safety: resumable for free. Every chapter's own needs_run() check
(does _timing.json already exist?) means re-running this exact script
after an interruption just skips everything already completed and picks
up where it left off -- no separate checkpoint file needed.

Usage:
    python tools/run_gpu_redo.py
    python tools/run_gpu_redo.py --limit 500   # smoke-test a subset first
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent))

from align_pipeline import needs_run  # noqa: E402
from align_verse_words import process_chapter_verse_only  # noqa: E402
from mms_align_words import load_mms_model, select_device  # noqa: E402
from whisper_transcribe import discover_chapter_files  # noqa: E402
from download_language_content import ensure_chapter_ready  # noqa: E402
from text_processing import load_language_config  # noqa: E402
from purge_aligned_audio import purge_iso_audio  # noqa: E402

OUTPUT_DIR = Path("export/timing-data")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=str, default="_runs/gpu_redo_final_report.json")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N chapters (smoke test)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    redo = json.loads(Path(args.report).read_text())
    chapters = redo["chapters"]
    if args.limit:
        chapters = chapters[: args.limit]

    # Group by (iso, canon, distinct_id, book) -> sorted chapter numbers,
    # parsed back out of each entry's own timing-file path rather than
    # trusting a separate field, since prepare_gpu_redo.py's report only
    # stores {"path": ..., "reason": ...}.
    groups = defaultdict(set)
    for entry in chapters:
        p = Path(entry["path"])
        book = p.parent.name
        distinct_id = p.parent.parent.name
        iso = p.parent.parent.parent.name
        canon = p.parent.parent.parent.parent.name
        stem = p.name.replace("_timing.json", "")
        chapter_str = stem.split("_", 2)[1]
        groups[(iso, canon, distinct_id, book)].add(int(chapter_str))

    log(f"Loaded {len(chapters)} chapters across {len(groups)} (iso/canon/distinct_id/book) groups")

    bundle, model, tokenizer, aligner, uroman = load_mms_model(select_device(args.device))

    config_cache = {}
    total_ok = 0
    total_failed = 0
    total_skipped = 0
    isos_touched_since_purge = set()

    group_items = sorted(groups.items())
    for gi, ((iso, canon, distinct_id, book), chapter_nums) in enumerate(group_items):
        if iso not in config_cache:
            try:
                config_cache[iso] = load_language_config(iso)
            except Exception:
                config_cache[iso] = load_language_config("default")
        config = config_cache[iso]

        for ch in sorted(chapter_nums):
            ensure_chapter_ready(iso, canon, distinct_id, book, ch)

        required = {book: chapter_nums}
        found, _skipped = discover_chapter_files(iso, canon, distinct_id, OUTPUT_DIR, force=True, required_chapters=required)
        if not found:
            log(f"{iso}/{distinct_id} {book} {sorted(chapter_nums)}: no chapters discovered after fetch attempt -- skipping")
            total_skipped += len(chapter_nums)
            continue

        for chapter in found:
            ch_num = chapter["chapter"]
            chapter_str = chapter["chapter_str"]
            audio_fileset = chapter["audio_fileset"]
            out_book_dir = OUTPUT_DIR / canon / iso / distinct_id / book
            timing_path = out_book_dir / f"{book}_{chapter_str}_{audio_fileset}_timing.json"
            words_path = out_book_dir / f"{book}_{chapter_str}_{audio_fileset}_words.json"
            quality_path = Path(str(words_path).replace("_words.json", "_words_quality.json"))

            if not needs_run(timing_path, force=False):
                total_skipped += 1
                continue

            item = {
                "book": book, "chapter": ch_num, "chapter_str": chapter_str,
                "audio_path": chapter["audio_path"], "text_path": chapter["text_path"],
                "timing_path": timing_path, "words_path": words_path, "quality_path": quality_path,
            }
            try:
                stats = process_chapter_verse_only(item, bundle, model, tokenizer, aligner, uroman, config)
            except Exception as e:
                log(f"{iso}/{distinct_id} {book} {ch_num}: EXCEPTION {e}")
                total_failed += 1
                continue
            if "error" in stats:
                log(f"{iso}/{distinct_id} {book} {ch_num}: ERROR {stats['error']}")
                total_failed += 1
                continue
            total_ok += 1
            isos_touched_since_purge.add(iso)

        if (gi + 1) % 50 == 0 or gi == len(group_items) - 1:
            log(f"... {gi + 1}/{len(group_items)} groups done "
                f"(ok={total_ok}, failed={total_failed}, skipped={total_skipped})")
            for iso in isos_touched_since_purge:
                try:
                    deleted, freed = purge_iso_audio(iso)
                    if deleted:
                        log(f"  purged {deleted} now-aligned mp3(s) for {iso}, {freed / 1e9:.2f} GB freed")
                except Exception as e:
                    log(f"  purge failed for {iso} (non-fatal): {e}")
            isos_touched_since_purge = set()

    log(f"DONE. ok={total_ok} failed={total_failed} skipped={total_skipped}")


if __name__ == "__main__":
    main()
