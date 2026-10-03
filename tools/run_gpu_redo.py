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
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent))

from align_pipeline import needs_run  # noqa: E402
from align_verse_words import ALIGNMENT_METHOD, process_chapter_verse_only  # noqa: E402
from chapter_state import alignment_method, has_real_timing  # noqa: E402
import shutil  # noqa: E402
from mms_align_words import load_mms_model, select_device  # noqa: E402
from whisper_transcribe import discover_chapter_files  # noqa: E402
from download_language_content import ensure_chapter_ready  # noqa: E402
from text_processing import load_language_config  # noqa: E402
from purge_aligned_audio import purge_iso_audio  # noqa: E402
from hw_config import load_hw_config  # noqa: E402
import mms_align_words  # noqa: E402

OUTPUT_DIR = Path("export/timing-data")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=str, default="_runs/gpu_redo_final_report.json")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N chapters (smoke test)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--no-download", action="store_true",
                         help="Skip ensure_chapter_ready() entirely -- only process chapters whose "
                             "audio is ALREADY local, under whatever fileset happens to be there. "
                             "discover_chapter_files() never downloads on its own, so this guarantees "
                             "zero new fetches; a chapter with no local audio at all is just skipped. "
                             "Use when the --report's whole point is 'audio already downloaded, no "
                             "more fetching' (e.g. sparse-list editions we don't want expanded).")
    parser.add_argument("--keep-fusion-mode-audio", action="store_true",
                         help="Don't purge audio for fusion-mode (not verse_only_mode) isos -- this "
                             "job only ever writes MMS-only output (process_chapter_verse_only()), "
                             "never Whisper, so a fusion-mode chapter still needs a later Whisper "
                             "backfill pass before it's truly done. Purging right after MMS for those "
                             "means re-downloading the same audio again when that pass eventually "
                             "runs -- confirmed 2026-10-01: a real, avoidable re-fetch cost, not just "
                             "a theoretical one. verse_only_mode isos are unaffected -- MMS-only genuinely "
                             "is their final state, so purging them immediately is still correct.")
    parser.add_argument("--inflight-marker", type=str, default=None,
                         help="Path to overwrite with {timing_path, iso, distinct_id, book, chapter} "
                             "before each chapter's actual alignment call -- lets a supervisor process "
                             "identify which chapter was running if this process dies to a signal "
                             "(SIGSEGV from a native library can't be caught or logged from Python). "
                             "See tools/run_gpu_redo_supervisor.py.")
    parser.add_argument("--exclude-chapters-file", type=str, default=None,
                         help="JSON file with a top-level 'chapters' list of {'path': ...} entries "
                             "(same shape as --report) to skip entirely -- e.g. chapters a previous "
                             "supervised run already confirmed crash this process. Resumable for free "
                             "via needs_run() otherwise, but a chapter that crashes BEFORE writing any "
                             "output needs an explicit exclude or it would be retried forever.")
    parser.add_argument("--redo-older-method", action="store_true",
                         help="Also re-align chapters that already have real timing, unless their "
                             "quality file is tagged with the current ALIGNMENT_METHOD "
                             "(align_verse_words.py). Without this, existing output is skipped.")
    parser.add_argument("--backup-dir", type=str, default=None,
                         help="Before overwriting a chapter's existing _timing/_words/_words_quality "
                             "files, copy them here (same relative layout under export/timing-data). "
                             "An existing backup is never overwritten, so re-runs keep the original.")
    parser.add_argument("--prefetch-groups", type=int, default=3,
                         help="Download this many upcoming (iso/canon/distinct_id/book) groups in a "
                             "background thread while the current one aligns, so downloading and GPU "
                             "work overlap instead of alternating. 0 = old sequential behaviour.")
    args = parser.parse_args()

    # Same machine tuning align_pipeline.py applies (conf/hw.local.json) --
    # without it this job ran with the built-in 2-minute model passes and
    # took ~6.5 GB of an 8 GB card shared with the Whisper backfill.
    hw = load_hw_config()
    if hw.get("mms_cpu"):
        mms_align_words._MMS_FORCE_CPU = True
    if hw.get("mms_chunk_minutes") is not None:
        mms_align_words._MAX_CHUNK_SAMPLES = int(hw["mms_chunk_minutes"] * 60 * 16000)
        log(f"MMS chunk size {hw['mms_chunk_minutes']} min (conf/hw.local.json)")
    if hw.get("ctc_chunk_threshold_cells") is not None:
        mms_align_words._CTC_CHUNK_THRESHOLD_CELLS = hw["ctc_chunk_threshold_cells"]
    if args.device is None and hw.get("mms_device"):
        args.device = hw["mms_device"]

    redo = json.loads(Path(args.report).read_text())
    chapters = redo["chapters"]
    if args.exclude_chapters_file:
        exclude_data = json.loads(Path(args.exclude_chapters_file).read_text())
        excluded = {e["path"] if isinstance(e, dict) else e for e in exclude_data["chapters"]}
        before = len(chapters)
        chapters = [c for c in chapters if (c["path"] if isinstance(c, dict) else c) not in excluded]
        if before - len(chapters) > 0:
            log(f"Excluded {before - len(chapters)} chapter(s) via --exclude-chapters-file")
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

    # Report order is the processing order (callers put priority work first).
    group_items = list(groups.items())

    # One background thread (not several: download_job() shares module-level
    # stats and catalog caches) fetching the next few groups while the GPU
    # works on the current one.
    prefetch = None
    pending: dict[int, object] = {}
    if not args.no_download and args.prefetch_groups > 0:
        prefetch = ThreadPoolExecutor(max_workers=1)

    def _fetch_group(idx):
        (f_iso, f_canon, f_did, f_book), f_chapters = group_items[idx]
        for f_ch in sorted(f_chapters):
            try:
                ensure_chapter_ready(f_iso, f_canon, f_did, f_book, f_ch)
            except Exception as e:  # a failed fetch just means discovery finds nothing below
                log(f"{f_iso}/{f_did} {f_book} {f_ch}: fetch failed ({e})")

    for gi, ((iso, canon, distinct_id, book), chapter_nums) in enumerate(group_items):
        if prefetch is not None:
            for ahead in range(gi, min(gi + 1 + args.prefetch_groups, len(group_items))):
                if ahead not in pending:
                    pending[ahead] = prefetch.submit(_fetch_group, ahead)
        if iso not in config_cache:
            try:
                config_cache[iso] = load_language_config(iso)
            except Exception:
                config_cache[iso] = load_language_config("default")
        config = config_cache[iso]

        if prefetch is not None:
            pending.pop(gi).result()
        elif not args.no_download:
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

            if args.redo_older_method and has_real_timing(timing_path):
                if alignment_method(quality_path) == ALIGNMENT_METHOD:
                    total_skipped += 1
                    # Already redone (e.g. before a restart) but the prefetch
                    # just downloaded its audio again -- nothing else will
                    # purge it if this whole language is skipped.
                    if prefetch is not None and (getattr(config, "verse_only_mode", False) or not args.keep_fusion_mode_audio):
                        Path(chapter["audio_path"]).unlink(missing_ok=True)
                    continue
            elif not needs_run(timing_path, force=False):
                total_skipped += 1
                continue

            if args.backup_dir:
                for src in (timing_path, words_path, quality_path):
                    if src.exists():
                        dst = Path(args.backup_dir) / src.relative_to(OUTPUT_DIR)
                        if not dst.exists():
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)

            item = {
                "book": book, "chapter": ch_num, "chapter_str": chapter_str,
                "audio_path": chapter["audio_path"], "text_path": chapter["text_path"],
                "timing_path": timing_path, "words_path": words_path, "quality_path": quality_path,
            }
            # Overwritten per chapter (never appended) -- the supervisor's
            # only way to know which chapter was in-flight if this process
            # dies to a SIGSEGV, which bypasses Python's exception handling
            # entirely and can't be logged from inside the try/except below.
            # See tools/run_gpu_redo_supervisor.py.
            if args.inflight_marker:
                Path(args.inflight_marker).write_text(json.dumps(
                    {"timing_path": str(timing_path), "iso": iso, "distinct_id": distinct_id,
                     "book": book, "chapter": ch_num}))
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
            # With prefetch, a big language's purge can be deferred for many
            # groups (see the purge block below); drop this chapter's audio
            # now so disk use stays bounded by the prefetch window.
            if prefetch is not None and (getattr(config, "verse_only_mode", False) or not args.keep_fusion_mode_audio):
                Path(chapter["audio_path"]).unlink(missing_ok=True)

        if (gi + 1) % 50 == 0 or gi == len(group_items) - 1:
            log(f"... {gi + 1}/{len(group_items)} groups done "
                f"(ok={total_ok}, failed={total_failed}, skipped={total_skipped})")
            # Never purge a language whose upcoming groups are already
            # prefetched: their chapters still carry old output, so the purge
            # would treat the fresh audio as "already aligned" and delete it.
            # Those languages are purged at a later purge point instead.
            prefetched_isos = {group_items[k][0][0] for k in pending}
            deferred_purge = isos_touched_since_purge & prefetched_isos
            for iso in isos_touched_since_purge - deferred_purge:
                if args.keep_fusion_mode_audio and not getattr(config_cache.get(iso), "verse_only_mode", False):
                    log(f"  keeping audio for {iso} (fusion-mode, still needs a later Whisper pass)")
                    continue
                try:
                    deleted, freed = purge_iso_audio(iso)
                    if deleted:
                        log(f"  purged {deleted} now-aligned mp3(s) for {iso}, {freed / 1e9:.2f} GB freed")
                except Exception as e:
                    log(f"  purge failed for {iso} (non-fatal): {e}")
            isos_touched_since_purge = deferred_purge

    if prefetch is not None:
        prefetch.shutdown(wait=True)
    log(f"DONE. ok={total_ok} failed={total_failed} skipped={total_skipped}")


if __name__ == "__main__":
    main()
