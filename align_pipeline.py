#!/usr/bin/env python3
"""
Unified Bible audio alignment pipeline.

Runs all three alignment steps in sequence for each chapter:
  Step 1a: Whisper transcription    → *_whisper_words.json  (whisper_transcribe.py)
  Step 1b: MMS forced alignment     → *_mms_words.json      (mms_align_words.py)
  Step 2:  Fusion alignment         → *_timing.json + *_words.json  (align_words.py)

Accepts the same scoping parameters as whisper_transcribe.py: --iso, --iso-list,
--tier, --all, --template, --testament, --force, --book, --chapter.

Supports two modes for selecting chapters:
  --template John    Use Bible references from template .md files
  --books JHN        Process full Bible books (no template needed)

Usage:
    # Process a single language (full pipeline, all templates)
    python align_pipeline.py --iso nld

    # Process only chapters from a specific template
    python align_pipeline.py --iso nld --template John

    # Process full Bible books (no template)
    python align_pipeline.py --iso heb --books JHN
    python align_pipeline.py --iso heb --books JHN,MAT,LUK
    python align_pipeline.py --iso swe --books NT
    python align_pipeline.py --iso heb --books OT
    python align_pipeline.py --iso heb --books ALL
    python align_pipeline.py --iso heb --books GEN:1-3,17

    # Process multiple languages
    python align_pipeline.py --iso-list nld,pol,ell

    # Only NT or OT
    python align_pipeline.py --iso nld --testament nt

    # Check what would be done without doing it
    python align_pipeline.py --iso nld --dry-run

    # Filter to specific chapter
    python align_pipeline.py --iso heb --book GEN --chapter 17

    # Skip individual steps
    python align_pipeline.py --iso heb --skip-whisper
    python align_pipeline.py --iso heb --skip-mms
    python align_pipeline.py --iso heb --skip-fusion

    # Skip auto-download (only process already-downloaded files)
    python align_pipeline.py --iso heb --books JHN --no-download

Prerequisites:
    1. pip install -r requirements-whisper.txt  (for Whisper)
       - Apple Silicon Mac: uses mlx-whisper (Metal GPU)
       - NVIDIA CUDA GPU:   uses faster-whisper (CUDA)
       - Other/CPU:         uses faster-whisper (CPU)
    2. pip install torch torchaudio uroman       (for MMS-FA)
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from text_processing import load_language_config

# ─── Reuse infrastructure from whisper_transcribe.py ─────────────────────

from whisper_transcribe import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    NT_BOOKS,
    OT_BOOKS,
    WORD_TIMING_DIR,
    _USE_MLX,
    _USE_CUDA,
    classify_template_refs,
    load_whisper_model,
    set_whisper_cpu,
    discover_chapter_files,
    download_audio_for_chapters,
    format_duration,
    generate_work_items,
    get_whisper_language,
    load_all_template_refs,
    load_priority_languages,
    refs_to_books_spec,
    resolve_languages,
)

# ─── Logging ─────────────────────────────────────────────────────────────

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ─── Book specification parser ────────────────────────────────────────────

def build_refs_from_books(books_spec: str) -> Dict[str, Set[int]]:
    """Build chapter refs from --books specification.

    Supports:
        JHN           → full book (all chapters)
        JHN,MAT       → multiple full books
        NT / OT / ALL → all books in testament or full Bible
        GEN:1-3       → specific chapter range
        GEN:1-3,17    → range + individual chapters

    Commas separate items; bare numbers attach to the preceding book.
    Example: "JHN,GEN:1-3,17,MAT" → JHN(all), GEN(1,2,3,17), MAT(all)
    """
    spec_upper = books_spec.strip().upper()

    # Handle shorthand for full testament/Bible
    all_books = {**OT_BOOKS, **NT_BOOKS}
    if spec_upper == "NT":
        return {b: set(range(1, c + 1)) for b, c in NT_BOOKS.items()}
    if spec_upper == "OT":
        return {b: set(range(1, c + 1)) for b, c in OT_BOOKS.items()}
    if spec_upper == "ALL":
        return {b: set(range(1, c + 1)) for b, c in all_books.items()}

    refs: Dict[str, Set[int]] = defaultdict(set)
    current_book = None

    for part in books_spec.split(","):
        part = part.strip().upper()
        if not part:
            continue

        if ":" in part:
            # Book with chapter spec: "GEN:1-3" or "GEN:17"
            book, chapter_spec = part.split(":", 1)
            if book in all_books:
                current_book = book
                for range_part in chapter_spec.split("-"):
                    pass  # handled below
                # Parse "1-3" or "17"
                if "-" in chapter_spec:
                    start, end = chapter_spec.split("-", 1)
                    refs[book].update(range(int(start), int(end) + 1))
                else:
                    refs[book].add(int(chapter_spec))
        elif part.isdigit():
            # Bare number: attach to current book
            if current_book:
                refs[current_book].add(int(part))
        elif part in all_books:
            # Full book name
            current_book = part
            refs[part] = set(range(1, all_books[part] + 1))
        else:
            log(f"Unknown book or spec: {part}", "WARN")

    return dict(refs)


# ─── Step runners ────────────────────────────────────────────────────────

def run_whisper_chapter(chapter: dict, model_name: str, whisper_lang: Optional[str],
                        whisper_model=None) -> dict:
    """Run Whisper transcription for a single chapter (Step 1a).

    Pass a pre-loaded faster-whisper model via `whisper_model` to avoid
    reloading it on every chapter call.

    Returns stats dict with keys: duration, transcribe_time, words, segments.
    """
    from whisper_transcribe import process_chapter as whisper_process
    return whisper_process(chapter, model_name, whisper_lang, _whisper_model=whisper_model)


def run_mms_chapter(
    item: dict, bundle, model, tokenizer, aligner, uroman, config,
    header_skip_time: Optional[float] = None,
) -> dict:
    """Run MMS forced alignment for a single chapter (Step 1b).

    Returns stats dict with keys: words, aligned, avg_score, elapsed.
    """
    from mms_align_words import process_chapter as mms_process
    return mms_process(item, bundle, model, tokenizer, aligner, uroman, config,
                       header_skip_time=header_skip_time,
                       whisper_path=item.get("whisper_path"))


def detect_whisper_header(whisper_path: Path, text_path: Path, config) -> Tuple[Optional[float], Optional[str]]:
    """Check Whisper output for a spoken audio header before verse text.

    Returns (verse_start_time, header_text) if header detected, else (None, None).
    """
    import json
    from align_words import detect_audio_header
    from text_processing import strip_markers

    if not whisper_path.exists() or not text_path.exists():
        return None, None

    with open(whisper_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    whisper_words = data.get("words", [])
    if not whisper_words:
        return None, None

    with open(text_path, "r", encoding="utf-8") as f:
        verse_texts = [strip_markers(line.rstrip("\n"), config) for line in f.readlines()]

    return detect_audio_header(whisper_words, verse_texts, config)


def run_fusion_chapter(item: dict, config, mms_components=None) -> dict:
    """Run fusion alignment for a single chapter (Step 2).

    Args:
        mms_components: Optional tuple (bundle, model, tokenizer, aligner, uroman)
            for gap-fill re-alignment during fusion.

    Returns stats dict.
    """
    from align_words import process_chapter as fusion_process
    return fusion_process(item, config, mms_components=mms_components)


# ─── MMS work item builder ──────────────────────────────────────────────

def build_mms_item(chapter: dict, canon: str, iso: str, distinct_id: str) -> dict:
    """Build an mms_align_words.py-compatible work item from a whisper chapter dict."""
    book = chapter["book"]
    chapter_str = chapter["chapter_str"]
    audio_fileset = chapter["audio_fileset"]

    word_book_dir = WORD_TIMING_DIR / canon / iso / distinct_id / book
    mms_path = word_book_dir / f"{book}_{chapter_str}_{audio_fileset}_mms_words.json"
    whisper_path = word_book_dir / f"{book}_{chapter_str}_{audio_fileset}_whisper_words.json"

    return {
        "audio_path": chapter["audio_path"],
        "text_path": chapter["text_path"],
        "mms_path": mms_path,
        "whisper_path": whisper_path if whisper_path.exists() else None,
        "book": book,
        "chapter": chapter["chapter"],
        "chapter_str": chapter_str,
        "canon": canon,
        "iso": iso,
        "distinct_id": distinct_id,
        "audio_fileset": audio_fileset,
    }


# ─── Fusion work item builder ───────────────────────────────────────────

def build_fusion_item(
    chapter: dict, canon: str, iso: str, distinct_id: str,
    output_dir: Path,
) -> dict:
    """Build an align_words.py-compatible work item from a whisper chapter dict.

    For contributions that ship authoritative verse-level _timing.json next
    to the audio (e.g. CSV-derived deu/DEUSOL), the fusion is told to keep
    only the word-level outputs and leave the verse timing untouched.
    """
    book = chapter["book"]
    chapter_str = chapter["chapter_str"]
    audio_fileset = chapter["audio_fileset"]
    word_book_dir = WORD_TIMING_DIR / canon / iso / distinct_id / book

    mms_path = word_book_dir / f"{book}_{chapter_str}_{audio_fileset}_mms_words.json"
    whisper_path = word_book_dir / f"{book}_{chapter_str}_{audio_fileset}_whisper_words.json"

    out_book_dir = output_dir / canon / iso / distinct_id / book
    timing_path = out_book_dir / f"{book}_{chapter_str}_{audio_fileset}_timing.json"
    words_path = out_book_dir / f"{book}_{chapter_str}_{audio_fileset}_words.json"

    # Detect contributor-provided verse timing (lives next to the audio file)
    contrib_timing = (
        chapter["audio_path"].parent
        / f"{book}_{chapter_str}_{audio_fileset}_timing.json"
    )
    preserve_existing_timing = (
        contrib_timing.exists()
        and "contrib" in chapter["audio_path"].parts
    )

    return {
        "mms_path": mms_path if mms_path.exists() else None,
        "whisper_path": whisper_path if whisper_path.exists() else None,
        "ref_text_path": chapter["text_path"],
        "timing_path": timing_path,
        "words_path": words_path,
        "preserve_existing_timing": preserve_existing_timing,
        "book": book,
        "chapter": chapter["chapter"],
        "chapter_str": chapter_str,
        "canon": canon,
        "iso": iso,
        "distinct_id": distinct_id,
        "audio_fileset": audio_fileset,
    }


# ─── Main Pipeline ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified Bible audio alignment pipeline (Whisper + MMS + Fusion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --iso nld
  %(prog)s --iso-list nld,pol,ell
  %(prog)s --tier 1
  %(prog)s --all
  %(prog)s --iso nld --template John
  %(prog)s --iso nld --testament nt
  %(prog)s --iso nld --dry-run
  %(prog)s --iso heb --book GEN --chapter 17
        """,
    )

    # Language selection
    lang_group = parser.add_argument_group("Language selection (at least one required)")
    lang_group.add_argument("--iso", type=str, help="Single language ISO 639-3 code (e.g., nld)")
    lang_group.add_argument("--iso-list", type=str, help="Comma-separated ISO codes (e.g., nld,pol,ell)")
    lang_group.add_argument("--tier", type=int, choices=[1, 2, 3], help="Process all languages of this tier")
    lang_group.add_argument("--all", action="store_true", help="Process all priority languages")

    # Scope
    scope_group = parser.add_argument_group("Scope")
    scope_group.add_argument(
        "--template", type=str,
        help="Only process chapters from this template (e.g., John, OBS). Default: all templates",
    )
    scope_group.add_argument(
        "--books", type=str, default=None,
        help="Process full books instead of templates (e.g. JHN, JHN,MAT, NT, OT, ALL, GEN:1-3,17)",
    )
    scope_group.add_argument(
        "--testament", type=str, choices=["nt", "ot", "both"], default="both",
        help="Which testament to process (default: both)",
    )
    scope_group.add_argument("--book", type=str, default=None, help="Filter to a specific book (e.g. GEN)")
    scope_group.add_argument("--chapter", type=int, default=None, help="Filter to a specific chapter number")

    # Whisper options
    whisper_group = parser.add_argument_group("Whisper options")
    whisper_group.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Whisper model (default: {DEFAULT_MODEL})",
    )
    whisper_group.add_argument(
        "--whisper-cpu", action="store_true",
        help="Force faster-whisper to run on CPU even when a CUDA GPU is available "
             "(useful when GPU RAM is needed by the MMS model). "
             "No effect on Apple Silicon (mlx-whisper always uses Metal).",
    )

    # MMS options
    mms_group = parser.add_argument_group("MMS options")
    mms_group.add_argument(
        "--mms-cpu", action="store_true",
        help="Force MMS to run on CPU even when a CUDA GPU is available. "
             "Useful on GPUs with limited VRAM shared with a desktop environment.",
    )
    mms_group.add_argument(
        "--mms-chunk-minutes", type=float, default=None,
        help="Maximum audio chunk size (in minutes) for MMS inference. "
             "Smaller values use less VRAM but may reduce alignment accuracy at chunk boundaries. "
             "Default: per-device (CPU=5 min, CUDA=2 min, MPS=1 min). Try 2 on a 6 GB GPU.",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing options")
    proc_group.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    proc_group.add_argument("--force", action="store_true", help="Re-process even if output exists")
    proc_group.add_argument("--check-audio", action="store_true", help="Only report audio availability")
    proc_group.add_argument("--no-download", action="store_true",
        help="Skip auto-download, only process already-downloaded files")

    # Step skipping
    skip_group = parser.add_argument_group("Step control")
    skip_group.add_argument("--skip-whisper", action="store_true", help="Skip Whisper transcription (Step 1a)")
    skip_group.add_argument("--skip-mms", action="store_true", help="Skip MMS forced alignment (Step 1b)")
    skip_group.add_argument("--skip-fusion", action="store_true", help="Skip fusion alignment (Step 2)")

    proc_group.add_argument("--device", type=str, default=None,
                            choices=["cpu", "mps", "cuda"],
                            help="Device for MMS model (default: auto — mps on Apple Silicon, cuda if available, else cpu)")

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # Apply --whisper-cpu flag to module-level state in whisper_transcribe so
    # load_whisper_model() forces CPU regardless of available VRAM.
    if getattr(args, "whisper_cpu", False):
        import whisper_transcribe as _wt
        _wt._WHISPER_FORCE_CPU = True

    # Apply --mms-cpu / --mms-chunk-minutes flags to mms_align_words module state.
    if getattr(args, "mms_cpu", False) or getattr(args, "mms_chunk_minutes", None) is not None:
        import mms_align_words as _mms
        if getattr(args, "mms_cpu", False):
            _mms._MMS_FORCE_CPU = True
        if getattr(args, "mms_chunk_minutes", None) is not None:
            _mms._MAX_CHUNK_SAMPLES = int(args.mms_chunk_minutes * 60 * 16000)
            log(f"MMS chunk size set to {args.mms_chunk_minutes:.1f} min "
                f"({_mms._MAX_CHUNK_SAMPLES:,} samples)")

    # Validate selection
    if not any([args.iso, args.iso_list, args.tier is not None, args.all]):
        parser.error("Specify at least one of: --iso, --iso-list, --tier, --all")

    log("=" * 70)
    log("Unified Bible Audio Alignment Pipeline")
    log("=" * 70)

    steps = []
    if not args.skip_whisper:
        steps.append("Whisper")
    if not args.skip_mms:
        steps.append("MMS-FA")
    if not args.skip_fusion:
        steps.append("Fusion")
    log(f"Active steps: {' → '.join(steps)}")

    # ── Step 1: Determine required chapters ──

    if args.books:
        log(f"Building refs from --books: {args.books}")
        all_refs = build_refs_from_books(args.books)
        if not all_refs:
            log(f"No valid books in: {args.books}", "ERROR")
            sys.exit(1)
    else:
        template_label = args.template if args.template else "all templates"
        log(f"Scanning {template_label} for Bible references...")
        all_refs = load_all_template_refs(args.template)
        if not all_refs:
            log(f"No Bible references found in {template_label}", "ERROR")
            sys.exit(1)

    refs_by_canon = classify_template_refs(all_refs)

    # Apply testament filter
    if args.testament == "nt":
        refs_by_canon["ot"] = {}
    elif args.testament == "ot":
        refs_by_canon["nt"] = {}

    total_chapters = sum(len(chs) for canon_refs in refs_by_canon.values() for chs in canon_refs.values())
    for canon in ("nt", "ot"):
        canon_refs = refs_by_canon.get(canon, {})
        if canon_refs:
            books_list = sorted(canon_refs.keys())
            if len(books_list) > 8:
                book_summary = f"{len(books_list)} books ({', '.join(books_list[:5])}...)"
            else:
                book_summary = ", ".join(
                    f"{b}({len(chs)}ch)" for b, chs in sorted(canon_refs.items())
                )
            log(f"  {canon.upper()}: {sum(len(chs) for chs in canon_refs.values())} chapters — {book_summary}")
    log(f"  Total: {total_chapters} chapters")

    # ── Step 2: Resolve languages ──

    all_languages = load_priority_languages()
    languages = resolve_languages(args, all_languages)
    log(f"Languages selected: {len(languages)}")
    for lang in languages:
        log(f"  {lang['iso']} ({lang['language']}) - tier {lang['tier']}")

    # Generate work items
    work_items = generate_work_items(languages, args.testament, refs_by_canon)
    if not work_items:
        log("No syncable filesets found for the selected languages/testament", "WARN")
        sys.exit(0)

    known = [w for w in work_items if w["distinct_id"] is not None]
    pending = [w for w in work_items if w["distinct_id"] is None]
    log(f"Work items: {len(known)} fileset(s)" + (f", {len(pending)} pending download" if pending else ""))

    # Check audio mode
    if args.check_audio:
        from whisper_transcribe import check_and_report_audio
        check_and_report_audio(work_items, args.output_dir, refs_by_canon)
        return

    # ── Auto-download missing audio+text ──

    if not args.no_download and not args.dry_run:
        log("Checking downloads...")
        # Two channels:
        #   - DBT: bulk-download per (iso) into downloads/BB/ via
        #     download_language_content.py
        #   - External (contrib sermon-online, helloAO BSB+reader):
        #     bulk-download per (iso, distinct_id) into the matching
        #     downloads/contrib/ or downloads/helloao/aligned/ tree via
        #     remote_audio.bulk_download_external_audio
        # Both channels write into the standard {canon}/{iso}/{distinct_id}/{BOOK}/
        # layout so the cleanup script handles them uniformly.
        from remote_audio import bulk_download_external_audio
        # Dedup DBT downloads per (iso, canon) — books_spec is canon-scoped,
        # so triggering per-iso would only ever cover the first canon seen.
        downloaded_iso_canons = set()
        for item in work_items:
            iso = item["iso"]
            canon = item["canon"]
            canon_refs = refs_by_canon.get(canon, {})
            if not canon_refs:
                continue
            distinct_id = item["distinct_id"]

            contrib_dir = Path("downloads/contrib") / canon / iso / (distinct_id or "")
            helloao_dir = Path("downloads/helloao/aligned") / canon / iso / (distinct_id or "")

            if distinct_id is not None and contrib_dir.is_dir():
                d, a, f = bulk_download_external_audio(
                    canon, iso, distinct_id, Path("downloads/contrib"), canon_refs,
                )
                if d or f:
                    log(f"  Contrib audio for {iso}/{distinct_id}: "
                        f"{d} downloaded, {a} already present, {f} failed")
            elif distinct_id is not None and helloao_dir.is_dir():
                d, a, f = bulk_download_external_audio(
                    canon, iso, distinct_id, Path("downloads/helloao/aligned"), canon_refs,
                )
                if d or f:
                    log(f"  helloAO audio for {iso}/{distinct_id}: "
                        f"{d} downloaded, {a} already present, {f} failed")
            else:
                # DBT path: download once per (iso, canon).
                # download_language_content.py groups by canon internally and
                # the books_spec we pass is canon-scoped, so dedup must include
                # canon — otherwise OT would be skipped after NT runs first.
                key = (iso, canon)
                if key in downloaded_iso_canons:
                    continue
                books_spec = refs_to_books_spec(canon_refs)
                download_audio_for_chapters(iso, books_spec)
                downloaded_iso_canons.add(key)

        # Re-resolve placeholder work items (distinct_id=None) now that downloads exist
        # (Version-exclusion is handled upstream by core — batch jobs are pre-filtered.)
        resolved = []
        for item in work_items:
            if item["distinct_id"] is not None:
                resolved.append(item)
            else:
                # Scan disk for what was downloaded across all source roots
                seen = set()
                any_dir = False
                for base in (
                    Path("downloads/BB"),
                    Path("downloads/contrib"),
                    Path("downloads/helloao/aligned"),
                ):
                    iso_dir = base / item["canon"] / item["iso"]
                    if not iso_dir.is_dir():
                        continue
                    any_dir = True
                    for distinct_dir in sorted(iso_dir.iterdir()):
                        if not distinct_dir.is_dir() or distinct_dir.name in seen:
                            continue
                        seen.add(distinct_dir.name)
                        resolved.append({
                            **item,
                            "distinct_id": distinct_dir.name,
                            "has_downloads": True,
                        })
                        log(f"  Discovered fileset: {item['iso']}/{item['canon']}/{distinct_dir.name}")
                if not any_dir:
                    log(f"  No downloads found for {item['iso']}/{item['canon']}", "WARN")
        work_items = resolved

    # ── Verify dependencies and pre-load models (unless dry-run) ──
    #
    # IMPORTANT: MMS is loaded FIRST so that the VRAM check for Whisper
    # reflects the actual remaining memory after MMS occupies the GPU.
    # This allows Whisper to fall back to CPU (int8) automatically when
    # there is not enough VRAM for both models simultaneously.

    # Apply --whisper-cpu override before any Whisper model loading
    if args.whisper_cpu:
        set_whisper_cpu(True)

    mms_loaded = None   # will be (bundle, model, tokenizer, aligner, uroman)
    whisper_model_instance = None

    if not args.dry_run:
        # ── Load MMS first (so VRAM check for Whisper is accurate) ──
        if not args.skip_mms:
            try:
                import torch  # noqa: F401
                import torchaudio  # noqa: F401
            except ImportError:
                log("torch/torchaudio not installed. Run: pip install torch torchaudio uroman", "ERROR")
                sys.exit(1)
            from mms_align_words import load_mms_model, select_device
            mms_loaded = load_mms_model(select_device(args.device))

        # ── Load Whisper second (VRAM check now sees MMS already loaded) ──
        if not args.skip_whisper:
            if _USE_MLX:
                try:
                    import mlx_whisper  # noqa: F401
                except ImportError:
                    log("mlx-whisper not installed. Run: pip install -r requirements-whisper.txt", "ERROR")
                    sys.exit(1)
                log("Whisper backend: mlx-whisper (Apple Silicon / Metal)")
            else:
                try:
                    from faster_whisper import WhisperModel  # noqa: F401
                except ImportError:
                    log("faster-whisper not installed. Run: pip install faster-whisper", "ERROR")
                    sys.exit(1)
                # Pre-load the model once so it is not reloaded for every chapter.
                # load_whisper_model() checks free VRAM *after* MMS is loaded and
                # automatically falls back to CPU (int8) if VRAM is insufficient.
                whisper_model_instance = load_whisper_model(args.model)
                from whisper_transcribe import _enough_vram_for_whisper
                actual_device = "CUDA" if _USE_CUDA and _enough_vram_for_whisper(args.model) else "CPU"
                log(f"Whisper backend: faster-whisper ({actual_device})")

    # ── Process each fileset ──

    total_stats = {
        "languages_processed": set(),
        "whisper_done": 0,
        "mms_done": 0,
        "fusion_done": 0,
        "chapters_skipped": 0,
        "chapters_failed": 0,
        "total_audio_duration": 0,
        "total_time": 0,
    }

    pipeline_start = time.time()

    for item_idx, item in enumerate(work_items):
        iso = item["iso"]
        canon = item["canon"]
        distinct_id = item["distinct_id"]
        lang_name = item["language"]
        tier = item["tier"]
        canon_refs = refs_by_canon.get(canon, {})

        log("")
        log(f"[{item_idx + 1}/{len(work_items)}] {iso} ({lang_name}), tier {tier}, "
            f"{canon.upper()}/{distinct_id}")
        log("-" * 50)

        if not item["has_downloads"]:
            log("No audio downloaded", "WARN")
            continue

        # Discover chapters (filtered to template refs and optional book/chapter)
        # For the discovery, we build chapter refs incorporating --book/--chapter filters
        filtered_refs = canon_refs if canon_refs else None
        if args.book or args.chapter is not None:
            if filtered_refs:
                # Further filter by book/chapter
                new_refs = {}
                for book, chs in filtered_refs.items():
                    if args.book and book != args.book:
                        continue
                    if args.chapter is not None:
                        chs = {c for c in chs if c == args.chapter}
                    if chs:
                        new_refs[book] = chs
                # If book/chapter filter produces nothing for this canon,
                # use empty dict to skip all chapters (not None which means "all")
                filtered_refs = new_refs if new_refs else {}
            else:
                # No template refs but book/chapter filter — let discover_chapter_files handle it
                pass

        chapters, skipped = discover_chapter_files(
            iso, canon, distinct_id, args.output_dir,
            force=args.force,  # Only force-rediscover when user passes --force
            required_chapters=filtered_refs,
        )
        total_stats["chapters_skipped"] += skipped

        if not chapters:
            if skipped > 0:
                log("All chapters already processed")
            else:
                log("No audio+text pairs found", "WARN")
            continue

        log(f"Chapters found: {len(chapters)}")

        if args.dry_run:
            for ch in chapters:
                whisper_path = ch["whisper_words_path"]
                mms_book_dir = WORD_TIMING_DIR / canon / iso / distinct_id / ch["book"]
                mms_path = mms_book_dir / f"{ch['book']}_{ch['chapter_str']}_{ch['audio_fileset']}_mms_words.json"
                out_book_dir = args.output_dir / canon / iso / distinct_id / ch["book"]
                timing_path = out_book_dir / f"{ch['book']}_{ch['chapter_str']}_{ch['audio_fileset']}_timing.json"

                status_parts = []
                if not args.skip_whisper:
                    status_parts.append(f"whisper={'exists' if whisper_path.exists() else 'TODO'}")
                if not args.skip_mms:
                    status_parts.append(f"mms={'exists' if mms_path.exists() else 'TODO'}")
                if not args.skip_fusion:
                    status_parts.append(f"fusion={'exists' if timing_path.exists() else 'TODO'}")
                log(f"  {ch['book']} {ch['chapter']} — {', '.join(status_parts)}")
            continue

        # Load language config
        config = load_language_config(iso)

        # Get Whisper language hint
        whisper_lang = get_whisper_language(iso)
        if not args.skip_whisper:
            if whisper_lang:
                log(f"Whisper language: {whisper_lang}")
            else:
                log("Whisper language: auto-detect")

        total_stats["languages_processed"].add(iso)

        # Process each chapter through the pipeline
        for ch_idx, chapter in enumerate(chapters):
            book = chapter["book"]
            ch_num = chapter["chapter"]
            label = f"[{ch_idx + 1}/{len(chapters)}] {book} {ch_num}"

            try:
                # Ensure local audio for external sources (sermon-online, helloAO).
                # No-op if mp3 is already present.
                if not chapter["audio_path"].exists():
                    from remote_audio import ensure_chapter_audio
                    if not ensure_chapter_audio(chapter["audio_path"], chapter["book"], chapter["chapter"]):
                        log(f"{label} Audio missing and could not be fetched, skipping", "WARN")
                        total_stats["chapters_failed"] += 1
                        continue
                    else:
                        log(f"{label} Audio fetched on demand")

                # ── Step 1a: Whisper ──
                # Prefer standard (non-drama) audio for Whisper transcription
                if not args.skip_whisper and not chapter.get("whisper_prefer", True):
                    log(f"{label} Whisper: skipped (drama — prefer standard)")
                elif not args.skip_whisper:
                    whisper_path = chapter["whisper_words_path"]
                    if whisper_path.exists() and not args.force:
                        log(f"{label} Whisper: skipped (exists)")
                    else:
                        log(f"{label} Whisper: transcribing...")
                        t0 = time.time()
                        stats = run_whisper_chapter(chapter, args.model, whisper_lang,
                                                    whisper_model=whisper_model_instance)
                        elapsed = time.time() - t0
                        duration_str = format_duration(stats["duration"])
                        speed = stats["duration"] / elapsed if elapsed > 0 else 0
                        log(f"{label} Whisper: {stats['words']} words, "
                            f"{duration_str} audio, {speed:.1f}x realtime")
                        total_stats["whisper_done"] += 1
                        total_stats["total_audio_duration"] += stats["duration"]

                # ── Detect header from Whisper output ──
                header_skip_time = None
                if not args.skip_mms:
                    whisper_path = chapter["whisper_words_path"]
                    if whisper_path.exists():
                        verse_start, header_text = detect_whisper_header(
                            whisper_path, chapter["text_path"], config,
                        )
                        if verse_start:
                            header_skip_time = verse_start
                            log(f"{label} Header detected ({header_skip_time:.1f}s): \"{header_text}\"")

                # ── Step 1b: MMS ──
                if not args.skip_mms:
                    mms_item = build_mms_item(chapter, canon, iso, distinct_id)
                    if mms_item["mms_path"].exists() and not args.force:
                        log(f"{label} MMS: skipped (exists)")
                    else:
                        log(f"{label} MMS: aligning...")
                        bundle, model, tokenizer, aligner, uroman = mms_loaded
                        stats = run_mms_chapter(
                            mms_item, bundle, model, tokenizer, aligner, uroman, config,
                            header_skip_time=header_skip_time,
                        )
                        if "error" in stats:
                            log(f"{label} MMS: {stats['error']}", "ERROR")
                        else:
                            extras = []
                            if stats.get("header_skipped"):
                                extras.append(f"header={stats['header_skipped']}s")
                            if stats.get("restarted"):
                                extras.append(f"restarted@verse{stats['restart_verse']}")
                            extra_str = f", {', '.join(extras)}" if extras else ""
                            log(f"{label} MMS: {stats['aligned']}/{stats['words']} words, "
                                f"score={stats['avg_score']}, {stats['elapsed']}s{extra_str}")
                            total_stats["mms_done"] += 1

                # ── Step 2: Fusion ──
                if not args.skip_fusion:
                    fusion_item = build_fusion_item(
                        chapter, canon, iso, distinct_id, args.output_dir,
                    )

                    # Check if fusion output is stale (inputs newer than output)
                    fusion_stale = False
                    if fusion_item["timing_path"].exists() and not args.force:
                        out_mtime = fusion_item["timing_path"].stat().st_mtime
                        for src_key in ("mms_path", "whisper_path"):
                            src = fusion_item[src_key]
                            if src and src.exists() and src.stat().st_mtime > out_mtime:
                                fusion_stale = True
                                break

                    if fusion_item["timing_path"].exists() and not args.force and not fusion_stale:
                        log(f"{label} Fusion: skipped (exists)")
                    else:
                        if fusion_stale:
                            log(f"{label} Fusion: re-running (inputs updated)")
                        # Check that at least one source exists
                        if not fusion_item["mms_path"] and not fusion_item["whisper_path"]:
                            log(f"{label} Fusion: no MMS or Whisper data available", "WARN")
                        else:
                            # Pass MMS components for gap-fill re-alignment
                            fusion_mms = None
                            if mms_loaded and not args.skip_mms:
                                fusion_mms = mms_loaded
                            # Add audio path for segment re-alignment
                            fusion_item["audio_path"] = chapter.get("audio_path")
                            stats = run_fusion_chapter(fusion_item, config, mms_components=fusion_mms)
                            if "error" in stats:
                                log(f"{label} Fusion: {stats['error']}", "ERROR")
                            else:
                                parts = [f"{stats['verses']} verses", f"source={stats['source']}"]
                                if stats.get("fusion"):
                                    fs = stats["fusion"]
                                    parts.append(
                                        f"{fs['from_whisper']}/{fs['total_words']} from whisper"
                                    )
                                if stats.get("quality"):
                                    q = stats["quality"]
                                    parts.append(f"avg={q['avg_score']}")
                                    if q["low_quality_count"] > 0:
                                        parts.append(
                                            f"{q['low_quality_count']} low-q in verses {','.join(q['low_quality_verses'][:5])}"
                                        )
                                log(f"{label} Fusion: {', '.join(parts)}")
                                total_stats["fusion_done"] += 1

            except KeyboardInterrupt:
                log("Interrupted by user — stopping pipeline", "WARN")
                total_time = time.time() - pipeline_start
                _print_summary(total_stats, total_time)
                sys.exit(1)
            except Exception as e:
                log(f"{label} Failed: {e}", "ERROR")
                total_stats["chapters_failed"] += 1

    # ── Summary ──

    total_time = time.time() - pipeline_start
    _print_summary(total_stats, total_time)


def _print_summary(total_stats: dict, total_time: float):
    """Print pipeline summary."""
    log("")
    log("=" * 70)
    log("Pipeline Summary")
    log("=" * 70)
    log(f"  Languages processed:    {len(total_stats['languages_processed'])}")
    log(f"  Whisper transcriptions: {total_stats['whisper_done']}")
    log(f"  MMS alignments:         {total_stats['mms_done']}")
    log(f"  Fusion alignments:      {total_stats['fusion_done']}")
    log(f"  Failed:                 {total_stats['chapters_failed']}")
    log(f"  Total audio duration:   {format_duration(total_stats['total_audio_duration'])}")
    log(f"  Total pipeline time:    {format_duration(total_time)}")
    log("=" * 70)


if __name__ == "__main__":
    main()
