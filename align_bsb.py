#!/usr/bin/env python3
"""
Word-level alignment for helloAO translations (currently BSB).

Fetches helloAO chapter JSON + reader-specific audio mp3, then runs the
existing Whisper + MMS + fusion pipeline directly on the downloaded files.

Reader selection:
  CLI flag --reader  >  config/helloao.toml `default_reader`

Storage:
  downloads/helloao/{translation}/                 -- catalog cache (existing)
  downloads/helloao/aligned/{canon}/{iso}/{distinct_id}/{BOOK}/
    BOOK_CCC_{distinct_id}{N2DA|O2DA}.mp3
    BOOK_CCC_{distinct_id}_{NT|OT}_ET.txt
    BOOK_CCC_{distinct_id}{N2DA|O2DA}_timing.json
    BOOK_CCC_{distinct_id}{N2DA|O2DA}_words.json

Usage:
    python align_bsb.py --book JHN --chapter 3
    python align_bsb.py --book JHN --chapters 1-21
    python align_bsb.py --book MAT --chapters 5,13,28 --reader gilbert
    python align_bsb.py --book GEN --chapters all --force
    python align_bsb.py --translation BSB --book JHN --chapter 3 --skip-whisper
"""

import argparse
import json
import re
import sys
import tomllib
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

HELLOAO_CACHE = Path("downloads/helloao")
ALIGNED_DIR = HELLOAO_CACHE / "aligned"
WORD_TIMING_DIR = Path("word-timing-data")
EXPORT_TIMING_DIR = Path("export/timing-data")
HELLOAO_API_BASE = "https://bible.helloao.org/api"
HELLOAO_CONFIG_PATH = Path("config/helloao.toml")


# Books standardly classified as OT
OT_BOOKS = {
    "GEN","EXO","LEV","NUM","DEU","JOS","JDG","RUT","1SA","2SA","1KI","2KI",
    "1CH","2CH","EZR","NEH","EST","JOB","PSA","PRO","ECC","SNG","ISA","JER",
    "LAM","EZK","DAN","HOS","JOL","AMO","OBA","JON","MIC","NAM","HAB","ZEP",
    "HAG","ZEC","MAL",
}


def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")


def load_helloao_config() -> dict:
    if not HELLOAO_CONFIG_PATH.exists():
        log(f"{HELLOAO_CONFIG_PATH} not found", "ERROR")
        sys.exit(1)
    with open(HELLOAO_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def resolve_reader(config: dict, translation: str, requested: Optional[str]) -> tuple[str, str, str]:
    """Return (reader, distinct_id, language_iso) — CLI flag overrides default_reader."""
    tr = config.get("translations", {}).get(translation)
    if not tr:
        log(f"Translation {translation} not found in {HELLOAO_CONFIG_PATH}", "ERROR")
        sys.exit(1)
    reader = requested or tr.get("default_reader")
    if not reader:
        log(f"No reader specified and no default_reader for {translation}", "ERROR")
        sys.exit(1)
    reader_meta = tr.get("readers", {}).get(reader)
    if not reader_meta:
        available = ", ".join(tr.get("readers", {}).keys())
        log(f"Reader '{reader}' not configured for {translation}. Available: {available}", "ERROR")
        sys.exit(1)
    distinct_id = reader_meta["distinct_id"]
    language = tr.get("language", "eng")
    return reader, distinct_id, language


def parse_chapter_spec(spec: str) -> List[int]:
    """Parse '1', '1-21', '5,13,28', 'all' (handled by caller)."""
    chapters = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            chapters.update(range(int(a), int(b) + 1))
        else:
            chapters.add(int(part))
    return sorted(chapters)


def fetch_chapter_json(translation: str, book: str, chapter: int, force: bool = False) -> dict:
    """Fetch helloAO chapter JSON, caching to downloads/helloao/{translation}/{canon}/."""
    canon = "ot" if book in OT_BOOKS else "nt"
    cache_dir = HELLOAO_CACHE / translation / canon
    cache_path = cache_dir / f"{book}_{chapter:03d}.json"

    if cache_path.exists() and not force:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    url = f"{HELLOAO_API_BASE}/{translation}/{book}/{chapter}.json"
    log(f"  Fetching {url}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read())
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def extract_verses(chapter_json: dict) -> List[str]:
    """Extract verse text, one verse per line, preserving order.

    helloAO chapter JSON has a `chapter.content` list of items; verses are
    objects with `type: 'verse'`, containing `content` (list of strings or
    objects with a `text` field). Footnotes are skipped.
    """
    chapter = chapter_json.get("chapter", {})
    items = chapter.get("content", [])

    verses = {}  # verse_num -> list of text fragments
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "verse":
            continue
        vnum = item.get("number")
        if vnum is None:
            continue
        parts = []
        for frag in item.get("content", []):
            if isinstance(frag, str):
                parts.append(frag)
            elif isinstance(frag, dict):
                # Skip footnotes; collect plain text fragments
                if frag.get("noteId") is not None:
                    continue
                if "text" in frag:
                    parts.append(frag["text"])
        text = " ".join(parts)
        text = re.sub(r"\s+", " ", text).strip()
        verses.setdefault(vnum, []).append(text)

    # Build ordered list — fill any gaps with empty (shouldn't happen but be safe)
    if not verses:
        return []
    max_v = max(verses.keys())
    return [" ".join(verses.get(v, [])).strip() for v in range(1, max_v + 1)]


def download_audio(url: str, target: Path, force: bool = False) -> bool:
    """Download mp3 from URL to target. Returns True on success or already-exists."""
    if target.exists() and not force:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"  Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        with open(target, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        log(f"  ERROR downloading {url}: {e}", "ERROR")
        return False


def get_audio_url(chapter_json: dict, reader: str) -> Optional[str]:
    links = chapter_json.get("thisChapterAudioLinks") or {}
    return links.get(reader)


def prepare_chapter(translation: str, book: str, chapter: int, reader: str,
                    distinct_id: str, language: str, force: bool = False) -> Optional[dict]:
    """Fetch+stage one chapter. Returns dict of paths or None on failure."""
    canon = "ot" if book in OT_BOOKS else "nt"
    chapter_str = f"{chapter:03d}"
    audio_suffix = "O2DA" if canon == "ot" else "N2DA"
    audio_fileset = f"{distinct_id}{audio_suffix}"
    text_fileset = f"{distinct_id}_{canon.upper()}_ET"

    book_dir = ALIGNED_DIR / canon / language / distinct_id / book
    book_dir.mkdir(parents=True, exist_ok=True)

    audio_path = book_dir / f"{book}_{chapter_str}_{audio_fileset}.mp3"
    text_path = book_dir / f"{book}_{chapter_str}_{text_fileset}.txt"

    # Fetch chapter JSON (cached)
    try:
        chapter_json = fetch_chapter_json(translation, book, chapter, force=force)
    except Exception as e:
        log(f"  ERROR fetching {book} {chapter}: {e}", "ERROR")
        return None

    # Extract text
    if not text_path.exists() or force:
        verses = extract_verses(chapter_json)
        if not verses:
            log(f"  ERROR: no verses extracted for {book} {chapter}", "ERROR")
            return None
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(verses) + "\n")

    # Download audio
    audio_url = get_audio_url(chapter_json, reader)
    if not audio_url:
        log(f"  ERROR: no '{reader}' audio link for {book} {chapter}", "ERROR")
        return None
    if not download_audio(audio_url, audio_path, force=force):
        return None

    return {
        "canon": canon,
        "book": book,
        "chapter": chapter,
        "chapter_str": chapter_str,
        "iso": language,
        "distinct_id": distinct_id,
        "audio_fileset": audio_fileset,
        "text_fileset": text_fileset,
        "audio_path": audio_path,
        "text_path": text_path,
        "book_dir": book_dir,
    }


# ─── Pipeline glue ──────────────────────────────────────────────────────────

def whisper_paths_for(prep: dict) -> tuple[Path, Path]:
    """Return (whisper_words_path, srt_path) for a prepared chapter."""
    book = prep["book"]; chapter_str = prep["chapter_str"]
    audio_fileset = prep["audio_fileset"]
    canon = prep["canon"]; iso = prep["iso"]; distinct_id = prep["distinct_id"]
    book = prep["book"]
    base = f"{book}_{chapter_str}_{audio_fileset}"
    whisper_words = (
        WORD_TIMING_DIR / canon / iso / distinct_id / book / f"{base}_whisper_words.json"
    )
    srt = EXPORT_TIMING_DIR / canon / iso / distinct_id / book / f"{base}.srt"
    return whisper_words, srt


def run_whisper(prep: dict, model_name: str, language_code: str, force: bool):
    from whisper_transcribe import process_chapter as whisper_process
    whisper_words, srt = whisper_paths_for(prep)
    if whisper_words.exists() and not force:
        log(f"  Whisper: skipped (exists)")
        return
    whisper_words.parent.mkdir(parents=True, exist_ok=True)
    srt.parent.mkdir(parents=True, exist_ok=True)
    chapter_arg = {
        "book": prep["book"],
        "chapter": prep["chapter"],
        "audio_path": prep["audio_path"],
        "whisper_words_path": whisper_words,
        "srt_path": srt,
    }
    log(f"  Whisper: transcribing...")
    stats = whisper_process(chapter_arg, model_name, language_code)
    log(f"  Whisper: {stats.get('words', '?')} words")


def run_mms(prep: dict, mms_components, force: bool):
    from mms_align_words import process_chapter as mms_process
    from align_words import detect_audio_header
    from text_processing import load_language_config, strip_markers

    whisper_words, _ = whisper_paths_for(prep)
    mms_path = (
        WORD_TIMING_DIR / prep["canon"] / prep["iso"] / prep["distinct_id"] /
        prep["book"] / f"{prep['book']}_{prep['chapter_str']}_{prep['audio_fileset']}_mms_words.json"
    )
    if mms_path.exists() and not force:
        log(f"  MMS: skipped (exists)")
        return
    mms_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_language_config(prep["iso"])

    header_skip = None
    if whisper_words.exists():
        with open(whisper_words, encoding="utf-8") as f:
            ww = json.load(f).get("words", [])
        with open(prep["text_path"], encoding="utf-8") as f:
            verse_texts = [strip_markers(l.rstrip("\n"), config) for l in f]
        vs, ht = detect_audio_header(ww, verse_texts, config)
        if vs:
            header_skip = vs
            log(f"  Header detected ({vs:.1f}s): {ht!r}")

    item = {
        "book": prep["book"],
        "chapter_str": prep["chapter_str"],
        "audio_path": prep["audio_path"],
        "text_path": prep["text_path"],
        "mms_path": mms_path,
        "iso": prep["iso"],
    }
    bundle, model, tokenizer, aligner, uroman = mms_components
    log(f"  MMS: aligning...")
    stats = mms_process(item, bundle, model, tokenizer, aligner, uroman, config,
                        header_skip_time=header_skip,
                        whisper_path=whisper_words if whisper_words.exists() else None)
    if "error" in stats:
        log(f"  MMS: {stats['error']}", "ERROR")
    else:
        log(f"  MMS: {stats['aligned']}/{stats['words']} words, score={stats['avg_score']}, {stats['elapsed']}s")


def run_fusion(prep: dict, mms_components, force: bool):
    from align_words import process_chapter as fusion_process
    from text_processing import load_language_config

    timing_path = (
        EXPORT_TIMING_DIR / prep["canon"] / prep["iso"] / prep["distinct_id"] /
        prep["book"] / f"{prep['book']}_{prep['chapter_str']}_{prep['audio_fileset']}_timing.json"
    )
    words_path = timing_path.with_name(timing_path.name.replace("_timing.json", "_words.json"))
    if timing_path.exists() and not force:
        log(f"  Fusion: skipped (exists)")
        return
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    whisper_words, _ = whisper_paths_for(prep)
    mms_path = (
        WORD_TIMING_DIR / prep["canon"] / prep["iso"] / prep["distinct_id"] /
        prep["book"] / f"{prep['book']}_{prep['chapter_str']}_{prep['audio_fileset']}_mms_words.json"
    )

    item = {
        "book": prep["book"],
        "chapter": prep["chapter"],
        "chapter_str": prep["chapter_str"],
        "ref_text_path": prep["text_path"],
        "mms_path": mms_path if mms_path.exists() else None,
        "whisper_path": whisper_words if whisper_words.exists() else None,
        "audio_path": prep["audio_path"],
        "timing_path": timing_path,
        "words_path": words_path,
    }
    config = load_language_config(prep["iso"])
    log(f"  Fusion: ...")
    stats = fusion_process(item, config, mms_components=mms_components)
    if "error" in stats:
        log(f"  Fusion: {stats['error']}", "ERROR")
    else:
        log(f"  Fusion: {stats.get('verses', '?')} verses, source={stats.get('source')}")

    # Also drop a copy into the aligned dir so it's discoverable as an export source
    aligned_timing = prep["book_dir"] / timing_path.name
    aligned_words = prep["book_dir"] / words_path.name
    if timing_path.exists():
        aligned_timing.write_bytes(timing_path.read_bytes())
    if words_path.exists():
        aligned_words.write_bytes(words_path.read_bytes())


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Align helloAO translations (BSB by default)")
    p.add_argument("--translation", default="BSB", help="helloAO translation id (default: BSB)")
    p.add_argument("--reader", default=None, help="Audio reader (default: from config)")
    p.add_argument("--book", required=True, help="Book code, e.g. JHN, GEN, MAT")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--chapter", type=int, help="Single chapter")
    g.add_argument("--chapters", help="Range or list, e.g. '1-21' or '5,13,28' or 'all'")
    p.add_argument("--force", action="store_true", help="Re-fetch and re-align even if outputs exist")
    p.add_argument("--skip-whisper", action="store_true")
    p.add_argument("--skip-mms", action="store_true")
    p.add_argument("--skip-fusion", action="store_true")
    p.add_argument("--whisper-model", default="mlx-community/whisper-large-v3-mlx")
    p.add_argument("--whisper-language", default=None,
                   help="ISO-639-1 lang hint for Whisper (e.g. 'en'); default: derive from config")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None,
                   help="Device for MMS (default: auto)")
    args = p.parse_args()

    config = load_helloao_config()
    reader, distinct_id, language = resolve_reader(config, args.translation, args.reader)
    whisper_lang = args.whisper_language or {"eng": "en"}.get(language, language)
    log(f"Translation={args.translation}, reader={reader}, distinct_id={distinct_id}, language={language}")

    # Determine chapter list
    if args.chapter is not None:
        chapters = [args.chapter]
    elif args.chapters == "all":
        # Need to fetch books.json to know chapter count
        books_url = f"{HELLOAO_API_BASE}/{args.translation}/books.json"
        with urllib.request.urlopen(books_url, timeout=60) as resp:
            books_data = json.loads(resp.read())
        book_meta = next(
            (b for b in books_data.get("books", []) if b.get("id") == args.book), None
        )
        if not book_meta:
            log(f"Book {args.book} not found in {args.translation}", "ERROR")
            sys.exit(1)
        chapters = list(range(1, book_meta["numberOfChapters"] + 1))
    else:
        chapters = parse_chapter_spec(args.chapters)

    log(f"Processing {args.book} chapters: {chapters}")

    # Prepare files for all chapters first
    preps = []
    for ch in chapters:
        log(f"\n[{args.book} {ch}] preparing...")
        prep = prepare_chapter(args.translation, args.book, ch, reader,
                               distinct_id, language, force=args.force)
        if prep:
            preps.append(prep)

    if not preps:
        log("No chapters prepared; aborting", "ERROR")
        sys.exit(1)

    # Run pipeline steps (load models lazily and only if needed)
    mms_components = None
    if not args.skip_mms or not args.skip_fusion:
        from mms_align_words import load_mms_model, select_device
        mms_components = load_mms_model(select_device(args.device))

    for prep in preps:
        log(f"\n=== {prep['book']} {prep['chapter']} ===")
        if not args.skip_whisper:
            run_whisper(prep, args.whisper_model, whisper_lang, args.force)
        if not args.skip_mms:
            run_mms(prep, mms_components, args.force)
        if not args.skip_fusion:
            run_fusion(prep, mms_components, args.force)

    log(f"\nDone. Outputs in: {ALIGNED_DIR}/")


if __name__ == "__main__":
    main()
