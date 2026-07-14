#!/usr/bin/env python3
"""
On-demand audio fetcher for external (non-DBT) sources.

Used by the alignment pipeline when a chapter's mp3 is missing locally
because the audio is hosted remotely:

  - Contrib `audio.json` with type=sermon-online (e.g. deu/DEUSOL):
      URL = {baseUrl}{code}{chapter:0Nd}-{title}_Kapitel-{chapter:03d}.mp3
      where N pads the chapter so the leading prefix is 5 chars.

  - helloAO chapter JSON (e.g. eng/BSBHAY): the chapter's
      `thisChapterAudioLinks[reader]` field has the direct mp3 URL.

The alignment pipeline calls `ensure_chapter_audio()` for each chapter
before Whisper. If the mp3 already exists locally, it's a no-op.
"""

import json
import urllib.request
from pathlib import Path
from typing import Optional


HELLOAO_API_BASE = "https://bible.helloao.org/api"
HELLOAO_CONFIG_PATH = Path("config/helloao.toml")
HELLOAO_CACHE = Path("downloads/helloao")

# Some hosts (sermon-online.com) reject requests without a browser-like UA.
_USER_AGENT = "Mozilla/5.0 (compatible; bible-story-builder/1.0)"


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ─── Source detection ──────────────────────────────────────────────────────

def detect_remote_source(book_dir: Path):
    """Return (kind, meta) for the remote source serving this book_dir, or
    (None, None) if audio is local-only.

    Walks up from {book_dir} to find:
      - audio.json sibling at the version dir → ("sermon-online", audio_meta)
      - parent matches downloads/helloao/aligned/{canon}/{iso}/{distinct_id}/
        → ("helloao-bsb", config_for_distinct_id)
    """
    # Version dir is one level above book_dir (book_dir = .../{distinct_id}/{BOOK})
    version_dir = book_dir.parent
    audio_json = version_dir / "audio.json"
    if audio_json.exists():
        try:
            with open(audio_json, encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("type", "external"), meta
        except Exception:
            pass

    # helloAO aligned content?
    parts = book_dir.parts
    if "helloao" in parts and "aligned" in parts:
        idx = parts.index("aligned")
        if idx + 3 < len(parts):
            distinct_id = parts[idx + 3]
            cfg = _load_helloao_config()
            for trans_id, trans_meta in cfg.get("translations", {}).items():
                for reader_name, reader_meta in trans_meta.get("readers", {}).items():
                    if reader_meta.get("distinct_id") == distinct_id:
                        return "helloao-bsb", {
                            "translation": trans_id,
                            "reader": reader_name,
                            "distinct_id": distinct_id,
                        }
    return None, None


def _load_helloao_config() -> dict:
    if not HELLOAO_CONFIG_PATH.exists():
        return {}
    try:
        import tomllib
        with open(HELLOAO_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


# ─── URL builders ──────────────────────────────────────────────────────────

def sermon_online_url(audio_meta: dict, book: str, chapter_num: int) -> Optional[str]:
    """Build a sermon-online mp3 URL from the audio.json metadata.

    Pattern: {baseUrl}{code}{chapter:0Nd}-{title}_Kapitel-{chapter:03d}.mp3
    The {code}{chapter} prefix pads to 5 chars total.
    """
    base = audio_meta.get("baseUrl", "")
    book_meta = audio_meta.get("books", {}).get(book)
    if not book_meta:
        return None
    code = book_meta["code"]
    title = book_meta["title"]
    chapter_pad = max(2, 5 - len(code))
    prefix = f"{code}{chapter_num:0{chapter_pad}d}"
    return f"{base}{prefix}-{title}_Kapitel-{chapter_num:03d}.mp3"


def helloao_chapter_audio_url(translation: str, reader: str, book: str, chapter_num: int) -> Optional[str]:
    """Fetch helloAO chapter JSON (cached) and return the reader's mp3 URL."""
    canon_dir = "ot" if book in {
        "GEN","EXO","LEV","NUM","DEU","JOS","JDG","RUT","1SA","2SA","1KI","2KI",
        "1CH","2CH","EZR","NEH","EST","JOB","PSA","PRO","ECC","SNG","ISA","JER",
        "LAM","EZK","DAN","HOS","JOL","AMO","OBA","JON","MIC","NAM","HAB","ZEP",
        "HAG","ZEC","MAL",
    } else "nt"
    cache = HELLOAO_CACHE / translation / canon_dir / f"{book}_{chapter_num:03d}.json"
    if cache.exists():
        try:
            with open(cache, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
    else:
        try:
            url = f"{HELLOAO_API_BASE}/{translation}/{book}/{chapter_num}.json"
            data = json.loads(_http_get(url, timeout=60))
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            data = None
    if not data:
        return None
    links = data.get("thisChapterAudioLinks") or {}
    return links.get(reader)


# ─── Public API ────────────────────────────────────────────────────────────

def ensure_chapter_audio(audio_path: Path, book: str, chapter_num: int) -> bool:
    """Ensure {audio_path} exists. If missing and the version dir indicates a
    known remote source, download. Returns True on success.

    No-op (returns True) when the file already exists.
    """
    if audio_path.exists():
        return True

    book_dir = audio_path.parent
    kind, meta = detect_remote_source(book_dir)
    if kind is None:
        return False

    if kind == "sermon-online":
        url = sermon_online_url(meta, book, chapter_num)
    elif kind == "helloao-bsb":
        url = helloao_chapter_audio_url(
            meta["translation"], meta["reader"], book, chapter_num,
        )
    else:
        return False

    if not url:
        return False

    try:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        data = _http_get(url, timeout=120)
        with open(audio_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def bulk_download_external_audio(
    canon: str,
    iso: str,
    distinct_id: str,
    base_dir: Path,
    refs: dict,
) -> tuple[int, int, int]:
    """Bulk-download external audio for one fileset upfront.

    Walks the requested refs ({BOOK: {chapter_nums}}) and downloads each
    missing mp3 into the standard layout under base_dir (downloads/contrib/
    or downloads/helloao/aligned/). Files end up in the same per-fileset
    structure as DBT downloads under downloads/BB/, so the cleanup script
    treats them uniformly.

    Returns (downloaded, already_present, failed).
    """
    suffix = "O2DA" if canon == "ot" else "N2DA"
    audio_fileset = f"{distinct_id}{suffix}"

    downloaded = 0
    already = 0
    failed = 0

    version_dir = base_dir / canon / iso / distinct_id
    if not version_dir.is_dir():
        return 0, 0, 0
    # detect_remote_source expects a book_dir and inspects its parent for
    # audio.json or "/aligned/" in its path. Use a synthetic book_dir under
    # version_dir to probe.
    kind, _meta = detect_remote_source(version_dir / "_probe")
    if kind is None:
        return 0, 0, 0

    for book, chapters in sorted(refs.items()):
        book_dir = version_dir / book
        for ch in sorted(chapters):
            target = book_dir / f"{book}_{ch:03d}_{audio_fileset}.mp3"
            if target.exists():
                already += 1
                continue
            if ensure_chapter_audio(target, book, ch):
                downloaded += 1
            else:
                failed += 1
    return downloaded, already, failed
