"""
Per-edition versification scheme lookup, from the bibles-repo CDN index.

WHY THIS EXISTS
---------------
Our alignment output is keyed to whatever chapter/verse numbering the
SOURCE TEXT uses, and until now we published no statement of what that
numbering is. That's a real correctness hazard for consumers, not a
cosmetic gap: 28 of the editions we've already aligned use a non-`eng`
scheme (10 `orgw`, 9 `org`, 8 `rso`, 1 `vul` as of 2026-09-04). For the
`rso` ones the Psalm numbering is shifted a whole chapter relative to what
most consumers would assume — so timings we publish for `kaz/KAZKAZ` PSA 34
describe rso Psalm 34, which is org Psalm 35, and nothing in the artifact
said so.

This is the mirror image of the bug we reported to the bibles repo (their
audio/text mismatch on bul/BULCBV), pointed at ourselves.

SOURCE
------
https://cdn.bibel.wiki/dbt/_vrs/index.json, published by the bibles repo,
shaped {"l": {"<iso>/<distinct_id>": "<scheme>"}, "schemes": [...],
"sentinels": {...}}. Scheme is a standard Paratext .vrs name (eng, org,
orgw, rso, lxx, vul, catm) or a sentinel:

    "irregular"    — no standard scheme; use the text's own verse numbers
    "undetermined" — examined, insufficient data to classify

Coverage is partial (~234 of 437 DBT text filesets with Psalms as of
2026-09-04), so an unknown edition is the normal case, not an error.

CONTRACT
--------
lookup_vrs() returns None for anything we can't positively identify —
unknown edition, sentinel value, CDN unreachable. Callers must OMIT the
field entirely rather than emitting a guess: a consumer seeing no `vrs`
knows to fall back to its own assumptions, whereas a consumer seeing a
WRONG `vrs` has been actively misled, which is worse than the silence we
started with. Never let a lookup failure block alignment.
"""

import json
import urllib.request
from pathlib import Path

VRS_INDEX_URL = "https://cdn.bibel.wiki/dbt/_vrs/index.json"
VRS_CACHE_PATH = Path("api-cache/vrs/index.json")

# Values that are sentinels rather than real scheme names — see the module
# docstring. Treated exactly like "not found": we know nothing publishable.
_SENTINELS = frozenset({"irregular", "undetermined"})

_index_cache: dict | None = None


def _load_index() -> dict:
    """The {"<iso>/<distinct_id>": scheme} map, cached on disk then in memory.

    Same fetch-then-cache shape as download_language_content.py's DBT
    catalog loader. Returns {} on any failure so a lookup degrades to
    "unknown" instead of crashing a batch.
    """
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    if VRS_CACHE_PATH.exists():
        try:
            with open(VRS_CACHE_PATH, encoding="utf-8") as f:
                _index_cache = json.load(f).get("l", {})
            return _index_cache
        except (OSError, json.JSONDecodeError):
            pass

    try:
        req = urllib.request.Request(
            VRS_INDEX_URL, headers={"User-Agent": "audio-sync"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        data = json.loads(raw)
    except Exception:
        # Deliberately silent-ish: this is decoration on the output, never a
        # reason to fail a run. Cached as {} so we don't retry per chapter.
        _index_cache = {}
        return _index_cache

    try:
        VRS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        VRS_CACHE_PATH.write_bytes(raw)
    except OSError:
        pass

    _index_cache = data.get("l", {})
    return _index_cache


def lookup_vrs(iso: str, distinct_id: str) -> str | None:
    """Versification scheme for one edition's TEXT, or None if not known.

    The returned name describes the numbering our own output is keyed to,
    because we key output to the reference text's chapters and verses —
    including under a chapter_map, where the audio's own numbering is
    remapped onto the text's (see map_audio_chapter_to_text()).
    """
    if not iso or not distinct_id:
        return None
    scheme = _load_index().get(f"{iso}/{distinct_id}")
    if not scheme or scheme in _SENTINELS:
        return None
    return scheme


def reset_cache() -> None:
    """Drop the in-memory index (tests; forcing a re-read after refresh)."""
    global _index_cache
    _index_cache = None
