"""Shared text processing for the alignment pipeline.

Provides language-configurable text normalization used by:
  - mms_align_words.py   (Step 1b — MMS forced alignment)
  - align_words.py       (Step 2  — fusion)
  - whisper_transcribe.py (Step 1a — Whisper transcription)

Language-specific rules (pronunciation maps, marker patterns, character
replacements) are loaded from TOML config files in config/languages/.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

CONFIG_DIR = Path(__file__).parent / "config" / "languages"


@dataclass
class LanguageConfig:
    """Language-specific text processing configuration."""
    iso: str
    pronunciation_map: dict[str, str] = field(default_factory=dict)
    strip_marker_rules: list[dict[str, str]] = field(default_factory=list)
    char_replacements: dict[str, str] = field(default_factory=dict)
    strip_unicode_categories: list[str] = field(default_factory=lambda: ["Mn"])
    mms_fallback_threshold: float = 0.3
    aramaic_passages: list[str] = field(default_factory=list)
    verse_only_mode: bool = False
    chapter_map: list[dict] = field(default_factory=list)
    whisper: dict = field(default_factory=dict)


def map_audio_chapter_to_text(
    config: LanguageConfig,
    book: str,
    audio_fileset: str,
    audio_chapter: int,
) -> int | None:
    """Translate an AUDIO chapter number into the TEXT chapter it actually reads.

    An audio fileset and its paired text can follow different versification
    schemes, in which case audio chapter N does not contain text chapter N.
    The confirmed case (2026-09-04) is bul/BULCBV: the audio is Orthodox/LXX-
    numbered while the text is Masoretic-numbered, so e.g. the file named
    PSA_050 actually reads Psalm 51 — the narration even announces both
    numbers ("Псалом петдесети. По-еврейски петдесет и първи"). Nothing
    catches this today: DBT publishes no versification field (only
    cdn.bibel.wiki/pkf/manifest.json has a "vrs" key, and it covers 2 of our
    128 aligned languages), so the mapping is derived empirically from the
    Whisper transcript and recorded in the language config.

    Returns the text chapter number, or None when this audio chapter must be
    skipped entirely. A None means the audio/text correspondence is a MERGE
    (one audio chapter spans two text chapters, e.g. LXX Ps 9 = Hebrew Ps
    9+10) or a SPLIT (two audio chapters cover one text chapter, e.g. LXX Ps
    114+115 = Hebrew Ps 116). Neither is expressible as a chapter offset —
    they need verse-range handling that doesn't exist yet — and skipping is
    strictly better than the status quo, which aligns them against the wrong
    text and silently emits confident-looking nonsense.

    With no chapter_map configured (every language but bul today) this is an
    identity function, so callers can apply it unconditionally.
    """
    if not config.chapter_map:
        return audio_chapter

    for rule in config.chapter_map:
        if rule.get("book") != book:
            continue
        fileset = rule.get("audio_fileset")
        if fileset and fileset != audio_fileset:
            continue
        if audio_chapter in rule.get("skip", []):
            return None
        for shift in rule.get("shifts", []):
            lo, hi = shift.get("from"), shift.get("to")
            if lo is None or hi is None:
                continue
            if lo <= audio_chapter <= hi:
                return audio_chapter + shift.get("offset", 0)
        # A rule matched this book/fileset but no range covered this chapter.
        # That's an incomplete map rather than an intentional identity, so
        # skip instead of guessing — an unmapped chapter in a book known to
        # be misnumbered is exactly the case that produces silent nonsense.
        return None

    return audio_chapter


_config_cache: dict[str, LanguageConfig] = {}


def load_language_config(iso: str) -> LanguageConfig:
    """Load language config from config/languages/{iso}.toml.

    Falls back to default.toml if no language-specific config exists.
    Results are cached per ISO code.
    """
    if iso in _config_cache:
        return _config_cache[iso]

    config_path = CONFIG_DIR / f"{iso}.toml"
    if not config_path.exists():
        config_path = CONFIG_DIR / "default.toml"

    if config_path.exists():
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        config = LanguageConfig(
            iso=iso,
            pronunciation_map=data.get("pronunciation_map", {}),
            strip_marker_rules=data.get("strip_marker_rules", []),
            char_replacements=data.get("char_replacements", {}),
            strip_unicode_categories=data.get("strip_unicode_categories", ["Mn"]),
            mms_fallback_threshold=data.get("mms_fallback_threshold", 0.3),
            aramaic_passages=data.get("aramaic_passages", []),
            verse_only_mode=data.get("verse_only_mode", False),
            chapter_map=data.get("chapter_map", []),
            whisper=data.get("whisper", {}),
        )
    else:
        config = LanguageConfig(iso=iso)

    _config_cache[iso] = config
    return config


def format_verse_id(book: str, chapter_str: str) -> str:
    """Build the compact "id" field used by the word-timing JSON format:
    "<BOOK> <chapter>", chapter without zero-padding (e.g. "NUM 7", not
    "NUM 007").
    """
    return f"{book} {int(chapter_str)}"


def is_aramaic_chapter(book: str, chapter: int, config: LanguageConfig) -> bool:
    """Check if a book/chapter overlaps with a configured Aramaic passage.

    Parses formats like: "DAN 3", "DAN 2:4-49", "EZR 4:8-24"
    Returns True if the chapter is fully or partially Aramaic.
    """
    for passage in config.aramaic_passages:
        parts = passage.split()
        if len(parts) < 2:
            continue
        p_book = parts[0]
        if p_book != book:
            continue
        ch_part = parts[1]
        if ":" in ch_part:
            p_ch = int(ch_part.split(":")[0])
        else:
            p_ch = int(ch_part)
        if p_ch == chapter:
            return True
    return False


def strip_markers(text: str, config: LanguageConfig) -> str:
    """Remove non-spoken markers from reference text.

    For Hebrew: removes parashah/setumah markers (פ/ס).
    For other languages: no-op if no patterns configured.
    """
    for rule in config.strip_marker_rules:
        text = re.sub(rule["pattern"], rule.get("replacement", ""), text)
    return text.strip()


def clean_for_alignment(text: str, config: LanguageConfig) -> str:
    """Clean text for forced alignment and word counting.

    This is the canonical cleaning function. It must be used identically by
    mms_align_words.py (to prepare words for MMS-FA) and align_words.py
    (to count words per verse for word-to-verse mapping).

    Steps: NFD-decompose, strip diacritics, apply char replacements, apply
    pronunciation map, strip punctuation, collapse whitespace. Does NOT
    lowercase.

    NFD decomposition is essential: precomposed characters like Greek `Ἦ`
    (U+1F26) or `ί` (U+03AF) bundle the diacritic with the base letter, so
    stripping `Mn` alone does nothing. After NFD, the diacritic becomes a
    separate combining mark and gets stripped.
    """
    text = unicodedata.normalize("NFD", text)
    categories = set(config.strip_unicode_categories)
    text = "".join(c for c in text if unicodedata.category(c) not in categories)
    for old, new in config.char_replacements.items():
        text = text.replace(old, new)
    for original, replacement in config.pronunciation_map.items():
        text = text.replace(original, replacement)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str, config: LanguageConfig) -> str:
    """Normalize text for fuzzy matching.

    Same as clean_for_alignment but also lowercases for case-insensitive
    comparison. Used by Whisper verse alignment and fusion matching.
    """
    return clean_for_alignment(text.lower(), config)
