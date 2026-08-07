"""Shared text processing for the alignment pipeline.

Provides language-configurable text normalization used by:
  - mms_align_words.py   (Step 1b — MMS forced alignment)
  - align_words.py       (Step 2  — fusion)
  - whisper_transcribe.py (Step 1a — Whisper transcription)

Language-specific rules (pronunciation maps, marker patterns, character
replacements) are loaded from TOML config files in config/languages/.
"""

import re
import tomllib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

CONFIG_DIR = Path(__file__).parent / "config" / "languages"


@dataclass
class LanguageConfig:
    """Language-specific text processing configuration."""
    iso: str
    pronunciation_map: Dict[str, str] = field(default_factory=dict)
    strip_marker_rules: List[Dict[str, str]] = field(default_factory=list)
    char_replacements: Dict[str, str] = field(default_factory=dict)
    strip_unicode_categories: List[str] = field(default_factory=lambda: ["Mn"])
    mms_fallback_threshold: float = 0.3
    aramaic_passages: List[str] = field(default_factory=list)
    verse_only_mode: bool = False


_config_cache: Dict[str, LanguageConfig] = {}


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
        )
    else:
        config = LanguageConfig(iso=iso)

    _config_cache[iso] = config
    return config


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
