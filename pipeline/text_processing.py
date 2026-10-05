"""Shared text processing for the alignment pipeline.

Provides language-configurable text normalization used by:
  - mms_align_words.py   (Step 1b — MMS forced alignment)
  - align_words.py       (Step 2  — fusion)
  - whisper_transcribe.py (Step 1a — Whisper transcription)

Language-specific rules (pronunciation maps, marker patterns, character
replacements) are loaded from TOML config files in config/languages/.
"""

import json
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


def _is_sofria_json(raw: str) -> bool:
    """True if raw looks like a Proskomma "sofria" nested-USJ document
    rather than plain line-per-verse text.

    Confirmed 2026-10-02: DBT's JSON/USX text filesets are downloaded and
    written verbatim as the ".txt" file by download_language_content.py's
    download_text() (type == "path" branch) -- no extraction ever
    happened. 1,365 chapters across 10 isos (adx, bpx, fuh, hak, kmh, mai,
    por, sdq, syl, tzm) carry this raw JSON under a .txt extension today,
    named with a telltale "-json" fileset-tag suffix. Without this check,
    read_verse_texts() below silently misreads the whole JSON blob as
    "one giant verse" (its schema/metadata keys leaking in as "text"),
    which is exactly what produced a ~29,000-character pseudo-verse for
    adx/ADXNVS PSA 119 -- windowed against the full chapter's audio, that
    fed CTC forced_align() a cells count large enough to segfault
    (confirmed via PYTHONFAULTHANDLER=1). Root cause was the missing
    parse, not chapter length.
    """
    stripped = raw.lstrip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False
    constraints = data.get("schema", {}).get("constraints", [])
    return any(c.get("name") == "sofria" for c in constraints if isinstance(c, dict))


def _parse_sofria_json(raw: str) -> dict[str, str]:
    """Extract {verse_number_str: plain_text} from a sofria/USJ document.

    Walks the nested block/wrapper/mark/graft tree (see Proskomma's own
    "sofria" schema): a "wrapper" with subtype "verses" opens at a nested
    "mark" with subtype "verses_label" (atts.number is the verse number)
    and its own plain-string content is that verse's text. A "graft" --
    footnotes, headings, cross-references -- is always a SEPARATE
    sub-sequence, never part of the enclosing verse's spoken text, so it
    is skipped entirely rather than descended into. A combined-verse
    label (e.g. "3-4") files its text under the first number in the
    range -- not yet confirmed against a real example, kept simple until
    one surfaces.

    This is a focused verse-text extractor, not a full renderer. For the
    authoritative, more complete walk of this same schema (handles
    tables, milestones, meta_content -- none seen in this corpus so far),
    see bcv-commons/bcv-query's node_modules/proskomma-json-tools
    (render/renderers/SofriaRenderFromJson.js) -- a generic Node.js event
    walker over the identical block/content/graft/mark shape this
    function reads. Reconcile against it if a chapter's extracted verse
    text ever looks wrong in a way this function's simpler rules don't
    explain.
    """
    verses: dict[str, list[str]] = {}
    current_verse: str | None = None

    def walk_content(items) -> None:
        nonlocal current_verse
        for item in items:
            if isinstance(item, str):
                if current_verse is not None and item.strip():
                    verses.setdefault(current_verse, []).append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "graft":
                continue
            if item_type == "mark":
                if item.get("subtype") == "verses_label":
                    number = str(item.get("atts", {}).get("number", ""))
                    current_verse = number.split("-")[0] or None
                continue
            if "content" in item:
                walk_content(item["content"])

    def walk_blocks(blocks) -> None:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "graft":
                continue
            if "content" in block:
                walk_content(block["content"])
            elif "blocks" in block:
                walk_blocks(block["blocks"])

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    walk_blocks(data.get("sequence", {}).get("blocks", []))
    return {k: " ".join(v).strip() for k, v in verses.items() if v}


def read_verse_texts(text_path: Path, config: LanguageConfig) -> list[str]:
    """Read a reference text file into one string per verse.

    Replaces the naive "one physical line = one verse" reading previously
    duplicated across 5 call sites (align_verse_words.py, align_words.py,
    align_pipeline.py, mms_align_words.py x2). DBT's own ET text exports
    sometimes spread a single verse across multiple physical lines — an
    embedded blank line mid-verse, or indented continuation lines used for
    quoted/poetic material (e.g. an Isaiah quotation set as several
    indented sub-lines within a Luke verse). Treating each of those extra
    lines as its own verse silently shifts every later verse in the
    chapter onto the wrong timestamp for the rest of the chapter — not
    just a count mismatch, a real misattribution.

    Confirmed 2026-09-15 via a corpus-wide audit against DBT's own
    verse-count metadata (downloads/BB/.../*_timing.json's own verse_start
    entries — ground truth for verse count regardless of alignment
    quality): 2,278 chapters affected this way (median +3 extra verses,
    worst case 891 vs a real 22 in mai/MAIWBT ACT 1; LUK 3's genealogy —
    heavily poetic-quotation-formatted — was a repeat offender across
    several languages).

    Fix: a line beginning with whitespace is a continuation of the verse
    still being accumulated (its content is appended, not started fresh);
    a blank line is never itself a verse boundary or content. Validated
    against the confirmed-mismatch corpus (_obs_verify/validate_merge_
    heuristic.py): resolves 87.8% of over-split cases to DBT's own verse
    count exactly. Does NOT address the separate, much rarer under-split
    pattern (DBT reporting more verses than physical lines exist in our
    copy of the text) — confirmed that has a different, not-yet-understood
    cause (e.g. bgq/BGQWBT: no indentation/blank-line pattern at all, just
    fewer physical lines than DBT's verse count), unaffected by this
    change either way.
    """
    raw = text_path.read_text(encoding="utf-8")

    if _is_sofria_json(raw):
        by_number = _parse_sofria_json(raw)
        ordered = sorted(by_number.items(), key=lambda kv: int(kv[0]))
        return [strip_markers(v, config) for _, v in ordered]

    verses: list[str] = []
    for raw_line in raw.splitlines():
        if not raw_line.strip():
            continue
        if raw_line[:1].isspace():
            if verses:
                verses[-1] = f"{verses[-1]} {raw_line.strip()}"
            continue
        verses.append(raw_line)
    return [strip_markers(v, config) for v in verses]


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


# Scripts whose vowels (and stacked consonants) are combining marks.
# clean_for_alignment() strips every combining mark, which turns these texts
# into bare consonant strings ("भारत का" -> "भरत क"; Tibetan also loses its
# syllable separator and glues phrases into one word). Keeping the marks was
# measured 2026-10-03 on verse-only alignment: 54 languages, verses scoring
# below 0.5 8.1% -> 5.5%, chapters held back by the gate 8 -> 4 (Tibetan
# 75% -> 23% low-scoring), and 5.5% -> 5.4% of verses >1 s off DBT. Myanmar,
# Arabic, Hebrew and fusion mode got slightly worse, so they are not listed
# and keep the plain cleaning.
VOWEL_SIGN_SCRIPTS = (
    "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU",
    "KANNADA", "MALAYALAM", "TIBETAN", "KHMER", "LAO", "THAI", "THAANA", "KAYAH",
)


def _vowel_sign_mark(c: str) -> bool:
    return unicodedata.category(c)[0] == "M" and unicodedata.name(c, "").startswith(VOWEL_SIGN_SCRIPTS)


def uses_vowel_sign_script(text: str) -> bool:
    """True if text contains combining marks of a VOWEL_SIGN_SCRIPTS script."""
    return any(_vowel_sign_mark(c) for c in unicodedata.normalize("NFC", text))


def clean_for_alignment_keep_vowel_signs(text: str, config: LanguageConfig) -> str:
    """clean_for_alignment(), but keeping the combining marks (vowel signs,
    viramas, stacked consonants) of VOWEL_SIGN_SCRIPTS and treating the
    Tibetan tsheg as the word separator it is. Identical to
    clean_for_alignment() for text without such marks. Used by the
    verse-only aligner only (align_verse_words.py)."""
    if not uses_vowel_sign_script(text):
        return clean_for_alignment(text, config)
    text = unicodedata.normalize("NFC", text).replace("\u0f0b", " ").replace("\u0f0c", " ")
    for old, new in config.char_replacements.items():
        text = text.replace(old, new)
    for original, replacement in config.pronunciation_map.items():
        text = text.replace(original, replacement)
    kept = []
    for c in text:
        cat = unicodedata.category(c)
        if cat[0] in "LN" or c.isspace() or _vowel_sign_mark(c):
            kept.append(c)
        elif cat[0] == "M" or cat == "Cf":
            continue                  # other marks; zero-width (non-)joiners sit INSIDE words
        else:
            kept.append(" ")          # punctuation etc. separates words
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def normalize_text(text: str, config: LanguageConfig) -> str:
    """Normalize text for fuzzy matching.

    Same as clean_for_alignment but also lowercases for case-insensitive
    comparison. Used by Whisper verse alignment and fusion matching.
    """
    return clean_for_alignment(text.lower(), config)
