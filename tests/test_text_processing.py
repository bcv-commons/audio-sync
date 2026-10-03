"""Unit tests for text_processing.py's verse-text reading -- both the
plain-text continuation-line merge heuristic and the sofria-JSON parser
(see internal-docs/single-pipeline-plan-2026-10-02.md for the bug this
second half fixes: DBT's text_json fileset was being written verbatim as
a .txt file and misread as one giant garbage verse).
"""
import json
import tempfile
from pathlib import Path

from text_processing import (
    LanguageConfig,
    _is_sofria_json,
    _parse_sofria_json,
    read_verse_texts,
)


def _default_config() -> LanguageConfig:
    return LanguageConfig(iso="test")


def _write_tmp(text: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    return Path(f.name)


class TestPlainTextVerseReading:
    def test_one_line_per_verse(self):
        p = _write_tmp("In the beginning\nGod created\nThe earth was formless")
        assert read_verse_texts(p, _default_config()) == [
            "In the beginning", "God created", "The earth was formless",
        ]

    def test_indented_continuation_line_merges_into_previous_verse(self):
        # A continuation line (leading whitespace) is NOT its own verse --
        # confirmed 2026-09-15 this over-splitting affected 2,278 chapters.
        p = _write_tmp("Verse one starts here\n  and continues here\nVerse two")
        verses = read_verse_texts(p, _default_config())
        assert len(verses) == 2
        assert verses[0] == "Verse one starts here and continues here"
        assert verses[1] == "Verse two"

    def test_blank_lines_are_never_verse_boundaries(self):
        p = _write_tmp("Verse one\n\n\nVerse two")
        assert read_verse_texts(p, _default_config()) == ["Verse one", "Verse two"]

    def test_leading_indented_line_with_nothing_to_merge_into_is_dropped(self):
        p = _write_tmp("  orphan continuation\nReal verse")
        assert read_verse_texts(p, _default_config()) == ["Real verse"]


def _sofria_doc(verses: dict[str, str]) -> str:
    """Minimal valid sofria document with the given {number: text} verses,
    shaped like a real DBT text_json fileset response (confirmed against
    adx/ADXNVS PSA 119's actual file)."""
    content = []
    for number, text in verses.items():
        content.append({
            "type": "wrapper", "subtype": "verses",
            "content": [
                {"type": "mark", "subtype": "verses_label", "atts": {"number": number}},
                text,
            ],
        })
    return json.dumps({
        "schema": {"structure": "nested", "structure_version": "0.2.1",
                   "constraints": [{"name": "sofria", "version": "0.2.1"}]},
        "metadata": {},
        "sequence": {"type": "main", "blocks": [
            {"type": "paragraph", "subtype": "usfm:p", "content": content},
        ]},
    })


class TestSofriaJsonVerseReading:
    def test_is_sofria_json_detects_real_shape(self):
        assert _is_sofria_json(_sofria_doc({"1": "hello"})) is True

    def test_is_sofria_json_rejects_plain_text(self):
        assert _is_sofria_json("In the beginning\nGod created") is False

    def test_is_sofria_json_rejects_other_json(self):
        assert _is_sofria_json(json.dumps({"not": "sofria"})) is False

    def test_is_sofria_json_rejects_empty_file(self):
        assert _is_sofria_json("") is False

    def test_parse_sofria_json_extracts_verses_in_order(self):
        raw = _sofria_doc({"1": "First verse text", "2": "Second verse text", "3": "Third"})
        by_number = _parse_sofria_json(raw)
        assert by_number == {"1": "First verse text", "2": "Second verse text", "3": "Third"}

    def test_parse_sofria_json_skips_footnote_grafts(self):
        raw = json.dumps({
            "schema": {"constraints": [{"name": "sofria"}]},
            "sequence": {"blocks": [{
                "type": "paragraph", "content": [{
                    "type": "wrapper", "subtype": "verses",
                    "content": [
                        {"type": "mark", "subtype": "verses_label", "atts": {"number": "1"}},
                        "Real verse text",
                        {"type": "graft", "subtype": "footnote",
                         "sequence": {"blocks": [{"content": ["footnote text that must not leak in"]}]}},
                        "more real text",
                    ],
                }],
            }]},
        })
        by_number = _parse_sofria_json(raw)
        assert by_number == {"1": "Real verse text more real text"}
        assert "footnote" not in by_number["1"]

    def test_read_verse_texts_routes_sofria_through_json_parser(self):
        raw = _sofria_doc({"1": "Alpha", "2": "Beta"})
        p = _write_tmp(raw)
        assert read_verse_texts(p, _default_config()) == ["Alpha", "Beta"]

    def test_read_verse_texts_does_not_produce_one_giant_pseudo_verse(self):
        # Regression guard for the actual bug: a 176-verse sofria document
        # must read back as 176 verses, never collapse to ~2 with one
        # pathologically long "verse" absorbing the whole schema/metadata.
        raw = _sofria_doc({str(i): f"verse number {i} text" for i in range(1, 177)})
        p = _write_tmp(raw)
        verses = read_verse_texts(p, _default_config())
        assert len(verses) == 176
        assert all(len(v) < 100 for v in verses)
