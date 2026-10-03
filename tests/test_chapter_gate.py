"""Verse-only chapter gate: defer to DBT / hold back when too many verses score low."""
import json
from pathlib import Path

import align_verse_words as avw
import checks as ppc


def _dbt(path: Path, stamps):
    path.write_text(json.dumps([{"verse_start": str(i + 1), "timestamp": t} for i, t in enumerate(stamps)]))


def test_usable_dbt_timing(tmp_path):
    mp3 = tmp_path / "MAT_001_XN1DA.mp3"
    assert not avw._usable_dbt_timing(mp3)
    _dbt(tmp_path / "MAT_001_XN1DA_timing.json", [1.0, 5.0, 9.0])
    assert avw._usable_dbt_timing(mp3)
    _dbt(tmp_path / "MAT_001_XN1DA_timing.json", [0, 0, 0])
    assert not avw._usable_dbt_timing(mp3)
    _dbt(tmp_path / "MAT_001_XN1DA_timing.json", [1.0, 9.0, 5.0])
    assert not avw._usable_dbt_timing(mp3)


def _run(tmp_path, monkeypatch, scores, with_dbt):
    d = tmp_path / "dl"; d.mkdir()
    mp3 = d / "MAT_001_XN1DA.mp3"; mp3.write_bytes(b"")
    if with_dbt:
        _dbt(d / "MAT_001_XN1DA_timing.json", [1.0 + i for i in range(len(scores))])
    txt = d / "MAT_001_X_ET.txt"; txt.write_text("\n".join(f"word{i} more" for i in range(len(scores))))
    out = tmp_path / "export/nt/xxx/XXXV/MAT"; out.mkdir(parents=True)
    results = [{"verse_index": i, "words": 2, "expected_start": i, "local_score": sc, "start": float(i), "end": i + 0.9,
                "word_results": [{"text": "w", "start": float(i), "end": i + 0.5, "score": sc}] * 2,
                "source": "local" if sc >= 0.35 else "interpolated"} for i, sc in enumerate(scores)]
    monkeypatch.setattr(avw, "verse_anchored_align", lambda *a, **k: results)

    class Cfg:
        mms_fallback_threshold = 0.3
    monkeypatch.setattr(avw, "read_verse_texts", lambda p, c: txt.read_text().split("\n"))
    monkeypatch.setattr(avw, "clean_for_alignment", lambda v, c: v)
    item = {"book": "MAT", "chapter": 1, "chapter_str": "001", "audio_path": mp3, "text_path": txt,
            "timing_path": out / "MAT_001_XN1DA_timing.json", "words_path": out / "MAT_001_XN1DA_words.json",
            "quality_path": out / "MAT_001_XN1DA_words_quality.json"}
    stats = avw.process_chapter_verse_only(item, None, None, None, None, None, Cfg())
    return stats, item


def test_gate_passes_good_chapter(tmp_path, monkeypatch):
    stats, item = _run(tmp_path, monkeypatch, [0.9] * 19 + [0.4], with_dbt=True)
    assert stats["gate"] == "pass"
    assert "pos" in json.loads(item["timing_path"].read_text())


def test_gate_defers_when_dbt_exists(tmp_path, monkeypatch):
    stats, item = _run(tmp_path, monkeypatch, [0.9] * 8 + [0.4] * 2, with_dbt=True)
    assert stats["gate"] == "defer_to_dbt"
    assert json.loads(item["timing_path"].read_text())["status"] == "defer_to_dbt"
    assert json.loads(item["words_path"].read_text())["status"] == "defer_to_dbt"
    q = json.loads(item["quality_path"].read_text())["summary"]
    assert q["gate"] == "defer_to_dbt" and q["low_score_share"] == 0.2


def test_gate_holds_back_without_dbt(tmp_path, monkeypatch):
    stats, item = _run(tmp_path, monkeypatch, [0.9] * 8 + [0.4] * 2, with_dbt=False)
    assert stats["gate"] == "held_back"
    assert "pos" in json.loads(item["timing_path"].read_text())
    assert ppc.is_held_back(item["quality_path"])


def test_quality_file_carries_method_tag(tmp_path, monkeypatch):
    from chapter_state import alignment_method
    _, item = _run(tmp_path, monkeypatch, [0.9] * 10, with_dbt=False)
    assert alignment_method(item["quality_path"]) == avw.ALIGNMENT_METHOD
    assert alignment_method(tmp_path / "missing.json") is None
