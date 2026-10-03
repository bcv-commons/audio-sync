"""Unit tests for the publish-gate checks: backwards-jump detection,
fallback-corruption detection, and legacy-format detection.
"""
import json
from pathlib import Path

from checks import has_backwards_jump, has_fallback_corruption, is_legacy_format


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data))


class TestHasBackwardsJump:
    def test_false_for_monotonic_pos(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, {"pos": [1.0, 2.0, 3.0]})
        assert has_backwards_jump(p) is False

    def test_true_for_a_backwards_jump(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, {"pos": [1.0, 5.0, 2.0]})
        assert has_backwards_jump(p) is True

    def test_nulls_are_skipped_not_treated_as_zero(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, {"pos": [1.0, None, 2.0]})
        assert has_backwards_jump(p) is False

    def test_false_for_legacy_verbose_list_format(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, [
            {"verse_start": "1", "timestamp": 1.0},
            {"verse_start": "2", "timestamp": 2.0},
        ])
        assert has_backwards_jump(p) is False

    def test_true_for_backwards_jump_in_legacy_format(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, [
            {"verse_start": "1", "timestamp": 5.0},
            {"verse_start": "2", "timestamp": 1.0},
        ])
        assert has_backwards_jump(p) is True

    def test_false_for_unrelated_shape_eg_obs_or_defer_to_dbt(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, {"status": "defer_to_dbt"})
        assert has_backwards_jump(p) is False

    def test_false_for_corrupt_json(self, tmp_path):
        p = tmp_path / "x_timing.json"
        p.write_text("not json")
        assert has_backwards_jump(p) is False

    def test_false_for_missing_file(self, tmp_path):
        assert has_backwards_jump(tmp_path / "nope.json") is False


class TestHasFallbackCorruption:
    def test_false_for_healthy_chapter(self, tmp_path):
        p = tmp_path / "x_words_quality.json"
        _write(p, {"summary": {"total_words": 100, "null_count": 2, "avg_score": 0.85}})
        assert has_fallback_corruption(p) is False

    def test_true_for_all_null_poisoned_context(self, tmp_path):
        p = tmp_path / "x_words_quality.json"
        _write(p, {"summary": {"total_words": 100, "null_count": 100, "avg_score": 0.0}})
        assert has_fallback_corruption(p) is True

    def test_false_when_no_summary(self, tmp_path):
        p = tmp_path / "x_words_quality.json"
        _write(p, {})
        assert has_fallback_corruption(p) is False

    def test_false_when_total_words_zero(self, tmp_path):
        p = tmp_path / "x_words_quality.json"
        _write(p, {"summary": {"total_words": 0, "null_count": 0, "avg_score": 0.0}})
        assert has_fallback_corruption(p) is False


class TestIsLegacyFormat:
    def test_true_for_list_shape(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, [{"verse_start": "1", "timestamp": 1.0}])
        assert is_legacy_format(p) is True

    def test_false_for_compact_dict_shape(self, tmp_path):
        p = tmp_path / "x_timing.json"
        _write(p, {"pos": [1.0]})
        assert is_legacy_format(p) is False

    def test_false_for_obs_path_even_if_list_shaped(self, tmp_path):
        obs_dir = tmp_path / "export" / "timing-data" / "obs" / "dub"
        obs_dir.mkdir(parents=True)
        p = obs_dir / "01_timing.json"
        _write(p, [{"segment": "01", "timestamp": 1.0}])
        assert is_legacy_format(p) is False
