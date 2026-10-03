"""Chunked-emission bookkeeping and wildcard-padded alignment.

No model weights and no GPU: the chunker is exercised with a stand-in model
whose output depends only on local samples, and the aligner with hand-built
emissions.
"""
import math

import pytest
import torch
import torchaudio

import mms_align_words as mms

HOP = 320
FIELD = 400


class LocalModel(torch.nn.Module):
    """Same frame arithmetic as wav2vec2's conv front end, but each frame is
    a pure function of its own 400 samples -- so a correct chunker must
    reproduce the single-pass output exactly."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.pass_lengths = []

    def forward(self, waveform):
        n = waveform.shape[1]
        self.pass_lengths.append(n)
        frames = waveform[0].unfold(0, FIELD, HOP)
        feats = torch.stack([frames.mean(dim=1), frames.amax(dim=1)], dim=1)
        assert feats.shape[0] == (n - FIELD) // HOP + 1
        return feats.unsqueeze(0), None


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(mms, "_MAX_CHUNK_SAMPLES", 32_000)
    return 32_000


@pytest.mark.parametrize("n_samples", [
    32_001, 40_000, 48_000, 48_050, 48_100, 48_399, 48_400, 64_000, 100_003, 163_217,
])
def test_chunked_emission_equals_single_pass(small_chunks, n_samples):
    torch.manual_seed(n_samples)
    waveform = torch.randn(1, n_samples)
    model = LocalModel()
    expected, _ = model(waveform)
    model.pass_lengths.clear()

    got = mms._compute_emission_chunked(waveform, model)

    assert got.shape == expected.shape
    assert torch.equal(got, expected)
    assert len(model.pass_lengths) > 1
    assert max(model.pass_lengths) <= small_chunks


def test_short_input_is_one_pass(small_chunks):
    model = LocalModel()
    mms._compute_emission_chunked(torch.randn(1, 32_000), model)
    assert model.pass_lengths == [32_000]


def test_interior_chunks_get_context_on_both_sides(small_chunks):
    model = LocalModel()
    mms._compute_emission_chunked(torch.randn(1, 200_000), model)
    context = min(mms._CHUNK_CONTEXT, small_chunks // 4)
    core = small_chunks - 2 * context
    # first pass has no left context, interior passes are core + both contexts
    assert model.pass_lengths[0] == core + context
    assert all(n == core + 2 * context for n in model.pass_lengths[1:-2])


# ── wildcard-padded alignment ────────────────────────────────────────────────

class IdentityRomanizer:
    def romanize_string(self, text):
        return text


@pytest.fixture(scope="module")
def fa():
    bundle = torchaudio.pipelines.MMS_FA
    return bundle, bundle.get_tokenizer(), bundle.get_aligner()


def _emission(tokenizer, text_start_frame, text, total_frames):
    """Frames outside the text are 'someone else talking': every real letter
    equally likely, blank very unlikely. The text itself is one sharp frame
    per letter with a blank frame between letters."""
    d = tokenizer.dictionary
    star = d[mms.STAR_WORD]
    vocab = len(d)
    letters = [i for i in range(vocab) if i not in (0, star)]

    foreign = torch.full((vocab,), -20.0)
    foreign[letters] = math.log(1.0 / len(letters))
    em = foreign.repeat(total_frames, 1)

    f = text_start_frame
    for ch in text.replace(" ", ""):
        row = torch.full((vocab,), -20.0); row[d[ch]] = -0.01
        em[f] = row
        blank = torch.full((vocab,), -20.0); blank[0] = -0.01
        em[f + 1] = blank
        f += 2
    em[:, star] = 0.0
    return em.unsqueeze(0), text_start_frame * 0.02, f * 0.02


def test_star_edges_finds_text_inside_a_wider_window(fa):
    bundle, tokenizer, aligner = fa
    emission, true_start, true_end = _emission(tokenizer, 400, "abc de fgh", 1500)
    words = mms._align_emission(
        emission, 1500 * HOP, "abc de fgh", bundle, tokenizer, aligner, IdentityRomanizer(),
        star_edges=True,
    )
    assert [w["text"] for w in words] == ["abc", "de", "fgh"]
    assert words[0]["start"] == pytest.approx(true_start, abs=0.04)
    assert words[-1]["end"] == pytest.approx(true_end, abs=0.06)


def test_without_star_the_text_is_smeared_over_the_window(fa):
    """Documents the failure star_edges exists to prevent."""
    bundle, tokenizer, aligner = fa
    emission, true_start, true_end = _emission(tokenizer, 400, "abc de fgh", 1500)
    words = mms._align_emission(
        emission, 1500 * HOP, "abc de fgh", bundle, tokenizer, aligner, IdentityRomanizer(),
    )
    smeared_left = true_start - words[0]["start"] > 1.0
    smeared_right = words[-1]["end"] - true_end > 1.0
    assert smeared_left or smeared_right


def test_star_edges_returns_only_the_real_words(fa):
    bundle, tokenizer, aligner = fa
    emission, _, _ = _emission(tokenizer, 10, "ab", 60)
    words = mms._align_emission(
        emission, 60 * HOP, "ab", bundle, tokenizer, aligner, IdentityRomanizer(), star_edges=True,
    )
    assert len(words) == 1 and words[0]["text"] == "ab"


def test_infeasible_window_returns_unaligned_words(fa):
    bundle, tokenizer, aligner = fa
    emission, _, _ = _emission(tokenizer, 0, "a", 4)
    words = mms._align_emission(
        emission, 4 * HOP, "abcdef ghijkl", bundle, tokenizer, aligner, IdentityRomanizer(),
        star_edges=True,
    )
    assert [w["start"] for w in words] == [None, None]


# ── whole-file emission, sliced per window ───────────────────────────────────

def test_align_window_reports_times_in_the_files_own_timeframe(fa):
    bundle, tokenizer, aligner = fa
    total_frames = 3000                                   # a 60 s "file"
    emission, true_start, true_end = _emission(tokenizer, 1400, "abc de fgh", total_frames)
    words = mms.align_window(
        emission, total_frames * 0.02, 20.0, 45.0, "abc de fgh",
        bundle, tokenizer, aligner, IdentityRomanizer(),
    )
    assert [w["text"] for w in words] == ["abc", "de", "fgh"]
    assert words[0]["start"] == pytest.approx(true_start, abs=0.04)      # 28.0 s
    assert words[-1]["end"] == pytest.approx(true_end, abs=0.06)


def test_align_window_same_answer_wherever_the_window_starts(fa):
    bundle, tokenizer, aligner = fa
    emission, true_start, _ = _emission(tokenizer, 1400, "abc de fgh", 3000)
    starts = {
        mms.align_window(emission, 60.0, lo, hi, "abc de fgh", bundle, tokenizer, aligner,
                         IdentityRomanizer())[0]["start"]
        for lo, hi in [(20.0, 45.0), (5.0, 59.0), (27.5, 30.0), (0.0, 60.0)]
    }
    assert len(starts) == 1 and starts.pop() == pytest.approx(true_start, abs=0.04)


def test_align_window_empty_slice(fa):
    bundle, tokenizer, aligner = fa
    emission, _, _ = _emission(tokenizer, 10, "ab", 60)
    assert mms.align_window(emission, 1.2, 5.0, 6.0, "ab", bundle, tokenizer, aligner, IdentityRomanizer()) == []
