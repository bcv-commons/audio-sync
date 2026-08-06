#!/usr/bin/env python3
"""Compare MMS_FA vs CTC-forced-aligner (wav2vec2-xlsr) on Hebrew audio.

Usage:
    python test_xlsr_alignment.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from uroman import Uroman

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from text_processing import load_language_config, strip_markers, clean_for_alignment

# ─── Config ──────────────────────────────────────────────────────────────

AUDIO_PATH = Path("downloads/BB/ot/syncable/heb/HEBM95/GEN/GEN_001_HBRHMTO2DA.mp3")
TEXT_PATH = Path("downloads/BB/ot/syncable/heb/HEBM95/GEN/GEN_001_HBRHMTO_ET.txt")
MMS_WORDS_PATH = Path("word-timing-data/ot/heb/HEBM95/GEN/GEN_001_HBRHMTO2DA_mms_words.json")


def load_reference_text():
    config = load_language_config("heb")
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        verse_texts = [strip_markers(line.rstrip("\n"), config) for line in f.readlines()]
    while verse_texts and not verse_texts[-1].strip():
        verse_texts.pop()
    cleaned = [clean_for_alignment(v, config) for v in verse_texts]
    non_empty = [v for v in cleaned if v]
    return " ".join(non_empty), non_empty


def run_ctc_aligner():
    """Run CTC forced aligner (wav2vec2-xlsr based) on Hebrew GEN 1."""
    from ctc_forced_aligner import (
        Alignment, Tokenizer, MODEL_URL, ensure_onnx_model,
        load_audio, generate_emissions, preprocess_text,
        get_alignments, get_spans, postprocess_results,
    )

    print("\n=== CTC Forced Aligner (wav2vec2-xlsr) ===")

    # Load model
    model_path = str(Path.home() / ".cache" / "ctc_forced_aligner" / "model.onnx")
    ensure_onnx_model(model_path, MODEL_URL)

    t0 = time.time()

    alignment = Alignment(model_path)
    tokenizer = alignment.tokenizer

    # Load audio
    audio_waveform = load_audio(str(AUDIO_PATH), ret_type="np")

    # Generate emissions
    emissions, stride = generate_emissions(alignment.model, audio_waveform)

    full_text, _ = load_reference_text()

    # Preprocess text with romanization for Hebrew
    tokens_starred, text_starred = preprocess_text(
        full_text, romanize=True, language="heb",
        split_size="word",
    )

    # Get alignments
    segments, scores, blank_token = get_alignments(emissions, tokens_starred, tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)

    # Post-process
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    elapsed = time.time() - t0
    print(f"Aligned {len(word_timestamps)} words in {elapsed:.1f}s")

    # Extract scores
    word_scores = [w['score'] for w in word_timestamps]
    avg = np.mean(word_scores) if word_scores else 0
    below_03 = sum(1 for s in word_scores if s < 0.3)
    below_015 = sum(1 for s in word_scores if s < 0.15)
    print(f"Avg score: {avg:.3f}")
    print(f"Words < 0.3: {below_03}/{len(word_scores)}")
    print(f"Words < 0.15: {below_015}/{len(word_scores)}")

    # Show first 15 words
    print("\nFirst 15 words:")
    for i, w in enumerate(word_timestamps[:15]):
        print(f"  [{i:3d}] {w['start']:7.2f}-{w['end']:7.2f}  "
              f"score={w['score']:.3f}  {w['text']}")

    return word_timestamps


def load_mms_results():
    """Load existing MMS alignment results for comparison."""
    print("\n=== MMS_FA (existing results) ===")
    data = json.load(open(MMS_WORDS_PATH))
    words = data["words"]
    scores = [w["score"] for w in words if w["score"] > 0]
    avg = sum(scores) / len(scores) if scores else 0
    below_03 = sum(1 for s in scores if s < 0.3)
    below_015 = sum(1 for s in scores if s < 0.15)

    print(f"Total words: {len(words)}")
    print(f"Avg score: {avg:.3f}")
    print(f"Words < 0.3: {below_03}/{len(words)}")
    print(f"Words < 0.15: {below_015}/{len(words)}")

    print("\nFirst 15 words:")
    for i, w in enumerate(words[:15]):
        print(f"  [{i:3d}] {w['start']:7.2f}-{w['end']:7.2f}  "
              f"score={w['score']:.3f}  {w['text']}")

    return words


def compare(mms_words, xlsr_words):
    """Compare timing placement between the two models."""
    print("\n=== Comparison ===")
    n = min(len(mms_words), len(xlsr_words))
    diffs = []
    for i in range(n):
        mms_start = mms_words[i]["start"]
        xlsr_start = xlsr_words[i]["start"]
        diffs.append(abs(mms_start - xlsr_start))

    if diffs:
        print(f"Word count: MMS={len(mms_words)}, XLSR={len(xlsr_words)}")
        print(f"Start-time differences (first {n} words):")
        print(f"  Mean: {np.mean(diffs):.3f}s")
        print(f"  Median: {np.median(diffs):.3f}s")
        print(f"  Max: {np.max(diffs):.3f}s")
        print(f"  Within 0.5s: {sum(1 for d in diffs if d < 0.5)}/{n}")
        print(f"  Within 1.0s: {sum(1 for d in diffs if d < 1.0)}/{n}")

        # Show biggest divergences
        indexed = [(d, i) for i, d in enumerate(diffs)]
        indexed.sort(reverse=True)
        print(f"\nLargest divergences:")
        for d, i in indexed[:10]:
            mw = mms_words[i]
            xw = xlsr_words[i]
            print(f"  [{i:3d}] MMS={mw['start']:7.2f} (score={mw['score']:.3f})  "
                  f"XLSR={xw['start']:7.2f} (score={xw['score']:.3f})  "
                  f"diff={d:.2f}s  {mw['text']}")


if __name__ == "__main__":
    mms_words = load_mms_results()
    xlsr_words = run_ctc_aligner()
    compare(mms_words, xlsr_words)
