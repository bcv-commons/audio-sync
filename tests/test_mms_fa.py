#!/usr/bin/env python3
"""
Proof-of-concept: MMS forced alignment on 3 Hebrew chapters.
Tests torchaudio MMS_FA pipeline with uroman romanization.
Uses chunked processing to handle long audio files.
"""

import time
import unicodedata
import re
from pathlib import Path

import torch
import torchaudio
from uroman import Uroman

# ─── Config ─────────────────────────────────────────────────────────────────

CHUNK_DURATION_SEC = 30  # seconds per chunk
CHUNK_OVERLAP_SEC = 2    # overlap between chunks to avoid edge artifacts

CHAPTERS = [
    {
        "label": "GEN 17 (Whisper hallucinated)",
        "audio": "downloads/BB/ot/syncable/heb/HEBM95/GEN/GEN_017_HBRHMTO2DA.mp3",
        "text": "downloads/BB/ot/syncable/heb/HEBM95/GEN/GEN_017_HBRHMTO_ET.txt",
    },
    {
        "label": "GEN 24",
        "audio": "downloads/BB/ot/syncable/heb/HEBM95/GEN/GEN_024_HBRHMTO2DA.mp3",
        "text": "downloads/BB/ot/syncable/heb/HEBM95/GEN/GEN_024_HBRHMTO_ET.txt",
    },
    {
        "label": "EXO 12",
        "audio": "downloads/BB/ot/syncable/heb/HEBM95/EXO/EXO_012_HBRHMTO2DA.mp3",
        "text": "downloads/BB/ot/syncable/heb/HEBM95/EXO/EXO_012_HBRHMTO_ET.txt",
    },
]


def strip_niqqud(text: str) -> str:
    """Remove Hebrew niqqud/cantillation marks."""
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def clean_verse(text: str) -> str:
    """Clean a verse line for alignment: strip niqqud, maqaf, punctuation."""
    text = strip_niqqud(text)
    text = text.replace("\u05be", " ")  # maqaf → space
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_verses(text_path: str):
    """Load verse texts (one per line), strip trailing blanks."""
    with open(text_path, "r", encoding="utf-8") as f:
        verses = [line.rstrip("\n") for line in f.readlines()]
    while verses and not verses[-1].strip():
        verses.pop()
    return verses


def load_model():
    """Load model, tokenizer, aligner once on CPU."""
    bundle = torchaudio.pipelines.MMS_FA
    print("Loading MMS_FA model on CPU ...")
    t0 = time.time()
    model = bundle.get_model()  # CPU
    model.eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return bundle, model, tokenizer, aligner


def prepare_words(text, uroman, tokenizer):
    """Romanize text and prepare token-clean word lists."""
    romanized = uroman.romanize_string(text)
    orig_words = text.split()
    rom_words = romanized.split()

    dict_keys = set(tokenizer.dictionary.keys())
    clean_rom_words = []
    for w in rom_words:
        cleaned = "".join(c for c in w if c in dict_keys)
        clean_rom_words.append(cleaned if cleaned else "*")

    return orig_words, rom_words, clean_rom_words


def align_chunk(waveform_chunk, words_chunk, rom_words_chunk, orig_words_chunk,
                time_offset, bundle, model, tokenizer, aligner):
    """Align a single audio chunk with its corresponding words. Returns word results."""
    tokens = tokenizer(rom_words_chunk)

    with torch.no_grad():
        emission, _ = model(waveform_chunk)

    token_spans = aligner(emission[0], tokens)
    ratio = waveform_chunk.shape[1] / emission.shape[1] / bundle.sample_rate

    results = []
    for word_i, word_spans in enumerate(token_spans):
        if not word_spans:
            continue
        start_sec = word_spans[0].start * ratio + time_offset
        end_sec = word_spans[-1].end * ratio + time_offset
        score = sum(s.score for s in word_spans) / len(word_spans)

        orig_word = orig_words_chunk[word_i] if word_i < len(orig_words_chunk) else rom_words_chunk[word_i]
        rom_word = rom_words_chunk[word_i] if word_i < len(rom_words_chunk) else "?"
        results.append({
            "text": orig_word,
            "romanized": rom_word,
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "score": round(score, 3),
        })
    return results


def run_forced_alignment_chunked(audio_path, text, bundle, model, tokenizer, aligner, uroman):
    """
    Run MMS_FA forced alignment on long audio by chunking.
    Splits audio into ~30s chunks, distributes words proportionally,
    aligns each chunk, and stitches results.
    """
    # Load and resample audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    total_samples = waveform.shape[1]
    total_duration = total_samples / bundle.sample_rate
    print(f"  Audio duration: {total_duration:.1f}s ({total_samples} samples @ {bundle.sample_rate}Hz)")

    # Romanize and prepare words
    orig_words, rom_words, clean_rom_words = prepare_words(text, uroman, tokenizer)
    print(f"  Original (first 80):  {text[:80]}...")
    romanized_preview = " ".join(clean_rom_words[:12])
    print(f"  Romanized (first 12 words): {romanized_preview}...")

    total_words = len(clean_rom_words)

    # Calculate chunks
    chunk_samples = int(CHUNK_DURATION_SEC * bundle.sample_rate)
    overlap_samples = int(CHUNK_OVERLAP_SEC * bundle.sample_rate)
    step_samples = chunk_samples - overlap_samples

    all_results = []
    word_offset = 0

    chunk_idx = 0
    sample_start = 0
    while sample_start < total_samples and word_offset < total_words:
        sample_end = min(sample_start + chunk_samples, total_samples)
        chunk_waveform = waveform[:, sample_start:sample_end]
        chunk_duration = (sample_end - sample_start) / bundle.sample_rate
        time_offset = sample_start / bundle.sample_rate

        # Estimate how many words belong to this chunk (proportional to duration)
        # For the last chunk, take all remaining words
        if sample_start + step_samples >= total_samples:
            words_in_chunk = total_words - word_offset
        else:
            fraction = chunk_duration / total_duration
            words_in_chunk = max(1, round(total_words * fraction))
            # Don't exceed remaining words
            words_in_chunk = min(words_in_chunk, total_words - word_offset)

        chunk_orig = orig_words[word_offset:word_offset + words_in_chunk]
        chunk_rom = clean_rom_words[word_offset:word_offset + words_in_chunk]

        chunk_idx += 1
        print(f"  Chunk {chunk_idx}: {time_offset:.1f}s-{time_offset + chunk_duration:.1f}s, "
              f"words {word_offset+1}-{word_offset + words_in_chunk} "
              f"({words_in_chunk} words)")

        try:
            chunk_results = align_chunk(
                chunk_waveform, chunk_rom, chunk_rom, chunk_orig,
                time_offset, bundle, model, tokenizer, aligner
            )
            all_results.extend(chunk_results)
        except Exception as e:
            print(f"    Chunk {chunk_idx} ERROR: {e}")

        word_offset += words_in_chunk
        sample_start += step_samples

    return all_results


def main():
    print("=" * 70)
    print("MMS Forced Alignment Test — Hebrew (chunked)")
    print("=" * 70)

    bundle, model, tokenizer, aligner = load_model()
    uroman = Uroman()

    for chapter in CHAPTERS:
        print(f"\n{'─' * 60}")
        print(f"Chapter: {chapter['label']}")
        print(f"{'─' * 60}")

        audio_path = chapter["audio"]
        text_path = chapter["text"]

        if not Path(audio_path).exists():
            print(f"  SKIP: audio not found")
            continue
        if not Path(text_path).exists():
            print(f"  SKIP: text not found")
            continue

        # Load verses and join into one text
        verses = load_verses(text_path)
        cleaned_verses = [clean_verse(v) for v in verses]
        full_text = " ".join(v for v in cleaned_verses if v)

        word_count = len(full_text.split())
        print(f"  Verses: {len(verses)}, Words: {word_count}")

        # Run alignment
        t0 = time.time()
        try:
            results = run_forced_alignment_chunked(
                audio_path, full_text, bundle, model, tokenizer, aligner, uroman
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        elapsed = time.time() - t0

        print(f"\n  Aligned {len(results)} words in {elapsed:.1f}s")

        # Show first 10 and last 5 words
        print(f"\n  First 10 words:")
        for w in results[:10]:
            print(f"    {w['start']:7.2f}s - {w['end']:7.2f}s  score={w['score']:.3f}  "
                  f"{w['text']}")

        print(f"\n  Last 5 words:")
        for w in results[-5:]:
            print(f"    {w['start']:7.2f}s - {w['end']:7.2f}s  score={w['score']:.3f}  "
                  f"{w['text']}")

        # Stats
        scores = [w["score"] for w in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        low_score = sum(1 for s in scores if s < 0.5)
        print(f"\n  Avg score: {avg_score:.3f}, Low-confidence (<0.5): {low_score}/{len(scores)}")

        # Check monotonicity
        non_mono = 0
        for i in range(1, len(results)):
            if results[i]["start"] < results[i-1]["start"]:
                non_mono += 1
        print(f"  Non-monotonic: {non_mono}")

        # Show per-verse breakdown (first word timestamp per verse)
        print(f"\n  Per-verse first-word timestamps:")
        word_idx = 0
        for vi, verse in enumerate(cleaned_verses):
            if not verse:
                print(f"    v{vi+1:3d}: (empty)")
                continue
            vwords = verse.split()
            if word_idx < len(results):
                w = results[word_idx]
                print(f"    v{vi+1:3d}: {w['start']:7.2f}s  score={w['score']:.3f}  "
                      f"'{vwords[0][:20]}' → '{w['romanized']}'")
            else:
                print(f"    v{vi+1:3d}: (no alignment)")
            word_idx += len(vwords)


if __name__ == "__main__":
    main()
