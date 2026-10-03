#!/usr/bin/env python3
"""CLI diagnostic for pipeline/vowel_pacing.py's constant-speech-rate
fingerprint -- prints the per-verse deviation table for one chapter (both
DBT's own timing and our pipeline's own timing, each anchored on its own
first/last verse) plus a chapter-level trust comparison (RMS deviation:
lower means that source's own claimed pacing looks more like real constant-
rate speech). Also runs detect_repeated_phrases() and warns if the chapter
has a formulaic/repeated passage where content-based signals (this one
included) can't be trusted to disambiguate.

This is a diagnostic only -- it doesn't touch the arbiter, the correction
tool, or the merge pipeline. See pipeline/vowel_pacing.py's own docstring
for the method and its confirmed failure mode.

Usage:
    python tools/vowel_pacing_diagnose.py --iso mtr --distinct-id MTRNLC \
        --book MAT --chapter 013 --fileset MTRNLCN1DA
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from timing_files import load_timing_verses, DOWNLOADS_DIR  # noqa: E402
from text_processing import read_verse_texts, load_language_config  # noqa: E402
from vowel_pacing import verse_vowel_counts, fingerprint, detect_repeated_phrases  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", required=True)
    parser.add_argument("--distinct-id", required=True)
    parser.add_argument("--book", required=True)
    parser.add_argument("--chapter", required=True, help="Zero-padded, e.g. 013")
    parser.add_argument("--fileset", required=True, help="e.g. MTRNLCN1DA")
    parser.add_argument("--canon", default="nt", choices=["nt", "ot"])
    parser.add_argument("--full", action="store_true", help="Print every verse, not a thinned view")
    args = parser.parse_args()

    book_dir = DOWNLOADS_DIR / args.canon / args.iso / args.distinct_id / args.book
    dl_path = book_dir / f"{args.book}_{args.chapter}_{args.fileset}_timing.json"
    pl_path = Path("export/timing-data") / args.canon / args.iso / args.distinct_id / args.book / \
        f"{args.book}_{args.chapter}_{args.fileset}_timing.json"
    text_matches = list(book_dir.glob(f"{args.book}_{args.chapter}_*.txt"))

    if not dl_path.exists():
        print(f"DBT timing not found: {dl_path}")
        sys.exit(1)
    if not pl_path.exists():
        print(f"Pipeline timing not found: {pl_path}")
        sys.exit(1)
    if not text_matches:
        print(f"Reference text not found under {book_dir}")
        sys.exit(1)

    dl_v = load_timing_verses(dl_path)
    pl_v = load_timing_verses(pl_path)
    try:
        config = load_language_config(args.iso)
    except Exception:
        config = load_language_config("default")
    ref_verses = read_verse_texts(text_matches[0], config)

    from uroman import Uroman
    uroman = Uroman()

    vcounts = verse_vowel_counts(ref_verses, config, uroman)

    dl_fp = fingerprint(dl_v, vcounts)
    pl_fp = fingerprint(pl_v, vcounts)

    if dl_fp is None or pl_fp is None:
        print("Not enough usable verses to build a fingerprint for one or both sources.")
        sys.exit(1)

    print(f"Chapter {args.book} {args.chapter} ({args.iso}/{args.distinct_id})")
    print(f"DBT anchors:  v{dl_fp.anchor_lo} -> v{dl_fp.anchor_hi}")
    print(f"Ours anchors: v{pl_fp.anchor_lo} -> v{pl_fp.anchor_hi}")
    print()

    all_verses = sorted(set(dl_fp.per_verse) | set(pl_fp.per_verse), key=lambda x: int(x))
    step = 1 if args.full else max(1, len(all_verses) // 25)
    print(f"{'v':>4} {'vowels':>6} | {'DBT_t':>8} {'DBT_exp':>8} {'DBT_dev':>8} | {'ours_t':>8} {'ours_exp':>8} {'ours_dev':>8}")
    for i, v in enumerate(all_verses):
        if i % step != 0 and i != len(all_verses) - 1:
            continue
        d = dl_fp.per_verse.get(v)
        p = pl_fp.per_verse.get(v)
        d_str = f"{d['t']:>8.1f} {d['expected']:>8.1f} {d['dev']:>+8.1f}" if d else f"{'--':>8} {'--':>8} {'--':>8}"
        p_str = f"{p['t']:>8.1f} {p['expected']:>8.1f} {p['dev']:>+8.1f}" if p else f"{'--':>8} {'--':>8} {'--':>8}"
        print(f"{v:>4} {vcounts.get(v, 0):>6} | {d_str} | {p_str}")

    print()
    print(f"DBT  RMS deviation: {dl_fp.rms_dev:6.2f}s   (worst: {dl_fp.max_abs_dev:.1f}s at v{dl_fp.max_abs_dev_verse})")
    print(f"Ours RMS deviation: {pl_fp.rms_dev:6.2f}s   (worst: {pl_fp.max_abs_dev:.1f}s at v{pl_fp.max_abs_dev_verse})")
    verdict = "DBT" if dl_fp.rms_dev < pl_fp.rms_dev else "OURS"
    print(f"=> more constant-rate-plausible pacing: {verdict}")

    repeats = detect_repeated_phrases(ref_verses, config, uroman)
    if repeats:
        print()
        print(f"WARNING: {len(repeats)} repeated/formulaic verse pair(s) found -- content-based "
              f"signals (including this one) may not be able to disambiguate near these verses:")
        for va, vb, jac in repeats:
            print(f"  v{va} <-> v{vb}  (jaccard={jac})")


if __name__ == "__main__":
    main()
