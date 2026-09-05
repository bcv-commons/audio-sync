#!/usr/bin/env python3
"""
Report audio filesets whose ACTUAL chapter count differs from EXPECTED.

Audio-side versification is published nowhere — the bibles repo declares a
text scheme per edition (cdn.bibel.wiki/dbt/_vrs/index.json) but nothing
states what numbering an audio fileset uses, or even how many chapters it
really contains. That gap is what let bul/BULCBV go unnoticed: its audio is
LXX/Orthodox-numbered against Masoretic text, so audio chapter N reads text
chapter N+1, for 135+ chapters.

We're better placed to produce this than they are, since we already
download and decode every audio fileset during alignment. But a full census
of chapter counts is mostly noise — for the overwhelming majority the count
is simply correct. The signal is the DIFFERENCES, so that is all this
reports:

  * FEWER than expected  — chapters the catalog claims that don't exist
    (confirmed real: DBT's metadata claimed 4 chapters of BULCBV Malachi;
    chapter 4 is a genuine 404, so Hebrew Mal 3:19-24 has no audio at all)
  * MORE than expected   — a versification mismatch is likely: the audio is
    using a scheme that splits where the text merges

EXPECTED comes from the edition's declared text versification shape
(_vrs/<scheme>.vrs). ACTUAL is reconstructed from durable evidence, because
source audio is purged after a language completes (tools/purge_aligned_audio
.py) so counting mp3s on disk would under-report almost everywhere. In
preference order per book:

    1. mp3 files still on disk          (most direct, rarely available)
    2. distinct chapters in word-timing-data/  (Whisper/MMS intermediates —
       these exist only where audio really existed)
    3. distinct chapters in export/timing-data/

Note (3) is the weakest: a chapter skipped for having no matching text
never produces output, so a book can under-report. Findings are therefore
leads to confirm, not conclusions — which is also why this prints a report
rather than writing config.

    python tools/audio_chapter_anomalies.py
    python tools/audio_chapter_anomalies.py --iso bul
"""

import argparse
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from versification import lookup_vrs  # noqa: E402

VRS_BASE = "https://cdn.bibel.wiki/_vrs/"
CACHE_DIR = Path("api-cache/vrs")
DOWNLOADS = Path("downloads/BB")
WORD_TIMING = Path("word-timing-data")
TIMING = Path("export/timing-data")

_shape_cache: dict[str, dict] = {}


def load_shape(scheme: str) -> dict:
    """{book: {chapter: verse_count}} for a versification scheme."""
    if scheme in _shape_cache:
        return _shape_cache[scheme]
    cache_path = CACHE_DIR / f"{scheme}.vrs"
    raw = None
    if cache_path.exists():
        try:
            raw = cache_path.read_bytes()
        except OSError:
            pass
    if raw is None:
        try:
            req = urllib.request.Request(
                f"{VRS_BASE}{scheme}.vrs", headers={"User-Agent": "audio-sync"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw)
        except Exception:
            _shape_cache[scheme] = {}
            return {}
    out = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        m = re.match(r"^([A-Z0-9]{3}) (.+)$", line.strip())
        if not m:
            continue
        d = {}
        for tok in m.group(2).split():
            if ":" in tok:
                c, v = tok.split(":")
                try:
                    d[int(c)] = int(v)
                except ValueError:
                    pass
        if d:
            out[m.group(1)] = d
    _shape_cache[scheme] = out
    return out


def chapters_from(paths, pattern) -> set[int]:
    found = set()
    for p in paths:
        m = re.match(pattern, p.name)
        if m:
            try:
                found.add(int(m.group(1)))
            except ValueError:
                pass
    return found


def actual_chapters(canon: str, iso: str, did: str, book: str) -> tuple[set[int], str]:
    """Which audio chapters we have evidence ever existed.

    UNION of all three sources rather than a priority order. Each is partial
    evidence of the same fact ("audio for chapter N existed"), and no single
    one is complete: source audio is purged after a language finishes, so
    surviving mp3s are a near-empty sample — preferring them as "most
    direct" made bul/BULCBV PSA report 1 chapter instead of 150, since
    exactly one mp3 survived. Taking the union avoids letting the sparsest
    evidence veto the richest.
    """
    found = set()
    sources = []
    for base, pattern, label in (
        (DOWNLOADS, "*.mp3", "mp3"),
        (WORD_TIMING, "*_words.json", "word-timing"),
        (TIMING, "*_timing.json", "timing"),
    ):
        d = base / canon / iso / did / book
        if not d.exists():
            continue
        got = chapters_from(d.glob(pattern), rf"{book}_(\d+)_")
        if got:
            found |= got
            sources.append(label)
    return found, "+".join(sources) if sources else "none"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", help="Only this language")
    args = ap.parse_args()

    findings = []
    editions = 0
    unknown_scheme = 0

    for canon_dir in sorted(TIMING.glob("*")):
        canon = canon_dir.name
        if canon not in ("ot", "nt"):
            continue
        for iso_dir in sorted(canon_dir.glob("*")):
            iso = iso_dir.name
            if args.iso and iso != args.iso:
                continue
            for did_dir in sorted(iso_dir.glob("*")):
                did = did_dir.name
                scheme = lookup_vrs(iso, did)
                if not scheme:
                    unknown_scheme += 1
                    continue
                shape = load_shape(scheme)
                if not shape:
                    continue
                editions += 1
                for book_dir in sorted(did_dir.glob("*")):
                    book = book_dir.name
                    if book not in shape:
                        continue
                    expected = max(shape[book])
                    actual, source = actual_chapters(canon, iso, did, book)
                    if not actual:
                        continue
                    got = max(actual)
                    # Distinguish real anomalies from incomplete alignment
                    # coverage, which otherwise drowns the report: a book we
                    # only partly aligned shows scattered interior gaps
                    # (urd/URDIRV DEU: expected 34, present 3, missing
                    # 1,2,4,5...) and says nothing about the audio.
                    #   got > expected  -> always real; excess chapters can't
                    #                      come from aligning less
                    #   got < expected  -> only trust a clean SUFFIX, i.e.
                    #                      1..got all present and only the
                    #                      tail absent
                    missing_all = set(range(1, expected + 1)) - actual
                    is_suffix = missing_all and all(m > got for m in missing_all)
                    if got == expected or not (got > expected or is_suffix):
                        continue
                    findings.append({
                        "iso": iso, "canon": canon, "did": did, "book": book,
                        "scheme": scheme, "expected": expected, "actual": got,
                        "source": source,
                        # Only meaningful when the audio is SHORT of the
                        # scheme; when it has more chapters than expected
                        # the interesting part is the excess, not gaps.
                        "missing": sorted(missing_all)[:8] if got < expected else [],
                    })

    print(f"Editions checked (scheme known): {editions}")
    print(f"Editions skipped (scheme unknown): {unknown_scheme}\n")
    if not findings:
        print("No audio chapter-count anomalies found.")
        return

    print(f"ANOMALIES — {len(findings)} (edition, book) pair(s) where the highest "
          f"chapter present differs from the scheme's chapter count:\n")
    fmt = "  {:<5} {:<12} {:<5} {:<5} {:>8} {:>7}  {:<12} {}"
    print(fmt.format("iso", "fileset", "book", "vrs", "expected", "actual",
                     "evidence", "missing"))
    print("  " + "-" * 78)
    for f in sorted(findings, key=lambda f: (f["iso"], f["did"], f["book"])):
        miss = ",".join(str(m) for m in f["missing"]) or "-"
        print(fmt.format(f["iso"], f["did"], f["book"], f["scheme"],
                         f["expected"], f["actual"], f["source"], miss))
    print("\nFEWER than expected  -> catalog claims chapters the audio doesn't have")
    print("MORE than expected   -> likely audio/text versification mismatch")


if __name__ == "__main__":
    main()
