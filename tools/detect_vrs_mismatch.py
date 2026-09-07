#!/usr/bin/env python3
"""
Detect audio/text versification mismatches WITHOUT needing a transcript.

THE GAP THIS CLOSES
-------------------
We found bul/BULCBV (LXX-numbered audio against Masoretic text, 135+ Psalms
silently mispaired) by scoring each chapter's Whisper transcript against
every candidate text chapter. That worked, and we then swept the corpus
with it and reported "BULCBV is the only mismatch we can see".

That claim was only ever true of chapters that HAVE a transcript. 83 of our
128 NT/OT languages run in verse_only_mode, which skips Whisper entirely —
so the sweep was structurally blind to them. As of 2026-09-05 that leaves
19 editions with a non-`eng` declared text scheme and no transcript
anywhere: exactly the population a second BULCBV would hide in.

HOW THIS WORKS INSTEAD
----------------------
It needs no ASR, and no audio: it reconstructs the one audio-derived
quantity we already publish. Each chapter's *_timing.json carries "pos",
the verse start times, so max(pos) is a lower-bound proxy for how long that
audio chapter runs. Chapter length in the TEXT is known independently from
the declared scheme's .vrs shape (verses per chapter).

For a correctly paired book those two vectors track each other: a long
chapter takes longer to read. Under an off-by-one mismatch they still track
each other, but only after shifting one of them — which is precisely what
makes it detectable. So for each book we correlate durations against verse
counts at several offsets and report the best one:

    offset 0 wins   -> pairing is consistent (the overwhelming majority)
    offset +/-1 wins clearly -> the audio is numbered in a different scheme

This is a WEAKER signal than transcript matching — it is a structural
correlation, not evidence about content — so it is tuned to surface leads,
not to conclude. A hit means "look at this edition", not "this is broken".
Confirm with tools/derive_chapter_map.py (what the published versification
data says the mapping should be) before writing any config.

VALIDATION
----------
Run with --self-test to check it against the two editions whose answers we
established independently:

    bul/BULCBV PSA -> must find a shift (its audio really is offset)
    kaz/KAZKAZ PSA -> must find none  (rso text AND rso audio, correctly
                                       paired; 110 of 111 transcripts
                                       agreed at offset 0)

    python tools/detect_vrs_mismatch.py --self-test
    python tools/detect_vrs_mismatch.py                 # blind editions
    python tools/detect_vrs_mismatch.py --all           # every edition
    python tools/detect_vrs_mismatch.py --iso bul
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from text_processing import load_language_config  # noqa: E402
from versification import lookup_vrs  # noqa: E402

TIMING = Path("export/timing-data")
CACHE_DIR = Path("api-cache/vrs")
VRS_BASE = "https://cdn.bibel.wiki/_vrs/"
CONFIG_DIR = Path("pipeline/config/languages")

# A book needs enough chapters for a correlation to mean anything. Below
# this, one unusually long chapter dominates and the "best offset" is
# noise — most false leads in early runs were 3-4 chapter books.
MIN_CHAPTERS = 8

# How much better a shifted offset must correlate than offset 0 before it
# is worth a human looking. Chosen from the observed spread: bul/BULCBV's
# true shift beats offset 0 by a wide margin, while correctly-paired books
# cluster near zero difference.
MIN_GAIN = 0.15

# Offset 0 has to be genuinely poor as well — a book can be well-paired AND
# correlate slightly better one step over, purely by chance.
MAX_BASELINE = 0.80

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
    out: dict[str, dict[int, int]] = {}
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


def chapter_durations(book_dir: Path, book: str) -> dict[int, float]:
    """{chapter: duration proxy} from our own published timings.

    max(pos) is the last verse's START, so this under-reports every chapter
    by one verse's worth of speech. That is fine: the correlation only
    needs the values to scale together, and the bias applies uniformly.
    """
    out = {}
    for p in book_dir.glob(f"{book}_*_timing.json"):
        m = re.match(rf"{book}_(\d+)_", p.name)
        if not m:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        pos = data.get("pos") if isinstance(data, dict) else None
        if not pos:
            continue  # legacy verbose format, or an empty chapter
        try:
            out[int(m.group(1))] = float(max(pos))
        except (ValueError, TypeError):
            continue
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def best_offset(durations: dict[int, float], verses: dict[int, int],
                offsets=(-2, -1, 0, 1, 2)) -> list[tuple[int, float, int]]:
    """[(offset, correlation, n_points)] for each candidate offset."""
    scored = []
    for off in offsets:
        xs, ys = [], []
        for ch, dur in sorted(durations.items()):
            v = verses.get(ch + off)
            if v is None:
                continue
            xs.append(dur)
            ys.append(float(v))
        if len(xs) >= max(3, MIN_CHAPTERS // 2):
            scored.append((off, pearson(xs, ys), len(xs)))
    return scored



# ── Pacing check: the short-book method ────────────────────────────────────
#
# The correlation above needs MIN_CHAPTERS points, so it cannot see a book
# like JOL or MAL at all — which is exactly where bul/BULCBV's non-Psalm
# damage was. This second method has no such floor because it judges each
# chapter on its own.
#
# A narrator reads at a roughly constant rate, so seconds-per-verse is
# near-constant across an edition. Pair a chapter's audio with the wrong
# text and that breaks immediately: the verse count comes from the text
# while the duration comes from the audio, so the ratio moves.
#
# MEASURED SENSITIVITY — this is a weak signal, treat it as such.
# Scored against bul/BULCBV's four known-mispaired PSA chapters (the ones
# the chapter_map orphans), using that book's OWN median as the baseline:
#
#     PSA 116  0.14x  flagged
#     PSA 147  0.39x  flagged
#     PSA 115  0.45x  MISSED
#     PSA  10  0.46x  MISSED
#
# 2 of 4. An earlier calibration claimed 4 of 4, but that scored them
# against the edition-wide median (13.4 s/verse, dominated by JOL, which is
# read far more slowly than the Psalms). Against the correct per-book
# baseline of 7.8 the real separation is only about 2.2x, not the 4-15x
# that comparison suggested. Loosening PACING_FACTOR to catch all four
# would put the threshold near 0.47, which sweeps in a large amount of
# ordinary poetic-passage noise — por/PORNLH's DEU 32/33 (Song and Blessing
# of Moses), NUM 23 (Balaam's oracles) and HOS 11/14 all sit at 0.27-0.40
# purely because poetry is segmented into short lines.
#
# So: this finds GROSS mispairings (a merge or split, where the verse count
# is wildly wrong for the audio) and misses subtle ones (an off-by-one
# between chapters of similar length). It does NOT replace the correlation
# method above where that method can run — it only reaches where that one
# structurally cannot, on books too short to correlate.
#
# Negative controls, all correctly silent: heb/HEBM95, hun/HUNHUN and
# por/PORB09 JOL 3, each suspected of a 4->3 chapter shift, pace at 6.8-8.2
# s/verse against their own 8.4-10.9. Their missing JOL 4 is a genuinely
# absent chapter — confirmed independently: only 3 JOL mp3s exist on disk.
PACING_MIN_VERSES = 5      # below this the ratio is too jumpy to trust
PACING_FACTOR = 2.5        # deviation from the edition median to flag


def pacing_outliers(canon: str, iso: str, did: str) -> list[dict]:
    """Chapters whose seconds-per-verse is out of step with their edition."""
    chapters = []
    did_dir = TIMING / canon / iso / did
    for book_dir in sorted(did_dir.glob("*")):
        book = book_dir.name
        for p in book_dir.glob(f"{book}_*_timing.json"):
            m = re.match(rf"{book}_(\d+)_", p.name)
            if not m:
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            pos = data.get("pos") if isinstance(data, dict) else None
            if not pos or len(pos) < PACING_MIN_VERSES:
                continue
            try:
                dur = float(max(pos))
            except (TypeError, ValueError):
                continue
            if dur <= 0:
                continue
            chapters.append((book, int(m.group(1)), len(pos), dur / len(pos)))
    if len(chapters) < 10:
        return []            # too little to establish the edition's rate

    # Calibrate PER BOOK, not per edition. Verse segmentation is a property
    # of the text, and it varies by book within one edition: por/PORNLH
    # splits Psalms by poetic line (PSA 18 carries 168 "verses" against
    # eng's 50), so its whole Psalter sits at 0.28x the edition median and
    # a per-edition baseline flags all 150 chapters as outliers. None of
    # them is mispaired — they are all consistent with each other, which is
    # exactly the point: a mismatch makes a chapter differ from its OWN
    # book, not from the average of every book in the edition.
    by_book: dict[str, list[float]] = {}
    for book, ch, nv, rate in chapters:
        by_book.setdefault(book, []).append(rate)

    edition_rates = sorted(c[3] for c in chapters)
    edition_median = edition_rates[len(edition_rates) // 2]

    out = []
    for book, ch, nv, rate in chapters:
        rates = sorted(by_book[book])
        # A book needs a few chapters before its own median means anything;
        # below that fall back to the edition, accepting the weaker baseline.
        median = rates[len(rates) // 2] if len(rates) >= 3 else edition_median
        if median <= 0:
            continue
        ratio = rate / median
        if ratio > PACING_FACTOR or ratio < 1 / PACING_FACTOR:
            out.append({"iso": iso, "canon": canon, "did": did, "book": book,
                        "chapter": ch, "verses": nv, "rate": rate,
                        "median": median, "ratio": ratio})
    return out


def verse_only_isos() -> set[str]:
    out = set()
    for f in CONFIG_DIR.glob("*.toml"):
        if f.stem == "default":
            continue
        try:
            if load_language_config(f.stem).verse_only_mode:
                out.add(f.stem)
        except Exception:
            continue
    return out


def analyse(iso: str, canon: str, did: str, scheme: str) -> list[dict]:
    shape = load_shape(scheme)
    if not shape:
        return []
    findings = []
    did_dir = TIMING / canon / iso / did
    for book_dir in sorted(did_dir.glob("*")):
        book = book_dir.name
        verses = shape.get(book)
        if not verses or len(verses) < MIN_CHAPTERS:
            continue
        durations = chapter_durations(book_dir, book)
        if len(durations) < MIN_CHAPTERS:
            continue
        scored = best_offset(durations, verses)
        if not scored:
            continue
        base = next((c for o, c, _ in scored if o == 0), 0.0)
        top_off, top_corr, n = max(scored, key=lambda r: r[1])
        if top_off == 0:
            continue
        if base > MAX_BASELINE or (top_corr - base) < MIN_GAIN:
            continue
        findings.append({
            "iso": iso, "canon": canon, "did": did, "book": book,
            "scheme": scheme, "offset": top_off, "corr": top_corr,
            "base": base, "n": n, "chapters": len(durations),
        })
    return findings


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", help="Only this language")
    ap.add_argument("--all", action="store_true",
                    help="Every edition, not just the transcript-blind ones")
    ap.add_argument("--self-test", action="store_true",
                    help="Check against bul/BULCBV and kaz/KAZKAZ")
    ap.add_argument("--pacing", action="store_true",
                    help="Per-chapter seconds-per-verse outliers — works on "
                         "SHORT books, which the correlation cannot see")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.pacing:
        return run_pacing(args)

    vo = verse_only_isos()
    findings, checked, skipped = [], 0, 0
    for canon_dir in sorted(TIMING.glob("*")):
        canon = canon_dir.name
        if canon not in ("ot", "nt"):
            continue
        for iso_dir in sorted(canon_dir.glob("*")):
            iso = iso_dir.name
            if args.iso and iso != args.iso:
                continue
            if not args.all and not args.iso and iso not in vo:
                continue
            for did_dir in sorted(iso_dir.glob("*")):
                did = did_dir.name
                scheme = lookup_vrs(iso, did)
                if not scheme:
                    skipped += 1
                    continue
                checked += 1
                findings.extend(analyse(iso, canon, did, scheme))

    print(f"Editions analysed          : {checked}")
    print(f"Skipped (scheme unknown)   : {skipped}")
    print(f"Scope                      : "
          f"{'all editions' if args.all or args.iso else 'transcript-blind (verse_only) editions'}\n")
    if not findings:
        print("No versification mismatch leads found.")
        return

    # Rank by how decisively the shift beats the current pairing, and say
    # which leads actually look like a scheme mismatch. Two things separate
    # the real thing from short-book noise, and both are visible above:
    #   * a real shift is measured over MANY chapters (bul/BULCBV PSA:
    #     n=149, gain 0.84) — Pearson on 7 points is close to meaningless
    #   * a scheme applies to the whole EDITION, so a genuine mismatch
    #     shows up in several books at once, not one in isolation
    by_edition: dict[tuple, int] = {}
    for f in findings:
        key = (f["iso"], f["did"])
        by_edition[key] = by_edition.get(key, 0) + 1
    for f in findings:
        f["gain"] = f["corr"] - f["base"]
        f["strong"] = (f["n"] >= MIN_CHAPTERS * 2 and f["gain"] >= 0.40)

    print(f"LEADS — {len(findings)} (edition, book) pair(s) where a SHIFTED pairing "
          f"correlates better than the current one:\n")
    fmt = "  {:<5} {:<8} {:<5} {:<5} {:>7} {:>8} {:>8} {:>7} {:>5} {:>6} {}"
    print(fmt.format("iso", "fileset", "book", "vrs", "offset", "corr",
                     "at 0", "gain", "n", "books", ""))
    print("  " + "-" * 78)
    for f in sorted(findings, key=lambda f: (not f["strong"], -f["gain"])):
        print(fmt.format(
            f["iso"], f["did"], f["book"], f["scheme"], f"{f['offset']:+d}",
            f"{f['corr']:.3f}", f"{f['base']:.3f}", f"{f['gain']:.3f}",
            f["n"], by_edition[(f["iso"], f["did"])],
            "STRONG" if f["strong"] else "weak"))

    strong = [f for f in findings if f["strong"]]
    print(f"\n{len(strong)} strong, {len(findings) - len(strong)} weak.")
    print("weak   = few chapters and/or a small margin; short books correlate")
    print("         erratically and produce these routinely — usually noise.")
    print("STRONG = many chapters and a decisive margin. Where an edition has")
    print("         several LONG books, a real scheme mismatch shows in more than")
    print("         one ('books' column). That corroboration is unavailable for a")
    print("         shift confined to Psalms, since the other affected books (JOL,")
    print("         MAL) are too short to score — as with bul/BULCBV, which is")
    print("         genuine despite showing books=1.")
    print("\nThese are LEADS, not conclusions — a structural correlation only.")
    print("Confirm with tools/derive_chapter_map.py before writing any config.")



def run_pacing(args) -> int:
    """Per-chapter pacing outliers — the method that works on short books."""
    vo = verse_only_isos()
    findings, checked = [], 0
    for canon_dir in sorted(TIMING.glob("*")):
        canon = canon_dir.name
        if canon not in ("ot", "nt"):
            continue
        for iso_dir in sorted(canon_dir.glob("*")):
            iso = iso_dir.name
            if args.iso and iso != args.iso:
                continue
            if not args.all and not args.iso and iso not in vo:
                continue
            for did_dir in sorted(iso_dir.glob("*")):
                checked += 1
                findings.extend(pacing_outliers(canon, iso, did_dir.name))

    print(f"Editions analysed : {checked}")
    print("Method            : seconds-per-verse vs the edition's own median")
    print(f"Flagging          : >{PACING_FACTOR}x or <{1/PACING_FACTOR:.2f}x the median, "
          f"min {PACING_MIN_VERSES} verses\n")
    if not findings:
        print("No pacing outliers found.")
        return 0

    # Cluster by edition+book: a versification shift hits a run of chapters,
    # whereas a single odd chapter is usually just a chapter with a long
    # musical intro or a truncated recording.
    from collections import Counter
    per_book = Counter((f["iso"], f["did"], f["book"]) for f in findings)
    print(f"{len(findings)} outlier chapter(s) in {len(per_book)} (edition, book) group(s). "
          f"Largest groups first:\n")
    fmt = "  {:<5} {:<8} {:<5} {:>5} {:>7} {:>9} {:>8} {:>7}"
    print(fmt.format("iso", "fileset", "book", "ch", "verses", "s/verse",
                     "median", "ratio"))
    print("  " + "-" * 62)
    shown = 0
    for (iso, did, book), _n in per_book.most_common():
        for f in sorted((f for f in findings
                         if (f["iso"], f["did"], f["book"]) == (iso, did, book)),
                        key=lambda f: f["chapter"]):
            print(fmt.format(f["iso"], f["did"], f["book"], f["chapter"],
                             f["verses"], f"{f['rate']:.1f}", f"{f['median']:.1f}",
                             f"{f['ratio']:.2f}x"))
            shown += 1
            if shown >= 60:
                break
        if shown >= 60:
            print(f"  ... {len(findings) - shown} more")
            break
    print("\nA RUN of consecutive chapters in one book suggests a versification")
    print("shift. An isolated chapter is more often a long musical intro, a")
    print("truncated recording, or a chapter that failed to align — check the")
    print("chapter before concluding anything.")
    return 0


def self_test() -> int:
    """Check the detector against two independently-established answers."""
    # bul/BULCBV was the positive control until 2026-09-05, when its
    # chapter_map was applied and the mismatch was actually fixed: PSA went
    # from offset +1 (corr 0.965, against 0.124 at offset 0) to offset 0 at
    # 0.983. Fixing the bug consumed the fixture. The real pre-fix output is
    # gone (overwritten by the re-alignment), so the case is inverted into a
    # REGRESSION test — if bul ever drifts back to a shift, the chapter_map
    # has stopped being applied — and the "can this detect a shift at all"
    # property is covered by a synthetic case below instead.
    cases = [
        ("bul", "ot", "BULCBV", "PSA", False,
         "chapter_map applied — mismatch fixed 2026-09-05, must stay fixed"),
        ("kaz", "ot", "KAZKAZ", "PSA", False,
         "rso text AND rso audio — correctly paired"),
    ]
    ok = True
    for iso, canon, did, book, expect_shift, why in cases:
        scheme = lookup_vrs(iso, did)
        book_dir = TIMING / canon / iso / did / book
        if not scheme or not book_dir.is_dir():
            print(f"  SKIP {iso}/{did} {book} — no data on disk "
                  f"(scheme={scheme}, dir={book_dir.is_dir()})")
            continue
        shape = load_shape(scheme)
        durations = chapter_durations(book_dir, book)
        scored = best_offset(durations, shape.get(book, {}))
        base = next((c for o, c, _ in scored if o == 0), 0.0)
        top_off, top_corr, n = max(scored, key=lambda r: r[1])
        got_shift = top_off != 0 and (top_corr - base) >= MIN_GAIN and base <= MAX_BASELINE
        verdict = "PASS" if got_shift == expect_shift else "FAIL"
        ok &= verdict == "PASS"
        print(f"  {verdict} {iso}/{did} {book}: best offset {top_off:+d} "
              f"(corr {top_corr:.3f} vs {base:.3f} at 0, n={n}, "
              f"{len(durations)} chapters) — expected "
              f"{'a shift' if expect_shift else 'no shift'}: {why}")

    # Synthetic positive control: the algorithm must still find a shift when
    # one is really there. Durations follow verse counts exactly, but the
    # pairing is displaced by one chapter — the bul signature, without
    # depending on bul's (now corrected) data.
    verses = {c: v for c, v in enumerate(
        [17, 6, 8, 8, 12, 10, 17, 9, 20, 18, 7, 8, 6, 7, 5, 11, 15, 50, 14, 13], start=1)}
    durations = {c: verses[c + 1] * 9.0 for c in range(1, 20) if (c + 1) in verses}
    scored = best_offset(durations, verses)
    base = next((c for o, c, _ in scored if o == 0), 0.0)
    top_off, top_corr, n = max(scored, key=lambda r: r[1])
    got = top_off == 1 and (top_corr - base) >= MIN_GAIN and base <= MAX_BASELINE
    ok &= got
    print(f"  {'PASS' if got else 'FAIL'} synthetic +1 shift: best offset "
          f"{top_off:+d} (corr {top_corr:.3f} vs {base:.3f} at 0, n={n}) — "
          f"expected +1: the detector must still catch a real shift")

    print("\nself-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
