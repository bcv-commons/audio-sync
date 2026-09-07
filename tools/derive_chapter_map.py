#!/usr/bin/env python3
"""
Generate a language config's [[chapter_map]] block from published versification.

When an edition's audio and text use different versification schemes, audio
chapter N does not contain text chapter N (confirmed for bul/BULCBV, whose
audio is LXX/Orthodox-numbered against Masoretic-numbered text). The
pipeline handles that via a chapter_map in pipeline/config/languages/
<iso>.toml — see map_audio_chapter_to_text() in pipeline/text_processing.py.

This derives that block from the versification data the bibles repo
publishes, instead of reverse-engineering it per edition against Whisper
transcripts:

    <vrs_base>/map/<src>-to-eng.json   verse-precise mapping rows
    <vrs_base>/<scheme>.vrs            per-chapter verse counts

Both schemes map through `eng` as pivot, so audio->text composes as
audio-to-eng then the inverse of text-to-eng.

TWO SUBTLETIES, both learned the hard way:

  * Use the `map` field, NOT `crosswalk`. `crosswalk` is book-code renames
    only (empty for org/rso, which need none). `map` holds the real
    verse-precise rows — 1,054 for org-to-eng, 2,485 for rso-to-eng in PSA.
  * The files carry single-verse DIFFS only; identity rows are omitted. So
    a verse with no row keeps its verse number, and takes its chapter from
    the surrounding rows (nearest preceding, else following). Defaulting an
    unmapped verse to its own source chapter instead produces spurious
    merges all over the book.

Deliberately a GENERATOR, not runtime derivation: an alignment batch should
not depend on fetching versification files mid-run, and a generated config
block stays auditable and diffable in review.

KNOWN LIMIT — review SPLIT skips before pasting them in. This derives from
the two schemes' declared SHAPES; it has no idea which audio chapters the
fileset actually contains. So when two audio chapters split one text
chapter, both are skipped (see to_toml) even if one of them does not exist
in this particular recording — in which case it is not really a split, and
skipping the surviving half throws away audio that aligns fine. Real case:
bul/BULCBV MAL, where rso 3+4 -> org 3 looks exactly like the PSA splits,
but audio chapter 4 is a genuine 404; skipping both would have discarded
the 18 recoverable verses in chapter 3 to avoid 6 that no mapping can
recover. tools/audio_chapter_anomalies.py is the cross-check — it reports
where a fileset's actual chapter count differs from its scheme's expected.

Verified 2026-09-04: for bul/BULCBV PSA (audio rso, text org) the output
matches, on all 150 chapters, a mapping derived independently by scoring
Whisper transcripts against every candidate text chapter.

    python tools/derive_chapter_map.py --iso bul --distinct-id BULCBV \
        --audio-fileset BULCBVO2DA --audio-vrs rso --books PSA,JOL,MAL
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from versification import lookup_vrs  # noqa: E402

VRS_BASE = "https://cdn.bibel.wiki/_vrs/"
CACHE_DIR = Path("api-cache/vrs")


def _fetch(url: str, cache_name: str) -> bytes | None:
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        try:
            return cache_path.read_bytes()
        except OSError:
            pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "audio-sync"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
    except Exception as e:
        print(f"  ! fetch failed {url}: {e}", file=sys.stderr)
        return None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
    except OSError:
        pass
    return raw


def _parse_ref(ref: str):
    m = re.match(r"^([A-Z0-9]{3}) (\d+):(\d+)", (ref or "").strip())
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def load_scheme_map(scheme: str, book: str) -> dict:
    """{(chapter, verse): (chapter, verse)} from <scheme>-to-eng.json's `map`."""
    if scheme == "eng":
        return {}
    raw = _fetch(f"{VRS_BASE}map/{scheme}-to-eng.json", f"map-{scheme}-to-eng.json")
    if not raw:
        return {}
    out = {}
    for row in json.loads(raw).get("map") or []:
        s, t = _parse_ref(row.get("s", "")), _parse_ref(row.get("t", ""))
        if s and t and s[0] == book and t[0] == book:
            out[(s[1], s[2])] = (t[1], t[2])
    return out


def load_shape(scheme: str, book: str) -> dict:
    """{chapter: verse_count} from <scheme>.vrs."""
    raw = _fetch(f"{VRS_BASE}{scheme}.vrs", f"{scheme}.vrs")
    if not raw:
        return {}
    for line in raw.decode("utf-8", "replace").splitlines():
        if line.startswith(book + " "):
            d = {}
            for tok in line.split()[1:]:
                if ":" in tok:
                    c, v = tok.split(":")
                    try:
                        d[int(c)] = int(v)
                    except ValueError:
                        pass
            return d
    return {}


def derive(audio_vrs: str, text_vrs: str, book: str) -> dict:
    """{audio_chapter: [text_chapter, ...]} for one book."""
    a2e = load_scheme_map(audio_vrs, book)
    t2e = load_scheme_map(text_vrs, book)
    e2t = {v: k for k, v in t2e.items()}
    shape = load_shape(audio_vrs, book)
    if not shape:
        return {}

    result = {}
    for ch in sorted(shape):
        n = shape[ch]
        explicit = {v: a2e[(ch, v)] for v in range(1, n + 1) if (ch, v) in a2e}
        targets = []
        for v in range(1, n + 1):
            if v in explicit:
                eng = explicit[v]
            else:
                # Identity verse: number unchanged, chapter inherited from
                # the surrounding rows (see module docstring).
                prev = [k for k in explicit if k < v]
                nxt = [k for k in explicit if k > v]
                if prev:
                    eng = (explicit[max(prev)][0], v)
                elif nxt:
                    eng = (explicit[min(nxt)][0], v)
                else:
                    eng = (ch, v)
            targets.append(e2t.get(eng, eng)[0])
        result[ch] = sorted(set(targets))
    return result


def to_toml(book: str, audio_fileset: str, mapping: dict, text_max: int) -> str:
    """Render as a [[chapter_map]] block: contiguous shifts, plus skips.

    A source chapter covering two text chapters (MERGE) still maps — the
    mapped text chapter aligns fine against its portion of the audio. Two
    source chapters covering ONE text chapter (SPLIT) can't be expressed as
    an offset at all and is emitted as a skip.
    """
    covers = {}
    for ch, tgt in mapping.items():
        for t in tgt:
            covers.setdefault(t, []).append(ch)

    shifts, skips = [], []
    for ch in sorted(mapping):
        tgt = mapping[ch]
        if not tgt:
            skips.append(ch)
            continue
        primary = tgt[0]  # first text chapter this audio begins with
        if len(covers.get(primary, [])) > 1 and len(tgt) == 1:
            skips.append(ch)   # SPLIT
        else:
            shifts.append((ch, primary - ch))

    runs = []
    for ch, off in shifts:
        if runs and runs[-1][2] == off and ch == runs[-1][1] + 1:
            runs[-1][1] = ch
        else:
            runs.append([ch, ch, off])

    lines = ["[[chapter_map]]", f'book = "{book}"',
             f'audio_fileset = "{audio_fileset}"', "shifts = ["]
    for lo, hi, off in runs:
        lines.append(f"    {{ from = {lo}, to = {hi}, offset = {off} }},")
    lines.append("]")
    lines.append(f"skip = [{', '.join(str(s) for s in skips)}]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", required=True)
    ap.add_argument("--distinct-id", required=True)
    ap.add_argument("--audio-fileset", required=True)
    ap.add_argument("--audio-vrs", required=True,
                    help="Audio fileset's scheme (not published anywhere — "
                         "determine empirically, e.g. rso)")
    ap.add_argument("--text-vrs", default=None,
                    help="Text scheme (default: look up the edition on the CDN index)")
    ap.add_argument("--books", default="PSA", help="Comma-separated book codes")
    args = ap.parse_args()

    text_vrs = args.text_vrs or lookup_vrs(args.iso, args.distinct_id)
    if not text_vrs:
        print(f"Text versification for {args.iso}/{args.distinct_id} is unknown — "
              f"pass --text-vrs explicitly.", file=sys.stderr)
        sys.exit(1)

    print(f"# audio={args.audio_vrs}  text={text_vrs}  "
          f"({args.iso}/{args.distinct_id})\n", file=sys.stderr)

    for book in [b.strip().upper() for b in args.books.split(",") if b.strip()]:
        mapping = derive(args.audio_vrs, text_vrs, book)
        if not mapping:
            print(f"# {book}: no data", file=sys.stderr)
            continue
        if all(v == [k] for k, v in mapping.items()):
            print(f"# {book}: schemes agree, no chapter_map needed", file=sys.stderr)
            continue
        print(to_toml(book, args.audio_fileset, mapping, max(mapping)))
        print()


if __name__ == "__main__":
    main()
