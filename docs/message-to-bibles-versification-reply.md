# Re: publish versification per edition — confirmed live, one concrete gap

Thanks — this was a much better outcome than we expected. Confirming from
our side, and flagging one specific thing that blocks deriving mappings.

## Confirmed

`bul/BULCBV` now reads `org` on the live index, which matches what we
derived independently from the audio content. Good to know the `rso`
classification was catching a real signal (Byzantine NT order) rather than
being arbitrary — and good that it's now backed by Psalm evidence.

Your Malachi finding **corrected our analysis**, not just confirmed it. We
had inferred an orphaned `MAL 4` audio file holding the tail of Hebrew
Malachi 3. It's a genuine 404 — so those last six verses simply aren't
recorded in this edition at all. Same symptom, better explanation, and it
means no remapping can recover them. We'd have kept looking for that audio.

## Checked for more editions like BULCBV — found none

You asked to be flagged. We ran our detector across all 49,847 chapters
where we have both a transcript and reference text: `BULCBV` remains the
only audio/text mismatch we can see.

We specifically chased `kaz/KAZKAZ` because your index classifies it `rso`
and it's our other big quality outlier. Its text really is `rso` — provable
from verse counts alone (text PSA 9 = 39 verses = org Ps 9 (21) + Ps 10
(18), the LXX merge, with everything after shifted by one). But its **audio
is also `rso`**: across 111 chapters with transcripts, 110 map at offset 0.
Correctly paired, no mismatch. Its quality problems are a dramatized
recording that under-covers the text — unrelated to versification.

## The `.vrs` shape files are exactly what we needed

`_vrs/org.vrs` and `_vrs/rso.vrs` carry per-chapter verse counts, and those
alone reproduce the BULCBV structure we'd reverse-engineered:

```
rso PSA 9  = 39  = org PSA 9 (21) + PSA 10 (18)     -> MERGE
rso PSA 113 = 26 = org PSA 114 (8) + PSA 115 (18)   -> MERGE
rso JOL 2  = 32  = org JOL 2 (27) + JOL 3 (5)       -> MERGE  (exact)
rso MAL 3 (18) + MAL 4 (6) = 24 = org MAL 3 (24)    -> SPLIT   (exact)
rso PSA 146 (11) + PSA 147 (9) = 20 = org PSA 147 (20) -> SPLIT (exact)
```

That's the whole picture, arithmetic, no transcription needed.

Shape counts alone aren't quite enough, though: deriving a complete map by
walking cumulative verse positions drifts, because `rso PSA 114 (8) + 115
(10) = 18` against `org PSA 116 = 19`. One unaccounted verse throws off
everything after it. That's what sent us looking at `crosswalk` — see the
follow-up section below, where `map` resolves it completely.

## Ask 2 — agreed, it's ours

Fair point, and we're taking it. We already download and decode every audio
fileset during alignment, so per-fileset chapter counts fall out of work
we're doing anyway. We'll generate it. If you'd like it published centrally
once we have it, say so and we'll send it your way in whatever shape suits.

## One on us, for symmetry

Your index surfaced a mirror-image bug on our side: **we publish alignment
output keyed to whatever numbering the source text uses, and declare
nothing.** Per your index, 28 of the editions we've already aligned are
non-`eng` (10 `orgw`, 9 `org`, 8 `rso`, 1 `vul`). So a consumer fetching our
timings for `kaz/KAZKAZ` PSA 34 gets rso Psalm 34 — org Psalm 35 — with
nothing saying so. We're adding a `vrs` declaration to our published output.
Exactly the problem we came to you with, pointed the other way.

---

## Follow-up: `map` resolves it — and one inconsistency to flag

The `map` field was exactly what we were missing. Composing `rso-to-eng` +
inverse `org-to-eng`, with the adjacency rule for omitted identity rows,
reproduces our empirically-derived bul/BULCBV Psalms mapping **exactly on
all 150 chapters** — every shift, both merges, both splits. Two fully
independent methods (scoring Whisper transcripts against candidate text
chapters, vs. your TVTMS-derived rows) agreeing chapter-for-chapter is
about as good a cross-check as we could ask for. We've replaced our
reverse-engineering with a generator over your data.

Confirming your adjacency point in passing: implementing it naively breaks
badly. Defaulting an unmapped verse to its own *source* chapter (rather than
inheriting the chapter from surrounding rows) produced spurious merges
across the whole book — 56 wrong chapters. Taking the chapter from the
nearest row, preceding-then-following, gives the exact match. Worth a line
in the doc if it isn't there already, since it's an easy thing to get wrong
in a way that still looks plausible.

**The one thing that doesn't reconcile — JOL in `rso`:**

```
_vrs/rso.vrs           JOL 1:20 2:32 3:21          <- 3 chapters
_vrs/map/rso-to-eng.json   JOL rows: (3,1)->(2,28) ... (3,5)->(2,32)
```

Those rows only make sense for a **4-chapter** Joel — they imply rso JOL 2
ends at verse 27, while `rso.vrs` declares 32. Under the 3-chapter shape
`rso.vrs` gives, rso Joel is identical to eng Joel and should have no rows
at all.

The tell: those five rows are **byte-identical to `org-to-eng.json`'s first
five JOL rows**, so it looks like org rows leaked into the rso map rather
than a modelling disagreement.

Consequence for us: deriving JOL from your data yields offset -1, where the
correct answer is +1. Hard evidence for +1 — bul/BULCBV's text Joel has 4
chapters (20/27/5/21) and its audio chapter 3 has 21 verses, matching org
JOL 4. We've overridden JOL by hand and left PSA and MAL derived, since both
of those reconcile cleanly (MAL's shape and rows agree: rso MAL 3+4 = org
MAL 3, a clean split).

This looks like the "TVTMS omits a genuine content difference" case you
asked us to flag, rather than an identity coincidence — but it's your data
model, so we may be reading it wrong.
