# Request: publish versification per edition (text *and* audio filesets)

Coming from `audio-sync`, which force-aligns audio against text to produce
per-word timings. We hit a class of bug that we can't fix on our side
without versification metadata, and it turns out we're currently
reverse-engineering something you're already partly publishing.

## What we hit

The Bulgarian edition `BULCBV`: **its audio and its text use different
versification schemes.**

- text = `org` (Hebrew/Masoretic)
- audio = an LXX/Orthodox scheme (consistent with `rso`)

So the mp3 named `PSA_050` actually reads **Psalm 51**. The narration even
says so out loud — *"Псалом петдесети. По-еврейски петдесет и първи."*
("Psalm 50. In Hebrew, 51.")

Because nothing declares either side's scheme, our aligner paired audio
chapter N with text chapter N and produced confident-looking, completely
wrong timings for **135+ chapters**, across three books:

| book | `org` (text) | LXX/`rso` (audio) | effect |
|---|---|---|---|
| PSA | 150, Hebrew numbering | LXX numbering | +1 shift for ch 10–147, plus 2 merges and 2 splits |
| JOL | 4 chapters | 3 chapters | audio ch 2 contains text ch 2+3 |
| MAL | 3 chapters | 4 chapters | audio ch 4 orphaned; text ch 3's last 6 verses have no audio paired to them |

All of it follows from that one scheme pair. We verified the Psalms mapping
empirically by scoring each chapter's transcript against every text chapter
in a ±4 window: it reproduces the standard LXX↔Masoretic correspondence
exactly, including all four irregular points (Heb 9+10 = Grk 9; Heb 114+115
= Grk 113; Heb 116 = Grk 114+115; Heb 147 = Grk 146+147). Match scores are
unambiguous — ~0.75–0.85 against the correct chapter vs ~0.20–0.30 against
the same-numbered one.

We've since scanned our whole corpus (49,847 chapters that have both a
transcript and reference text). `BULCBV` is the only edition affected that
we can currently see — but see the blind spot below.

## What's published today

| source | versification? | notes |
|---|---|---|
| `cdn.bibel.wiki/pkf/manifest.json` | ✅ `vrs` per collection | 589 languages. Values: `eng` ×285, `org` ×5, `rso` ×1, plus ~130 distinct content-hash fingerprints for custom schemes |
| `cdn.bibel.wiki/dbt/_catalog.json` | ❌ none | 3,360 version entries, shaped `[iso, id, canon, audio_fileset, text_fileset]` |
| helloAO `available_translations.json` | ❌ none | has verse/chapter counts, no scheme identifier |

So the `vrs` concept already exists and is already modelled correctly in
the PKF path — it just doesn't reach the other two sources. Concretely, of
the **128 languages we've aligned, 2** (`ilo`, `ind`) have any versification
data available. `bul` is not among them; `BULCBV` comes from DBT.

## Ask 1 — fingerprint and publish text versification for DBT/helloAO editions

This looks straightforward, because **the text's own verse counts are a
sufficient fingerprint**. We confirmed this locally: comparing `BULCBV`'s
Psalms against a Masoretic count table gives 0 deviations, and three
independent discriminators agree on `org` —

- JOL chapter count (3 = `eng`, 4 = `org`)
- MAL chapter count (4 = `eng`, 3 = `org`)
- whether psalm superscriptions are numbered (PSA 3/51/60 = 8/19/12 in
  `eng`, 9/21/14 in `org`)

Publishing `vrs` alongside each text fileset — a scheme name where it
matches a standard Paratext `.vrs`, a content hash where it doesn't,
exactly as the PKF manifest already does — would let every consumer stop
guessing.

## Ask 2 — the one that actually matters: audio fileset versification

**Text fingerprinting alone would not have caught this bug.** `BULCBV`'s
text is perfectly standard and self-consistent. The mismatch is entirely on
the *audio* side, and audio versification is not published anywhere, by
anyone, in any of the three sources.

It's also much harder to fingerprint: an audio fileset has no verse counts
to inspect. The only signals we know of are:

- **chapter count per book** — cheap, and already enough to flag JOL (3 vs
  4) and MAL (4 vs 3). It would *not* have flagged Psalms, since both
  schemes have 150.
- **transcribing and matching against candidate texts** — what we ended up
  doing. Reliable, but far too expensive to be the detection mechanism, and
  it needs a working transcript, which we don't always have.

If DBT exposes the scheme upstream (or if it can be inferred once per
fileset at ingest and cached), that would be worth much more to us than the
text side. Even just **"audio chapter count per book, per fileset"** would
be a large improvement — it's a cheap, purely structural signal that
catches the whole class where chapter counts differ.

## Our blind spot, in case it informs priority

Our corpus-wide detector only works for languages where we run Whisper. For
the ~83 languages we align without a transcript, we cannot detect this at
all. That's roughly half our corpus (49k of ~99k chapters) invisible to us.

Worse, it's self-concealing: a versification mismatch causes exactly the
alignment failure that makes our pipeline give up on Whisper for that
language — which then stops producing the transcript that would have
revealed the mismatch. A language can get permanently written off as "hard
to transcribe" when the real problem is that it was handed the wrong text.
Published versification would close that loop for us entirely.

## Happy to share

We have a working detector and the fully derived `BULCBV` mapping
(including the merge/split points) if either is useful to you — say the
word and we'll send them over, or run the detector across anything you'd
like checked.
