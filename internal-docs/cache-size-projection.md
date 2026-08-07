# Cache size projection — full download footprint

Computed 2026-08-07, for capacity planning ahead of scaling batch alignment
beyond the initial 8-language `--iso-list` (fra, arb, hin, cmn, ben, kor, tur,
uzn) toward the full set of languages already present under `downloads/BB/`.

## Scope

This box has audio downloaded for **1,888 languages** under `downloads/BB/`
(mostly leftover from an earlier bulk-download operation predating the
current batch-alignment work), across **1,708 editions** (iso/canon/
distinct_id combinations) with at least some audio present.

Of those, **1,101 editions** pass `has_usable_text_source()`
(`pipeline/download_language_content.py`) — i.e. have a real text source
(DBT catalog fileset or a verified helloAO overlap match), not just audio.
The other ~600 editions have audio but no alignable text (same class of gap
as `fra`'s `FRALSN`/`FRAPDV` — audio downloaded, but no DBT text fileset and
no verified helloAO match for that specific edition).

## Chapter-count assumption

Per-edition exact book coverage isn't available for all 1,101 usable
editions (no reliable per-edition mapping). Per-language testament coverage
*is* available via `catalog-index.json`'s `[iso, canon, source]` entries,
where `canon` is one of `nt`, `ntp`, `ot`, `otp` (the `p` suffix marking
partial-testament coverage). Agreed approximation:

- `nt` or `ot` listed (full testament) → assume the full book count for
  that testament: **NT = 260 chapters** (27 books), **OT = 929 chapters**
  (39 books)
- `ntp` or `otp` listed (partial testament) → assume **half** the full
  count: NT = 130, OT ≈ 465

This is a language-level signal applied uniformly to every edition of that
language/canon — not an edition-specific figure. It will over- or
under-count individual editions, but is the best available proxy without
per-edition coverage data.

## Resulting totals

| Metric | Value |
|---|---|
| Usable editions (audio + real text source) | 1,101 |
| Total assumed chapters (per the rule above) | 338,963 |
| Already aligned (final `_timing.json` exists) | 10,506 |
| **Remaining to process** | **328,457** |

## Measured per-chapter sizes (from real, existing cached data)

| Layer | Files sampled | Avg size/chapter |
|---|---|---|
| Audio (`downloads/BB/*.mp3`) | 19,712 | 2.263 MB |
| Text (`downloads/BB/*.txt`) | 19,430 | 4.63 KB |
| Raw Whisper (`word-timing-data/*_whisper_words.json`) | 10,576 | 55.10 KB |
| Raw MMS (`word-timing-data/*_mms_words.json`) | 11,026 | 53.82 KB |
| Published/final (`export/timing-data/*_timing.json` + `*_words.json` + `*_words_quality.json`) | 10,921 | 61.23 KB |

Note: raw Whisper/MMS cache (10,480–11,026 files) is slightly *smaller*
than the published-output count (10,921) — some chapters have had their
raw intermediate output cleaned up at some point while the final merged
result was kept. Not all currently-published chapters still have their
raw cache.

## Projected totals at full scope (338,963 chapters)

| Layer | Projected size | Storage plan |
|---|---|---|
| Audio | **749.2 GB** | → external 1TB drive (once connected) |
| Cached text | 1.5 GB | internal disk |
| Cached raw Whisper output | 17.8 GB | internal disk |
| Cached raw MMS output | 17.4 GB | internal disk |
| **Published (merged) output** | **19.8 GB** | internal disk |
| **Internal-disk total** (text + raw + published, excl. audio) | **~56.5 GB** | internal disk |

Current internal disk: 438 GB total, 313 GB free (as of 2026-08-07) — the
~56.5 GB internal-disk footprint fits comfortably even without offloading
audio. The 749.2 GB audio figure is what drives the external-drive plan.

## Caveats

- **328,457 remaining is a theoretical ceiling across all 1,888
  languages**, not a near-term processing target. Most of that footprint
  predates the current batch-alignment effort. Real near-term work is
  whatever subset actually gets queued via `--iso-list`.
- The `ntp`/`otp` half-count assumption is a coarse, language-level
  approximation — no per-edition precision is available. Real per-edition
  totals will vary above and below this estimate.
- **749 GB of audio against a 1TB external drive leaves only ~250 GB of
  headroom** — fine for the current footprint as measured, but tight if
  the addressable scope grows (more languages added, or `ntp`/`otp`
  editions turning out to have more real chapters than the half-count
  estimate).
- No cleanup/pruning of `downloads/BB/` exists anywhere in the current
  pipeline (confirmed by code inspection — only `compare_timing.py` has
  any `unlink` calls, unrelated to routine audio cache management, and
  `make clean` explicitly excludes `downloads/BB/`). Audio accumulates
  permanently unless a cleanup strategy is deliberately built.
