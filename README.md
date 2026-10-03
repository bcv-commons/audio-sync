# audio-sync

A batch alignment worker for Bible audio. A GPU (Apple Silicon or NVIDIA
CUDA) makes it much faster, but it is **not required** — both alignment
steps have a CPU fallback and will run on a plain CPU-only box, just
slower. Its whole job is:

1. **Whisper** — transcribe Bible audio to text
2. **MMS** — force-align the reference text to the audio (CTC alignment)
3. **Fuse** — merge the two into a final, per-word timing file

On CPU-only hardware, expect this to be viable for small or occasional
batches rather than for chewing through a large backlog — a modest laptop-
class CPU can take noticeably longer
per chapter than a GPU box would. `make check` (below) tells you which
mode a given machine will actually run in.

The output (word-level timing for every verse of every chapter it
processes) is uploaded to CDN so other services can use it — this repo
does not serve anything itself, and it does not decide what to align.
Another system (referred to below as "core") decides which languages and
chapters need alignment and hands this repo a **batch manifest** to work
through.

If you're setting this repo up on a new machine, you're standing up one
more worker that can pull manifests and grind through alignment jobs.

## How a batch flows through this repo

```
core (elsewhere) → batch manifest → this repo → CDN
                    (what to align)   (does the      (word timing +
                                        alignment)     run report)
```

1. **Get a batch manifest.** A batch manifest lists which languages,
   versions and chapters need aligning. `batch_manifest.py` finds one by
   `BATCH_ID`:
   - first it looks for a local file at `_batches/<id>.json`
   - if that's not there, it fetches
     `https://cdn.bibel.wiki/_batches/queue/{high,normal,low}/<id>.json`
     and caches it locally

   There is currently no automatic "keep pulling the next manifest off the
   queue" loop — you (or a cron job / shell loop, see below) need to supply
   a `BATCH_ID` each time. In practice every run so far has been started
   directly with `--iso` / `--iso-list` (or from a hand-written
   `_batches/<name>.json`); core has not yet published to the CDN queue.

2. **Fetch audio + text.** For each job in the manifest,
   `download_language_content.py` fetches the DBT audio/text fileset for
   that language+version+chapter. It resolves which exact fileset IDs to
   use either from the manifest itself (if core already resolved them) or
   by looking them up in the DBT catalogs published on CDN
   (`cdn.bibel.wiki/catalog/{text,audio,overlap,index}.json`) — no local
   database or setup script needed for this step.

3. **Align.** `align_pipeline.py` runs Whisper transcription, MMS forced
   alignment, and fusion for every chapter in the batch, writing
   `_timing.json`/`_words.json` files to `export/timing-data/`.
   See "How chapters are aligned" below for the two modes.

4. **Publish.** The pipeline writes a run manifest to `_runs/<batch_id>.json`
   (per-chapter status, alignment scores) and — if asked to — uploads both
   the timing output and the run manifest to `cdn.bibel.wiki/align/` via
   `scripts/publish-align.sh`.

Once published, this repo's job for that batch is done. Nothing here
needs to stay running for the results to be usable.

## How chapters are aligned

There are two modes, chosen per language (`verse_only_mode` in
`pipeline/config/languages/<iso>.toml`) and per recording:

- **Fusion mode** (default): Whisper transcribes the chapter, MMS
  force-aligns the whole chapter's text, and fusion merges the two.
- **Verse-only mode** (languages Whisper can't handle, set automatically
  for 175+ languages, and every dramatized `2DA`/`2SA` recording): each
  verse is aligned on its own inside a search window around where reading
  pace says it should be (`align_verse_words.py`). The verse's text is
  padded with MMS's `<star>` wildcard so the neighbouring verses' speech in
  the window is absorbed instead of smearing the verse. The model runs once
  per chapter and each window is a slice of that output. OBS stories use
  the same approach per segment (`align_obs_words.py`).

Verse-only output carries a few extra safeguards:

- **Chapter gate.** If more than 10% of a chapter's verses score below 0.5,
  the chapter is not published as our own timing. If DBT has usable timing
  for the same recording, the chapter is written as a `defer_to_dbt`
  record pointing clients at DBT; otherwise it is written but marked
  `gate: held_back`, and `tools/pre_publish_check.py` keeps it off the CDN.
- **Method tag.** Each chapter's local `_words_quality.json` records the
  method that produced it (`summary.method`, see `ALIGNMENT_METHOD` in
  `align_verse_words.py`). Bump it when a change makes old output worth
  redoing; `tools/run_gpu_redo.py --redo-older-method` then re-aligns
  exactly the chapters with an older tag (backing up the old files with
  `--backup-dir`), so long redo runs can be stopped and resumed freely.
- **Non-DBT audio.** Editions listed in `config/helloao.toml` (e.g. BSB read
  by Hays, `eng/ENGBSBHAY`) get their audio from helloAO, not DBT; every
  tool that fetches audio must take that route for them.

Long-running jobs are started from `tools/` (`run_gpu_redo.py` with
`run_gpu_redo_supervisor.py`, `whisper_backfill.py` with
`fusion_sweep_watcher.py`); `tools/diag/redo_spot_check.py` summarizes a
running redo per language. Plans and measurements behind all of this:
`internal-docs/alignment-fix-rollout-plan-2026-10-02.md` and
`internal-docs/single-pipeline-plan-2026-10-02.md`.

## Running one batch

```bash
BATCH_ID=<id> make align            # steps 1–3 in one go
BATCH_ID=<id> make align ARGS="--publish"   # ...and publish when done
```

Or run just one stage (useful for debugging): `align_pipeline.py` takes
`--skip-whisper`/`--skip-mms`/`--skip-fusion` flags directly, e.g.
`BATCH_ID=<id> make align ARGS="--skip-mms --skip-fusion"` for transcription
only. There's no separate step-only CLI -- one driver, one process, same
model-loading cost either way.

```bash
make publish-align                  # publish whatever's in export/ + _runs/
make publish-align-dry              # same, but don't actually upload
```

Run `make help` for the full command list, including quality-check tools
and content-prep scripts (`import-contrib`, `prepare-cross-source`).

## Running unattended on a dedicated box

There's no built-in "watch the queue forever" mode yet (see the note in
step 1 above). Until that lands, the simplest way to keep a dedicated
machine busy in the background is a small polling loop of your own, e.g.:

```bash
#!/usr/bin/env bash
# poll-and-align.sh — naive stopgap until a real queue worker exists
while true; do
    BATCH_ID="$(curl -fsS https://cdn.bibel.wiki/_batches/queue/high/latest.json | jq -r .id)"
    if [ -n "$BATCH_ID" ] && [ "$BATCH_ID" != "null" ]; then
        BATCH_ID="$BATCH_ID" make align ARGS="--publish"
    fi
    sleep 60
done
```

Run it under `nohup`, `tmux`/`screen`, or a `systemd` unit so it survives
your SSH session ending. (The `curl .../latest.json` endpoint above is
illustrative — check with core/`internal-docs/audio-sync-interface.md` in
MONO for whatever the actual "give me the next batch" convention is before
wiring this up for real.)

## Setup

```bash
make install
```

This runs `pip install -r conf/requirements-whisper.txt`, then installs the
~1.5GB `conf/requirements-cuda.txt` (CUDA 12 runtime libs for
`faster-whisper`) only if `nvidia-smi` is found on the machine — so a
CPU-only box like this one doesn't waste bandwidth/disk on CUDA libraries
it can never use. If you're setting up manually instead of via
`make install`, mirror the same logic: `pip install -r
conf/requirements-whisper.txt`, and only add `pip install -r
conf/requirements-cuda.txt` if you know the box has an NVIDIA GPU.
Alignment dependencies (torch/torchaudio/uroman) still need to be
installed separately — see the module docstring in `pipeline/align_pipeline.py`.

Every machine tends to need different GPU/CPU tuning (Whisper model size,
MMS chunk size, forcing CPU when VRAM is tight, ...). Rather than passing
those as flags on every invocation, copy `conf/hw.local.json.example` to
`conf/hw.local.json` (gitignored, machine-specific) and set them there —
`make align` picks it up automatically, and
any CLI flag you do pass still overrides it for that one run. See the
`_notes` block in the example file for what each setting does.

Create a `.env` file with:

```
BIBLE_API_KEY=...              # DBT API key, for downloading audio/text
BIBLE_API_BASE_URL=...         # DBT API base URL

R2_ACCESS_KEY_ID=...           # Cloudflare R2 credentials, for publishing
R2_SECRET_ACCESS_KEY=...       # results to cdn.bibel.wiki (used by
R2_ACCOUNT_ID=...              # scripts/publish-align.sh)
R2_BUCKET=...
```

Check everything's wired up, including which compute device this machine
will actually align on:

```bash
make check      # confirms python/torch/torchaudio are importable,
                 # and reports cuda / mps / cpu
```

## Where things live

- `pipeline/` — all pipeline code (`align_pipeline.py`, `whisper_transcribe.py`,
  `mms_align_words.py`, `batch_manifest.py`, etc.). Invoked via `make`, not
  directly — see `make help`.
- `conf/` — dependency + machine tuning files: `requirements-whisper.txt`,
  `requirements-cuda.txt`, `hw.local.json.example` (and your own gitignored
  `hw.local.json`, see Setup above)
- `tools/` — long-running jobs (redo, Whisper backfill, fusion watcher),
  checks and reports; `tools/diag/` holds investigation scripts
- `tests/` — pytest suite (`make test`); no model or GPU needed. Real-audio
  fixtures under `tests/fixtures/` are kept locally, not in git
- `scripts/` — shell scripts (`publish-align.sh`)
- `config/helloao.toml` — editions whose audio/text come from helloAO
- `export/timing-data/` — final output, mirrors the CDN layout
  (`<canon>/<iso>/<version>/<BOOK>/`)
- `_runs/<batch_id>.json` — one run manifest per batch (status + scores)
- `downloads/` — fetched audio/text, working storage (not published)
- `api-cache/` — cached copies of CDN catalogs/queue manifests
- `_batches/` — cached/hand-crafted batch manifests
- `internal-docs/` — plans, measurements, correspondence (gitignored)

Apart from `pipeline/`, `tools/`, `tests/`, `scripts/`, `conf/` and `config/`, none of
these are committed to git — they're working state, not source.

## Three-repo split

This repo is one of three that coordinate only through CDN artifacts (no
shared code or database):

- **bible-story-builder** ("core") — decides what needs aligning, emits
  batch manifests
- **bibles** — publishes DBT catalog/metadata to CDN
- **audio-sync** (this repo) — does the actual alignment work

See `CLAUDE.local.md` in this repo (not checked in) for more background
on the split and pending work.
