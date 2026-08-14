#!/usr/bin/env bash
#
# Publish export/timing-data/ (+ the run manifest for one batch) to
# cdn.bibel.wiki/align/ via rclone (Cloudflare R2).
#
# align/ is the intermediate/staging tree (Contract B) — distinct from the
# client-facing /dbt/<iso>/timing/<BOOK>.json that `bibles` publishes after
# aggregation. See internal-docs/audio-sync-interface.md §3 in MONO.
#
# Timing/words files are effectively immutable once written (one set per
# chapter per run), so this is a straight upload of new/changed files, no
# delta/cleanup pass like bible-story-builder's publish-dbt.sh.
#
# Credentials from .env (gitignored):
#   R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, R2_BUCKET
#   (also accepts CLOUDFLARE_* names, same fallback as MONO's publish scripts)
#
# Usage:
#   scripts/publish-align.sh                 # publish timing-data + all run manifests
#   scripts/publish-align.sh --batch-id <id> # also true without this — kept for
#                                             # symmetry with other publish scripts;
#                                             # the run manifest for <id> is just one
#                                             # more small file in _runs/
#   make publish-align              # same as above
#   make publish-align-dry          # dry-run (no writes)
#   DRY_RUN=1 scripts/publish-align.sh   # same as dry target
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

TIMING_SOURCE_DIR="export/timing-data"
RUNS_SOURCE_DIR="_runs"
CDN_PREFIX="align"

if [ ! -d "$TIMING_SOURCE_DIR" ]; then
    echo "[ERROR] $TIMING_SOURCE_DIR not found. Run: make align ARGS=\"...\""
    exit 1
fi

# ── Load credentials ──
if [ -f .env ]; then
    # shellcheck disable=SC1091
    set -a; source .env; set +a
fi

R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:-${CLOUDFLARE_ACCESS_KEY_ID:-}}"
R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:-${CLOUDFLARE_SECRET_ACCESS_KEY:-}}"
R2_ACCOUNT_ID="${R2_ACCOUNT_ID:-${CLOUDFLARE_ACCOUNT_ID:-}}"
R2_BUCKET="${R2_BUCKET:-${CLOUDFLARE_BUCKET:-}}"

if [ -z "$R2_ACCESS_KEY_ID" ] || [ -z "$R2_SECRET_ACCESS_KEY" ] || [ -z "$R2_ACCOUNT_ID" ] || [ -z "$R2_BUCKET" ]; then
    echo "[ERROR] Missing R2 credentials. Set R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, R2_BUCKET in .env"
    exit 1
fi

export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export RCLONE_CONFIG_R2_ACL=private
export RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true

REMOTE="R2:${R2_BUCKET}/${CDN_PREFIX}"

DRY_FLAG=""
if [ "${DRY_RUN:-}" = "1" ]; then
    DRY_FLAG="--dry-run"
    echo "[DRY RUN] No files will be written to CDN."
fi

# ── Pre-publish plausibility gate ──
# Scans for chapters with a backwards timestamp jump (verse N+1 timestamped
# earlier than verse N — never legitimate, always a mis-alignment; see the
# ENGBSB/ZLMAVB ISA 51 case in session history that motivated this check).
# Quarantines just the flagged chapters rather than blocking the whole
# publish — see tools/pre_publish_check.py's docstring for the reasoning
# and why this is scoped to backwards-only, not the fuller dupes/gaps
# signal set check_timing_quality.py also tracks (those need per-chapter
# score cross-referencing to tell a real issue from a legitimate one).
QUARANTINE_FILE="_runs/pre_publish_quarantine.txt"
python3 tools/pre_publish_check.py --out "$QUARANTINE_FILE"

# ── Pass 1: timing-data tree -> align/<canon>/<iso>/<version>/<BOOK>/ ──
# Local layout already mirrors the CDN layout 1:1 (see internal-docs/
# audio-sync-interface.md §3), so this is a plain recursive copy — except
# for two file types that stay local-only:
#   *_words_quality.json — per-word confidence/source, only consumed by
#     this repo's own tooling (tools/quality_report.py,
#     check_timing_quality.py, requeue_dupes_chapters.py, compare_timing.py)
#   *.srt — no longer generated at all (see whisper_transcribe.py); excluded
#     here too so any already-on-disk leftovers from before that change
#     don't get published on a future run.
echo "── Publishing $TIMING_SOURCE_DIR -> cdn.bibel.wiki/${CDN_PREFIX}/ ..."
rclone copy "$TIMING_SOURCE_DIR" "$REMOTE" \
    --header-upload "Cache-Control: max-age=3600" \
    --transfers 16 \
    --checkers 16 \
    --exclude "**/*_words_quality.json" \
    --exclude "**/*.srt" \
    --exclude-from "$QUARANTINE_FILE" \
    $DRY_FLAG \
    -v

# ── Pass 2: run manifests -> align/_runs/ (DISABLED 2026-08-14) ──
#
# Contract B calls for publishing per-batch run manifests (scores,
# coverage) so core (bible-story-builder/MONO) doesn't have to rescan our
# raw timing/words output to know what happened. But core's batch-manifest
# *emitter* (the thing that would tell us what to align in the first
# place) isn't built yet ("Core's batch emitter (Phase 3 of the repo
# split) is not yet implemented" — see this repo's own CLAUDE.local.md).
# Every batch run so far has been driven by hand-crafted local invocations,
# not a real batch manifest from core — consistent with nobody being
# downstream of our run manifests either. No confirmed consumer exists.
#
# Publishing ~15MB of internal run-history data with no known reader is
# pure overhead for no current benefit. Disabled rather than deleted —
# turn it back on (the shape-filtering logic below already solves the
# "_runs/ is full of our own working data, not just manifests" problem,
# confirmed real 2026-08-14) once there's an actual confirmed consumer
# and we know what it actually needs from the manifest.
#
# MANIFEST_LIST="$(mktemp)"
# trap 'rm -f "$MANIFEST_LIST"' EXIT
# python3 - "$RUNS_SOURCE_DIR" "$MANIFEST_LIST" <<'PYEOF'
# import json, sys
# from pathlib import Path
#
# runs_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
# names = []
# for f in sorted(runs_dir.glob("*.json")):  # top-level only — no shard-*/, no logs/
#     try:
#         d = json.loads(f.read_text())
#     except Exception:
#         continue
#     if isinstance(d, dict) and "batch_id" in d and "results" in d:
#         names.append(f.name)
# out_path.write_text("".join(n + "\n" for n in names))
# print(f"[publish-align] {len(names)} real run manifest(s) found in {runs_dir}/")
# PYEOF
#
# if [ -s "$MANIFEST_LIST" ]; then
#     echo "── Publishing $RUNS_SOURCE_DIR run manifests -> cdn.bibel.wiki/${CDN_PREFIX}/_runs/ ..."
#     rclone copy "$RUNS_SOURCE_DIR" "R2:${R2_BUCKET}/${CDN_PREFIX}/_runs" \
#         --header-upload "Cache-Control: max-age=300" \
#         --files-from "$MANIFEST_LIST" \
#         --no-traverse \
#         $DRY_FLAG \
#         -v
# else
#     echo "── No run manifests in $RUNS_SOURCE_DIR/ — skipping."
# fi
echo "── Pass 2 (run manifests) disabled — see comment above. Skipping."

echo "── Done."
