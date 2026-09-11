#!/usr/bin/env bash
#
# Publish export/timing-data/obs-video/ to cdn.bibel.wiki/obs-video/ via
# rclone (Cloudflare R2) — deliberately a SEPARATE top-level root from
# align/ (see scripts/publish-align.sh), not nested under it. align/ is
# Contract B's text-anchored audio-alignment staging tree (per
# internal-docs/audio-sync-interface.md §3); this data is structurally
# nothing like it — no canon/version/BOOK breakdown, no _words.json, just
# video-frame-detected position markers ("pos"/"images"/"run_bounds") for
# OBS-OBS4All's video-only languages, each file's own "note" field saying
# plainly "video-frame-detected, no text yet". Publishing it under align/
# would misrepresent it to any consumer expecting real word/verse timing.
#
# Local layout mirrors the CDN layout 1:1 — export/timing-data/obs-video/
# <iso>/<story>_timing.json -> obs-video/<iso>/<story>_timing.json, same
# filenames, no renaming — same "local IS the CDN shape" convention
# publish-align.sh already uses for align/.
#
# No quarantine/exclude pass: unlike align/'s backwards-timestamp check,
# there's no equivalent plausibility signal here yet, and a story still
# carrying an unresolved "gaps" field (e.g. a confirmed-truncated source
# video) is published as-is — the gap is documented transparently in the
# file itself (see tools/detect_obs_video_segments.py's module docstring),
# not something to hide by holding the file back.
#
# Credentials from .env (gitignored), same as publish-align.sh:
#   R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, R2_BUCKET
#   (also accepts CLOUDFLARE_* names)
#
# Usage:
#   scripts/publish-obs-video.sh
#   make publish-obs-video          # same as above
#   make publish-obs-video-dry      # dry-run (no writes)
#   DRY_RUN=1 scripts/publish-obs-video.sh   # same as dry target
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

SOURCE_DIR="export/timing-data/obs-video"
CDN_PREFIX="obs-video"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "[ERROR] $SOURCE_DIR not found. Run tools/detect_obs_video_segments.py first."
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

echo "── Publishing $SOURCE_DIR -> cdn.bibel.wiki/${CDN_PREFIX}/ ..."
rclone copy "$SOURCE_DIR" "$REMOTE" \
    --header-upload "Cache-Control: max-age=3600" \
    --transfers 16 \
    --checkers 16 \
    $DRY_FLAG \
    -v

echo "── Done."
