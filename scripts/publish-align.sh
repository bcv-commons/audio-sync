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

# ── Pass 1: timing-data tree -> align/<canon>/<iso>/<version>/<BOOK>/ ──
# Local layout already mirrors the CDN layout 1:1 (see internal-docs/
# audio-sync-interface.md §3), so this is a plain recursive copy.
echo "── Publishing $TIMING_SOURCE_DIR -> cdn.bibel.wiki/${CDN_PREFIX}/ ..."
rclone copy "$TIMING_SOURCE_DIR" "$REMOTE" \
    --header-upload "Cache-Control: max-age=3600" \
    --transfers 16 \
    --checkers 16 \
    $DRY_FLAG \
    -v

# ── Pass 2: run manifests -> align/_runs/ ──
if [ -d "$RUNS_SOURCE_DIR" ] && [ -n "$(ls -A "$RUNS_SOURCE_DIR" 2>/dev/null)" ]; then
    echo "── Publishing $RUNS_SOURCE_DIR -> cdn.bibel.wiki/${CDN_PREFIX}/_runs/ ..."
    rclone copy "$RUNS_SOURCE_DIR" "R2:${R2_BUCKET}/${CDN_PREFIX}/_runs" \
        --header-upload "Cache-Control: max-age=300" \
        $DRY_FLAG \
        -v
else
    echo "── No run manifests in $RUNS_SOURCE_DIR/ — skipping."
fi

echo "── Done."
