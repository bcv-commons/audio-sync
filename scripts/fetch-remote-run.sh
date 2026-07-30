#!/usr/bin/env bash
#
# Pull alignment output back from a rented GPU box (Vast.ai/RunPod/etc) to
# this machine, so publishing can happen here — using this machine's own
# .env R2 credentials — instead of on rented, untrusted compute.
#
# This is the other half of the Dockerfile's security design: the
# container never holds R2 credentials and never runs with --publish, so
# results just sit in export/timing-data/ + _runs/ on the remote box until
# you pull them with this script. After that, run scripts/publish-align.sh
# here as normal — it doesn't care whether the local export/timing-data/
# tree came from a local run or a synced-back remote one.
#
# Usage:
#   scripts/fetch-remote-run.sh <user@host> [ssh_port]
#   scripts/fetch-remote-run.sh root@203.0.113.5 40022
#
# Remote paths are assumed to be /app/export and /app/_runs (this repo's
# Dockerfile WORKDIR) — override via REMOTE_APP_DIR if you built with a
# different WORKDIR.
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@host> [ssh_port]" >&2
    exit 1
fi

REMOTE_HOST="$1"
SSH_PORT="${2:-22}"
REMOTE_APP_DIR="${REMOTE_APP_DIR:-/app}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "── Pulling export/timing-data/ from ${REMOTE_HOST}:${REMOTE_APP_DIR}/export/timing-data/ ..."
rsync -avz --progress -e "ssh -p ${SSH_PORT}" \
    "${REMOTE_HOST}:${REMOTE_APP_DIR}/export/timing-data/" \
    "export/timing-data/"

echo "── Pulling _runs/ from ${REMOTE_HOST}:${REMOTE_APP_DIR}/_runs/ ..."
rsync -avz --progress -e "ssh -p ${SSH_PORT}" \
    "${REMOTE_HOST}:${REMOTE_APP_DIR}/_runs/" \
    "_runs/"

echo "── Done. Review the run manifest(s) in _runs/, then publish with:"
echo "    make publish-align"
echo "    make publish-align-dry   # to preview first"
