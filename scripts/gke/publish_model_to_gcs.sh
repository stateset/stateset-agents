#!/usr/bin/env bash
set -euo pipefail

# Publish a local model artifact directory to a versioned GCS prefix.
#
# Recommended layout:
#   gs://<bucket>/kimi-k25/runs/<run_id>/
#     serving_manifest.json
#     merged/
#     (optionally) adapters/, logs/, metrics/
#
# This script uses `gcloud storage rsync` which is generally faster and more
# reliable than ad-hoc `cp` for directories.

LOCAL_DIR=""
GCS_URI=""
PROJECT=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --local-dir PATH --gcs-uri gs://BUCKET/prefix [--project PROJECT]

Examples:
  $(basename "$0") --local-dir /models/kimi-k25 --gcs-uri gs://stateset-models-prod/kimi-k25/runs/2026-02-07_001
  $(basename "$0") --local-dir ./outputs/kimi_k25_gspo --gcs-uri gs://stateset-models-dev/kimi-k25/runs/local-test
EOF
}

while [ "${#}" -gt 0 ]; do
  case "$1" in
    --local-dir) LOCAL_DIR="${2:-}"; shift 2 ;;
    --gcs-uri) GCS_URI="${2:-}"; shift 2 ;;
    --project) PROJECT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "${LOCAL_DIR}" ] || [ -z "${GCS_URI}" ]; then
  usage
  exit 2
fi

if [ ! -d "${LOCAL_DIR}" ]; then
  echo "local dir not found: ${LOCAL_DIR}" >&2
  exit 2
fi

if [[ "${GCS_URI}" != gs://* ]]; then
  echo "gcs uri must start with gs:// (got: ${GCS_URI})" >&2
  exit 2
fi

set -x
if [ -n "${PROJECT}" ]; then
  gcloud storage rsync --recursive "${LOCAL_DIR}" "${GCS_URI}" --project="${PROJECT}"
else
  gcloud storage rsync --recursive "${LOCAL_DIR}" "${GCS_URI}"
fi
