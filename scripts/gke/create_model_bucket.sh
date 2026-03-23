#!/usr/bin/env bash
set -euo pipefail

# Create a versioned GCS bucket intended for model artifacts.
#
# This script is safe to run multiple times (it will no-op if the bucket exists).
#
# Requirements:
# - gcloud authenticated to the target project
# - permissions to create/update Cloud Storage buckets

BUCKET=""
PROJECT=""
LOCATION="us-central1"
STORAGE_CLASS="STANDARD"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --bucket BUCKET --project PROJECT [--location LOCATION] [--storage-class CLASS]

Example:
  $(basename "$0") --bucket example-project-id-agents-models --project example-project-id --location us-central1
EOF
}

while [ "${#}" -gt 0 ]; do
  case "$1" in
    --bucket) BUCKET="${2:-}"; shift 2 ;;
    --project) PROJECT="${2:-}"; shift 2 ;;
    --location) LOCATION="${2:-}"; shift 2 ;;
    --storage-class) STORAGE_CLASS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "${BUCKET}" ] || [ -z "${PROJECT}" ]; then
  usage
  exit 2
fi

bucket_uri="gs://${BUCKET}"

if gcloud storage buckets describe "${bucket_uri}" --project="${PROJECT}" >/dev/null 2>&1; then
  echo "bucket exists: ${bucket_uri}"
else
  echo "creating bucket: ${bucket_uri}"
  gcloud storage buckets create "${bucket_uri}" \
    --project="${PROJECT}" \
    --location="${LOCATION}" \
    --default-storage-class="${STORAGE_CLASS}" \
    --uniform-bucket-level-access
fi

echo "enabling versioning: ${bucket_uri}"
gcloud storage buckets update "${bucket_uri}" --versioning --project="${PROJECT}"

echo "enforcing public access prevention: ${bucket_uri}"
gcloud storage buckets update "${bucket_uri}" \
  --public-access-prevention \
  --project="${PROJECT}"

echo "ok"
