#!/usr/bin/env bash
set -euo pipefail

# Creates a GPU node pool on a Standard GKE cluster.
#
# This is intentionally "thin": it prints and runs a single `gcloud container node-pools create`
# using a profile (a100|h100|b200) and a few flags. Adjust machine/accelerator values for
# your project/region if needed.
#
# Note: Some GPU families require reservations/quota. For those, add the appropriate
# reservation flags/labels and align your workload's nodeSelector accordingly.

PROFILE=""
CLUSTER=""
REGION=""
POOL_NAME=""
MIN_NODES="0"
MAX_NODES="3"
NUM_NODES="0"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --cluster CLUSTER --region REGION --profile (a100|h100|b200) [options]

Options:
  --name NAME        Node pool name (default: "\${profile}-8g")
  --min-nodes N      Autoscaler min nodes (default: ${MIN_NODES})
  --max-nodes N      Autoscaler max nodes (default: ${MAX_NODES})
  --num-nodes N      Initial nodes (default: ${NUM_NODES})

Examples:
  $(basename "$0") --cluster my-cluster --region us-central1 --profile a100
  $(basename "$0") --cluster my-cluster --region us-central1 --profile h100 --max-nodes 1
EOF
}

while [ "${#}" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --cluster) CLUSTER="${2:-}"; shift 2 ;;
    --region) REGION="${2:-}"; shift 2 ;;
    --name) POOL_NAME="${2:-}"; shift 2 ;;
    --min-nodes) MIN_NODES="${2:-}"; shift 2 ;;
    --max-nodes) MAX_NODES="${2:-}"; shift 2 ;;
    --num-nodes) NUM_NODES="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "${PROFILE}" ] || [ -z "${CLUSTER}" ] || [ -z "${REGION}" ]; then
  usage
  exit 2
fi

MACHINE_TYPE=""
ACCELERATOR_TYPE=""
ACCELERATOR_COUNT="8"

case "${PROFILE}" in
  a100)
    MACHINE_TYPE="a2-highgpu-8g"
    ACCELERATOR_TYPE="nvidia-tesla-a100"
    ;;
  h100)
    MACHINE_TYPE="a3-highgpu-8g"
    ACCELERATOR_TYPE="nvidia-h100-80gb"
    ;;
  b200)
    MACHINE_TYPE="a4-highgpu-8g"
    ACCELERATOR_TYPE="nvidia-b200"
    ;;
  *)
    echo "Unknown profile: ${PROFILE} (expected a100|h100|b200)" >&2
    exit 2
    ;;
esac

if [ -z "${POOL_NAME}" ]; then
  POOL_NAME="${PROFILE}-8g"
fi

set -x
gcloud container node-pools create "${POOL_NAME}" \
  --cluster="${CLUSTER}" \
  --region="${REGION}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT},gpu-driver-version=latest" \
  --num-nodes="${NUM_NODES}" \
  --enable-autoscaling --min-nodes="${MIN_NODES}" --max-nodes="${MAX_NODES}" \
  --enable-autorepair --enable-autoupgrade
