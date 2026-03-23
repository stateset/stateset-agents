#!/usr/bin/env bash
set -euo pipefail

# Bootstrap StateSet Agents resources on GKE.
#
# Creates:
# - Kubernetes namespace
# - Versioned GCS bucket for model artifacts (PAP enforced)
# - GCP Service Account (GSA) with bucket permissions
# - Kubernetes Service Account (KSA) annotated for Workload Identity
# - Workload Identity IAM binding between KSA and GSA
#
# Optional (if env vars are set):
# - Hugging Face token secret: hf-secret (key: hf_token)
# - W&B API key secret: wandb-secret (key: api_key)
#
# You can either specify --project/--location/--cluster explicitly or let the
# script infer them from the current kubectl context of the form:
#   gke_<project>_<location>_<cluster>

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--project PROJECT --location LOCATION --cluster CLUSTER] \\
    --namespace NAMESPACE \\
    --bucket BUCKET \\
    [--gsa GSA_NAME] [--ksa KSA_NAME] [--bucket-role ROLE]

Examples:
  $(basename "$0") \\
    --namespace stateset-agents \\
    --bucket example-project-id-agents-models

  $(basename "$0") \\
    --project example-project-id --location us-central1 --cluster example-gke-cluster \\
    --namespace stateset-agents \\
    --bucket example-project-id-agents-models \\
    --gsa agents-model-registry --ksa agents-model-registry

Notes:
- Set HUGGING_FACE_HUB_TOKEN to create hf-secret automatically.
- Set WANDB_API_KEY to create wandb-secret automatically.
EOF
}

PROJECT=""
LOCATION=""
CLUSTER=""
NAMESPACE=""
BUCKET=""
GSA="agents-model-registry"
KSA="agents-model-registry"
BUCKET_ROLE="roles/storage.objectAdmin"

while [ "${#}" -gt 0 ]; do
  case "$1" in
    --project) PROJECT="${2:-}"; shift 2 ;;
    --location) LOCATION="${2:-}"; shift 2 ;;
    --cluster) CLUSTER="${2:-}"; shift 2 ;;
    --namespace) NAMESPACE="${2:-}"; shift 2 ;;
    --bucket) BUCKET="${2:-}"; shift 2 ;;
    --gsa) GSA="${2:-}"; shift 2 ;;
    --ksa) KSA="${2:-}"; shift 2 ;;
    --bucket-role) BUCKET_ROLE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "${NAMESPACE}" ] || [ -z "${BUCKET}" ]; then
  usage
  exit 2
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found in PATH" >&2
  exit 2
fi
if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found in PATH" >&2
  exit 2
fi

context="$(kubectl config current-context 2>/dev/null || true)"
if [ -z "${PROJECT}" ] || [ -z "${LOCATION}" ] || [ -z "${CLUSTER}" ]; then
  if [[ "${context}" =~ ^gke_([^_]+)_([^_]+)_([^_]+)$ ]]; then
    PROJECT="${PROJECT:-${BASH_REMATCH[1]}}"
    LOCATION="${LOCATION:-${BASH_REMATCH[2]}}"
    CLUSTER="${CLUSTER:-${BASH_REMATCH[3]}}"
  fi
fi

if [ -z "${PROJECT}" ] || [ -z "${LOCATION}" ] || [ -z "${CLUSTER}" ]; then
  echo "Could not infer --project/--location/--cluster from kubectl context: ${context}" >&2
  echo "Provide --project, --location, and --cluster explicitly." >&2
  exit 2
fi

echo "context:   ${context}"
echo "project:   ${PROJECT}"
echo "location:  ${LOCATION}"
echo "cluster:   ${CLUSTER}"
echo "namespace: ${NAMESPACE}"
echo "bucket:    gs://${BUCKET}"
echo "gsa:       ${GSA}@${PROJECT}.iam.gserviceaccount.com"
echo "ksa:       ${KSA}"
echo ""

echo "1) Namespace"
if kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  echo "  ok: namespace exists"
else
  kubectl create namespace "${NAMESPACE}"
fi
echo ""

echo "2) Model Bucket"
bash "$(dirname "$0")/create_model_bucket.sh" --bucket "${BUCKET}" --project "${PROJECT}" --location "${LOCATION}"
echo ""

echo "3) Service Accounts (Workload Identity)"
gsa_email="${GSA}@${PROJECT}.iam.gserviceaccount.com"
if gcloud iam service-accounts describe "${gsa_email}" --project "${PROJECT}" >/dev/null 2>&1; then
  echo "  ok: gsa exists (${gsa_email})"
else
  gcloud iam service-accounts create "${GSA}" --project "${PROJECT}" --display-name "stateset-agents model registry"
fi

echo "  granting bucket role: ${BUCKET_ROLE}"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${gsa_email}" \
  --role="${BUCKET_ROLE}" \
  --project="${PROJECT}"

if kubectl -n "${NAMESPACE}" get serviceaccount "${KSA}" >/dev/null 2>&1; then
  echo "  ok: ksa exists (${KSA})"
else
  kubectl -n "${NAMESPACE}" create serviceaccount "${KSA}"
fi

kubectl -n "${NAMESPACE}" annotate serviceaccount "${KSA}" \
  iam.gke.io/gcp-service-account="${gsa_email}" \
  --overwrite

gcloud iam service-accounts add-iam-policy-binding "${gsa_email}" \
  --project "${PROJECT}" \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${PROJECT}.svc.id.goog[${NAMESPACE}/${KSA}]"
echo ""

echo "4) Optional Secrets"
if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  kubectl -n "${NAMESPACE}" create secret generic hf-secret \
    --from-literal=hf_token="${HUGGING_FACE_HUB_TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f -
  echo "  ok: hf-secret applied"
else
  echo "  skip: HUGGING_FACE_HUB_TOKEN not set (hf-secret not created)"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  kubectl -n "${NAMESPACE}" create secret generic wandb-secret \
    --from-literal=api_key="${WANDB_API_KEY}" \
    --dry-run=client -o yaml | kubectl apply -f -
  echo "  ok: wandb-secret applied"
else
  echo "  skip: WANDB_API_KEY not set (wandb-secret not created)"
fi

echo ""
echo "ok"

